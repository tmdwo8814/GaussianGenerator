"""Evaluate configured Gaussian budgets from one frozen Final checkpoint.

One context encoding per scene; reuse prior cameras and Final metrics when provided.
Entry point: scripts/eval_gaussian_control.sh. Never invokes Trainer.fit.
"""

from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import hydra
import torch
from lightning.pytorch import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from src.config import load_typed_root_config
from src.dataset.data_module import DataModule
from src.evaluation.gaussian_control_report import FRACTIONS, GaussianControlReport, retention_fractions
from src.evaluation.gaussian_control_reference import GaussianControlReference, scene_key
from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from src.global_cfg import set_cfg
from src.loss import get_losses
from src.misc.utils import get_overlap_tag
from src.model.decoder import get_decoder
from src.model.encoder import get_encoder
from src.model.encoder.heads.gaussian_control import (
    gaussian_statistics, redistribute_gaussians, select_slots, subset_gaussians,
)
from src.model.model_wrapper import ModelWrapper


@contextmanager
def elapsed_ms(device, timings, name):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    start = perf_counter()
    yield
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    timings[name] = (perf_counter() - start) * 1000


def prepare_config(cfg: DictConfig) -> Path:
    options = cfg.get('gaussian_control')
    if options is None or not options.get('checkpoint'):
        raise ValueError('Set +gaussian_control.checkpoint=/path/to/Final.ckpt')
    checkpoint = Path(options.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    encoder = cfg.model.encoder
    if encoder.gs_params_head_type != 'moment' or encoder.name != 'noposplat':
        raise ValueError('Gaussian control requires the two-view Final moment encoder (+experiment=re10k_moment)')
    moment = encoder.moment_decoder
    if not all(moment.get(key, False) for key in ('appearance_2d', 'separate_appearance', 'appearance_capacity')):
        raise ValueError('Final requires appearance_2d, separate_appearance and appearance_capacity')
    if cfg.train.distiller:
        raise ValueError('This analysis evaluates Final without a training distiller')
    if len(cfg.dataset) != 1 or cfg.data_loader.test.batch_size != 1 or cfg.trainer.num_nodes != 1:
        raise ValueError('Use one dataset, test batch size one, one node and one GPU')
    if cfg.checkpointing.load is not None:
        raise ValueError('Use gaussian_control.checkpoint instead of checkpointing.load')
    if not cfg.test.compute_scores:
        raise ValueError('Gaussian control requires test.compute_scores=true')
    if any(cfg.test[key] for key in ('save_image', 'save_video', 'save_compare')):
        raise ValueError('This entry point writes metrics and cameras; use save_image/video/compare=false')
    if cfg.test.align_pose and not options.get('reference_run') and cfg.test.pose_align_steps < 1:
        raise ValueError('Shared Final pose alignment requires pose_align_steps >= 1')
    if int(options.get('max_scenes', 0)) < 0:
        raise ValueError('gaussian_control.max_scenes must be nonnegative (0 = all scenes)')
    retention_fractions(options.get('fractions'))
    if options.get('reference_run'):
        reference = Path(options.reference_run).expanduser().resolve()
        for filename in ('metadata.json', 'per_scene.jsonl', 'run_status.txt'):
            if not (reference / filename).is_file():
                raise FileNotFoundError(reference / filename)

    cfg.mode = 'test'
    cfg.wandb.mode = 'disabled'
    # The full checkpoint supplies all learned weights; MASt3R initialization
    # is unnecessary and must never substitute for the trained Final weights.
    encoder.pretrained_weights = ''
    dataset_name, dataset = next(iter(cfg.dataset.items()))
    index = Path(options.get('index_path') or cfg.trainer.get('auto_eval_index_path')
                 or f'assets/evaluation_index_{dataset_name}.json').expanduser().resolve()
    if not index.is_file():
        raise FileNotFoundError(index)
    for root in dataset.roots:
        if not Path(root).is_dir():
            raise FileNotFoundError(f'Dataset root does not exist: {root}')
    dataset.view_sampler = {'name': 'evaluation', 'index_path': str(index), 'num_context_views': 2}
    return checkpoint


def load_reference_camera(reference, entry, batch, device):
    camera = torch.load(reference.camera_path(entry), map_location='cpu', weights_only=True)
    (scene,) = batch['scene']
    expected = scene_key(scene, batch['context']['index'][0].tolist(), batch['target']['index'][0].tolist())
    actual = scene_key(camera['scene'], camera['context_indices'].reshape(-1).tolist(),
                       camera['target_indices'].reshape(-1).tolist())
    if actual != expected or camera['protocol'] != reference.metadata['camera_protocol']:
        raise ValueError('Saved camera identity/protocol does not match this evaluation scene')
    result = {}
    for key in ('extrinsics', 'intrinsics', 'near', 'far'):
        saved, current = camera[key], batch['target'][key]
        if saved.shape != current.shape or not bool(torch.isfinite(saved).all()):
            raise ValueError(f'Invalid saved camera {key} for {scene}')
        if key != 'extrinsics' and not torch.allclose(saved, current.cpu(), rtol=1e-6, atol=1e-7):
            raise ValueError(f'Saved camera {key} differs from the current dataset for {scene}')
        result[key] = saved.to(device=device)
    return result


class GaussianControlModel(ModelWrapper):
    def _eval_step(self):
        return self.analysis_step

    def on_test_start(self):
        if self.trainer.world_size != 1:
            raise ValueError('Use one GPU; every scene evaluates all configured conditions together')
        self.eval()
        self.requires_grad_(False)
        self.analysis_fractions = getattr(self, 'analysis_fractions', FRACTIONS)
        self.analysis_reference = getattr(self, 'analysis_reference', None)
        self.reference_seen = set()
        self.report = GaussianControlReport(
            self.analysis_directory, self.analysis_metadata, fractions=self.analysis_fractions,
            reuse_baseline=self.analysis_reference is not None,
        )
        self.camera_directory = self.analysis_directory / 'cameras'
        self.camera_directory.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        batch = self.data_shim(batch)
        if batch['target']['image'].shape[0] != 1:
            raise ValueError('Analysis expects one scene per batch')
        (scene,) = batch['scene']
        overlap = get_overlap_tag(batch['context']['overlap'][0])
        shared, cache = {}, {}
        reference_entry = None
        if self.analysis_reference is not None:
            key, reference_entry = self.analysis_reference.lookup(
                scene, batch['context']['index'][0].tolist(), batch['target']['index'][0].tolist(),
            )
            if key in self.reference_seen:
                raise ValueError(f'Duplicate scene/frame identity in the new evaluation: {key}')
            if reference_entry['overlap_group'] != overlap:
                raise ValueError(f'Reference overlap group differs for {scene}')
            self.reference_seen.add(key)
            with elapsed_ms(self.device, shared, 'camera_reuse_ms'):
                target_camera = load_reference_camera(self.analysis_reference, reference_entry, batch, self.device)
            shared['shared_pose_alignment_ms'] = 0.

        with elapsed_ms(self.device, shared, 'shared_encoder_ms'):
            baseline = self.encoder(batch['context'], self.analysis_step, analysis_cache=cache)
        if not cache:
            raise RuntimeError('Final did not export its original allocation')

        if reference_entry is None:
            with elapsed_ms(self.device, shared, 'shared_pose_alignment_ms'):
                camera = {}
                if self.test_cfg.align_pose:
                    self.test_step_align(batch, baseline, camera_dump=camera)
                    extrinsics = camera['extrinsics']
                else:
                    extrinsics = batch['target']['extrinsics'].detach().clone()
            target_camera = {key: batch['target'][key] for key in ('intrinsics', 'near', 'far')}
            target_camera['extrinsics'] = extrinsics
        extrinsics = target_camera['extrinsics']
        camera_path = self.camera_directory / f'{batch_idx:06d}.pt'
        torch.save({
            'scene': scene, 'protocol': self.analysis_metadata['camera_protocol'],
            'context_indices': batch['context']['index'].cpu(),
            'target_indices': batch['target']['index'].cpu(),
            **{key: value.cpu() for key, value in target_camera.items()},
        }, camera_path)
        records = {}

        def evaluate(name, gaussians, slots, mass, timings, fallback=0.):
            if gaussians.means.shape[1] != len(slots):
                raise RuntimeError('Renderer must receive exactly L Gaussians')
            height, width = batch['target']['image'].shape[-2:]
            with elapsed_ms(self.device, timings, 'render_ms'):
                output = self.decoder(
                    gaussians, extrinsics, target_camera['intrinsics'],
                    target_camera['near'], target_camera['far'], (height, width),
                )
            target, predicted = batch['target']['image'][0], output.color[0]
            stats = gaussian_statistics(gaussians, cache, slots, mass, fallback)
            records[name] = {
                'psnr': float(compute_psnr(target, predicted).mean()),
                'ssim': float(compute_ssim(target, predicted).mean()),
                'lpips': float(compute_lpips(target, predicted).mean()),
                **stats, **shared, **timings,
                'render_ms_per_target': timings['render_ms'] / len(target),
            }

        if reference_entry is None:
            all_slots = torch.arange(len(cache['mass']), device=self.device)
            evaluate('final_100', baseline, all_slots, cache['mass'],
                     {'additional_decode_ms': 0., 'slot_selection_ms': 0.})
        for fraction in self.analysis_fractions:
            timings = {}
            with elapsed_ms(self.device, timings, 'slot_selection_ms'):
                slots = select_slots(cache['mass'], fraction)
            with elapsed_ms(self.device, timings, 'additional_decode_ms'):
                pruned = subset_gaussians(baseline, slots)
            suffix = f'{round(fraction * 100):03d}'
            evaluate(f'prune_{suffix}', pruned, slots, cache['mass'][slots], dict(timings))
            del pruned
            with elapsed_ms(self.device, timings, 'additional_decode_ms'):
                regenerated, mass, fallback = redistribute_gaussians(self.encoder.gaussian_decoder, cache, slots)
            evaluate(f'redistribute_{suffix}', regenerated, slots, mass, dict(timings), fallback)
            del regenerated

        self.report.add_scene({
            'scene': scene, 'context_indices': batch['context']['index'][0].tolist(),
            'target_indices': batch['target']['index'][0].tolist(),
            'camera_file': str(camera_path.relative_to(self.analysis_directory)),
        }, overlap, records, reference_final=(reference_entry['reference_final_100']
                                            if reference_entry is not None else None))
        if batch_idx % 10 == 0:
            print(f'[GaussianControl] {batch_idx + 1} scenes; {scene}; '
                  f'{len(records)} evaluated conditions; Final baseline '
                  f'{"reused" if reference_entry is not None else "evaluated"}', flush=True)

    def on_test_end(self):
        if self.analysis_reference is not None and not self.analysis_metadata.get('max_scenes', 0):
            self.analysis_reference.require_complete_coverage(self.reference_seen)
        print(self.report.finish(), flush=True)


@hydra.main(version_base=None, config_path='../config', config_name='main')
def evaluate(cfg_dict: DictConfig):
    checkpoint_path = prepare_config(cfg_dict)
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    seed_everything(cfg.seed, workers=True)
    directory = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    # Fail before allocating GPU/model memory if this output contains a prior run.
    if (directory / 'per_scene.jsonl').exists():
        raise FileExistsError(f'Use a new hydra.run.dir: {directory}')
    OmegaConf.save(cfg_dict, directory / 'analysis_config.yaml')

    encoder, visualizer = get_encoder(cfg.model.encoder)
    model = GaussianControlModel(cfg.optimizer, cfg.test, cfg.train, encoder, visualizer,
                                 get_decoder(cfg.model.decoder), get_losses(cfg.loss), step_tracker=None)
    # Strict restore of the ENTIRE Lightning state, including backbone/DPT/SH.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.analysis_step = int(checkpoint.get('global_step', 0))
    model.analysis_directory = directory
    model.analysis_fractions = retention_fractions(cfg_dict.gaussian_control.get('fractions'))
    dataset_name, dataset = next(iter(cfg_dict.dataset.items()))
    protocol = 'shared_final_pose' if cfg.test.align_pose else 'dataset_pose'
    model.analysis_metadata = {
        'checkpoint': str(checkpoint_path), 'checkpoint_step': model.analysis_step,
        'checkpoint_load': 'strict=True; entire model state',
        'dataset': dataset_name, 'dataset_roots': list(dataset.roots),
        'evaluation_index': dataset.view_sampler.index_path,
        'input_image_shape': list(dataset.input_image_shape),
        'camera_protocol': protocol, 'pose_align_steps': cfg.test.pose_align_steps if cfg.test.align_pose else 0,
        'seed': cfg.seed, 'selection': 'original incoming mass; stable descending rank; lower ID wins ties',
        'max_scenes': int(cfg_dict.gaussian_control.get('max_scenes', 0)),
        'timing': 'synchronized wall time, includes initial warmup; milliseconds per scene',
        'model_parameters_frozen': True,
        'fractions': list(model.analysis_fractions),
    }
    reference_run = cfg_dict.gaussian_control.get('reference_run')
    model.analysis_reference = (GaussianControlReference(Path(reference_run), model.analysis_metadata)
                                if reference_run else None)
    model.analysis_metadata['reference_run'] = (str(model.analysis_reference.directory)
                                               if model.analysis_reference is not None else None)
    model.analysis_metadata['final_evaluation_reused'] = model.analysis_reference is not None
    if model.analysis_reference is not None:
        model.analysis_metadata['reference_pose_align_steps'] = model.analysis_reference.metadata.get('pose_align_steps', 0)
        model.analysis_metadata['pose_align_steps'] = 0
    del checkpoint
    model.freeze()
    max_scenes = model.analysis_metadata['max_scenes']
    trainer = Trainer(
        accelerator='gpu', devices=1, num_nodes=1, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, inference_mode=False,
        limit_test_batches=max_scenes if max_scenes else 1.0,
    )
    data = DataModule(cfg.dataset, cfg.data_loader, step_tracker=None, global_rank=0)
    trainer.test(model, datamodule=data)


if __name__ == '__main__':
    evaluate()
