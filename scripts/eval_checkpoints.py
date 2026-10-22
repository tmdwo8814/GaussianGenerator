"""Evaluate selected saved checkpoints and append auto_eval to an existing run.

Run from the repository root with the same environment as train_re10k.sh.
Only main() imports the GPU/model stack, so --dry-run needs OmegaConf alone.
"""

import argparse
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from omegaconf import OmegaConf


def parse_run_path(value: str) -> tuple[str, str, str]:
    """Accept entity/project/id, wandb.ai URLs, or Forge's /wandb/... URLs."""
    value = value.strip().rstrip('/')
    if value.startswith(('wandb.ai/', 'forge.coreweave.com/')):
        value = 'https://' + value
    if '://' in value:
        parsed = urlsplit(value)
        if parsed.scheme != 'https' or parsed.hostname not in ('wandb.ai', 'forge.coreweave.com'):
            raise ValueError('Use a W&B/Forge run link or entity/project/run_id.')
        parts = parsed.path.strip('/').split('/')
        if parsed.hostname == 'forge.coreweave.com' and parts[:1] == ['wandb']:
            parts = parts[1:]
        if len(parts) != 4 or parts[2] != 'runs':
            raise ValueError('Expected a run URL ending in /entity/project/runs/run_id.')
        parts = [parts[0], parts[1], parts[3]]
    else:
        parts = value.split('/')
    if len(parts) != 3 or any(not re.fullmatch(r'[A-Za-z0-9_.-]+', p) or p in ('.', '..') for p in parts):
        raise ValueError('Expected entity/project/run_id.')
    return tuple(parts)


def select_checkpoints(directory: Path, steps: list[int]) -> list[tuple[int, Path]]:
    if not steps or any(step <= 0 for step in steps) or len(steps) != len(set(steps)):
        raise ValueError('Steps must be distinct positive integers.')
    candidates = {step: [] for step in steps}
    for path in directory.glob('*.ckpt'):
        match = re.search(r'step[_=](\d+)(?:\D|$)', path.name)
        if match and int(match[1]) in candidates:
            candidates[int(match[1])].append(path.resolve())
    selected = []
    for step in sorted(steps):
        paths = candidates[step]
        if len(paths) != 1:
            raise ValueError(f'Expected exactly one checkpoint for step {step} in {directory}; found {paths}')
        selected.append((step, paths[0]))
    return selected


def load_eval_config(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'Saved training config not found: {path}. Supply --config explicitly.')
    config = OmegaConf.load(path)
    config.mode = 'test'
    for name, dataset in config.dataset.items():
        views = dataset.view_sampler.get('num_context_views', 2)
        index = config.trainer.get('auto_eval_index_path') or f'assets/evaluation_index_{name}.json'
        if not Path(index).is_file():
            raise FileNotFoundError(f'Evaluation index not found: {index}. Run from the repository root.')
        dataset.view_sampler = dict(name='evaluation', index_path=str(index), num_context_views=views)
        for root in dataset.roots:
            if not (Path(root) / 'test').is_dir():
                raise FileNotFoundError(f'Test dataset missing at {Path(root) / "test"} (saved config).')
    for kind in ('image', 'video', 'compare'):
        config.test[f'save_{kind}'] = config.trainer.get(f'auto_eval_save_{kind}', False)
    if not config.test.compute_scores:
        raise ValueError('Saved config has test.compute_scores=false; no auto_eval scores would be produced.')
    if config.data_loader.test.batch_size != 1:
        raise ValueError('The existing test_step requires data_loader.test.batch_size=1.')
    return config


def logger_options(run_path: tuple[str, str, str], output_dir: Path) -> dict:
    entity, project, run_id = run_path
    # No name, tags, notes or config: preserve the original training metadata.
    return dict(entity=entity, project=project, id=run_id, resume='must',
                mode='online', log_model=False, save_dir=str(output_dir))


def evaluation_result(model, step: int) -> dict:
    """Persist the same already-all-reduced sums/counts used by ModelWrapper."""
    count = int(model._test_metric_count.item())
    if count == 0:
        raise RuntimeError(f'No test samples at step {step}; evaluation is incomplete.')
    result = {'auto_eval/ckpt_step': step}
    groups = [('overall', model._test_metric_sums, model._test_metric_count)]
    groups.extend((tag, model._test_overlap_sums[tag], model._test_overlap_counts[tag])
                  for tag in model._test_overlap_tags)
    for tag, sums, count_tensor in groups:
        count = int(count_tensor.item())
        if count:
            result[f'auto_eval/{tag}/num_samples'] = count
            for key in model._test_metric_names:
                result[f'auto_eval/{tag}/{key.removesuffix("_ours")}'] = (sums[key] / count_tensor).item()
    return result


def evaluate_checkpoints(trainer, model, data_module, checkpoints, output_dir: Path):
    for step, checkpoint in checkpoints:
        step_dir = output_dir / f'step_{step}'
        model.test_cfg.output_path = step_dir
        model._auto_eval_step = step
        model._auto_eval_log_prefix = 'auto_eval'
        if trainer.is_global_zero:
            print(f'[RemainingEval] Evaluating step {step}: {checkpoint}', flush=True)
        # This is the same test path as src.main's post-training evaluation.
        # Lightning restores model weights only for test; optimizer state is unnecessary.
        trainer.test(model, datamodule=data_module, ckpt_path=str(checkpoint))
        if trainer.is_global_zero:
            result = evaluation_result(model, step)
            step_dir.mkdir(parents=True, exist_ok=True)
            temporary = step_dir / 'metrics.json.tmp'
            temporary.write_text(json.dumps({'checkpoint': str(checkpoint), 'metrics': result},
                                            indent=2, allow_nan=False), encoding='utf-8')
            temporary.replace(step_dir / 'metrics.json')
            print(f'[RemainingEval] Completed step {step}; saved {step_dir / "metrics.json"}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='Training output folder containing checkpoints/.')
    parser.add_argument('--wandb-run', required=True, help='Existing entity/project/run_id or run URL.')
    parser.add_argument('--steps', nargs='+', type=int, default=[80000, 100000, 120000, 140000])
    parser.add_argument('--config', type=Path, help='Defaults to RUN_DIR/.hydra/config.yaml; no current-config fallback.')
    parser.add_argument('--dry-run', action='store_true', help='Check paths/settings without GPUs or W&B writes.')
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    run_path = parse_run_path(args.wandb_run)
    config_path = args.config or run_dir / '.hydra/config.yaml'
    config = load_eval_config(config_path)
    checkpoints = select_checkpoints(run_dir / 'checkpoints', args.steps)
    output_dir = run_dir / 'remaining_eval'
    print(f'[RemainingEval] Config: {config_path}', flush=True)
    print(f'[RemainingEval] W&B: {"/".join(run_path)} (resume=must)', flush=True)
    print(f'[RemainingEval] Pose alignment: {config.test.align_pose}; steps: {config.test.pose_align_steps}', flush=True)
    for step, path in checkpoints:
        print(f'  {step}: {path}', flush=True)
    if args.dry_run:
        print('[RemainingEval] Dry run OK; no GPU/W&B connection was made.')
        return

    # Reuse the exact model/data/loss factories and import hooks of the training entry point.
    from src.main import (DataModule, ModelWrapper, Trainer, WandbLogger, get_decoder,
                          get_distiller, get_encoder, get_losses, load_typed_root_config,
                          set_cfg, torch, wandb)

    if wandb.run is not None:
        raise RuntimeError('An active W&B run already exists in this process; use a fresh process.')
    if not torch.cuda.is_available():
        raise RuntimeError('Evaluation requires the CUDA renderer. Run this script on the training server.')
    set_cfg(config)
    cfg = load_typed_root_config(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = WandbLogger(**logger_options(run_path, output_dir))
    trainer = Trainer(
        accelerator='gpu', devices='auto', num_nodes=cfg.trainer.num_nodes,
        strategy='ddp_find_unused_parameters_true' if torch.cuda.device_count() > 1 else 'auto',
        logger=logger, enable_checkpointing=False, enable_progress_bar=False,
        inference_mode=not cfg.test.align_pose,
    )
    torch.manual_seed(cfg.seed + trainer.global_rank)
    encoder, visualizer = get_encoder(cfg.model.encoder)
    distiller = get_distiller(cfg.train.distiller).eval() if cfg.train.distiller else None
    model = ModelWrapper(cfg.optimizer, cfg.test, cfg.train, encoder, visualizer,
                         get_decoder(cfg.model.decoder), get_losses(cfg.loss),
                         step_tracker=None, distiller=distiller)
    model.strict_loading = True
    class EvaluationDataModule(DataModule):
        def setup(self, stage=None):
            # Unlike post-fit auto_eval, this process has not initialized DDP
            # when the data module is constructed. Set the rank AFTER launch,
            # before test_dataloader attaches it to the iterable dataset.
            self.global_rank = self.trainer.global_rank
            super().setup(stage)

    data_module = EvaluationDataModule(cfg.dataset, cfg.data_loader, step_tracker=None,
                                      global_rank=trainer.global_rank)
    # WandbLogger is lazy; ModelWrapper.on_test_start opens it only on rank zero.
    # That hook also defines auto_eval/* against auto_eval/ckpt_step. Never pass
    # an old checkpoint step as wandb.log(step=...), which would drop old steps.
    try:
        evaluate_checkpoints(trainer, model, data_module, checkpoints, output_dir)
    except BaseException:
        if trainer.is_global_zero and wandb.run is not None:
            wandb.finish(exit_code=1)
        raise
    else:
        if trainer.is_global_zero and wandb.run is not None:
            wandb.finish()


if __name__ == '__main__':
    main()
