"""Cache one frozen Final++ prediction per scene, then evaluate count controls.

Run via scripts/eval_gaussian_control.sh. All stages are resumable. Cache-only
evaluation loads the small frozen moment decoder, never the MASt3R backbone.
"""

from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import platform
import random
import socket

import hydra
from hydra.utils import to_absolute_path
import numpy as np
import scipy
from omegaconf import DictConfig, OmegaConf
import torch

from .gaussian_control.common import (
    METHODS, gaussian_dict, select_gaussians, synchronized_time, target_count,
    to_device, uniform_pixel_indices,
)
from .model.types import Gaussians


FORMAT_VERSION = 1


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(to_device(value, torch.device('cpu')), temporary)
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def freeze(module):
    return module.eval().requires_grad_(False)


def scene_key(scene, entry):
    return digest({'scene': scene, 'context': list(entry['context']), 'target': list(entry['target'])})


def cache_manifest(cfg, root):
    """Validate checkpoint/config/index identity before reusing any saved scene."""
    if list(cfg.dataset) != ['re10k']:
        raise ValueError('This control evaluation expects only RE10K')
    sampler = cfg.dataset.re10k.view_sampler
    if sampler.name != 'evaluation' or sampler.num_context_views != 2:
        raise ValueError('Use the RE10K evaluation sampler with two context views')
    index = read_json(to_absolute_path(str(sampler.index_path)))
    selected = {scene: entry for scene, entry in sorted(index.items()) if entry is not None}
    if cfg.control.max_scenes is not None:
        if cfg.control.max_scenes < 1:
            raise ValueError('control.max_scenes must be positive or null')
        selected = dict(list(selected.items())[:cfg.control.max_scenes])
    if not selected:
        raise ValueError('Evaluation index contains no selected scenes')
    checkpoint = Path(to_absolute_path(str(cfg.checkpointing.load)))
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Final++ checkpoint not found: {checkpoint}')
    identity = {
        'format_version': FORMAT_VERSION, 'checkpoint_sha256': file_digest(checkpoint),
        'encoder_sources': {path.relative_to(Path(__file__).parent).as_posix(): file_digest(path)
                            for path in sorted((Path(__file__).parent / 'model/encoder').rglob('*.py'))},
        'alignment_source': file_digest(Path(__file__)),
        'model': OmegaConf.to_container(cfg.model, resolve=True),
        'dataset': OmegaConf.to_container(cfg.dataset, resolve=True),
        'index': selected, 'test': OmegaConf.to_container(cfg.test, resolve=True),
        'loss': OmegaConf.to_container(cfg.loss, resolve=True),
        'seed': cfg.seed,
    }
    signature = digest(identity)
    path = root / 'manifest.json'
    if path.exists():
        manifest = read_json(path)
        if manifest['signature'] != signature:
            raise RuntimeError('Cache checkpoint/config/index differs; use a different control.cache_dir')
    else:
        manifest = {
            'signature': signature, 'identity': identity, 'checkpoint': str(checkpoint),
            'scenes': {scene: {'key': scene_key(scene, entry), 'entry': entry}
                       for scene, entry in selected.items()},
            'complete': False,
        }
        save_json(path, manifest)
    save_json(root / 'selected_index.json', selected)
    return manifest, checkpoint


def load_encoder_weights(encoder, checkpoint):
    value = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(value, dict):
        raise ValueError('Expected a Lightning or encoder state dictionary')
    state = value.get('state_dict', value)
    if any(key.startswith('encoder.') for key in state):
        state = {key[len('encoder.'):]: tensor for key, tensor in state.items() if key.startswith('encoder.')}
    # Never silently evaluate a partially initialized decoder/backbone.
    encoder.load_state_dict(state, strict=True)
    return int(value.get('global_step', 140000))


def align_cameras(renderer, losses, batch, gaussians, cfg, step):
    extrinsics = batch['target']['extrinsics'].detach().clone()
    if not cfg.align_pose:
        return extrinsics
    from .misc.cam_utils import update_pose
    b, views = extrinsics.shape[:2]
    shape = batch['target']['image'].shape[-2:]
    with torch.enable_grad():
        rotation = torch.nn.Parameter(torch.zeros(b, views, 3, device=extrinsics.device))
        translation = torch.nn.Parameter(torch.zeros_like(rotation))
        optimizer = torch.optim.Adam([
            {'params': [rotation], 'lr': cfg.rot_opt_lr},
            {'params': [translation], 'lr': cfg.trans_opt_lr},
        ])
        for _ in range(cfg.pose_align_steps):
            optimizer.zero_grad(set_to_none=True)
            output = renderer(gaussians, extrinsics, batch['target']['intrinsics'],
                              batch['target']['near'], batch['target']['far'], shape,
                              cam_rot_delta=rotation, cam_trans_delta=translation)
            loss = sum(fn.forward(output, batch, gaussians, step) for fn in losses)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite Final++ camera alignment loss')
            loss.backward()
            with torch.no_grad():
                optimizer.step()
                extrinsics = update_pose(
                    cam_rot_delta=rotation.reshape(-1, 3),
                    cam_trans_delta=translation.reshape(-1, 3),
                    extrinsics=extrinsics.reshape(-1, 4, 4),
                ).reshape(b, views, 4, 4)
                rotation.zero_()
                translation.zero_()
    return extrinsics.detach()


def validate_scene(payload, manifest, scene):
    expected = manifest['scenes'][scene]
    if payload['signature'] != manifest['signature'] or payload['key'] != expected['key']:
        raise RuntimeError(f'Cache identity mismatch for {scene}')
    for field, label in [('context_indices', 'context'), ('target_indices', 'target')]:
        if payload[field] != list(expected['entry'][label]):
            raise RuntimeError(f'Cached frame indices differ for {scene}: {label}')


def build_cache(cfg_dict, root, device):
    from .config import load_typed_root_config
    from .dataset.data_module import DataModule
    from .global_cfg import set_cfg
    from .loss import get_losses
    from .model.decoder import get_decoder
    from .model.encoder import get_encoder

    manifest, checkpoint = cache_manifest(cfg_dict, root)
    pending = {scene for scene, item in manifest['scenes'].items()
               if not (root / 'scenes' / f"{item['key']}.pt").exists()}
    if not pending and (root / 'decoder.pt').exists():
        manifest['complete'] = True
        save_json(root / 'manifest.json', manifest)
        print(f'[Cache] Reusing {len(manifest["scenes"])} saved scenes; no inference', flush=True)
        return manifest
    local_cfg = OmegaConf.create(OmegaConf.to_container(cfg_dict, resolve=True))
    local_cfg.dataset.re10k.view_sampler.index_path = str(root / 'selected_index.json')
    cfg = load_typed_root_config(local_cfg)
    set_cfg(local_cfg)
    if cfg.data_loader.test.batch_size != 1:
        raise ValueError('Control cache requires test.batch_size=1')
    encoder, _ = get_encoder(cfg.model.encoder)
    step = load_encoder_weights(encoder, checkpoint)
    encoder = freeze(encoder.to(device))
    head = encoder.gaussian_decoder
    if not head.cfg.appearance_capacity:
        raise ValueError('Expected the Final++ moment/appearance-capacity decoder')
    save_torch(root / 'decoder.pt', {
        'signature': manifest['signature'], 'cfg': asdict(head.cfg),
        'sh_degree': cfg.model.encoder.gaussian_adapter.sh_degree,
        'state_dict': head.state_dict(), 'render_cfg': asdict(cfg.model.decoder),
    })
    renderer = freeze(get_decoder(cfg.model.decoder).to(device))
    losses = freeze(torch.nn.ModuleList(get_losses(cfg.loss)).to(device))
    loader = DataModule(cfg.dataset, cfg.data_loader, step_tracker=None, global_rank=0).test_dataloader()
    if isinstance(loader, list):
        raise ValueError('Expected one RE10K test loader')
    shim = encoder.get_data_shim()
    for batch in loader:
        scene, = batch['scene']
        if scene not in pending:
            continue
        # Target RGB remains unnormalized; the shim modifies only context RGB.
        batch = shim(to_device(batch, device))
        support = {}
        with torch.no_grad():
            gaussians = encoder(batch['context'], step, control_dump=support)
        target = dict(batch['target'])
        target['extrinsics'] = align_cameras(renderer, losses, batch, gaussians, cfg.test, step)
        payload = {
            'signature': manifest['signature'], 'key': manifest['scenes'][scene]['key'],
            'scene': scene, 'context_indices': batch['context']['index'][0].tolist(),
            'target_indices': batch['target']['index'][0].tolist(),
            'overlap': float(batch['context']['overlap'].reshape(-1)[0]),
            'gaussians': gaussian_dict(gaussians), 'support': support, 'target': target,
        }
        validate_scene(payload, manifest, scene)
        path = root / 'scenes' / f"{payload['key']}.pt"
        save_torch(path, payload)
        pending.remove(scene)
        print(f'[Cache] {scene}: {gaussians.means.shape[1]} Gaussians, '
              f'{path.stat().st_size / 1024**2:.1f} MiB saved; {len(pending)} remaining', flush=True)
        del payload, gaussians, support, batch, target
    if pending:
        raise RuntimeError(f'Dataset did not yield {len(pending)} selected scenes: {sorted(pending)[:12]}')
    manifest['complete'] = True
    save_json(root / 'manifest.json', manifest)
    return manifest


def summarize(rows, methods, percentages):
    summary = []
    for method in methods:
        for percent in percentages:
            candidates = [r for r in rows if r['method'] == method and r['percent'] == percent]
            for overlap in ('overall', 'small', 'medium', 'large'):
                subset = [r for r in candidates if overlap == 'overall' or r['overlap_tag'] == overlap]
                if not subset:
                    continue
                row = {'method': method, 'percent': percent, 'overlap': overlap, 'scenes': len(subset)}
                for key in ('gaussians', 'actual_percent', 'psnr', 'ssim', 'lpips',
                            'postprocess_ms', 'render_ms_per_view'):
                    row[key] = float(np.mean([r[key] for r in subset]))
                row['postprocess_ms_p50'] = float(np.median([r['postprocess_ms'] for r in subset]))
                row['postprocess_ms_p95'] = float(np.percentile([r['postprocess_ms'] for r in subset], 95))
                summary.append(row)
    return summary


def write_reports(output, rows, methods, percentages, expected_scenes):
    summary = summarize(rows, methods, percentages)
    save_json(output / 'summary.json', summary)
    (output / 'per_scene.jsonl').write_text(
        ''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows), encoding='utf-8',
    )
    header = ('method percent overlap scenes Gaussians actual_percent PSNR SSIM LPIPS '
              'post_ms post_p50_ms post_p95_ms render_ms_per_view')
    def line(row):
        return (f"{row['method']:12s} {row['percent']:3d} {row['overlap']:7s} {row['scenes']:5d} "
                f"{row['gaussians']:10.1f} {row['actual_percent']:8.4f} "
                f"{row['psnr']:.6f} {row['ssim']:.6f} {row['lpips']:.6f} "
                f"{row['postprocess_ms']:.4f} {row['postprocess_ms_p50']:.4f} "
                f"{row['postprocess_ms_p95']:.4f} {row['render_ms_per_view']:.4f}")
    preamble = (f'Expected scenes per condition: {expected_scenes}\n'
                'Metrics/time: equal scene means; metrics average target views first.\n'
                'Postprocess: synchronized wall time from resident inputs to resident final Gaussians.\n'
                'Excludes cache I/O, backbone inference, camera alignment, rendering and metrics.\n'
                'NanoGS CPU/transfer time is included; other controls run on the GPU.\n'
                '100% is an identity shared by all methods; postprocess_ms=0.\n')
    (output / 'summary.txt').write_text(preamble + header + '\n' + '\n'.join(map(line, summary)) + '\n', encoding='utf-8')
    for method in methods:
        for percent in percentages:
            selected = [row for row in summary if row['method'] == method and row['percent'] == percent]
            (output / f'{method}_{percent:03d}.txt').write_text(
                preamble + header + '\n' + '\n'.join(map(line, selected)) + '\n', encoding='utf-8',
            )
    final = [row for row in summary if row['method'] == methods[0] and row['percent'] == 100]
    if final:
        (output / 'final_100.txt').write_text(preamble + header + '\n' + '\n'.join(map(line, final)) + '\n', encoding='utf-8')


@torch.no_grad()
def evaluate(cfg, root, output, device):
    from .evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
    from .gaussian_control.nanogs import UPSTREAM_COMMIT, merge
    from .gaussian_control.redistribute import redistribute
    from .misc.utils import get_overlap_tag
    from .model.decoder import get_decoder
    from .model.decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg
    from .model.encoder.heads.moment_gaussian_decoder import MomentDecoderCfg, MomentGaussianDecoder

    manifest = read_json(root / 'manifest.json')
    if not manifest.get('complete'):
        raise RuntimeError('Cache is incomplete; run control.stage=cache or all first')
    bundle = torch.load(root / 'decoder.pt', map_location='cpu', weights_only=True)
    if bundle['signature'] != manifest['signature']:
        raise RuntimeError('Cached decoder and scenes belong to different runs')
    head = MomentGaussianDecoder(MomentDecoderCfg(**bundle['cfg']), bundle['sh_degree'])
    head.load_state_dict(bundle['state_dict'], strict=True)
    head = freeze(head.to(device))
    renderer = freeze(get_decoder(DecoderSplattingCUDACfg(**bundle['render_cfg'])).to(device))
    percentages = list(map(int, cfg.control.percentages))
    methods = list(cfg.control.methods)
    settings = {
        'format_version': FORMAT_VERSION, 'cache_signature': manifest['signature'],
        'nanogs_commit': UPSTREAM_COMMIT, 'opacity_preprune': False, 'nanogs_k': 16,
        'timing_repeats': cfg.control.timing_repeats, 'warmup_runs': cfg.control.warmup_runs,
        'render_chunk_size': cfg.control.render_chunk_size,
        'control_sources': {name: file_digest(Path(__file__).parent / name) for name in (
            'gaussian_control/common.py', 'gaussian_control/redistribute.py',
            'gaussian_control/nanogs.py', 'eval_gaussian_control.py',
        )},
    }
    signature = digest(settings)
    metadata_path = output / 'metadata.json'
    if metadata_path.exists() and read_json(metadata_path)['signature'] != signature:
        raise RuntimeError('Result configuration differs; use a different control.output_dir')
    save_json(metadata_path, {
        'signature': signature, 'settings': settings, 'methods': methods, 'percentages': percentages,
        'gpu': torch.cuda.get_device_name(device), 'torch': torch.__version__,
        'numpy': np.__version__, 'scipy': scipy.__version__, 'python': platform.python_version(),
        'cuda': torch.version.cuda, 'host': socket.gethostname(), 'platform': platform.platform(),
        'cpu_threads': torch.get_num_threads(), 'cache_dir': str(root),
    })
    rows = []
    warmed = set()
    completed = 0
    try:
        for scene, item in manifest['scenes'].items():
            result_path = output / 'scenes' / f"{item['key']}.json"
            previous = read_json(result_path) if result_path.exists() else {'signature': signature, 'rows': []}
            if previous['signature'] != signature:
                raise RuntimeError(f'Stale scene results for {scene}')
            existing = {(r['method'], r['percent']): r for r in previous['rows']}
            requested = [(method, p) for p in percentages for method in methods]
            if all(key in existing for key in requested):
                rows.extend(existing[key] for key in requested)
                completed += 1
                continue
            path = root / 'scenes' / f"{item['key']}.pt"
            payload = torch.load(path, map_location='cpu', weights_only=True)
            validate_scene(payload, manifest, scene)
            payload = to_device(payload, device)  # outside every postprocess timer
            original = Gaussians(**payload['gaussians'])
            total = original.means.shape[1]
            target = payload['target']
            def render(gaussians):
                chunks = []
                views = target['image'].shape[1]
                for start in range(0, views, cfg.control.render_chunk_size):
                    stop = start + cfg.control.render_chunk_size
                    chunks.append(renderer(
                        gaussians, target['extrinsics'][:, start:stop], target['intrinsics'][:, start:stop],
                        target['near'][:, start:stop], target['far'][:, start:stop], target['image'].shape[-2:],
                    ).color)
                return torch.cat(chunks, 1)[0]
            for percent in percentages:
                count = target_count(total, percent)
                identity = next((r for (m, p), r in existing.items() if p == 100), None) if percent == 100 else None
                for method in methods:
                    key = (method, percent)
                    if key in existing:
                        continue
                    if percent == 100 and identity is not None:
                        existing[key] = {**identity, 'method': method}
                        continue
                    def operation():
                        if method == 'prune':
                            keep = uniform_pixel_indices(payload['support']['image_shape'], count, device)
                            return select_gaussians(original, keep), {'backend': 'torch_device'}
                        if method == 'merge':
                            return merge(original, count)
                        return redistribute(head, payload['support'], count)
                    if percent == 100:
                        variant, detail, samples = original, {'identity': True}, [0.0]
                    else:
                        if key not in warmed:
                            for _ in range(cfg.control.warmup_runs):
                                warm = operation()
                                del warm
                            torch.cuda.synchronize(device)
                            warmed.add(key)
                        samples = []
                        for repeat in range(cfg.control.timing_repeats):
                            if repeat:
                                del variant
                            (variant, detail), seconds = synchronized_time(operation, device)
                            samples.append(1000 * seconds)
                    if variant.means.shape[1] != count:
                        raise AssertionError(f'{method}/{percent}: wrong final Gaussian count')
                    if not all(bool(torch.isfinite(x).all()) for x in gaussian_dict(variant).values()):
                        raise FloatingPointError(f'{scene}/{method}/{percent}: non-finite Gaussian')
                    # Warm rasterization once per variant, separately from measured render time.
                    render(variant)
                    prediction, render_seconds = synchronized_time(lambda: render(variant), device)
                    gt = target['image'][0]
                    metric_values = {name: fn(gt, prediction) for name, fn in (
                        ('psnr', compute_psnr), ('ssim', compute_ssim), ('lpips', compute_lpips))}
                    row = {
                        'scene': scene, 'context_indices': payload['context_indices'],
                        'target_indices': payload['target_indices'], 'overlap': payload['overlap'],
                        'overlap_tag': get_overlap_tag(payload['overlap']),
                        'method': method, 'percent': percent, 'original_gaussians': total,
                        'gaussians': count, 'actual_percent': 100 * count / total,
                        'target_views': len(gt), 'postprocess_ms': float(np.mean(samples)),
                        'postprocess_ms_samples': samples,
                        'render_ms_per_view': 1000 * render_seconds / len(gt),
                        **{name: float(value.mean()) for name, value in metric_values.items()},
                        'per_view': {name: value.tolist() for name, value in metric_values.items()},
                        'detail': detail,
                    }
                    existing[key] = row
                    if percent == 100:
                        identity = row
                    if cfg.control.save_variants and percent != 100:
                        save_torch(output / 'gaussians' / item['key'] / f'{method}_{percent:03d}.pt', gaussian_dict(variant))
                    save_json(result_path, {'signature': signature, 'rows': list(existing.values())})
                    print(f"[Eval] {scene} {method}/{percent}% N={count} "
                          f"PSNR={row['psnr']:.3f} post={row['postprocess_ms']:.3f} ms", flush=True)
                    del variant, prediction
            save_json(result_path, {'signature': signature, 'rows': list(existing.values())})
            rows.extend(existing[key] for key in requested)
            completed += 1
            write_reports(output, rows, methods, percentages, len(manifest['scenes']))
            del payload, original, target
    finally:
        write_reports(output, rows, methods, percentages, len(manifest['scenes']))
    expected = len(manifest['scenes']) * len(methods) * len(percentages)
    if completed != len(manifest['scenes']) or len(rows) != expected:
        raise RuntimeError(f'Incomplete evaluation: {len(rows)}/{expected} condition-scene rows')


@hydra.main(version_base=None, config_path='../config', config_name='gaussian_control')
def main(cfg: DictConfig):
    control = cfg.control
    if control.stage not in ('all', 'cache', 'evaluate'):
        raise ValueError('control.stage must be all, cache or evaluate')
    percentages = list(control.percentages)
    if not percentages or len(set(percentages)) != len(percentages):
        raise ValueError('Use nonempty, distinct integer percentages')
    for value in percentages:
        if not isinstance(value, int):
            raise ValueError('Percentages must be integers')
        target_count(1, value)
    if not control.methods or len(set(control.methods)) != len(control.methods) or any(m not in METHODS for m in control.methods):
        raise ValueError(f'Unknown or duplicate methods; choose from {METHODS}')
    if control.timing_repeats < 1 or control.warmup_runs < 0 or control.render_chunk_size < 1:
        raise ValueError('Invalid timing/render settings')
    if not torch.cuda.is_available():
        raise RuntimeError('Final++ control evaluation requires a CUDA Gaussian renderer')
    device = torch.device('cuda', 0)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    root = Path(to_absolute_path(str(control.cache_dir)))
    output = Path(to_absolute_path(str(control.output_dir)))
    root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / 'analysis_config.yaml')
    status = output / 'run_status.txt'
    status.write_text('running\n', encoding='utf-8')
    try:
        if control.stage in ('all', 'cache'):
            build_cache(cfg, root, device)
            gc.collect()
            torch.cuda.empty_cache()
        if control.stage in ('all', 'evaluate'):
            evaluate(cfg, root, output, device)
    except BaseException as error:
        status.write_text(f'failed\n{type(error).__name__}: {error}\n', encoding='utf-8')
        raise
    status.write_text('complete\n', encoding='utf-8')


if __name__ == '__main__':
    main()
