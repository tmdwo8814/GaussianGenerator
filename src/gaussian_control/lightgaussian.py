"""LightGaussian pruning only, with a separately named CUDA count extension.

The score sums opacity * pre-splat transmittance over contributing rays, then
uses the released LightGaussian volume weight (v_pow=0.1, no clipping).
No recovery, SH reduction, quantization, or model-weight updates are performed.
"""

from functools import lru_cache
import os
from pathlib import Path

import torch

from .common import select_gaussians


RASTERIZER_ROOT = Path(__file__).resolve().parents[1] / 'third_party/lightgaussian_rasterizer'
RASTERIZER_COMMIT = '9893d8c607684bc1a432e179cc97035d72d84520'


@lru_cache(maxsize=1)
def load_count_extension(build_directory):
    from torch.utils.cpp_extension import CUDA_HOME, load
    if CUDA_HOME is None:
        raise RuntimeError('LightGaussian count extension needs nvcc/CUDA_HOME; the existing renderer is not replaced')
    os.environ.setdefault('MAX_JOBS', '1')
    destination = Path(build_directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    sources = [RASTERIZER_ROOT / name for name in (
        'cuda_rasterizer/rasterizer_impl.cu', 'cuda_rasterizer/forward.cu',
        'cuda_rasterizer/backward.cu', 'rasterize_points.cu', 'ext.cpp',
    )]
    return load(
        name='_gg_lightgaussian_counts_9893d8c', sources=list(map(str, sources)),
        extra_include_paths=[str(RASTERIZER_ROOT / 'third_party/glm')],
        extra_cuda_cflags=['-O2'], build_directory=str(destination), verbose=True,
    )


@torch.no_grad()
def count_view(extension, gaussians, extrinsics, intrinsics, near, far, shape,
               scale_invariant=True):
    from ..geometry.projection import get_fov
    from ..model.decoder.cuda_splatting import get_projection_matrix
    means, covariance = gaussians.means[0].float(), gaussians.covariances[0].float()
    pose = extrinsics.float().clone()
    if scale_invariant:
        scale = 1 / near
        means = means * scale
        covariance = covariance * scale.square()
        pose[:3, 3] *= scale
        near, far = near * scale, far * scale
    fov = get_fov(intrinsics[None]).squeeze(0)
    tanfov = (fov / 2).tan()
    projection = get_projection_matrix(near[None], far[None], fov[None, 0], fov[None, 1])[0].T.contiguous()
    view = pose.inverse().T.contiguous()
    row, col = torch.triu_indices(3, 3, device=means.device)
    empty = means.new_empty(0)
    # RGB does not enter the significance criterion. Precomputed zero colors
    # avoid the upstream SH<=3 restriction while keeping all original SH75.
    result = extension.count_gaussians(
        means.new_zeros(3), means.contiguous(), means.new_zeros(len(means), 3),
        gaussians.opacities[0, :, None].float().contiguous(), empty, empty, 1.,
        covariance[:, row, col].contiguous(), view, (view @ projection).contiguous(),
        float(tanfov[0]), float(tanfov[1]), int(shape[0]), int(shape[1]),
        empty, 0, pose[:3, 3].contiguous(), False, False, True,
    )
    counts, importance = result[:2]
    return counts, importance


def volume_weighted_score(covariances, importance, exponent=.1):
    # Volume is proportional to sqrt(det(Sigma)); the 4*pi/3 factor cancels.
    volume = torch.linalg.det(covariances.double()).clamp_min(0).sqrt()
    reference = volume.sort(descending=True).values[int(len(volume) * .9)]
    if not bool(torch.isfinite(volume).all()) or reference <= 0:
        raise ValueError('LightGaussian requires finite positive Gaussian volumes')
    return ((volume / reference).pow(exponent) * importance.double()).float()


@torch.no_grad()
def significance(extension, gaussians, cameras, shape, scale_invariant=True):
    importance = gaussians.opacities.new_zeros(gaussians.means.shape[1])
    hits = torch.zeros_like(importance, dtype=torch.int64)
    for index in range(len(cameras['extrinsics'])):
        count, value = count_view(
            extension, gaussians, cameras['extrinsics'][index], cameras['intrinsics'][index],
            cameras['near'][index], cameras['far'][index], shape, scale_invariant,
        )
        importance += value
        hits += count.long()
    scores = volume_weighted_score(gaussians.covariances[0], importance)
    if not bool(torch.isfinite(scores).all()) or not bool((scores > 0).any()):
        raise RuntimeError('Invalid/zero LightGaussian significance; check context-camera calibration')
    return scores, hits


def prune_with_scores(gaussians, scores, count):
    if not 1 <= count <= len(scores):
        raise ValueError('Invalid LightGaussian count')
    # Exact shared budget even when many unobserved Gaussians have zero scores.
    keep = torch.argsort(scores, descending=True, stable=True)[:count].sort().values
    return select_gaussians(gaussians, keep)


@torch.no_grad()
def prune(extension, gaussians, cameras, shape, count, scale_invariant=True):
    scores, hits = significance(extension, gaussians, cameras, shape, scale_invariant)
    result = prune_with_scores(gaussians, scores, count)
    return result, {'zero_score_gaussians': int((scores == 0).sum()),
                    'contributing_ray_hits': int(hits.sum())}
