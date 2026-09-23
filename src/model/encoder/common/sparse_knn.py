"""Exact 3D neighbors without constructing an M x M distance matrix."""

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor


@torch.no_grad()
def build_knn(points: Tensor, k: int = 16, workers: int = 1,
              backend: str = 'auto') -> Tensor:
    """Return [M, min(k, M)] candidate slot indices for ONE scene.

    Row j belongs to support j; column zero is always its own slot. The
    remaining columns contain distinct nearest slots in the common 3D frame.
    Only this discrete search uses detached coordinates. Callers must
    gather the ORIGINAL tensors when computing differentiable attributes.
    auto = exact GPU KD-tree on CUDA, SciPy on CPU. Missing GPU dependencies
    fail explicitly rather than silently restoring the expensive CPU path.
    """
    if points.ndim != 2 or points.shape[-1] != 3 or len(points) == 0:
        raise ValueError("points must have shape [M, 3], with M > 0")
    if k < 1 or workers < 1:
        raise ValueError("k and workers must be positive")
    if backend not in ('auto', 'cupy', 'scipy'):
        raise ValueError('knn_backend must be auto, cupy or scipy')

    count = len(points)
    k = min(k, count)
    if k == 1:
        if not bool(torch.isfinite(points).all()):
            raise ValueError("Cannot build 3D neighbors from non-finite support points")
        return torch.arange(count, device=points.device, dtype=torch.long)[:, None]
    if backend == 'cupy' or (backend == 'auto' and points.is_cuda):
        from .cuda_knn import build_cuda_knn
        return build_cuda_knn(points, k)
    own = np.arange(count, dtype=np.int64)[:, None]
    coordinates = points.detach().float().cpu().numpy()
    if not np.isfinite(coordinates).all():
        raise ValueError("Cannot build 3D neighbors from non-finite support points")
    # eps=0 means exact Euclidean kNN. A tree avoids quadratic storage/work
    # for ordinary 3D point clouds; CPU search/transfer should still be profiled.
    _, candidates = cKDTree(coordinates).query(
        coordinates, k=k, eps=0.0, p=2, workers=workers
    )
    # Equal coordinates can cause the tree to omit self or return it anywhere.
    # Explicitly reserve self, then retain the first k-1 OTHER indices.
    other = candidates != own
    keep = other & (np.cumsum(other, axis=1) <= k - 1)
    neighbors = np.concatenate(
        (own, candidates[keep].reshape(count, k - 1)), axis=1
    )
    return torch.from_numpy(neighbors.astype(np.int64, copy=False)).to(points.device)
