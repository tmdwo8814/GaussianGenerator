"""Exact GPU KD-tree search, detached from autograd.

CuPy builds AND queries the tree on the GPU. DLPack and a shared CUDA stream
avoid per-scene host point/index transfers. This is not a dense distance matrix
or a truncated/approximate neighborhood search.
"""

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor


_validated = set()


def self_first(candidates: Tensor) -> Tensor:
    """Reserve self, retaining the first K-1 OTHER candidates without compaction.

    KD-tree ties may omit self entirely. Sorting small integer column keys moves
    an existing self to the end without changing the order of other neighbors.
    The final output is owned by PyTorch, not the external allocator.
    """
    count, k = candidates.shape
    own = torch.arange(count, device=candidates.device, dtype=torch.long)[:, None]
    if k == 1:
        return own
    columns = torch.arange(k, device=candidates.device).expand(count, k)
    order = (columns + (candidates == own) * k).argsort(dim=1)[:, :k - 1]
    return torch.cat((own, candidates.gather(1, order)), dim=1)


def _validate_first_query(points: Tensor, neighbors: Tensor):
    """One-time sampled exact-distance check on the real first scene per rank.

    Only this startup check transfers points to CPU. Later steps stay on GPU.
    Tied neighbor IDs need not match SciPy's unspecified tie ordering.
    """
    key = (points.device.index, neighbors.shape[1])
    if key in _validated:
        return
    xyz = points.detach().float().cpu().numpy().astype(np.float64)
    rows = np.linspace(0, len(xyz) - 1, min(32, len(xyz)), dtype=np.int64)
    indices = neighbors[torch.as_tensor(rows, device=neighbors.device)].cpu().numpy()
    k = neighbors.shape[1]
    if not (indices[:, 0] == rows).all() or (indices < 0).any() or (indices >= len(xyz)).any():
        raise RuntimeError('GPU kNN returned invalid/self-missing indices')
    if any(len(np.unique(row)) != k for row in indices):
        raise RuntimeError('GPU kNN returned duplicate neighbors')
    expected, _ = cKDTree(xyz).query(xyz[rows], k=k, eps=0.0, p=2, workers=1)
    actual = np.linalg.norm(xyz[indices] - xyz[rows, None], axis=-1)
    if not np.allclose(np.sort(actual, axis=1), np.asarray(expected).reshape(-1, k),
                       rtol=1e-8, atol=1e-10):
        raise RuntimeError('GPU KD-tree failed the startup exact-kNN check against SciPy')
    _validated.add(key)
    print(f'[moment kNN] CuPy exact GPU KD-tree verified on {points.device}, K={k}', flush=True)


@torch.no_grad()
def build_cuda_knn(points: Tensor, k: int) -> Tensor:
    if not points.is_cuda:
        raise ValueError('CuPy kNN requires CUDA points')
    try:
        import cupy as cp
        from cupyx.scipy.spatial import KDTree
    except ImportError as error:
        raise ImportError(
            'GPU moment decoding requires CuPy 14.2: '
            'python -m pip install -r requirements-fast.txt. '
            'For an explicit slow CPU fallback set model.encoder.moment_decoder.knn_backend=scipy.'
        ) from error

    if not bool(torch.isfinite(points).all()):
        raise ValueError('Cannot build 3D neighbors from non-finite support points')
    # Match the old SciPy path: first convert to FP32, then search in FP64.
    # Search is discrete; differentiable attributes still gather original points.
    with torch.cuda.device(points.device), cp.cuda.Device(points.device.index):
        stream = torch.cuda.current_stream(points.device)
        with cp.cuda.ExternalStream(stream.cuda_stream, device_id=points.device.index):
            coordinates = points.detach().float().to(torch.float64).contiguous()
            array = cp.from_dlpack(coordinates)
            tree = KDTree(array)
            _, candidates = tree.query(array, k=k, eps=0.0, p=2.0)
            candidates = torch.from_dlpack(candidates).to(dtype=torch.long).reshape(len(points), k)
            neighbors = self_first(candidates)
        _validate_first_query(points, neighbors)
    return neighbors
