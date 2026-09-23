"""Exact GPU KD-tree search, detached from autograd.

CuPy builds the tree; the specialized or general query runs on the same GPU.
DLPack and a shared CUDA stream avoid host point/index transfers. This is not
a dense distance matrix or a truncated/approximate neighborhood search.
"""

import torch
from torch import Tensor


def self_first(candidates: Tensor) -> Tensor:
    """Reserve self, retaining the first K-1 OTHER candidates without compaction.

    KD-tree ties may omit self entirely. Skip self by shifting column indices
    after its position, retaining the order of other neighbors without sorting.
    The final output is owned by PyTorch, not the external allocator.
    """
    count, k = candidates.shape
    own = torch.arange(count, device=candidates.device, dtype=torch.long)[:, None]
    if k == 1:
        return own
    columns = torch.arange(k, device=candidates.device).expand(count, k)
    self_column = torch.where(candidates == own, columns, k).amin(dim=1, keepdim=True)
    order = columns[:, :k - 1]
    order = order + (order >= self_column)
    return torch.cat((own, candidates.gather(1, order)), dim=1)


def query_tree(tree, array, k: int, backend: str = 'specialized'):
    if backend not in ('specialized', 'cupy'):
        raise ValueError('knn_query_backend must be specialized or cupy')
    if backend == 'specialized' and k == 16 and 16 <= len(array) < 2**30:
        from .knn_query16 import query_knn16
        return query_knn16(tree, array)
    # Other K values keep the exact general query, including tiny scenes.
    _, candidates = tree.query(array, k=k, eps=0.0, p=2.0)
    return candidates


@torch.no_grad()
def build_cuda_knn(points: Tensor, k: int, *, check_finite: bool = True,
                   query_backend: str = 'specialized') -> Tensor:
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

    if check_finite and not bool(torch.isfinite(points).all()):
        raise ValueError('Cannot build 3D neighbors from non-finite support points')
    # Match the old SciPy path: first convert to FP32, then search in FP64.
    # Search is discrete; differentiable attributes still gather original points.
    with torch.cuda.device(points.device), cp.cuda.Device(points.device.index):
        stream = torch.cuda.current_stream(points.device)
        with cp.cuda.ExternalStream(stream.cuda_stream, device_id=points.device.index):
            coordinates = points.detach().float().to(torch.float64).contiguous()
            array = cp.from_dlpack(coordinates)
            tree = KDTree(array)
            candidates = query_tree(tree, array, k, backend=query_backend)
            candidates = torch.from_dlpack(candidates).to(dtype=torch.long).reshape(len(points), k)
            neighbors = self_first(candidates)
    return neighbors
