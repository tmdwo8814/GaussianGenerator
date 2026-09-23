"""K=16 query on CuPy's existing tree; the tree builder is unchanged.

Only this module and its .cu file specialize the search. CuPy NVRTC compiles
the kernel on first use, then caches it. No torch extension or nvcc build step.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np


_checked_devices = set()


@lru_cache(maxsize=None)
def _kernel(device_id):
    import cupy as cp
    # Cache separately per device in case this process uses more than one GPU.
    with cp.cuda.Device(device_id):
        source = Path(__file__).with_suffix('.cu').read_text(encoding='utf-8')
        return cp.RawKernel(source, 'query_knn16', options=('--std=c++11', '--fmad=false'))


def _launch(tree, queries):
    import cupy as cp
    result = cp.empty((len(queries), 16), dtype=cp.int64)
    if len(queries):
        _kernel(queries.device.id)(
            ((len(queries) + 127) // 128,), (128,),
            (queries, tree.tree, tree.index, np.int32(len(tree.tree)),
             np.int32(len(queries)), result),
        )
    return result


def _check_kernel_once(device_id):
    """Run the actual CUDA kernel against SciPy before using it for training.

    Small non-full trees, planar points, duplicate ties and off-point queries
    exercise traversal edge cases. Runs only at startup on each CUDA device.
    The caller also validates a sample from the first full real scene.
    """
    if device_id in _checked_devices:
        return
    import cupy as cp
    from cupyx.scipy.spatial import KDTree
    from scipy.spatial import cKDTree
    xyz = np.random.default_rng(43).normal(size=(257, 3)).astype(np.float32).astype(np.float64)
    planar = xyz.copy()
    planar[:, 2] = 0.
    for points in (xyz[:17], xyz[:33], planar, np.zeros((33, 3))):
        queries = np.concatenate((points, points[:16] + .125))
        indices = cp.asnumpy(_launch(KDTree(cp.asarray(points)), cp.asarray(queries)))
        if ((indices < 0).any() or (indices >= len(points)).any()
                or any(len(np.unique(row)) != 16 for row in indices)):
            raise RuntimeError('Specialized kNN startup check: invalid/duplicate indices')
        actual = np.linalg.norm(points[indices] - queries[:, None], axis=-1)
        expected, _ = cKDTree(points).query(queries, k=16, eps=0., p=2.)
        if not np.allclose(actual, expected, rtol=1e-10, atol=1e-12):
            raise RuntimeError('Specialized kNN startup check failed against SciPy; '
                               'use model.encoder.moment_decoder.knn_query_backend=cupy')
    _checked_devices.add(device_id)
    print(f'[moment kNN] specialized FP64/K=16 kernel self-check passed on cuda:{device_id}', flush=True)


def query_knn16(tree, queries):
    import cupy as cp
    n = tree.tree.shape[0]
    if not (16 <= n < 2**30):
        raise ValueError('Specialized kNN requires 16 <= tree size < 2**30')
    if queries.ndim != 2 or queries.shape[1] != 3 or tree.tree.shape[1] != 3:
        raise ValueError('Specialized kNN requires 3D points')
    if queries.shape[0] >= 2**31:
        raise ValueError('Too many queries for specialized kNN')
    if tree.copy_query_points:
        raise ValueError('Specialized kNN does not support periodic trees')
    if (queries.dtype != cp.float64 or tree.tree.dtype != cp.float64
            or tree.index.dtype != cp.int64):
        raise ValueError('Specialized kNN requires FP64 coordinates and int64 tree indices')
    if not all(x.flags.c_contiguous for x in (queries, tree.tree, tree.index)):
        raise ValueError('Specialized kNN requires contiguous arrays')
    if len({x.device.id for x in (queries, tree.tree, tree.index)}) != 1:
        raise ValueError('Specialized kNN arrays must share a CUDA device')
    # The caller already selected PyTorch's CUDA stream via ExternalStream.
    with cp.cuda.Device(queries.device.id):
        _check_kernel_once(queries.device.id)
        return _launch(tree, queries)
