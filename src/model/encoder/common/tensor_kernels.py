"""Optional regional compilation; no precision, parameter or CUDA-graph changes.

Only bounded tensor kernels are compiled. kNN, Python scene loops, logging and
checkpoint scheduling stay eager. CPU and higher-order pooling gradients keep
the ordinary PyTorch path. Compiled functions are not registered as modules, so
checkpoint parameter names and optimizer ordering remain unchanged.
"""

from functools import lru_cache
from inspect import unwrap

import torch


@lru_cache(maxsize=None)
def _compiled(function):
    # The application's jaxtyping import hook wraps even internal functions.
    # Compile their individual bodies, not the shared type-check wrapper code
    # (which otherwise exhausts Dynamo's per-code recompilation limit).
    return torch.compile(unwrap(function), fullgraph=True, dynamic=False,
                         options={"triton.cudagraphs": False})


def run_tensor_kernel(function, *args, enabled=False):
    tensor = next((arg for arg in args if isinstance(arg, torch.Tensor)), None)
    if enabled and tensor is not None and tensor.is_cuda:
        return _compiled(function)(*args)
    return function(*args)
