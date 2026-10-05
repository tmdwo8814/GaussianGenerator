"""Sparse incoming feature pooling with a direct, chunked backward.

h_i = sum_(j,k: neighbors[j,k]=i) weights[j,k] * features[j]

The backward only needs features, weights and indices. It does not rebuild
forward messages through activation checkpointing or keep [M,K,D] activations.
The same PyTorch formulas support optional regional CUDA compilation.
Point/weight gradients outside this operation remain in autograd.
"""

import torch
from torch import Tensor

from .tensor_kernels import run_tensor_kernel


def _pool_add(pooled, features, weights, neighbors):
    messages = weights[..., None] * features[:, None, :]
    pooled.index_add_(0, neighbors.reshape(-1), messages.reshape(-1, features.shape[-1]))


def _pool_gradients(grad_output, features, weights, neighbors, need_features, need_weights):
    incoming = grad_output[neighbors]
    grad_features = (weights[..., None] * incoming).sum(1) if need_features else None
    grad_weights = (features[:, None, :] * incoming).sum(-1) if need_weights else None
    return grad_features, grad_weights


class _SparseFeaturePool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, weights, neighbors, chunk_size, compile_kernels):
        ctx.save_for_backward(features, weights, neighbors)
        ctx.chunk_size = chunk_size
        ctx.compile_kernels = compile_kernels
        pooled = torch.zeros_like(features)
        for start in range(0, len(features), chunk_size):
            stop = start + chunk_size
            run_tensor_kernel(_pool_add, pooled, features[start:stop], weights[start:stop],
                              neighbors[start:stop], enabled=compile_kernels)
        return pooled

    @staticmethod
    def backward(ctx, grad_output):
        features, weights, neighbors = ctx.saved_tensors
        need_features, need_weights = ctx.needs_input_grad[:2]
        grad_features = torch.empty_like(features) if need_features else None
        grad_weights = torch.empty_like(weights) if need_weights else None
        for start in range(0, len(features), ctx.chunk_size):
            stop = start + ctx.chunk_size
            df, dw = run_tensor_kernel(
                _pool_gradients, grad_output, features[start:stop], weights[start:stop],
                neighbors[start:stop], need_features, need_weights,
                enabled=ctx.compile_kernels and not torch.is_grad_enabled(),
            )
            # dL/df_j = sum_k w_jk * dL/dh_neighbor(j,k)
            if need_features:
                grad_features[start:stop] = df
            # dL/dw_jk = dot(f_j, dL/dh_neighbor(j,k))
            if need_weights:
                grad_weights[start:stop] = dw
        return grad_features, grad_weights, None, None, None


def pool_features(features: Tensor, weights: Tensor, neighbors: Tensor,
                  chunk_size: int, *, compile_kernels: bool = False) -> Tensor:
    return _SparseFeaturePool.apply(features, weights, neighbors, chunk_size, compile_kernels)
