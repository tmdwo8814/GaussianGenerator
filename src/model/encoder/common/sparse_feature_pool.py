"""Sparse incoming feature pooling with a direct, chunked backward.

h_i = sum_(j,k: neighbors[j,k]=i) weights[j,k] * features[j]

The backward only needs features, weights and indices. It does not rebuild
forward messages through activation checkpointing or keep [M,K,D] activations.
This uses ordinary PyTorch operations on CPU/CUDA, with no additional kernel
compiler. Point/weight gradients outside this operation remain in autograd.
"""

import torch
from torch import Tensor


class _SparseFeaturePool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, weights, neighbors, chunk_size):
        ctx.save_for_backward(features, weights, neighbors)
        ctx.chunk_size = chunk_size
        pooled = torch.zeros_like(features)
        for start in range(0, len(features), chunk_size):
            stop = start + chunk_size
            messages = weights[start:stop, :, None] * features[start:stop, None, :]
            pooled.index_add_(0, neighbors[start:stop].reshape(-1),
                              messages.reshape(-1, features.shape[-1]))
        return pooled

    @staticmethod
    def backward(ctx, grad_output):
        features, weights, neighbors = ctx.saved_tensors
        need_features, need_weights = ctx.needs_input_grad[:2]
        grad_features = torch.empty_like(features) if need_features else None
        grad_weights = torch.empty_like(weights) if need_weights else None
        for start in range(0, len(features), ctx.chunk_size):
            stop = start + ctx.chunk_size
            incoming = grad_output[neighbors[start:stop]]
            # dL/df_j = sum_k w_jk * dL/dh_neighbor(j,k)
            if need_features:
                grad_features[start:stop] = (weights[start:stop, :, None] * incoming).sum(dim=1)
            # dL/dw_jk = dot(f_j, dL/dh_neighbor(j,k))
            if need_weights:
                grad_weights[start:stop] = (features[start:stop, None, :] * incoming).sum(dim=-1)
        return grad_features, grad_weights, None, None


def pool_features(features: Tensor, weights: Tensor, neighbors: Tensor,
                  chunk_size: int) -> Tensor:
    return _SparseFeaturePool.apply(features, weights, neighbors, chunk_size)
