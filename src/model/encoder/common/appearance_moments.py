"""Parameter-free feature spread on the existing appearance allocation graph."""

from torch import Tensor

from .sparse_feature_pool import pool_features


def appearance_feature_std(features: Tensor, weights: Tensor, neighbors: Tensor,
                           mean: Tensor, chunk_size: int, *,
                           compile_kernels: bool = False, epsilon: float = 1e-6) -> Tensor:
    """sqrt(E_w[f^2] - E_w[f]^2 + eps), with incoming-normalized wA.

    Reuse the same weights/mean used by the SH path, not geometry weights or
    the 2D reader's attention. The decoder supplies FP32 tensors. Sparse
    pooling bounds edge messages by chunk_size and recomputes its backward;
    no full [supports, candidates, channels] activation is retained.
    """
    second = pool_features(features.square(), weights, neighbors, chunk_size,
                           compile_kernels=compile_kernels)
    # Roundoff can make an almost-zero variance slightly negative. Epsilon
    # keeps the sqrt derivative finite for single-support/constant features.
    return (second - mean.square()).clamp_min(0).add(epsilon).sqrt()
