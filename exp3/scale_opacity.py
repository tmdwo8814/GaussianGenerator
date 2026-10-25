"""Numerical helpers, independent of torch/CUDA and the training imports."""

from __future__ import annotations

import numpy as np


def log_max_scale(covariance: np.ndarray) -> np.ndarray:
    """log10(max_k sqrt(eig_k(Sigma))) from FINAL world-space covariance.

    No depth/focal multiplier, raw head parameters, or screen-space filter.
    Use float64 eigensolves to avoid losing the small axes of flat Gaussians.
    """
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim != 3 or covariance.shape[1:] != (3, 3):
        raise ValueError("Expected final covariance with shape (N, 3, 3)")
    if not np.isfinite(covariance).all():
        raise ValueError("Non-finite covariance")
    covariance = (covariance + covariance.transpose(0, 2, 1)) * 0.5
    eigenvalues = np.linalg.eigvalsh(covariance)
    largest = eigenvalues[:, -1]
    tolerance = 32 * np.finfo(np.float32).eps * np.maximum(
        np.abs(eigenvalues).max(axis=1), np.finfo(np.float64).tiny,
    )
    if np.any(eigenvalues[:, 0] < -tolerance):
        raise ValueError("Final covariance is not positive semidefinite")
    if np.any(largest <= 0):
        raise ValueError("A selected Gaussian has zero/nonpositive maximum scale")
    return 0.5 * np.log10(largest)


class PrioritySample:
    """Uniform sample of pooled Gaussians with bounded memory, not per-scene quotas.

    Independent uniform priorities are assigned before this class sees a chunk.
    Keeping the smallest priorities makes selection independent of chunk order.
    """

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("sample-size must be positive")
        self.capacity = capacity
        self.priority = np.empty(0, dtype=np.float64)
        self.values = {
            "log_s_max": np.empty(0, dtype=np.float64),
            "opacity": np.empty(0, dtype=np.float32),
            "scene_index": np.empty(0, dtype=np.int32),
            "gaussian_index": np.empty(0, dtype=np.int64),
        }

    def add(self, priority, **values):
        priority = np.asarray(priority, dtype=np.float64)
        if set(values) != set(self.values):
            raise ValueError("Supply all sample fields")
        if any(np.asarray(value).shape != priority.shape for value in values.values()):
            raise ValueError("Sample fields must have the same one-dimensional shape")
        merged_priority = np.concatenate((self.priority, priority))
        count = min(self.capacity, len(merged_priority))
        indices = (np.argpartition(merged_priority, count - 1)[:count]
                   if len(merged_priority) > count else np.arange(count))
        self.priority = merged_priority[indices]
        for key, previous in self.values.items():
            incoming = np.asarray(values[key], dtype=previous.dtype)
            self.values[key] = np.concatenate((previous, incoming))[indices]

    def arrays(self):
        indices = np.argsort(self.priority, kind="stable")
        return {key: value[indices] for key, value in self.values.items()}
