"""Distribution helpers independent of the model and CUDA renderer."""

import numpy as np


def log_max_scale(covariance):
    """log10(max_k sqrt(eig_k(final covariance))), before renderer rescaling."""
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim != 3 or covariance.shape[1:] != (3, 3):
        raise ValueError("Expected covariance (N,3,3)")
    if not np.isfinite(covariance).all():
        raise ValueError("Non-finite covariance")
    covariance = (covariance + covariance.transpose(0, 2, 1)) * 0.5
    eigenvalues = np.linalg.eigvalsh(covariance)
    maximum = eigenvalues[:, -1]
    tolerance = 32 * np.finfo(np.float32).eps * np.maximum(
        np.abs(eigenvalues).max(axis=1), np.finfo(np.float64).tiny,
    )
    if np.any(eigenvalues[:, 0] < -tolerance):
        raise ValueError("Covariance is not positive semidefinite")
    if np.any(maximum <= 0):
        raise ValueError("Selected Gaussian has zero/nonpositive maximum scale")
    return 0.5 * np.log10(maximum)


class PrioritySample:
    """Uniform sample of pooled Gaussians, independent of scene/chunk order."""

    def __init__(self, capacity):
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
        if (priority.ndim != 1 or set(values) != set(self.values)
                or any(np.asarray(v).shape != priority.shape for v in values.values())):
            raise ValueError("Supply all one-dimensional sample fields of equal length")
        merged = np.concatenate((self.priority, priority))
        count = min(self.capacity, len(merged))
        chosen = (np.argpartition(merged, count - 1)[:count]
                  if len(merged) > count else np.arange(count))
        self.priority = merged[chosen]
        for key, previous in self.values.items():
            value = np.asarray(values[key], dtype=previous.dtype)
            self.values[key] = np.concatenate((previous, value))[chosen]

    def arrays(self):
        chosen = np.argsort(self.priority, kind="stable")
        return {key: value[chosen] for key, value in self.values.items()}
