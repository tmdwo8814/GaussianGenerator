"""Support allocation -> moments -> renderable Gaussians.

Notation in this file: j is a source SUPPORT and i is a destination SLOT.
neighbors[j, k] = i, so softmax runs over outgoing candidates (k). Moments
use index_add over destination i, whose incoming degree is NOT limited to k.
The decoder takes only points/features, never cameras or target images.
"""

from dataclasses import dataclass
from math import log

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ...types import Gaussians
from ..common.sparse_knn import build_knn, build_partitioned_knn, validate_points
from ..common.sparse_feature_pool import pool_features


@dataclass
class MomentDecoderCfg:
    feature_dim: int = 64
    hidden_dim: int = 64
    num_neighbors: int = 16
    chunk_size: int = 32768
    checkpoint_chunks: bool = True
    knn_backend: str = 'auto'
    knn_query_backend: str = 'specialized'
    knn_workers: int = 1
    budget_max: float = 2.0
    budget_init: float = log(2.0)
    scale_min: float = 1.0
    scale_max: float = 3.0
    scale_init: float = 1.75
    covariance_floor: float = 1e-4  # standard deviation / scene scale
    mass_epsilon: float = 1e-6
    scene_epsilon: float = 1e-6


class MomentGaussianDecoder(nn.Module):
    def __init__(self, cfg: MomentDecoderCfg, sh_degree: int = 4):
        super().__init__()
        self.cfg = cfg
        if cfg.knn_backend not in ('auto', 'cupy', 'scipy'):
            raise ValueError('knn_backend must be auto, cupy or scipy')
        if cfg.knn_query_backend not in ('specialized', 'cupy'):
            raise ValueError('knn_query_backend must be specialized or cupy')
        if min(cfg.feature_dim, cfg.hidden_dim, cfg.num_neighbors,
               cfg.chunk_size, cfg.knn_workers) < 1:
            raise ValueError("Feature sizes, neighbor count, chunk size and workers must be positive")
        if not 0 < cfg.budget_init < cfg.budget_max:
            raise ValueError("Require 0 < budget_init < budget_max")
        if not 0 < cfg.scale_min < cfg.scale_init < cfg.scale_max:
            raise ValueError("Require 0 < scale_min < scale_init < scale_max")
        if min(cfg.covariance_floor, cfg.mass_epsilon, cfg.scene_epsilon) <= 0:
            raise ValueError("Covariance floor and epsilons must be positive")
        if sh_degree < 0:
            raise ValueError("sh_degree must be nonnegative")

        self.allocation_head = nn.Sequential(
            nn.Linear(2 * cfg.feature_dim + 4, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )
        self.budget_head = nn.Linear(cfg.feature_dim, 1)
        self.scale_head = nn.Linear(cfg.feature_dim, 1)
        self.d_sh = (sh_degree + 1) ** 2
        self.sh_head = nn.Linear(cfg.feature_dim, 3 * self.d_sh)

        # Nearly uniform initial allocation; bounded budget gives opacity ~0.5
        # when incoming mass equals outgoing mass. No per-slot mass guarantee.
        nn.init.normal_(self.allocation_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.allocation_head[-1].bias)
        nn.init.zeros_(self.budget_head.weight)
        nn.init.constant_(self.budget_head.bias,
                          log(cfg.budget_init / (cfg.budget_max - cfg.budget_init)))
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias,
                          log((cfg.scale_init - cfg.scale_min) /
                              (cfg.scale_max - cfg.scale_init)))

        # Preserve the baseline's bias toward DC appearance at initialization.
        sh_mask = torch.ones(self.d_sh)
        for degree in range(1, sh_degree + 1):
            sh_mask[degree**2 : (degree + 1)**2] = 0.1 * 0.25**degree
        self.register_buffer("sh_mask", sh_mask, persistent=False)

    def _run_chunk(self, function, *args):
        if self.cfg.checkpoint_chunks and self.training and torch.is_grad_enabled():
            # No mutation inside checkpointed functions; scatter happens outside.
            return checkpoint(function, *args, use_reentrant=False,
                              preserve_rng_state=False)
        return function(*args)

    def _allocation_chunk(self, source_points, source_projection, budget,
                          points, destination_projection, neighbors):
        delta = source_points[:, None] - points[neighbors]
        geometry = torch.cat((delta, delta.square().sum(-1, keepdim=True)), dim=-1)
        first = self.allocation_head[0]
        # W[fi,fj,g] + b = Wi fi + Wj fj + Wg g + b, BEFORE SiLU.
        # Retain the exact same parameters/state_dict and nonlinear function.
        hidden = (destination_projection[neighbors] + source_projection[:, None]
                  + F.linear(geometry, first.weight[:, 2 * self.cfg.feature_dim:], first.bias))
        logits = self.allocation_head[2](self.allocation_head[1](hidden)).squeeze(-1)
        allocation = logits.softmax(dim=-1)  # sum_k A[j, k] = 1
        return allocation * budget  # q[j, k] = A_ij * b_j

    def predict_allocation(self, points: Tensor, features: Tensor,
                           neighbors: Tensor) -> Tensor:
        """Normalized coordinates in; sparse outgoing mass q [M, K] out."""
        # Project each support once instead of repeating a 128->64 projection
        # for every edge. These projections remain in the autograd graph.
        weight = self.allocation_head[0].weight
        dim = self.cfg.feature_dim
        # One GEMM for the two independent projections; keep the original
        # parameters and split before the geometry sum / nonlinear activation.
        projection_weight = torch.cat((weight[:, :dim], weight[:, dim:2 * dim]), dim=0)
        destination_projection, source_projection = F.linear(features, projection_weight).split(
            self.cfg.hidden_dim, dim=-1
        )
        budget = self.cfg.budget_max * self.budget_head(features).sigmoid()
        chunks = []
        for start in range(0, len(points), self.cfg.chunk_size):
            stop = start + self.cfg.chunk_size
            chunks.append(self._run_chunk(
                self._allocation_chunk, points[start:stop], source_projection[start:stop],
                budget[start:stop], points, destination_projection, neighbors[start:stop],
            ))
        return torch.cat(chunks, dim=0)

    @staticmethod
    def _offset_chunk(source_points, points, neighbors, weights):
        # Accumulate offsets about each slot's support to avoid subtracting
        # large global second moments. This is exactly mu_i = sum_j w_ij x_j.
        offsets = source_points[:, None] - points[neighbors]
        return weights[..., None] * offsets

    @staticmethod
    def _covariance_chunk(source_points, means, neighbors, weights):
        delta = source_points[:, None] - means[neighbors]
        return weights[..., None, None] * delta[..., :, None] * delta[..., None, :]

    def aggregate_moments(self, points: Tensor, features: Tensor,
                          neighbors: Tensor, mass_per_edge: Tensor):
        """Incoming mass, weighted mean, CENTRAL covariance and pooled feature.

        All tensors use FP32 in forward. The epsilon self contribution only
        stabilizes moments; it does NOT contribute to opacity mass.
        """
        count = len(points)
        destinations = neighbors.reshape(-1)
        mass = points.new_zeros(count).index_add(
            0, destinations, mass_per_edge.reshape(-1)
        )
        self_prior = mass_per_edge.new_zeros(1, neighbors.shape[1])
        self_prior[:, 0] = self.cfg.mass_epsilon  # build_knn guarantees self first
        weights = (mass_per_edge + self_prior) / (
            mass[neighbors] + self.cfg.mass_epsilon
        )

        mean_offsets = torch.zeros_like(points)
        for start in range(0, count, self.cfg.chunk_size):
            stop = start + self.cfg.chunk_size
            indices = neighbors[start:stop]
            offsets = self._run_chunk(
                self._offset_chunk, points[start:stop],
                points, indices, weights[start:stop],
            )
            mean_offsets.index_add_(0, indices.reshape(-1), offsets.reshape(-1, 3))
        means = points + mean_offsets
        pooled = pool_features(features, weights, neighbors, self.cfg.chunk_size)

        covariance = points.new_zeros(count, 9)
        for start in range(0, count, self.cfg.chunk_size):
            stop = start + self.cfg.chunk_size
            indices = neighbors[start:stop]
            messages = self._run_chunk(
                self._covariance_chunk, points[start:stop], means,
                indices, weights[start:stop],
            )
            covariance.index_add_(0, indices.reshape(-1), messages.reshape(-1, 9))
        covariance = covariance.reshape(count, 3, 3)
        covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
        return mass, means, covariance, pooled

    def build_gaussians(self, mass, means, covariance, pooled, scene_scale):
        # Accept a single scene for inspection/tests, or all scenes for training.
        # Batch the small heads to avoid repeated GEMMs and a final SH concat.
        if means.ndim == 2:
            mass, means, covariance, pooled = (
                value.unsqueeze(0) for value in (mass, means, covariance, pooled)
            )
        scene_scale = scene_scale.reshape(-1, 1, 1)
        coverage = self.cfg.scale_min + (self.cfg.scale_max - self.cfg.scale_min) * (
            self.scale_head(pooled).sigmoid()
        )
        eye = torch.eye(3, device=means.device, dtype=means.dtype)
        covariance = coverage[..., None].square() * covariance
        # A fixed tiny floor can round away for wide/rank-deficient Gaussians.
        # Add a detached FP32 roundoff guard relative to the covariance trace.
        roundoff = 8 * torch.finfo(covariance.dtype).eps * (
            covariance.diagonal(dim1=-2, dim2=-1).sum(-1).detach()
        )
        floor = self.cfg.covariance_floor**2 + roundoff
        covariance = (covariance + floor[..., None, None] * eye) * scene_scale[..., None].square()
        harmonics = self.sh_head(pooled).reshape(*means.shape[:-1], 3, self.d_sh)
        # Linear backward saves its input/weight, not this output. Masking the
        # fresh output in place avoids allocating another full SH tensor.
        harmonics.mul_(self.sh_mask)
        return Gaussians(
            means=means * scene_scale,
            covariances=covariance,
            harmonics=harmonics,
            opacities=-torch.expm1(-mass),
        )

    def forward(self, points: Tensor, features: Tensor,
                partitions: list[tuple[int, ...] | None] | None = None) -> Gaussians:
        """[B, M, 3] points + [B, M, D] features -> standard Gaussians.

        M can represent pixels, voxels or tokens from any number of views.
        All points in one batch item must already share a coordinate frame.
        Output slot order is exactly the input support order.
        Optional per-scene partitions restrict search after failed registration;
        None retains the original global kNN path exactly.
        """
        if points.ndim != 3 or points.shape[-1] != 3 or min(points.shape[:2]) < 1:
            raise ValueError("points must have shape [B, M, 3], with B, M > 0")
        if features.shape != (*points.shape[:2], self.cfg.feature_dim):
            raise ValueError("features must have shape [B, M, feature_dim]")
        if points.device != features.device:
            raise ValueError("points and features must share a device")
        if partitions is None:
            partitions = [None] * len(points)
        if len(partitions) != len(points):
            raise ValueError("Provide one partition entry per scene")

        moments = []
        with torch.autocast(device_type=points.device.type, enabled=False):
            search_points = points.float()
            # One host-visible finite check per batch, not one per scene.
            validate_points(search_points)
            # Same detached lower median per scene, computed in one batch.
            scene_scales = search_points.detach().norm(dim=-1).median(dim=-1).values.clamp_min(
                self.cfg.scene_epsilon
            )
            for scene_points, scene_features, scene_scale, sizes in zip(
                    search_points, features.float(), scene_scales, partitions):
                if sizes is None:
                    neighbors = build_knn(scene_points, self.cfg.num_neighbors,
                                          self.cfg.knn_workers, self.cfg.knn_backend,
                                          check_finite=False,
                                          query_backend=self.cfg.knn_query_backend)
                else:
                    neighbors = build_partitioned_knn(
                        scene_points, sizes, self.cfg.num_neighbors,
                        self.cfg.knn_workers, self.cfg.knn_backend,
                        query_backend=self.cfg.knn_query_backend)
                normalized = scene_points / scene_scale
                q = self.predict_allocation(normalized, scene_features, neighbors)
                moments.append(self.aggregate_moments(normalized, scene_features, neighbors, q))
            batched_moments = [torch.stack(values) for values in zip(*moments)]
            del moments
            return self.build_gaussians(*batched_moments, scene_scales)
