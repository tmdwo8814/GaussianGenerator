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
from ..common.sparse_knn import build_knn


@dataclass
class MomentDecoderCfg:
    feature_dim: int = 64
    hidden_dim: int = 64
    num_neighbors: int = 16
    chunk_size: int = 32768
    checkpoint_chunks: bool = True
    knn_backend: str = 'auto'
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
        destination_projection = F.linear(features, weight[:, :dim])
        source_projection = F.linear(features, weight[:, dim:2 * dim])
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
    def _pool_chunk(source_points, source_features, points, neighbors, weights):
        # Accumulate offsets about each slot's support to avoid subtracting
        # large global second moments. This is exactly mu_i = sum_j w_ij x_j.
        offsets = source_points[:, None] - points[neighbors]
        return (weights[..., None] * offsets,
                weights[..., None] * source_features[:, None])

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
        pooled = torch.zeros_like(features)
        for start in range(0, count, self.cfg.chunk_size):
            stop = start + self.cfg.chunk_size
            indices = neighbors[start:stop]
            offsets, feature_messages = self._run_chunk(
                self._pool_chunk, points[start:stop], features[start:stop],
                points, indices, weights[start:stop],
            )
            mean_offsets.index_add_(0, indices.reshape(-1), offsets.reshape(-1, 3))
            pooled.index_add_(0, indices.reshape(-1),
                              feature_messages.reshape(-1, features.shape[-1]))
        means = points + mean_offsets

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
        covariance = (covariance + floor[:, None, None] * eye) * scene_scale.square()
        return Gaussians(
            means=(means * scene_scale).unsqueeze(0),
            covariances=covariance.unsqueeze(0),
            harmonics=(self.sh_head(pooled).reshape(-1, 3, self.d_sh) * self.sh_mask).unsqueeze(0),
            opacities=(-torch.expm1(-mass)).unsqueeze(0),
        )

    def forward(self, points: Tensor, features: Tensor) -> Gaussians:
        """[B, M, 3] points + [B, M, D] features -> standard Gaussians.

        M can represent pixels, voxels or tokens from any number of views.
        All points in one batch item must already share a coordinate frame.
        Output slot order is exactly the input support order.
        """
        if points.ndim != 3 or points.shape[-1] != 3 or min(points.shape[:2]) < 1:
            raise ValueError("points must have shape [B, M, 3], with B, M > 0")
        if features.shape != (*points.shape[:2], self.cfg.feature_dim):
            raise ValueError("features must have shape [B, M, feature_dim]")
        if points.device != features.device:
            raise ValueError("points and features must share a device")

        scenes = []
        with torch.autocast(device_type=points.device.type, enabled=False):
            for scene_points, scene_features in zip(points.float(), features.float()):
                neighbors = build_knn(scene_points, self.cfg.num_neighbors,
                                      self.cfg.knn_workers, self.cfg.knn_backend)
                scene_scale = scene_points.detach().norm(dim=-1).median().clamp_min(
                    self.cfg.scene_epsilon
                )
                normalized = scene_points / scene_scale
                q = self.predict_allocation(normalized, scene_features, neighbors)
                moments = self.aggregate_moments(normalized, scene_features, neighbors, q)
                scenes.append(self.build_gaussians(*moments, scene_scale))
        return Gaussians(
            means=torch.cat([g.means for g in scenes]),
            covariances=torch.cat([g.covariances for g in scenes]),
            harmonics=torch.cat([g.harmonics for g in scenes]),
            opacities=torch.cat([g.opacities for g in scenes]),
        )
