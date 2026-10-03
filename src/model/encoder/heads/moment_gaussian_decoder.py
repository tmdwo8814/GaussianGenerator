"""Support allocation -> moments -> renderable Gaussians.

Notation in this file: j is a source SUPPORT and i is a destination SLOT.
neighbors[j, k] = i, so softmax runs over outgoing candidates (k). Moments
use index_add over destination i, whose incoming degree is NOT limited to k.
The optional appearance path also reads context RGB and its raster layout,
never cameras or target images.
"""

from dataclasses import dataclass, field
from math import log

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ...types import Gaussians
from ..common.sparse_knn import build_knn, validate_points
from ..common.sparse_feature_pool import pool_features
from ..common.appearance_neighborhood import make_image_neighborhood
from ..common.image_neighborhood_appearance import ImageNeighborhoodAppearance
from ..common.appearance_capacity import MultiHeadAppearance
from ..common.separate_appearance import appearance_weights, appearance_weight_statistics


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
    appearance_2d: bool = False
    appearance_dim: int = 32
    appearance_2d_radii: list[int] = field(default_factory=lambda: [1, 4])
    separate_appearance: bool = False
    appearance_capacity: bool = False
    appearance_heads: int = 4
    appearance_mlp_dim: int = 256
    checkpoint_appearance: bool = True  # Recompute one complete capacity scene in backward.
    log_every_n_steps: int = 50  # 0 disables detached diagnostics


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
        if cfg.appearance_dim < 1 or cfg.log_every_n_steps < 0:
            raise ValueError("appearance_dim must be positive; log_every_n_steps must be nonnegative")
        if cfg.appearance_heads < 1 or cfg.appearance_mlp_dim < 1:
            raise ValueError("appearance_heads and appearance_mlp_dim must be positive")
        if cfg.appearance_capacity and not (cfg.appearance_2d and cfg.separate_appearance):
            raise ValueError("appearance_capacity requires appearance_2d and separate_appearance")
        if (any(not isinstance(radius, int) or radius < 1 for radius in cfg.appearance_2d_radii)
                or len(set(cfg.appearance_2d_radii)) != len(cfg.appearance_2d_radii)):
            raise ValueError("appearance_2d_radii must contain unique positive integers")

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
        self.appearance_head = None
        self.appearance_reader = None
        self.appearance_sh_head = None
        if cfg.separate_appearance:
            with torch.random.fork_rng(devices=[]):
                self.appearance_head = nn.Linear(cfg.hidden_dim, 1, bias=False)
            nn.init.zeros_(self.appearance_head.weight)
        if cfg.appearance_2d:
            # Preserve every original head and subsequent RNG initialization.
            with torch.random.fork_rng(devices=[]):
                if cfg.appearance_capacity:
                    self.appearance_reader = MultiHeadAppearance(
                        cfg.feature_dim, cfg.appearance_dim, cfg.appearance_heads,
                    )
                    self.appearance_sh_head = nn.Sequential(
                        nn.Linear(cfg.feature_dim + cfg.appearance_heads * cfg.appearance_dim,
                                  cfg.appearance_mlp_dim),
                        nn.SiLU(), nn.Linear(cfg.appearance_mlp_dim, 3 * self.d_sh),
                    )
                    nn.init.zeros_(self.appearance_sh_head[-1].weight)
                    nn.init.zeros_(self.appearance_sh_head[-1].bias)
                else:
                    self.appearance_reader = ImageNeighborhoodAppearance(cfg.feature_dim, cfg.appearance_dim)
                    self.appearance_sh_head = nn.Linear(cfg.appearance_dim, 3 * self.d_sh, bias=False)
                    nn.init.zeros_(self.appearance_sh_head.weight)

    def _run_chunk(self, function, *args):
        if self.cfg.checkpoint_chunks and self.training and torch.is_grad_enabled():
            # No mutation inside checkpointed functions; scatter happens outside.
            return checkpoint(function, *args, use_reentrant=False,
                              preserve_rng_state=False)
        return function(*args)

    def _allocation_chunk(self, source_points, source_projection, budget,
                          points, destination_projection, neighbors, return_appearance=False):
        delta = source_points[:, None] - points[neighbors]
        geometry = torch.cat((delta, delta.square().sum(-1, keepdim=True)), dim=-1)
        first = self.allocation_head[0]
        # W[fi,fj,g] + b = Wi fi + Wj fj + Wg g + b, BEFORE SiLU.
        # Retain the exact same parameters/state_dict and nonlinear function.
        hidden = (destination_projection[neighbors] + source_projection[:, None]
                  + F.linear(geometry, first.weight[:, 2 * self.cfg.feature_dim:], first.bias))
        hidden = self.allocation_head[1](hidden)
        logits = self.allocation_head[2](hidden).squeeze(-1)
        allocation = logits.softmax(dim=-1)  # sum_k A[j, k] = 1
        mass = allocation * budget  # q[j, k] = A_ij * b_j
        if return_appearance:
            return mass, self.appearance_head(hidden).squeeze(-1)
        return mass

    def predict_allocation(self, points: Tensor, features: Tensor,
                           neighbors: Tensor, return_appearance: bool = False):
        """Normalized coordinates in; sparse outgoing mass q [M, K] out."""
        # Project each support once instead of repeating a 128->64 projection
        # for every edge. These projections remain in the autograd graph.
        if return_appearance and self.appearance_head is None:
            raise ValueError("Appearance scores require separate_appearance=True")
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
                budget[start:stop], points, destination_projection, neighbors[start:stop], return_appearance,
            ))
        if return_appearance:
            return tuple(torch.cat(values, dim=0) for values in zip(*chunks))
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
                          neighbors: Tensor, mass_per_edge: Tensor, return_weights: bool = False):
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
        moments = mass, means, covariance, pooled
        return (*moments, weights) if return_weights else moments

    def _capacity_sh_chunk(self, pooled, context):
        return self.sh_head(pooled), self.appearance_sh_head(torch.cat((pooled, context), -1))

    def _capacity_harmonics(self, pooled, context, diagnostics):
        """Checkpoint the MLP per chunk instead of saving a [B,M,384] concat."""
        shape = pooled.shape[:-1]
        pooled, context = pooled.reshape(-1, pooled.shape[-1]), context.reshape(-1, context.shape[-1])
        harmonics, squared = [], []
        for start in range(0, len(pooled), self.cfg.chunk_size):
            stop = start + self.cfg.chunk_size
            base, extra = self._run_chunk(self._capacity_sh_chunk, pooled[start:stop], context[start:stop])
            harmonics.append(base + extra)
            if diagnostics is not None:
                with torch.no_grad():
                    squared.append(torch.stack([
                        (value.reshape(-1, 3, self.d_sh) * self.sh_mask).square().sum()
                        for value in (base, extra)
                    ]))
        if diagnostics is not None:
            rms = (torch.stack(squared).sum(0) / (len(pooled) * 3 * self.d_sh)).sqrt()
            diagnostics['base_sh_rms'], diagnostics['extra_sh_rms'] = rms.unbind()
        return torch.cat(harmonics).reshape(*shape, 3 * self.d_sh)

    def _capacity_scene(self, points, features, rgb, means, geometry_pooled,
                        geometry_weights, scores, neighbors, image, collect_statistics):
        """Pure checkpoint boundary: only final SH and detached metrics escape.

        kNN and geometry moments are computed outside and reused. Selected
        DPT features, expanded graph, reader projections/weights and contexts
        are recomputed for this scene instead of retained for the full batch.
        Existing inner chunk checkpoints still bound edge recomputation memory.
        """
        selected = appearance_weights(geometry_weights, scores, neighbors)
        appearance_pooled = pool_features(features, selected, neighbors, self.cfg.chunk_size)
        context, stats = self.appearance_reader(
            points, features, rgb, means, geometry_pooled, neighbors, image,
            chunk_size=self.cfg.chunk_size, checkpoint_chunks=self.cfg.checkpoint_chunks,
            collect_statistics=collect_statistics,
        )
        report = {} if collect_statistics else None
        raw_sh = self._capacity_harmonics(appearance_pooled, context, report)
        if report is not None:
            report.update(stats)
            report.update(appearance_weight_statistics(geometry_weights, selected, neighbors))
        return raw_sh, report

    def build_gaussians(self, mass, means, covariance, pooled, scene_scale, appearance_context=None,
                        appearance_pooled=None, diagnostics=None, appearance_harmonics=None):
        # Accept a single scene for inspection/tests, or all scenes for training.
        # Batch the small heads to avoid repeated GEMMs and a final SH concat.
        if means.ndim == 2:
            mass, means, covariance, pooled = (
                value.unsqueeze(0) for value in (mass, means, covariance, pooled)
            )
            if appearance_context is not None:
                appearance_context = appearance_context.unsqueeze(0)
            if appearance_pooled is not None:
                appearance_pooled = appearance_pooled.unsqueeze(0)
            if appearance_harmonics is not None:
                appearance_harmonics = appearance_harmonics.unsqueeze(0)
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
        sh_features = pooled if appearance_pooled is None else appearance_pooled
        if appearance_harmonics is not None:
            harmonics = appearance_harmonics
        elif self.cfg.appearance_capacity:
            if appearance_context is None or appearance_pooled is None:
                raise ValueError("appearance_capacity requires appearance pooled features and context")
            harmonics = self._capacity_harmonics(sh_features, appearance_context, diagnostics)
        else:
            harmonics = self.sh_head(sh_features)
        if appearance_context is not None and not self.cfg.appearance_capacity:
            # Equivalent to a linear SH head on [pooled_256, context_32].
            # Keep the old parameter keys; only the new columns start at zero.
            harmonics = harmonics + self.appearance_sh_head(appearance_context)
        harmonics = harmonics.reshape(*means.shape[:-1], 3, self.d_sh)
        # Linear backward saves its input/weight, not this output. Masking the
        # fresh output in place avoids allocating another full SH tensor.
        harmonics.mul_(self.sh_mask)
        return Gaussians(
            means=means * scene_scale,
            covariances=covariance,
            harmonics=harmonics,
            opacities=-torch.expm1(-mass),
        )

    def forward(self, points: Tensor, features: Tensor, *,
                image_shape: tuple[int, int, int] | None = None,
                rgb: Tensor | None = None, diagnostics: dict | None = None) -> Gaussians:
        """[B, M, 3] points + [B, M, D] features -> standard Gaussians.

        M can represent pixels, voxels or tokens from any number of views.
        All points in one batch item must already share a coordinate frame.
        Output slot order is exactly the input support order.
        appearance_2d requires image_shape=(V,H,W), M=V*H*W and context RGB
        [B,M,3] in [0,1], all in the SAME view-major raster order as points.
        """
        if points.ndim != 3 or points.shape[-1] != 3 or min(points.shape[:2]) < 1:
            raise ValueError("points must have shape [B, M, 3], with B, M > 0")
        if features.shape != (*points.shape[:2], self.cfg.feature_dim):
            raise ValueError("features must have shape [B, M, feature_dim]")
        if points.device != features.device:
            raise ValueError("points and features must share a device")
        image = None
        if self.appearance_reader is not None:
            if (image_shape is None or len(image_shape) != 3 or min(image_shape) < 1
                    or image_shape[0] * image_shape[1] * image_shape[2] != points.shape[1]):
                raise ValueError("appearance_2d requires image_shape=(V,H,W) with V*H*W=M")
            if rgb is None or rgb.shape != points.shape or rgb.device != points.device:
                raise ValueError("appearance_2d requires context RGB [B,M,3] on the points device")
            # Same source-image layout for the local batch; compute it once.
            image = make_image_neighborhood(image_shape, self.cfg.appearance_2d_radii, points.device)

        moments = []
        contexts, statistics, appearance_features = [], [], []
        completed_scenes = []
        with torch.autocast(device_type=points.device.type, enabled=False):
            search_points = points.float()
            # One host-visible finite check per batch, not one per scene.
            validate_points(search_points)
            # Same detached lower median per scene, computed in one batch.
            scene_scales = search_points.detach().norm(dim=-1).median(dim=-1).values.clamp_min(
                self.cfg.scene_epsilon
            )
            for scene_index, (scene_points, scene_features, scene_scale) in enumerate(zip(
                    search_points, features.float(), scene_scales)):
                neighbors = build_knn(scene_points, self.cfg.num_neighbors,
                                      self.cfg.knn_workers, self.cfg.knn_backend,
                                      check_finite=False,
                                      query_backend=self.cfg.knn_query_backend)
                normalized = scene_points / scene_scale
                scene_stats = {}
                if self.appearance_head is None:
                    q = self.predict_allocation(normalized, scene_features, neighbors)
                    scene_moments = self.aggregate_moments(normalized, scene_features, neighbors, q)
                else:
                    q, scores = self.predict_allocation(normalized, scene_features, neighbors,
                                                        return_appearance=True)
                    *scene_moments, geometry = self.aggregate_moments(
                        normalized, scene_features, neighbors, q, return_weights=True,
                    )
                    if self.cfg.appearance_capacity:
                        args = (normalized, scene_features, rgb[scene_index].float(),
                                scene_moments[1], scene_moments[3], geometry, scores,
                                neighbors, image, diagnostics is not None)
                        if (self.cfg.checkpoint_appearance and self.training
                                and torch.is_grad_enabled()):
                            raw_sh, scene_stats = checkpoint(
                                self._capacity_scene, *args,
                                use_reentrant=False, preserve_rng_state=False,
                            )
                        else:
                            raw_sh, scene_stats = self._capacity_scene(*args)
                        completed_scenes.append(self.build_gaussians(
                            *scene_moments, scene_scale, appearance_harmonics=raw_sh,
                        ))
                        if scene_stats is not None:
                            statistics.append(scene_stats)
                        # No Python list of per-scene DPT256/context tensors,
                        # and no full-batch copies of those tensors via stack.
                        del args, raw_sh, scene_moments, geometry, scores
                        continue
                    selected = appearance_weights(geometry, scores, neighbors)
                    appearance_features.append(pool_features(
                        scene_features, selected, neighbors, self.cfg.chunk_size,
                    ))
                    if diagnostics is not None:
                        scene_stats.update(appearance_weight_statistics(geometry, selected, neighbors))
                moments.append(scene_moments)
                if self.appearance_reader is not None:
                    context, stats = self.appearance_reader(
                        normalized, scene_features, rgb[scene_index].float(),
                        scene_moments[1], scene_moments[3], neighbors, image,
                        chunk_size=self.cfg.chunk_size, checkpoint_chunks=self.cfg.checkpoint_chunks,
                        collect_statistics=diagnostics is not None,
                    )
                    contexts.append(context)
                    if stats is not None:
                        scene_stats.update(stats)
                if scene_stats:
                    statistics.append(scene_stats)
            if completed_scenes:
                if diagnostics is not None:
                    for key in statistics[0]:
                        values = torch.stack([stats[key] for stats in statistics])
                        # Preserve the previous full-batch RMS, rather than
                        # accidentally changing it to a mean of scene RMSs.
                        diagnostics[key] = (values.square().mean().sqrt()
                                            if key in ('base_sh_rms', 'extra_sh_rms')
                                            else values.mean())
                return Gaussians(**{
                    key: torch.cat([getattr(scene, key) for scene in completed_scenes], dim=0)
                    for key in ('means', 'covariances', 'harmonics', 'opacities')
                })
            batched_moments = [torch.stack(values) for values in zip(*moments)]
            del moments
            appearance_context = torch.stack(contexts) if contexts else None
            appearance_pooled = torch.stack(appearance_features) if appearance_features else None
            del contexts, appearance_features
            gaussians = self.build_gaussians(*batched_moments, scene_scales,
                                            appearance_context=appearance_context,
                                            appearance_pooled=appearance_pooled, diagnostics=diagnostics)
            if diagnostics is not None and statistics:
                diagnostics.update({key: torch.stack([stats[key] for stats in statistics]).mean()
                                    for key in statistics[0]})
                if appearance_context is not None and not self.cfg.appearance_capacity:
                    with torch.no_grad():
                        extra_sh = self.appearance_sh_head(appearance_context).reshape(
                            *appearance_context.shape[:2], 3, self.d_sh,
                        ) * self.sh_mask
                        diagnostics['context_sh_rms'] = extra_sh.square().mean().sqrt()
            return gaussians
