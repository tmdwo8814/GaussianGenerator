"""Choose output slots from existing allocation evidence; retain every support.

Gate inputs detach the original features, coordinates and allocation. Only the
small controller receives the count-loss gradient. Image gradients still train
the original decoder through the reallocated mass. Binary gates use an STE;
coverage repair is discrete and is included in the *actual* active count.
"""

from dataclasses import dataclass
from math import log

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .sparse_feature_pool import pool_features


@dataclass
class SlotControlCfg:
    enabled: bool = False
    feature_dim: int = 32
    hidden_dim: int = 64
    target_fraction: float = 0.75
    loss_weight: float = 0.01
    warmup_steps: int = 10000
    ramp_steps: int = 10000
    initial_probability: float = 0.9


def repair_coverage(opened: Tensor, probabilities: Tensor, neighbors: Tensor,
                    mass: Tensor) -> Tensor:
    """Retain at least 1/K of each support's original allocation before rescaling.

    Reopen the largest allocation if too little mass survives. Its mass is at
    least budget/K, bounding the redistribution factor by K. Gate probability
    only breaks ties (and selects a destination for zero-budget rows). Slot
    openings are shared globally, so other supports can only gain coverage.
    """
    with torch.no_grad():
        edge_open = opened[neighbors]
        budget = mass.sum(-1)
        remaining = (mass * edge_open).sum(-1)
        # The explicit zero check also covers subnormal budget/K underflow.
        insufficient = (remaining < budget / neighbors.shape[1]) | ((remaining == 0) & (budget > 0))
        needs_repair = insufficient | ~edge_open.any(-1)
        largest = mass == mass.amax(-1, keepdim=True)
        scores = probabilities[neighbors].masked_fill(~largest, -torch.inf)
        chosen = neighbors.gather(1, scores.argmax(-1, keepdim=True)).squeeze(-1)
        additions = torch.zeros_like(probabilities).index_add(
            0, chosen, needs_repair.to(probabilities.dtype),
        )
        return opened | (additions > 0)


def redistribute_mass(mass: Tensor, neighbors: Tensor, gates: Tensor) -> Tensor:
    """q' = b * (q * z) / sum_k(q * z), conserving each support's budget."""
    edge_gates = gates[neighbors]
    weighted = mass * edge_gates
    remaining = weighted.sum(-1, keepdim=True)
    budget = mass.sum(-1, keepdim=True)
    tiny = torch.finfo(mass.dtype).tiny
    valid = remaining > tiny
    # Normalize before restoring the budget; do not form a huge budget/remaining
    # intermediate. Mask the denominator BEFORE dividing: masking an unsafe
    # result afterward does not keep its backward graph finite.
    denominator = torch.where(valid, remaining, torch.ones_like(remaining))
    normalized = weighted / denominator
    fallback = edge_gates / edge_gates.sum(-1, keepdim=True).clamp_min(1)
    return torch.where(valid, normalized, fallback) * budget


def global_active_fraction(active_count: Tensor, candidate_count: Tensor) -> Tensor:
    """Global-batch forward value and gradient correct under DDP averaging.

    The nonlinear budget penalty must be applied AFTER reducing counts. A mean
    of per-rank hinge losses would impose a different constraint on each rank.
    """
    if not dist.is_available() or not dist.is_initialized():
        return active_count / candidate_count
    counts = torch.stack((active_count.detach(), candidate_count.detach()))
    dist.all_reduce(counts)
    return (counts[0] / counts[1]
            + (active_count - active_count.detach()) * dist.get_world_size() / counts[1])


class SlotController(nn.Module):
    def __init__(self, feature_dim: int, cfg: SlotControlCfg, *, chunk_size: int,
                 epsilon: float, checkpoint_chunks: bool, compile_kernels: bool):
        super().__init__()
        if min(cfg.feature_dim, cfg.hidden_dim) < 1:
            raise ValueError('Slot controller dimensions must be positive')
        if not 0 < cfg.target_fraction <= 1 or cfg.loss_weight < 0:
            raise ValueError('Require 0 < target_fraction <= 1 and loss_weight >= 0')
        if min(cfg.warmup_steps, cfg.ramp_steps) < 0:
            raise ValueError('Slot warmup/ramp steps must be nonnegative')
        if not 0.5 < cfg.initial_probability < 1:
            raise ValueError('initial_probability must be in (0.5, 1) for all-open initialization')
        self.cfg = cfg
        self.chunk_size = chunk_size
        self.epsilon = epsilon
        self.checkpoint_chunks = checkpoint_chunks
        self.compile_kernels = compile_kernels
        self.projection = nn.Linear(feature_dim, cfg.feature_dim)
        self.gate = nn.Sequential(nn.Linear(2 * cfg.feature_dim + 4, cfg.hidden_dim),
                                  nn.SiLU(), nn.Linear(cfg.hidden_dim, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias,
                          log(cfg.initial_probability / (1 - cfg.initial_probability)))

    def progress(self, step: int | None) -> float:
        # None is ordinary inference; warmup only governs optimization.
        if step is None:
            return 1.0
        if step < self.cfg.warmup_steps:
            return 0.0
        if self.cfg.ramp_steps == 0:
            return 1.0
        return min(1.0, (step - self.cfg.warmup_steps) / self.cfg.ramp_steps)

    def summarize(self, points: Tensor, features: Tensor, neighbors: Tensor,
                  mass: Tensor) -> Tensor:
        """68 channels at d=32: own/pooled features and four scalar summaries."""
        points, features, mass = points.detach(), features.detach(), mass.detach()
        with torch.no_grad():
            received = mass.new_zeros(len(mass)).index_add(0, neighbors.flatten(), mass.flatten())
            prior = mass.new_zeros(1, neighbors.shape[1])
            prior[:, 0] = self.epsilon  # kNN self is first, as in geometry moments.
            weights = (mass + prior) / (received[neighbors] + self.epsilon)
            # Center before computing coordinate second moments for FP32 stability.
            centered = points - points.mean(0, keepdim=True)
            geometry = pool_features(
                torch.cat((centered, centered.square().sum(-1, keepdim=True)), -1),
                weights, neighbors, self.chunk_size, compile_kernels=self.compile_kernels,
            )
            spread = (geometry[:, 3:] - geometry[:, :3].square().sum(-1, keepdim=True)).clamp_min(0)
            relative_mass = torch.log1p(received / received.mean().clamp_min(self.epsilon))[:, None]
            probability = mass / mass.sum(-1, keepdim=True).clamp_min(torch.finfo(mass.dtype).tiny)
            alternatives = torch.zeros_like(probability)
            if neighbors.shape[1] > 1:
                top, positions = probability.topk(2, dim=-1)
                alternatives = top[:, :1].expand_as(probability).clone()
                alternatives.scatter_(1, positions[:, :1], top[:, 1:2])
            alternative_strength = received.new_zeros(len(received)).index_add(
                0, neighbors.flatten(), (weights * alternatives).flatten(),
            )[:, None]

        projected = self.projection(features)
        projected = F.layer_norm(projected, (self.cfg.feature_dim,))
        pooled = pool_features(
            torch.cat((projected, projected.square().mean(-1, keepdim=True)), -1),
            weights, neighbors, self.chunk_size, compile_kernels=self.compile_kernels,
        )
        mean = pooled[:, :-1]
        variance = (pooled[:, -1:] - mean.square().mean(-1, keepdim=True)).clamp_min(0)
        return torch.cat((projected, mean, variance, torch.log1p(spread),
                          relative_mass, alternative_strength), -1)

    def _probabilities(self, points, features, neighbors, mass):
        return self.gate(self.summarize(points, features, neighbors, mass)).squeeze(-1).sigmoid()

    def forward(self, points: Tensor, features: Tensor, neighbors: Tensor,
                mass: Tensor, *, global_step: int | None = None):
        args = (points.detach(), features.detach(), neighbors, mass.detach())
        if self.checkpoint_chunks and self.training and torch.is_grad_enabled():
            probabilities = checkpoint(self._probabilities, *args, use_reentrant=False,
                                       preserve_rng_state=False)
        else:
            probabilities = self._probabilities(*args)
        warming_up = self.progress(global_step) == 0.0
        raw = torch.ones_like(probabilities, dtype=torch.bool) if warming_up else probabilities > 0.5
        opened = raw if warming_up else repair_coverage(raw, probabilities, neighbors, mass)
        # Exact binary forward; no rounding from (z-p)+p. During warmup the zero
        # path keeps all controller parameters in DDP's graph with zero data gradients.
        surrogate = probabilities * (0.0 if warming_up else 1.0)
        gates = opened.to(mass.dtype) + (surrogate - surrogate.detach())
        allocated = mass if warming_up else redistribute_mass(mass, neighbors, gates)
        report = {
            'active_count': gates.sum(),
            'candidate_count': mass.new_tensor(len(mass)),
        }
        with torch.no_grad():
            budget = mass.sum(-1)
            positive_budget = budget > 0
            safe_budget = torch.where(positive_budget, budget, torch.ones_like(budget))
            raw_remaining = (mass * raw[neighbors]).sum(-1)
            remaining = (mass * opened[neighbors]).sum(-1)
            raw_fraction = torch.where(positive_budget, raw_remaining / safe_budget, 1.)
            fraction = torch.where(positive_budget, remaining / safe_budget, 1.)
            safe_remaining = torch.where(remaining > 0, remaining, torch.ones_like(remaining))
            amplification = torch.where(positive_budget, budget / safe_remaining, 1.)
            report.update(
                raw_active_fraction=raw.float().mean(),
                fallback_fraction=(opened & ~raw).float().mean(),
                probability_mean=probabilities.mean(),
                raw_retained_mass_fraction_min=raw_fraction.min(),
                retained_mass_fraction_min=fraction.min(),
                redistribution_scale_max=amplification.max(),
                mass_relative_error=((allocated.sum(-1) - mass.sum(-1)).abs()
                                     / mass.sum(-1).clamp_min(self.epsilon)).max(),
            )
        return allocated, opened, report

    def budget_loss(self, state: dict, step: int) -> tuple[Tensor, dict]:
        ratio = global_active_fraction(state['active_count'], state['candidate_count'])
        progress = self.progress(step)
        penalty = (ratio - self.cfg.target_fraction).clamp_min(0).square()
        weight = self.cfg.loss_weight * progress
        logs = {key: value.detach() for key, value in state.items()
                if key not in ('active_count', 'candidate_count', 'scene_count')}
        logs.update(
            active_fraction=ratio.detach(),
            removed_fraction=1 - ratio.detach(),
            gaussians_before=(state['candidate_count'] / state['scene_count']).detach(),
            gaussians_after=(state['active_count'] / state['scene_count']).detach(),
            budget_loss=penalty.detach(), budget_weight=weight,
            target_fraction=self.cfg.target_fraction, progress=progress,
        )
        return penalty * weight, logs
