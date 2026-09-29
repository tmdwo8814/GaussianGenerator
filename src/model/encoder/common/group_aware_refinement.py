"""One allocation refinement using the group currently gathered by each slot.

Rows j are source supports; neighbors[j, k] are destination slots i. The
outgoing softmax preserves each support's budget. Group statistics instead
normalize INCOMING mass, over all supports that sent mass to a slot.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .sparse_feature_pool import pool_features


@dataclass
class SlotGroups:
    mass: Tensor
    weights: Tensor
    means: Tensor
    appearance: Tensor
    disagreement: Tensor


def gather_slot_groups(points: Tensor, appearance: Tensor, neighbors: Tensor,
                       allocation: Tensor, epsilon: float, chunk_size: int) -> SlotGroups:
    """Differentiable first moments and scalar SH disagreement (no covariance).

The epsilon self contribution stabilizes empty slots without adding opacity.
Appearance is the existing SH head's masked coefficients, not target RGB.
"""
    mass = allocation.new_zeros(len(points)).index_add(
        0, neighbors.flatten(), allocation.flatten()
    )
    prior = allocation.new_zeros(1, neighbors.shape[1])
    prior[:, 0] = epsilon  # The existing kNN builder guarantees self first.
    weights = (allocation + prior) / (mass[neighbors] + epsilon)
    packed = torch.cat((points, appearance, appearance.square().sum(-1, keepdim=True)), -1)
    pooled = pool_features(packed, weights, neighbors, chunk_size)
    means, mean_appearance = pooled[:, :3], pooled[:, 3:-1]
    disagreement = (pooled[:, -1] - mean_appearance.square().sum(-1)).clamp_min(0)
    return SlotGroups(mass, weights, means, mean_appearance, disagreement)


class GroupAwareRefinement(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(7, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1, bias=False),
        )
        # q_final == q_coarse at initialization. A common output bias would
        # cancel under the outgoing softmax, so the last layer has no bias.
        nn.init.zeros_(self.score[-1].weight)

    def _score_chunk(self, source_points, source_appearance, logits, neighbors,
                     means, mean_appearance, disagreement, log_mass):
        delta = source_points[:, None] - means[neighbors]
        appearance_error = (source_appearance[:, None] - mean_appearance[neighbors]).square().sum(-1)
        inputs = torch.cat((
            logits[..., None], delta, appearance_error[..., None],
            disagreement[neighbors, None], log_mass[neighbors, None],
        ), dim=-1)
        return self.score(inputs).squeeze(-1)

    def forward(self, points: Tensor, appearance: Tensor, neighbors: Tensor,
                coarse: Tensor, logits: Tensor, budget: Tensor, *, epsilon: float,
                chunk_size: int, checkpoint_chunks: bool, collect_statistics: bool = False):
        groups = gather_slot_groups(points, appearance, neighbors, coarse, epsilon, chunk_size)
        log_mass = groups.mass.log1p()
        chunks = []
        for start in range(0, len(points), chunk_size):
            stop = start + chunk_size
            args = (points[start:stop], appearance[start:stop], logits[start:stop],
                    neighbors[start:stop], groups.means, groups.appearance,
                    groups.disagreement, log_mass)
            if checkpoint_chunks and self.training and torch.is_grad_enabled():
                update = checkpoint(self._score_chunk, *args, use_reentrant=False,
                                    preserve_rng_state=False)
            else:
                update = self._score_chunk(*args)
            chunks.append(update)
        update = torch.cat(chunks, 0)
        refined = budget * (logits + update).softmax(-1)
        statistics = None
        if collect_statistics:
            statistics = refinement_statistics(
                points, appearance, neighbors, groups, refined, logits, update, epsilon, chunk_size,
            )
        return refined, statistics


@torch.no_grad()
def refinement_statistics(points, appearance, neighbors, before, refined,
                          logits, update, epsilon, chunk_size):
    """Detached [M,8] diagnostics; only computed on requested logging steps.

Columns: outgoing TV, centered logit update squared, then incoming
concentration before/after, entropy before/after, SH variance before/after.
"""
    after = gather_slot_groups(points, appearance, neighbors, refined, epsilon, chunk_size)
    w0, w1 = before.weights, after.weights
    tiny = torch.finfo(w0.dtype).tiny
    edge_stats = torch.stack((w0.square(), w1.square(),
                              -w0 * w0.clamp_min(tiny).log(),
                              -w1 * w1.clamp_min(tiny).log()), -1)
    incoming = w0.new_zeros(len(points), 4).index_add(
        0, neighbors.flatten(), edge_stats.reshape(-1, 4)
    )
    tv = .5 * ((logits + update).softmax(-1) - logits.softmax(-1)).abs().sum(-1)
    centered_update = update - update.mean(-1, keepdim=True)
    return torch.cat((tv[:, None], centered_update.square().mean(-1, keepdim=True),
                      incoming, before.disagreement[:, None], after.disagreement[:, None]), -1)
