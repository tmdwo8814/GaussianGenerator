"""Read contextual appearance on the union of 3D and source-image neighbors.

Only the compact appearance context is returned. Geometry budgets, moments,
coverage and opacity never receive the extra image-grid edges.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .appearance_neighborhood import ImageNeighborhood, incoming_softmax, merge_appearance_neighbors


class ImageNeighborhoodAppearance(nn.Module):
    def __init__(self, feature_dim: int, context_dim: int):
        super().__init__()
        self.context_dim = context_dim
        # Project once per support/slot instead of per edge at DPT width 256.
        self.source = nn.Linear(feature_dim + 3, 2 * context_dim)
        self.query = nn.Linear(feature_dim, context_dim, bias=False)
        # xyz delta, squared distance, same-view xy delta, same-view flag,
        # new-image-edge flag. Cross-view image coordinate deltas are zero.
        self.score_position = nn.Linear(8, context_dim)
        self.value_position = nn.Linear(8, context_dim)
        self.score = nn.Linear(context_dim, 1, bias=False)
        nn.init.normal_(self.score.weight, std=1e-3)

    @staticmethod
    def _edge_geometry(source_points, means, source_xy, xy, source_views, view_ids,
                       neighbors, added):
        delta = source_points[:, None] - means[neighbors]
        same_view = source_views[:, None] == view_ids[neighbors]
        delta_xy = (source_xy[:, None] - xy[neighbors]) * same_view[..., None]
        return torch.cat((delta, delta.square().sum(-1, keepdim=True), delta_xy,
                          same_view[..., None].to(delta.dtype),
                          added[None, :, None].expand(len(delta), -1, -1)), -1)

    def _score_chunk(self, keys, queries, source_points, means, source_xy, xy,
                     source_views, view_ids, neighbors, added):
        geometry = self._edge_geometry(source_points, means, source_xy, xy,
                                       source_views, view_ids, neighbors, added)
        hidden = keys[:, None] + queries[neighbors] + self.score_position(geometry)
        return self.score(F.silu(hidden)).squeeze(-1)

    def _message_chunk(self, values, weights, source_points, means, source_xy, xy,
                       source_views, view_ids, neighbors, added):
        geometry = self._edge_geometry(source_points, means, source_xy, xy,
                                       source_views, view_ids, neighbors, added)
        message = F.silu(values[:, None] + self.value_position(geometry))
        return weights[..., None] * message

    def forward(self, points: Tensor, features: Tensor, rgb: Tensor, means: Tensor,
                pooled: Tensor, geometry_neighbors: Tensor, image: ImageNeighborhood, *,
                chunk_size: int, checkpoint_chunks: bool, collect_statistics: bool = False):
        neighbors, valid = merge_appearance_neighbors(geometry_neighbors, image, chunk_size)
        geometry_k = geometry_neighbors.shape[1]
        added = (torch.arange(neighbors.shape[1], device=points.device) >= geometry_k).to(points.dtype)
        keys, values = self.source(torch.cat((features, rgb), -1)).split(self.context_dim, -1)
        queries = self.query(pooled)

        def run(function, *args):
            if checkpoint_chunks and self.training and torch.is_grad_enabled():
                return checkpoint(function, *args, use_reentrant=False, preserve_rng_state=False)
            return function(*args)

        def edge_args(start, stop):
            return (points[start:stop], means, image.xy[start:stop], image.xy,
                    image.view_ids[start:stop], image.view_ids, neighbors[start:stop], added)

        scores = []
        for start in range(0, len(points), chunk_size):
            stop = start + chunk_size
            scores.append(run(self._score_chunk, keys[start:stop], queries, *edge_args(start, stop)))
        # This normalization includes the new edges independently of q_geometry.
        # Multiplying by q_geometry would set every new image edge to zero.
        weights = incoming_softmax(torch.cat(scores), neighbors, valid)
        context = points.new_zeros(len(points), self.context_dim)
        for start in range(0, len(points), chunk_size):
            stop = start + chunk_size
            messages = run(self._message_chunk, values[start:stop], weights[start:stop],
                           *edge_args(start, stop))
            context.index_add_(0, neighbors[start:stop].flatten(), messages.reshape(-1, self.context_dim))
        statistics = None
        if collect_statistics:
            statistics = appearance_statistics(weights, valid, image.valid, geometry_k)
        return context, statistics


@torch.no_grad()
def appearance_statistics(weights, valid, image_valid, geometry_k):
    count = len(weights)
    extra = valid[:, geometry_k:]
    extra_count = extra.sum()
    statistics = {
        'added_candidates': extra_count / count,
        'added_candidate_fraction': extra_count / valid.sum().clamp_min(1),
        'new_2d_fraction': extra_count / image_valid.sum().clamp_min(1),
        'incoming_candidates': valid.sum() / count,
        'image_weight': weights[:, geometry_k:].sum() / count,
        'entropy': -(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum() / count,
    }
    # Per-radius weights, ordered like appearance_2d_radii (default: 1, 4).
    for index in range(image_valid.shape[1] // 8):
        start = geometry_k + 8 * index
        statistics[f'image_weight_scale_{index}'] = weights[:, start:start + 8].sum() / count
    return statistics


@torch.no_grad()
def image_error_statistics(prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
    """Logging only: RGB MSE on strong image differences, not a new loss.

Target RGB is [0,1], [...,3,H,W]. An edge pixel has mean absolute horizontal
or vertical RGB difference > 0.05. This is a texture/edge heuristic, not a
semantic or depth boundary label. It is never passed into the encoder.
"""
    dx = (target[..., :, :, 1:] - target[..., :, :, :-1]).abs().mean(-3)
    dy = (target[..., :, 1:, :] - target[..., :, :-1, :]).abs().mean(-3)
    edge = torch.maximum(F.pad(dx, (0, 1)), F.pad(dy, (0, 0, 0, 1))) > .05
    error = (prediction - target).square().mean(-3)

    def masked_mean(mask):
        return torch.where(mask, error, 0).sum() / mask.sum().clamp_min(1)

    return {'edge_mse': masked_mean(edge), 'smooth_mse': masked_mean(~edge),
            'edge_fraction': edge.to(error.dtype).mean()}
