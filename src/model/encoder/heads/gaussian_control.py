"""Frozen Final analysis. Source rows always retain their original raster IDs.

Intermediate scatter buffers keep M slots to reuse Final's pooling/readers;
only the selected L slots enter the attribute heads and CUDA renderer.
"""

from math import floor, isfinite

import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch import Tensor

from ...types import Gaussians
from ..common.separate_appearance import appearance_weights
from ..common.sparse_feature_pool import pool_features


ATTRIBUTES = ('means', 'covariances', 'harmonics', 'opacities')


def select_slots(mass: Tensor, fraction: float) -> Tensor:
    """Raw incoming mass; ties favor the smaller ORIGINAL slot ID.

    Return raster order so both methods also share the renderer's input order.
    """
    if not isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('Retention fraction must be in (0, 1]')
    if mass.ndim != 1 or not len(mass) or not bool(torch.isfinite(mass).all()):
        raise ValueError('Selection requires finite incoming mass [M]')
    count = max(1, floor(fraction * len(mass)))
    return torch.argsort(mass, descending=True, stable=True)[:count].sort().values


def subset_gaussians(gaussians: Gaussians, slots: Tensor) -> Gaussians:
    """Gather attributes verbatim, physically shrinking the Gaussian dimension."""
    return Gaussians(**{key: getattr(gaussians, key).index_select(1, slots)
                        for key in ATTRIBUTES})


@torch.no_grad()
def restricted_allocation(decoder, cache: dict, slots: Tensor):
    points, features = cache['points'], cache['features']
    active = torch.zeros(len(points), dtype=torch.bool, device=points.device)
    active[slots] = True
    neighbors = cache['neighbors'].clone()
    valid = active[neighbors]
    orphan = ~valid.any(-1)
    logits = cache['logits'].clone()
    scores = cache['scores'].clone()

    if bool(orphan.any()):
        sources = orphan.nonzero().flatten()
        # Exact 3D query of ONLY orphan supports, against ONLY open slots.
        # No M x L distance tensor. CPU transfer/search is part of decode time.
        coordinates = cache['search_points'].detach().double().cpu().numpy()
        tree = cKDTree(coordinates[slots.cpu().numpy()])
        _, nearest = tree.query(coordinates[sources.cpu().numpy()], k=1, eps=0, workers=1)
        destinations = slots[torch.as_tensor(nearest, device=slots.device, dtype=torch.long)]
        neighbors[sources, 0] = destinations
        valid[sources, 0] = True
        logits[sources, 0] = 0  # The sole valid destination receives all budget.

        # Evaluate the existing frozen score head for each NEW geometry edge.
        # Match Final's split projections and operation order, in bounded chunks.
        first = decoder.allocation_head[0]
        dim = decoder.cfg.feature_dim
        for start in range(0, len(sources), decoder.cfg.chunk_size):
            src = sources[start:start + decoder.cfg.chunk_size]
            dst = destinations[start:start + decoder.cfg.chunk_size]
            delta = points[src] - points[dst]
            geometry = torch.cat((delta, delta.square().sum(-1, keepdim=True)), -1)
            hidden = (F.linear(features[dst], first.weight[:, :dim])
                      + F.linear(features[src], first.weight[:, dim:2 * dim])
                      + F.linear(geometry, first.weight[:, 2 * dim:], first.bias))
            scores[src, 0] = decoder.appearance_head(F.silu(hidden)).squeeze(-1)

    allocation = logits.masked_fill(~valid, -torch.inf).softmax(-1)
    q = allocation * cache['budget'][:, None]
    if not bool(torch.isfinite(q).all()):
        raise RuntimeError('Non-finite restricted allocation')
    return neighbors, valid, q, scores, active, orphan.float().mean()


@torch.no_grad()
def redistribute_gaussians(decoder, cache: dict, slots: Tensor):
    """Recompute moments, scale, appearance context and SH using frozen heads."""
    if decoder.training or not decoder.cfg.appearance_capacity:
        raise ValueError('Redistribution requires the frozen Final capacity decoder in eval mode')
    with torch.autocast(device_type=cache['points'].device.type, enabled=False):
        neighbors, valid, q, scores, active, fallback = restricted_allocation(decoder, cache, slots)
        mass, means, covariance, pooled, geometry = decoder.aggregate_moments(
            cache['points'], cache['features'], neighbors, q,
            return_weights=True, active_slots=active,
        )
        selected = appearance_weights(geometry, scores, neighbors)
        appearance_pooled = pool_features(
            cache['features'], selected, neighbors, decoder.cfg.chunk_size,
            compile_kernels=decoder.cfg.compile_kernels,
        )
        context, _ = decoder.appearance_reader(
            cache['points'], cache['features'], cache['rgb'], means, pooled,
            neighbors, cache['image'], chunk_size=decoder.cfg.chunk_size,
            checkpoint_chunks=False, geometry_valid=valid, destination_mask=active,
        )
        gaussians = decoder.build_gaussians(
            mass[slots], means[slots], covariance[slots], pooled[slots], cache['scene_scale'],
            appearance_context=context[slots], appearance_pooled=appearance_pooled[slots],
        )
    before = cache['budget'].double().sum()
    after = mass[slots].double().sum()
    relative_error = (after - before).abs() / before.clamp_min(1e-30)
    if not bool(torch.isfinite(relative_error)) or relative_error.item() > 1e-5:
        raise RuntimeError(f'Redistribution did not conserve budget: relative error={relative_error.item()}')
    if gaussians.means.shape[1] != len(slots):
        raise RuntimeError('Redistribution produced the wrong number of Gaussians')
    return gaussians, mass[slots], float(fallback)


@torch.no_grad()
def gaussian_statistics(gaussians: Gaussians, cache: dict, slots: Tensor,
                        mass: Tensor, fallback: float = 0.) -> dict[str, float]:
    before = cache['budget'].double().sum()
    after = mass.double().sum()
    original = cache['mass'].double().sum()
    result = {
        'num_gaussians': float(gaussians.means.shape[1]),
        'retained_fraction': len(slots) / len(cache['points']),
        'fallback_support_fraction': fallback,
        'budget_before': float(before),
        'mass_after': float(after),
        'budget_error_abs': float((after - before).abs()),
        'budget_error_relative': float((after - before).abs() / before.clamp_min(1e-30)),
        'original_mass_retained_fraction': float(cache['mass'][slots].double().sum() / original.clamp_min(1e-30)),
    }
    trace = gaussians.covariances[0].diagonal(dim1=-2, dim2=-1).sum(-1)
    result['covariance_trace_mean'] = float(trace.mean())
    for name, values in (('opacity', gaussians.opacities[0]),
                         ('covariance_radius', trace.clamp_min(0).sqrt())):
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError(f'Non-finite Gaussian {name}')
        result[f'{name}_mean'] = float(values.mean())
        result[f'{name}_min'] = float(values.min())
        result[f'{name}_max'] = float(values.max())
        for label, value in zip(('p10', 'p50', 'p90'), torch.quantile(values, values.new_tensor([.1, .5, .9]))):
            result[f'{name}_{label}'] = float(value)
    return result
