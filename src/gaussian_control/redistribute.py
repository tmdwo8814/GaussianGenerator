"""Mass selection and support reallocation on the original learned graph."""

import torch
from scipy.spatial import cKDTree

from ..model.encoder.common.appearance_moments import appearance_feature_std
from ..model.encoder.common.separate_appearance import appearance_weights
from ..model.encoder.common.sparse_feature_pool import pool_features


def redistribute_edges(decoder, cache, count):
    """Return open IDs, updated graph, q, and appearance logits (all sources kept)."""
    points, features = cache['points'], cache['features']
    size = len(points)
    # Stable mass ties keep the smaller original raster-slot ID.
    keep = torch.argsort(cache['mass'], descending=True, stable=True)[:count].sort().values
    opened = torch.zeros(size, device=points.device, dtype=torch.bool)
    opened[keep] = True
    neighbors = cache['neighbors'].clone()
    valid = opened[neighbors]
    logits = cache['logits'].masked_fill(~valid, -torch.inf)
    scores = cache['appearance_scores'].clone()
    missing = ~valid.any(-1)
    rows = missing.nonzero().flatten()
    if rows.numel():
        # Exact nearest OPEN ORIGINAL SUPPORT, not the decoded Gaussian mean.
        # This CPU fallback and its transfers are included in adaptation time.
        coordinates = points.detach().float().cpu().numpy()
        _, nearest = cKDTree(coordinates[keep.cpu().numpy()]).query(
            coordinates[rows.cpu().numpy()], k=1, workers=decoder.cfg.knn_workers,
        )
        destinations = keep[torch.as_tensor(nearest, device=points.device).long()]
        neighbors[rows, 0] = destinations
        logits[rows, 0] = 0  # sole candidate -> all source budget
        delta = points[rows] - points[destinations]
        pair = torch.cat((features[destinations], features[rows], delta,
                          delta.square().sum(-1, keepdim=True)), -1)
        # Re-evaluate the appearance score of the newly created fallback edge.
        hidden = decoder.allocation_head[1](decoder.allocation_head[0](pair))
        scores[rows, 0] = decoder.appearance_head(hidden).squeeze(-1)
    q = logits.softmax(-1) * cache['budget']
    return keep, opened, neighbors, q, scores, len(rows)


@torch.no_grad()
def redistribute(decoder, cache, count):
    points, features = cache['points'], cache['features']
    if not 1 <= count <= len(points):
        raise ValueError('Invalid redistribution target')
    keep, opened, neighbors, q, scores, fallback = redistribute_edges(decoder, cache, count)
    mass, means, covariance, pooled, geometry = decoder.aggregate_moments(
        points, features, neighbors, q, return_weights=True, destination_mask=opened,
    )
    selected = appearance_weights(geometry, scores, neighbors)
    appearance_pooled = pool_features(features, selected, neighbors, decoder.cfg.chunk_size,
                                      compile_kernels=decoder.cfg.compile_kernels)
    spread = None
    if decoder.cfg.appearance_moments:
        spread = appearance_feature_std(
            features, selected, neighbors, appearance_pooled, decoder.cfg.chunk_size,
            compile_kernels=decoder.cfg.compile_kernels,
            epsilon=decoder.cfg.appearance_variance_epsilon,
        )[keep]
    image = decoder._image_neighbors(cache['image_shape'], points.device)
    context, _ = decoder.appearance_reader(
        points, features, cache['rgb'], means, pooled, neighbors, image,
        chunk_size=decoder.cfg.chunk_size, checkpoint_chunks=False,
        destination_mask=opened,
    )
    sh = decoder._capacity_harmonics(appearance_pooled[keep], context[keep], None, spread)
    gaussians = decoder.build_gaussians(
        mass[keep], means[keep], covariance[keep], pooled[keep], cache['scene_scale'],
        appearance_harmonics=sh,
    )
    return gaussians, {'fallback_supports': fallback, 'open_slots': count}
