"""Independent compact appearance readers on ONE shared candidate graph.

Each head has the existing reader's own key/value, query, position and score
projections. Softmax is incoming per slot and per head. Values stay compact
on edges; only the resulting slot contexts are concatenated across heads.
"""

from itertools import combinations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .appearance_neighborhood import incoming_softmax, merge_appearance_neighbors
from .image_neighborhood_appearance import ImageNeighborhoodAppearance, appearance_statistics
from .tensor_kernels import run_tensor_kernel


class MultiHeadAppearance(nn.Module):
    def __init__(self, feature_dim: int, head_dim: int, num_heads: int, *,
                 compile_kernels: bool = False, batch_projections: bool = False):
        super().__init__()
        self.compile_kernels = compile_kernels
        self.batch_projections = batch_projections
        self.heads = nn.ModuleList([
            ImageNeighborhoodAppearance(feature_dim, head_dim, compile_kernels=compile_kernels)
            for _ in range(num_heads)
        ])

    def forward(self, points, features, rgb, means, pooled, geometry_neighbors, image, *,
                chunk_size: int, checkpoint_chunks: bool, collect_statistics: bool = False):
        neighbors, valid = merge_appearance_neighbors(geometry_neighbors, image, chunk_size)
        geometry_k = geometry_neighbors.shape[1]
        added = (torch.arange(neighbors.shape[1], device=points.device) >= geometry_k).to(points.dtype)
        source = torch.cat((features, rgb), -1)

        def run(head, function, *args):
            # Bind this head here; backward must not capture the loop's last head.
            def operation(*inputs):
                return run_tensor_kernel(function, head, *inputs, enabled=self.compile_kernels)
            if checkpoint_chunks and self.training and torch.is_grad_enabled():
                return checkpoint(operation, *args, use_reentrant=False, preserve_rng_state=False)
            return operation(*args)

        def edge_args(start, stop):
            return (points[start:stop], means, image.xy[start:stop], image.xy,
                    image.view_ids[start:stop], image.view_ids, neighbors[start:stop], added)

        all_sources = all_queries = None
        if self.batch_projections:
            # Batch only SUPPORT/SLOT projections, never [M,K,H,D] edge tensors.
            # Concatenation preserves independent parameters and their gradients.
            all_sources = F.linear(
                source, torch.cat([head.source.weight for head in self.heads]),
                torch.cat([head.source.bias for head in self.heads]),
            ).split([2 * head.context_dim for head in self.heads], dim=-1)
            all_queries = F.linear(
                pooled, torch.cat([head.query.weight for head in self.heads]),
            ).split([head.context_dim for head in self.heads], dim=-1)

        contexts, reports, diagnostic_weights = [], [], []
        for index, head in enumerate(self.heads):
            projected = head.source(source) if all_sources is None else all_sources[index]
            keys, values = projected.split(head.context_dim, -1)
            queries = head.query(pooled) if all_queries is None else all_queries[index]
            scores = []
            for start in range(0, len(points), chunk_size):
                stop = start + chunk_size
                scores.append(run(head, ImageNeighborhoodAppearance._score_chunk,
                                  keys[start:stop], queries,
                                  *edge_args(start, stop)))
            weights = incoming_softmax(torch.cat(scores), neighbors, valid)
            del scores  # Chunk logits need not coexist with messages of the next stage.
            context = points.new_zeros(len(points), head.context_dim)
            for start in range(0, len(points), chunk_size):
                stop = start + chunk_size
                if self.compile_kernels:
                    contribution = run(head, ImageNeighborhoodAppearance._aggregate_chunk,
                                       values[start:stop], weights[start:stop], *edge_args(start, stop))
                    context.add_(contribution)
                    del contribution
                else:
                    messages = run(head, ImageNeighborhoodAppearance._message_chunk,
                                   values[start:stop], weights[start:stop], *edge_args(start, stop))
                    context.index_add_(0, neighbors[start:stop].flatten(),
                                       messages.reshape(-1, head.context_dim))
                    del messages
            contexts.append(context)
            if collect_statistics:
                reports.append(appearance_statistics(weights, valid, image.valid, geometry_k))
                diagnostic_weights.append(weights.detach())

        statistics = None
        if collect_statistics:
            with torch.no_grad():
                statistics = {key: torch.stack([report[key] for report in reports]).mean()
                              for key in reports[0]}
                for index, (report, context) in enumerate(zip(reports, contexts)):
                    for key in ('entropy', 'image_weight'):
                        statistics[f'head_{index}/{key}'] = report[key]
                    statistics[f'head_{index}/context_rms'] = context.detach().square().mean().sqrt()
                # Mean incoming TV across all head pairs. Do not compare raw
                # logit magnitudes: head softmaxes have independent offsets.
                distances = [.5 * (a - b).abs().sum() / len(points)
                             for a, b in combinations(diagnostic_weights, 2)]
                statistics['head_attention_tv'] = (torch.stack(distances).mean() if distances
                                                   else points.new_zeros(()))
        return torch.cat(contexts, -1), statistics
