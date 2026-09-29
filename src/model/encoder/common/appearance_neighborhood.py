"""Geometry edges plus same-view image-grid edges, for appearance only.

neighbors[j, k] = destination slot i. Every row is an OUTGOING candidate set;
appearance softmax normalizes INCOMING edges per slot, not per source row.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class ImageNeighborhood:
    neighbors: Tensor
    valid: Tensor
    xy: Tensor
    view_ids: Tensor


def make_image_neighborhood(image_shape: tuple[int, int, int], radii: list[int],
                            device: torch.device) -> ImageNeighborhood:
    """View-major raster ordering; no border clamping/wrapping of valid edges.

Eight offsets for each radius, in row-major order, excluding self. Invalid
edges carry a safe self index but remain masked out everywhere downstream.
"""
    views, height, width = image_shape
    source = torch.arange(views * height * width, device=device)
    view_ids = source // (height * width)
    y, x = (source // width) % height, source % width
    offsets = [(dx * radius, dy * radius) for radius in radii
               for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy]
    offsets = torch.tensor(offsets, device=device, dtype=torch.long).reshape(-1, 2)
    nx, ny = x[:, None] + offsets[:, 0], y[:, None] + offsets[:, 1]
    valid = (nx >= 0) & (nx < width) & (ny >= 0) & (ny < height)
    indices = view_ids[:, None] * height * width + ny * width + nx
    indices = torch.where(valid, indices, source[:, None])
    xy = torch.stack((x / max(width - 1, 1), y / max(height - 1, 1)), -1)
    return ImageNeighborhood(indices, valid, xy, view_ids)


def merge_appearance_neighbors(geometry: Tensor, image: ImageNeighborhood,
                               chunk_size: int) -> tuple[Tensor, Tensor]:
    """Keep geometry first; mask image edges already present in that row.

No compaction is needed: masks avoid variable-size batches, and the number
of allocated columns stays K + 8*len(radii). Radii must be positive/unique.
"""
    added = image.valid.clone()
    for start in range(0, len(geometry), chunk_size):
        stop = start + chunk_size
        duplicate = (image.neighbors[start:stop, :, None]
                     == geometry[start:stop, None, :]).any(-1)
        added[start:stop] &= ~duplicate
    return (torch.cat((geometry, image.neighbors), -1),
            torch.cat((torch.ones_like(geometry, dtype=torch.bool), added), -1))


class _IncomingSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, neighbors, valid):
        destinations = neighbors.flatten()
        masked = logits.masked_fill(~valid, -torch.inf)
        maximum = logits.new_full((len(logits),), -torch.inf)
        maximum.scatter_reduce_(0, destinations, masked.flatten(), reduce='amax', include_self=True)
        exponential = torch.where(valid, masked - maximum[neighbors], -torch.inf).exp()
        denominator = logits.new_zeros(len(logits)).index_add(0, destinations, exponential.flatten())
        weights = exponential / denominator[neighbors].clamp_min(torch.finfo(logits.dtype).tiny)
        ctx.save_for_backward(weights, neighbors)
        return weights

    @staticmethod
    def backward(ctx, grad_output):
        weights, neighbors = ctx.saved_tensors
        weighted_grad = weights * grad_output
        incoming = weights.new_zeros(len(weights)).index_add(
            0, neighbors.flatten(), weighted_grad.flatten(),
        )
        return weights * (grad_output - incoming[neighbors]), None, None


def incoming_softmax(logits: Tensor, neighbors: Tensor, valid: Tensor) -> Tensor:
    """Stable sparse softmax with a direct backward and exactly zero masked edges."""
    return _IncomingSoftmax.apply(logits, neighbors, valid)
