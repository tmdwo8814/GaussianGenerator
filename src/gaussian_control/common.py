from dataclasses import fields
import math
from time import perf_counter

import torch

from ..model.types import Gaussians


METHODS = ('prune', 'merge', 'redistribute')
PERCENTAGES = (100, 75, 50, 25, 10, 5, 1)


def target_count(count: int, percent: float) -> int:
    if count < 1 or not math.isfinite(percent) or not 0 < percent <= 100:
        raise ValueError('Require count > 0 and 0 < percent <= 100')
    return max(1, math.floor(count * percent / 100))


def gaussian_dict(value: Gaussians) -> dict:
    return {field.name: getattr(value, field.name) for field in fields(Gaussians)}


def select_gaussians(value: Gaussians, indices: torch.Tensor) -> Gaussians:
    return Gaussians(**{key: tensor[:, indices] for key, tensor in gaussian_dict(value).items()})


def uniform_pixel_indices(image_shape, count, device='cpu'):
    """Exact per-view quotas; stratify BOTH image axes, without flatten strides."""
    views, height, width = map(int, image_shape)
    if min(views, height, width) < 1 or not 1 <= count <= views * height * width:
        raise ValueError('Invalid raster layout or target count')
    selected = []
    quota, remainder = divmod(count, views)
    for view in range(views):
        n = quota + (view < remainder)
        if not n:
            continue
        rows = min(height, n, max(math.ceil(n / width), round(math.sqrt(n * height / width))))
        per_row, extra = divmod(n, rows)
        # Spread the rows with one extra sample across the entire image.
        row_ids = torch.arange(rows, device=device)
        columns = per_row + ((row_ids + 1) * extra // rows - row_ids * extra // rows)
        row = torch.repeat_interleave(row_ids, columns, output_size=n)
        starts = columns.cumsum(0) - columns
        column = torch.arange(n, device=device) - starts[row]
        y = (2 * row + 1) * height // (2 * rows)
        x = (2 * column + 1) * width // (2 * columns[row])
        selected.append(view * height * width + y * width + x)
    return torch.cat(selected).long()


def synchronized_time(function, device):
    """Wall time includes CPU work and device transfers inside function only."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    start = perf_counter()
    result = function()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return result, perf_counter() - start


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(to_device(item, device) for item in value)
    return value
