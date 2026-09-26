"""Train-only RoMa correspondences supervising the small student descriptor.

No pose estimation, Gaussian attribute loss, or extra rendering is performed.
Teacher images are the augmented context RGB BEFORE encoder normalization.
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor
import torch.nn.functional as F

from .roma_matching import Correspondences, RoMaMatcher, RoMaMatcherCfg


@dataclass
class DescriptorTeacherCfg:
    enabled: bool = False
    weight: float = 0.05
    every_n_steps: int = 1
    max_pairs: int = 512
    min_pairs: int = 16
    temperature: float = 0.1
    exclusion_pixels: float = 3.0
    matcher: RoMaMatcherCfg = field(default_factory=RoMaMatcherCfg)

    def __post_init__(self):
        if self.weight <= 0 or self.every_n_steps < 1 or self.temperature <= 0:
            raise ValueError("Teacher weight, interval and temperature must be positive")
        if not 2 <= self.min_pairs <= self.max_pairs or self.exclusion_pixels < 0:
            raise ValueError("Invalid descriptor pair count/exclusion radius")


def sample_features(features: Tensor, grid: Tensor, height: int, width: int):
    """[H*W,D] -> [P,D], at RoMa's pixel-center normalized coordinates."""
    image = features.detach().float().T.reshape(1, -1, height, width)
    return F.grid_sample(image, grid.detach().reshape(1, -1, 1, 2),
                         align_corners=False, padding_mode="border")[0, :, :, 0].T


def descriptor_objective(student, features_a: Tensor, features_b: Tensor,
                         pairs: Correspondences | None, image_shape: tuple[int, int],
                         cfg: DescriptorTeacherCfg):
    """Symmetric correspondence classification; only student parameters train.

    Nearby pairs in EITHER image are not negatives (including duplicate teacher
    samples). Ambiguous rows with no remaining negatives do not train the head.
    Empty/scheduled-off batches retain a zero-gradient connection for DDP.
    """
    zero = sum(p.reshape(-1)[0] * 0 for p in student.parameters())
    metrics = {"pairs": zero.detach(), "accuracy": zero.detach()}
    if pairs is None or len(pairs.confidence) < cfg.min_pairs:
        return zero, metrics
    height, width = image_shape
    # Teacher correspondences can be constants produced by inference_mode.
    grid_a, grid_b = pairs.grid_a.detach(), pairs.grid_b.detach()
    da = student.describe(sample_features(features_a, grid_a, height, width))
    db = student.describe(sample_features(features_b, grid_b, height, width))
    logits = da @ db.T / cfg.temperature
    pixels = grid_a.new_tensor([width / 2, height / 2])
    nearby = ((torch.cdist(grid_a * pixels, grid_a * pixels) < cfg.exclusion_pixels)
              | (torch.cdist(grid_b * pixels, grid_b * pixels) < cfg.exclusion_pixels))
    nearby.fill_diagonal_(False)
    logits = logits.masked_fill(nearby, -torch.inf)
    usable = (~nearby).sum(-1) > 1
    weights = pairs.confidence.detach().clamp(0, 1) * usable
    labels = torch.arange(len(logits), device=logits.device)
    losses = (F.cross_entropy(logits, labels, reduction="none")
              + F.cross_entropy(logits.T, labels, reduction="none")) * .5
    loss = (losses * weights).sum() / weights.sum().clamp_min(1e-6)
    accuracy = ((logits.argmax(1) == labels).float() + (logits.argmax(0) == labels).float()) * .5
    metrics = {"pairs": usable.sum().float(),
               "accuracy": (accuracy * weights).sum().detach() / weights.sum().clamp_min(1e-6)}
    return loss + zero, metrics


class DescriptorTeacher:
    """Plain owner, so RoMa is never registered in the model or checkpoint."""

    def __init__(self, cfg: DescriptorTeacherCfg):
        self.cfg = cfg
        self.matcher = RoMaMatcher(cfg.matcher)

    @torch.no_grad()
    def prepare(self, rgb: Tensor, step: int):
        if step % self.cfg.every_n_steps:
            return None
        if rgb.ndim != 5 or rgb.shape[1:3] != (2, 3):
            raise ValueError("The current teacher adapter requires [B,2,3,H,W] RGB")
        # One rotating scene per rank/step, before allocating the encoder graph.
        scene = (step // self.cfg.every_n_steps) % len(rgb)
        pairs = self.matcher.match(rgb[scene].detach().float())
        # Balanced RoMa sampling is retained; avoid taking only highest scores,
        # which can concentrate all supervision in one easy image region.
        order = torch.randperm(len(pairs.confidence), device=rgb.device)[:self.cfg.max_pairs]
        pairs = Correspondences(pairs.grid_a[order], pairs.grid_b[order], pairs.confidence[order])
        return scene, pairs

    def loss(self, student, features: Tensor, prepared, image_shape: tuple[int, int]):
        split = image_shape[0] * image_shape[1]
        scene, pairs = (0, None) if prepared is None else prepared
        with torch.autocast(device_type=features.device.type, enabled=False):
            return descriptor_objective(student, features[scene, :split],
                                        features[scene, split:], pairs, image_shape, self.cfg)
