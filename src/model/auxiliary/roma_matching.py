"""Thin, frozen RoMaV2 adapter. No dependency on cameras or the Gaussian model."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch
from torch import Tensor


@contextmanager
def matching_precision():
    """Meet RoMaV2's FP32 requirement without changing backbone/decoder TF32."""
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


@dataclass
class RoMaMatcherCfg:
    setting: str = "fast"
    num_matches: int = 2048
    min_confidence: float = 0.5

    def __post_init__(self):
        if self.setting not in ("turbo", "fast", "base", "precise"):
            raise ValueError("Unsupported RoMaV2 setting")
        if self.num_matches < 8 or not 0 <= self.min_confidence <= 1:
            raise ValueError("Require num_matches >= 8 and confidence in [0, 1]")


@dataclass
class Correspondences:
    # RoMaV2 / grid_sample coordinates: [-1, 1], align_corners=False.
    grid_a: Tensor
    grid_b: Tensor
    confidence: Tensor


class RoMaMatcher:
    """Plain Python owner: matcher weights never enter Lightning/DDP checkpoints.

    Initialize after Lightning assigns the rank's CUDA device. RoMaV2 has several
    module-level torch.device("cuda") allocations, so guard EVERY upstream call
    with the correct current device. Imports and downloads are lazy.
    """

    def __init__(self, cfg: RoMaMatcherCfg):
        self.cfg = cfg
        self._model = None
        self._device = None

    def initialize(self, device: torch.device):
        device = torch.device(device)
        if self._model is not None:
            if device != self._device:
                raise RuntimeError("RoMaV2 was initialized on a different device")
            return
        if device.type != "cuda":
            raise RuntimeError("The RoMaV2 descriptor teacher requires CUDA")
        try:
            from romav2 import RoMaV2
        except ImportError as error:
            raise ImportError(
                "RoMaV2 is not installed. See docs/cross_view_verifier.md and "
                "install the local RoMaV2 clone into the training environment."
            ) from error
        with torch.cuda.device(device):
            model = RoMaV2(RoMaV2.Cfg(setting=self.cfg.setting, compile=False))
            model = model.to(device).eval().requires_grad_(False)
        self._model, self._device = model, device

    def release(self):
        self._model = None
        self._device = None

    @torch.no_grad()
    def match(self, images: Tensor) -> Correspondences:
        """Match one pair [2,3,H,W] of ACTUAL augmented RGB images in [0,1]."""
        if images.ndim != 4 or images.shape[:2] != (2, 3):
            raise ValueError("RoMaV2 expects one pair with shape [2,3,H,W]")
        self.initialize(images.device)
        guard = torch.cuda.device(images.device) if images.is_cuda else nullcontext()
        with guard, matching_precision(), torch.autocast(images.device.type, enabled=False):
            # Upstream tensors must be BCHW, even for a single pair.
            predictions = self._model.match(images[0:1], images[1:2])
            positive = 0
            for direction in ("AB", "BA"):
                warp = predictions.get("warp_" + direction)
                if warp is None:
                    continue
                key = "overlap_" + direction
                confidence = torch.nan_to_num(
                    predictions[key], nan=0.0, posinf=0.0, neginf=0.0
                ).clamp(0, 1)
                predictions[key] = confidence
                limit = 1 - 1 / warp.shape[1]
                valid = torch.isfinite(warp).all(-1) & (warp.abs().amax(-1) <= limit)
                positive += int(((confidence[..., 0] > 0) & valid).sum().item())
            # Upstream sample() draws 4*num_matches before balanced sampling.
            count = min(self.cfg.num_matches, positive // 4)
            if count == 0:
                empty = images.new_empty((0, 2))
                return Correspondences(empty, empty.clone(), images.new_empty(0))
            matches, confidence, _, _ = self._model.sample(predictions, count)

        # match() uses inference_mode. Clone OUTSIDE that context so the constants
        # can safely be saved by autograd during descriptor supervision.
        matches = matches.detach().float().clone()
        confidence = confidence.detach().float().reshape(-1).clone()
        height, width = images.shape[-2:]
        bounds = matches.new_tensor([1 - 1 / width, 1 - 1 / height] * 2)
        valid = (
            torch.isfinite(matches).all(-1) & torch.isfinite(confidence)
            & (confidence >= self.cfg.min_confidence)
            & (confidence > 0) & (matches.abs() <= bounds).all(-1)
        )
        return Correspondences(matches[valid, :2], matches[valid, 2:],
                               confidence[valid].clamp_max(1))
