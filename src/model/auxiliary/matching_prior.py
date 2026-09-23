"""Matching -> calibrated relative pose -> detached canonical-frame scale.

No GT extrinsics are accepted. The first context camera defines NoPoSplat's
canonical frame. Only the raw first-view support map sets the translation scale.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .roma_matching import Correspondences


@dataclass
class MatchingPriorCfg:
    min_inliers: int = 32
    min_inlier_ratio: float = 0.25
    ransac_threshold_px: float = 1.0
    max_reprojection_px: float = 2.0
    min_parallax_deg: float = 0.5
    max_alignment_error: float = 0.5
    ransac_iterations: int = 10000

    def __post_init__(self):
        if self.min_inliers < 8 or not 0 < self.min_inlier_ratio <= 1:
            raise ValueError("Invalid pose inlier requirements")
        if min(self.ransac_threshold_px, self.max_reprojection_px,
               self.min_parallax_deg, self.max_alignment_error,
               self.ransac_iterations) <= 0:
            raise ValueError("Pose thresholds must be positive")


@dataclass
class MatchingPrior:
    cameras: Tensor  # [2,4,4], camera-to-world, first camera is identity.
    grid_a: Tensor
    grid_b: Tensor
    confidence: Tensor
    scale: float
    alignment_error: float


def sample_image(image: Tensor, grid: Tensor) -> Tensor:
    """[C,H,W] and [N,2] -> [N,C]; used for points, RGB and rendered images."""
    return F.grid_sample(
        image.unsqueeze(0), grid.reshape(1, 1, -1, 2),
        mode="bilinear", padding_mode="zeros", align_corners=False,
    )[0, :, 0].transpose(0, 1)


@torch.no_grad()
def estimate_matching_prior(
    matches: Correspondences, intrinsics: Tensor, support_a: Tensor,
    cfg: MatchingPriorCfg,
) -> MatchingPrior | None:
    """Reject unreliable pairs rather than inventing a camera or a target image.

    intrinsics: [2,3,3], normalized to image width/height (repository convention).
    support_a: [H,W,3], RAW point-head output, before Gaussian moment aggregation.
    """
    n = len(matches.confidence)
    if n < cfg.min_inliers:
        return None
    if intrinsics.shape != (2, 3, 3) or support_a.ndim != 3 or support_a.shape[-1] != 3:
        raise ValueError("Expected intrinsics [2,3,3] and support_a [H,W,3]")
    if matches.grid_a.shape != (n, 2) or matches.grid_b.shape != (n, 2):
        raise ValueError("Correspondence shapes disagree")
    height, width = support_a.shape[:2]
    k = intrinsics.detach().double().cpu().numpy()
    if not np.isfinite(k).all() or (k[:, (0, 1), (0, 1)] <= 0).any():
        raise ValueError("Invalid normalized intrinsics")
    grids = [g.detach().double().cpu().numpy() for g in (matches.grid_a, matches.grid_b)]
    uv = [(g + 1) / 2 for g in grids]
    rays = [np.c_[p, np.ones(n)] @ np.linalg.inv(camera).T
            for p, camera in zip(uv, k)]
    xy = [r[:, :2] / r[:, 2:] for r in rays]
    focal_px = np.mean(k[:, (0, 1), (0, 1)] * np.array([width, height]))
    try:
        essential, ransac_mask = cv2.findEssentialMat(
            xy[0], xy[1], np.eye(3), method=cv2.RANSAC, prob=0.999,
            threshold=cfg.ransac_threshold_px / focal_px,
            maxIters=cfg.ransac_iterations,
        )
        if essential is None or ransac_mask is None:
            return None
        best = None
        for candidate in essential.reshape(-1, 3, 3):
            count, rotation, translation, pose_mask, _ = cv2.recoverPose(
                candidate, xy[0], xy[1], np.eye(3),
                distanceThresh=1e6, mask=ransac_mask.copy(),
            )
            if best is None or count > best[0]:
                best = (count, rotation, translation[:, 0], pose_mask[:, 0] != 0)
        if best is None or best[0] < max(cfg.min_inliers, cfg.min_inlier_ratio * n):
            return None
        _, rotation, translation, valid = best
        ids = np.flatnonzero(valid)
        homogeneous = cv2.triangulatePoints(
            np.c_[np.eye(3), np.zeros(3)],
            np.c_[rotation, translation], xy[0][ids].T, xy[1][ids].T,
        ).T
    except cv2.error:
        return None

    finite_w = np.abs(homogeneous[:, 3]) > 1e-10
    points = homogeneous[:, :3] / np.where(finite_w, homogeneous[:, 3], 1)[:, None]
    points_b = points @ rotation.T + translation
    valid = finite_w & np.isfinite(points).all(1) & (points[:, 2] > 0) & (points_b[:, 2] > 0)
    bearing_a = rays[0][ids] / np.linalg.norm(rays[0][ids], axis=1, keepdims=True)
    bearing_b = rays[1][ids] @ rotation
    bearing_b /= np.linalg.norm(bearing_b, axis=1, keepdims=True)
    parallax = np.degrees(np.arccos(np.clip((bearing_a * bearing_b).sum(1), -1, 1)))
    valid &= parallax >= cfg.min_parallax_deg
    for camera_points, camera_k, observed in zip((points, points_b), k, uv):
        projected = camera_points @ camera_k.T
        denominator = np.where(np.abs(projected[:, 2]) > 1e-10, projected[:, 2], 1)
        projected = projected[:, :2] / denominator[:, None]
        error = np.linalg.norm((projected - observed[ids]) * [width, height], axis=1)
        valid &= error <= cfg.max_reprojection_px
    ids, points = ids[valid], points[valid]
    if len(ids) < cfg.min_inliers:
        return None

    # Use actual pixel locations, never the post-allocation Gaussian means.
    all_supports = sample_image(support_a.detach().float().permute(2, 0, 1),
                                matches.grid_a.detach().float())
    supports = all_supports.double().cpu().numpy()[ids]
    weights = matches.confidence.detach().double().cpu().numpy()[ids]
    finite = np.isfinite(supports).all(1) & np.isfinite(weights) & (weights > 0)
    ids, points, supports, weights = ids[finite], points[finite], supports[finite], weights[finite]
    if len(ids) < cfg.min_inliers:
        return None
    ratios = (points * supports).sum(1) / np.maximum((points * points).sum(1), 1e-12)
    positive = np.isfinite(ratios) & (ratios > 0)
    if positive.sum() < cfg.min_inliers:
        return None
    initial_scale = np.median(ratios[positive])
    residual = np.linalg.norm(supports - initial_scale * points, axis=1) / np.maximum(
        np.linalg.norm(supports, axis=1), 1e-8
    )
    valid = positive & (residual <= cfg.max_alignment_error)
    ids, points, supports, weights = ids[valid], points[valid], supports[valid], weights[valid]
    if len(ids) < max(cfg.min_inliers, cfg.min_inlier_ratio * n):
        return None
    scale = float((weights * (points * supports).sum(1)).sum()
                  / max((weights * (points * points).sum(1)).sum(), 1e-12))
    residual = np.linalg.norm(supports - scale * points, axis=1) / np.maximum(
        np.linalg.norm(supports, axis=1), 1e-8
    )
    alignment_error = float(np.median(residual))
    if not np.isfinite(scale) or scale <= 0 or alignment_error > cfg.max_alignment_error:
        return None

    cameras = torch.eye(4, dtype=torch.float32, device=support_a.device).repeat(2, 1, 1)
    cameras[1, :3, :3] = torch.as_tensor(rotation.T.copy(), device=support_a.device).float()
    cameras[1, :3, 3] = torch.as_tensor(
        -rotation.T @ (scale * translation), device=support_a.device
    ).float()
    selected = torch.as_tensor(ids.copy(), device=matches.grid_a.device, dtype=torch.long)
    return MatchingPrior(
        cameras, matches.grid_a[selected].detach(), matches.grid_b[selected].detach(),
        matches.confidence[selected].detach(), scale, alignment_error,
    )
