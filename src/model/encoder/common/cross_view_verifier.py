"""Student-only descriptor matching and verified correction of canonical points.

No cameras, RGB, teacher or renderer are used here. A fitted SE(3) corrects a
predicted point map that is ALREADY canonical; it is not a camera extrinsic.
Discrete matching/fitting are detached. Rendering still trains the point heads
through the accepted, fixed transform. Failed scenes keep their original points.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class CrossViewVerifierCfg:
    enabled: bool = False
    descriptor_dim: int = 32
    hidden_dim: int = 64
    num_representatives: int = 1024
    min_similarity: float = 0.8
    min_margin: float = 0.02
    min_matches: int = 24
    ransac_trials: int = 64
    # Threshold relative to representative spacing, not distance from origin.
    distance_fraction: float = 0.5
    min_inlier_ratio: float = 0.6
    min_spread_ratio: float = 0.05

    def __post_init__(self):
        if min(self.descriptor_dim, self.hidden_dim, self.ransac_trials) < 1:
            raise ValueError("Descriptor dimensions and RANSAC trials must be positive")
        if not 12 <= self.min_matches <= self.num_representatives:
            raise ValueError("Require 12 <= min_matches <= num_representatives")
        if not -1 <= self.min_similarity <= 1 or not 0 < self.min_margin < 2:
            raise ValueError("Invalid cosine similarity/margin threshold")
        if not 0 < self.distance_fraction <= 1:
            raise ValueError("distance_fraction must be in (0, 1]")
        if not 0 < self.min_inlier_ratio <= 1 or not 0 < self.min_spread_ratio < 1:
            raise ValueError("Invalid registration verification threshold")


def pixel_representatives(height: int, width: int, budget: int, device) -> Tensor:
    """Uniform pixel centers; only this adapter assumes image-shaped supports."""
    rows = min(height, max(1, int(math.sqrt(budget * height / width))))
    cols = min(width, max(1, budget // rows))
    y = ((torch.arange(rows, device=device) + .5) * height / rows).long()
    x = ((torch.arange(cols, device=device) + .5) * width / cols).long()
    return (y[:, None] * width + x[None]).flatten()


@torch.no_grad()
def mutual_matches(a: Tensor, b: Tensor, cfg: CrossViewVerifierCfg):
    """Cosine mutual nearest neighbors, with ambiguity rejection on BOTH sides."""
    scores = a @ b.T
    if min(scores.shape) < 2:
        empty = torch.empty(0, device=a.device, dtype=torch.long)
        return empty, empty, a.new_empty(0)
    ab, ia = scores.topk(2, dim=1)
    ba, ib = scores.topk(2, dim=0)
    target = ia[:, 0]
    source = torch.arange(len(a), device=a.device)
    valid = (
        (ib[0, target] == source) & (ab[:, 0] >= cfg.min_similarity)
        & (ab[:, 0] - ab[:, 1] >= cfg.min_margin)
        & (ba[0, target] - ba[1, target] >= cfg.min_margin)
    )
    return source[valid], target[valid], ab[valid, 0].clamp_min(1e-6)


def weighted_rigid(source: Tensor, target: Tensor, weights: Tensor):
    """Batched weighted Kabsch. Row-vector convention: source @ R.T + t."""
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
    center_s = (source * weights[..., None]).sum(-2)
    center_t = (target * weights[..., None]).sum(-2)
    xs, xt = source - center_s[..., None, :], target - center_t[..., None, :]
    covariance = (xs * weights[..., None]).transpose(-1, -2) @ xt
    u, _, vh = torch.linalg.svd(covariance)
    v = vh.transpose(-1, -2)
    sign = torch.ones_like(center_s)
    sign[..., -1] = torch.where(torch.linalg.det(v @ u.transpose(-1, -2)) < 0, -1., 1.)
    rotation = (v * sign[..., None, :]) @ u.transpose(-1, -2)
    translation = center_t - (rotation @ center_s[..., None]).squeeze(-1)
    return rotation, translation


@torch.no_grad()
def representative_spacing(points: Tensor) -> Tensor:
    # Avoid cancellation in ||x||^2 + ||y||^2 - 2*x.y for translated/nearby points.
    centered = points - points.mean(0)
    distances = torch.cdist(centered, centered, compute_mode='donot_use_mm_for_euclid_dist')
    distances.fill_diagonal_(float("inf"))
    return distances.min(-1).values.median()


@torch.no_grad()
def verify_rigid(source: Tensor, target: Tensor, confidence: Tensor,
                 tolerance: Tensor, cfg: CrossViewVerifierCfg):
    """Robust fit on 3/4 of matches, validation on the untouched 1/4.

    Deterministic local sampling does not consume the training RNG. Planar
    surfaces are allowed; collinear/collapsed matches are rejected. Validation
    points are deliberately NOT used to refit the returned transform.
    """
    identity = torch.eye(3, device=source.device, dtype=source.dtype)
    zero = source.new_zeros(3)
    stats = source.new_zeros(3)  # accepted, validation inlier ratio, residual/tol
    if len(source) < cfg.min_matches or not bool(
        torch.isfinite(source).all() & torch.isfinite(target).all()
        & torch.isfinite(tolerance) & (tolerance > 1e-8)
    ):
        return identity, zero, stats
    generator = torch.Generator(device=source.device).manual_seed(1729)
    order = torch.randperm(len(source), generator=generator, device=source.device)
    validation, fit = order[::4], order[torch.arange(len(order), device=source.device) % 4 != 0]
    xs, xt, weight = source[fit], target[fit], confidence[fit]
    # Three distinct correspondences per hypothesis, in one small batched SVD.
    triples = torch.rand(cfg.ransac_trials, len(fit), generator=generator,
                         device=source.device).topk(3, dim=-1).indices
    rotation, translation = weighted_rigid(xs[triples], xt[triples], weight[triples])
    residual = (xs[None] @ rotation.transpose(-1, -2) + translation[:, None] - xt).norm(dim=-1)
    inliers = residual <= tolerance
    best = (inliers * weight).sum(-1).argmax()
    keep = inliers[best]
    if int(keep.sum()) < max(6, math.ceil(len(fit) * cfg.min_inlier_ratio)):
        return identity, zero, stats
    # Two refits of a single consensus, not iterative decoder refinement.
    for _ in range(2):
        rotation, translation = weighted_rigid(xs, xt, weight * keep)
        residual = (xs @ rotation.T + translation - xt).norm(dim=-1)
        keep = residual <= tolerance
    if int(keep.sum()) < max(6, math.ceil(len(fit) * cfg.min_inlier_ratio)):
        return identity, zero, stats
    # Validate geometry on BOTH sides. Two nontrivial axes suffice for SE(3).
    spreads = []
    for points in (xs[keep], xt[keep]):
        singular = torch.linalg.svdvals(points - points.mean(0)) / math.sqrt(len(points))
        spreads.append((singular[1] > singular[0] * cfg.min_spread_ratio)
                       & (singular[1] > 2 * tolerance))
    error = (source[validation] @ rotation.T + translation - target[validation]).norm(dim=-1)
    ratio, relative = (error <= tolerance).float().mean(), error.median() / tolerance
    accepted = (ratio >= cfg.min_inlier_ratio) & torch.stack(spreads).all()
    stats = torch.stack((accepted.float(), ratio, relative))
    if not bool(accepted):
        return identity, zero, stats
    return rotation, translation, stats


class CrossViewVerifier(nn.Module):
    """Generic two-group support adapter; representative indices come from caller."""

    def __init__(self, feature_dim: int, cfg: CrossViewVerifierCfg):
        super().__init__()
        self.cfg = cfg
        self.descriptor = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, cfg.hidden_dim),
            nn.GELU(), nn.Linear(cfg.hidden_dim, cfg.descriptor_dim),
        )

    def describe(self, features: Tensor) -> Tensor:
        return F.normalize(self.descriptor(features.float()), dim=-1, eps=1e-6)

    def forward(self, points: Tensor, features: Tensor, split: int,
                representatives_a: Tensor, representatives_b: Tensor):
        """Returns live corrected points, per-scene split/fuse flags, mean stats.

        Bounded matching: [representatives_a, representatives_b], never [M,M].
        The caller uses within-group kNN when a scene's fuse flag is False.
        """
        rotations, translations, reports = [], [], []
        with torch.no_grad(), torch.autocast(device_type=points.device.type, enabled=False):
            descriptors_a = self.describe(features[:, representatives_a].detach())
            descriptors_b = self.describe(features[:, split + representatives_b].detach())
            for p, da, db in zip(points.detach().float(), descriptors_a, descriptors_b):
                pa, pb = p[representatives_a], p[split + representatives_b]
                ia, ib, confidence = mutual_matches(da, db, self.cfg)
                tolerance = torch.minimum(representative_spacing(pa), representative_spacing(pb)) * self.cfg.distance_fraction
                rotation, translation, stats = verify_rigid(pb[ib], pa[ia], confidence, tolerance, self.cfg)
                rotations.append(rotation)
                translations.append(translation)
                reports.append(torch.cat((stats, stats.new_tensor([len(ia)]))))
            report = torch.stack(reports)
            fused = report[:, 0].bool().cpu().tolist()
        rotation, translation = torch.stack(rotations), torch.stack(translations)
        corrected = points[:, split:].float() @ rotation.transpose(-1, -2) + translation[:, None]
        # Identity on rejected scenes; preserve all points and their slot order.
        corrected = torch.cat((points[:, :split], corrected), dim=1)
        stats = dict(zip(("accepted", "validation_inliers", "relative_residual", "matches"), report.mean(0)))
        return corrected, fused, stats
