"""Local, visibility-aware analysis of geometry--opacity coupling in NoPoSplat.

This analysis is designed to answer a narrower question than final-RGB stress tests:

    For a local Gaussian group that actually affects a target view, can an opacity
    change compensate the change in the renderer's *pre-color contribution measure*
    caused by a geometry change?

For every selected local group, central finite differences estimate derivatives of
three color-independent alpha-blending moments:

    M0(p) = sum_i w_i(p)
    M1(p) = sum_i w_i(p) z_i
    M2(p) = sum_i w_i(p) z_i^2

where w_i = T_i alpha_i G_i includes projection, depth ordering, visibility, and
transmittance.  Pseudo-colors [1, z, z^2] obtain all three moments in one standard
rasterizer call.  Actual SH color is not involved.

For a geometry derivative J_g and opacity derivative J_o, the script reports:

    beta*    = - <J_o, J_g> / ||J_o||^2
    recovery = 1 - ||J_g + beta* J_o||^2 / ||J_g||^2

Thus beta* is the locally optimal change in log optical thickness per unit geometry
change, and recovery measures how much of the geometry-induced rendering-measure
change opacity can explain.  Invisible or unimportant groups have small ||J_g|| and
are separated by target-view-specific importance ranks.

Default scale is a full analysis, not a pilot:

* 100 RE10K scenes from each context-overlap group (300 total)
* 24 deterministic 16x16 local groups per scene, balanced across context views
* all target views
* uniform scale, anisotropy, covariance rotation, and radial-center perturbations
* scene-level chunks with safe resume support

Run:

    python -m src.scripts.analysis_local_rendering_coupling \
        +experiment=re10k mode=test wandb.mode=disabled \
        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
        checkpointing.load=./pretrained_weights/re10k.ckpt \
        test.save_image=false

The Gaussian decoder never receives target images or target poses.  Those are used
only after decoding for the same evaluation alignment and diagnostic rendering used
by the baseline test protocol.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import random
import zlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import hydra
import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from einops import rearrange, repeat
from jaxtyping import install_import_hook
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm

with install_import_hook(("src",), ("beartype", "beartype")):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.geometry.projection import get_fov, homogenize_points
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.step_tracker import StepTracker
    from src.misc.utils import get_overlap_tag
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.decoder.cuda_splatting import get_projection_matrix
    from src.model.encoder import get_encoder
    from src.model.types import Gaussians
    from src.scripts.analysis_scale_opacity import (
        OVERLAP_TAGS,
        _absolute_path,
        _align_target_extrinsics,
        _find_evaluation_sampler,
        _freeze,
        _git_info,
        _index_list,
        _json_safe,
        _load_encoder_checkpoint,
        _make_stratified_index,
        _scene_name,
        _to_device,
    )


GEOMETRY_MODES = (
    "uniform_scale",
    "anisotropy",
    "rotation",
    "radial_center",
)
MEASURE_CHANNELS = {
    "alpha": (0,),
    "depth_moments": (1, 2),
    "joint": (0, 1, 2),
}
IMPORTANCE_SUBSETS = {
    "all": 0.0,
    "top50": 0.5,
    "top25": 0.75,
}


@dataclass(frozen=True)
class CouplingAnalysisCfg:
    output_dir: str = "outputs/local_rendering_coupling/re10k"
    scenes_per_overlap: int = 100
    seed: int = 20260814
    tile_size: int = 16
    groups_per_scene: int = 24
    geometry_modes: tuple[str, ...] = GEOMETRY_MODES
    log_perturbation: float = 0.05
    rotation_perturbation_radians: float = 0.05
    minimum_rms_sensitivity: float = 1e-8
    alpha_identity_tolerance: float = 5e-5
    bootstrap_samples: int = 2000
    beta_abs_max_for_regression: float = 10.0
    regression_test_fraction: float = 0.2
    ridge_lambda: float = 1e-2
    capture_dpt_features: bool = True
    allow_checkpoint_mismatch: bool = False
    resume: bool = True


def _coupling_cfg(cfg_dict: DictConfig) -> CouplingAnalysisCfg:
    node = cfg_dict.get("coupling_analysis")
    if node is None:
        raw: dict[str, Any] = {}
    elif OmegaConf.is_config(node):
        container = OmegaConf.to_container(node, resolve=True)
        raw = {} if container is None else dict(container)
    else:
        raw = dict(node)
    if "geometry_modes" in raw:
        raw["geometry_modes"] = tuple(str(x) for x in raw["geometry_modes"])
    cfg = CouplingAnalysisCfg(**raw)

    if cfg.scenes_per_overlap <= 0:
        raise ValueError("coupling_analysis.scenes_per_overlap must be positive")
    if cfg.tile_size <= 0:
        raise ValueError("coupling_analysis.tile_size must be positive")
    if cfg.groups_per_scene <= 0:
        raise ValueError("coupling_analysis.groups_per_scene must be positive")
    if cfg.log_perturbation <= 0:
        raise ValueError("coupling_analysis.log_perturbation must be positive")
    if cfg.rotation_perturbation_radians <= 0:
        raise ValueError(
            "coupling_analysis.rotation_perturbation_radians must be positive"
        )
    if cfg.minimum_rms_sensitivity <= 0:
        raise ValueError("coupling_analysis.minimum_rms_sensitivity must be positive")
    if cfg.alpha_identity_tolerance <= 0:
        raise ValueError(
            "coupling_analysis.alpha_identity_tolerance must be positive"
        )
    if cfg.bootstrap_samples <= 0:
        raise ValueError("coupling_analysis.bootstrap_samples must be positive")
    if cfg.beta_abs_max_for_regression <= 0:
        raise ValueError(
            "coupling_analysis.beta_abs_max_for_regression must be positive"
        )
    if not 0 < cfg.regression_test_fraction < 1:
        raise ValueError("coupling_analysis.regression_test_fraction must be in (0,1)")
    if cfg.ridge_lambda < 0:
        raise ValueError("coupling_analysis.ridge_lambda must be non-negative")
    unknown = set(cfg.geometry_modes) - set(GEOMETRY_MODES)
    if unknown:
        raise ValueError(f"Unknown geometry modes: {sorted(unknown)}")
    if "uniform_scale" not in cfg.geometry_modes:
        raise ValueError("geometry_modes must include uniform_scale for importance ranks")
    return cfg


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(value), f, indent=2, allow_nan=False)


def _save_chunk(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, separators=(",", ":"), allow_nan=False)
    temporary.replace(path)


def _load_chunk(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


class DPTFeatureCapture:
    """Capture the feature entering each view's final Gaussian-output convolution."""

    def __init__(self, encoder: nn.Module, enabled: bool) -> None:
        self.enabled = enabled
        self.features: dict[int, torch.Tensor] = {}
        self.handles = []
        if not enabled:
            return
        heads = [encoder.gaussian_param_head, encoder.gaussian_param_head2]
        for view_index, head in enumerate(heads):
            try:
                final_layer = head.dpt.head[-1]
            except (AttributeError, IndexError, TypeError) as error:
                raise RuntimeError(
                    "Could not locate the final DPT Gaussian convolution for feature "
                    "capture. Disable with +coupling_analysis.capture_dpt_features=false."
                ) from error

            def capture(_module, inputs, vi=view_index):
                self.features[vi] = inputs[0].detach()

            self.handles.append(final_layer.register_forward_pre_hook(capture))

    def clear(self) -> None:
        self.features.clear()

    def stacked(self, expected_views: int) -> torch.Tensor | None:
        if not self.enabled:
            return None
        missing = [view for view in range(expected_views) if view not in self.features]
        if missing:
            raise RuntimeError(f"DPT feature hooks did not fire for views {missing}")
        return torch.stack([self.features[v][0] for v in range(expected_views)], dim=0)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def _render_contribution_measure(
    decoder: nn.Module,
    gaussians: Gaussians,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    near: torch.Tensor,
    far: torch.Tensor,
    image_shape: tuple[int, int],
) -> tuple[torch.Tensor, float]:
    """Render [M0, M1, M2] with pseudo-colors [1, z, z^2]."""
    b, v = extrinsics.shape[:2]
    if b != 1:
        raise ValueError("Contribution analysis requires batch_size=1")
    h, w = image_shape

    extrinsics_f = rearrange(extrinsics, "b v i j -> (b v) i j").clone()
    intrinsics_f = rearrange(intrinsics, "b v i j -> (b v) i j")
    near_f = rearrange(near, "b v -> (b v)").clone()
    far_f = rearrange(far, "b v -> (b v)").clone()
    means = repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v).clone()
    covariances = repeat(
        gaussians.covariances, "b g i j -> (b v) g i j", v=v
    ).clone()
    opacities = repeat(gaussians.opacities, "b g -> (b v) g", v=v)

    if decoder.make_scale_invariant:
        scene_scale = 1.0 / near_f
        extrinsics_f[..., :3, 3] *= scene_scale[:, None]
        covariances *= scene_scale[:, None, None, None] ** 2
        means *= scene_scale[:, None, None]
        near_f *= scene_scale
        far_f *= scene_scale

    fov_x, fov_y = get_fov(intrinsics_f).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()
    projection = get_projection_matrix(near_f, far_f, fov_x, fov_y)
    projection = rearrange(projection, "b i j -> b j i")
    view_matrix = rearrange(extrinsics_f.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection

    camera_points = torch.einsum(
        "bij,bgj->bgi", extrinsics_f.inverse(), homogenize_points(means)
    )
    depth = camera_points[..., 2]
    depth_normalized = (
        (depth - near_f[:, None])
        / (far_f[:, None] - near_f[:, None]).clamp_min(1e-8)
    ).clamp(0.0, 1.0)
    pseudo_color = torch.stack(
        [torch.ones_like(depth_normalized), depth_normalized, depth_normalized.square()],
        dim=-1,
    )

    row, col = torch.triu_indices(3, 3, device=means.device)
    images = []
    alpha_differences = []
    for view in range(v):
        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[view].item(),
            tanfovy=tan_fov_y[view].item(),
            bg=torch.zeros(3, dtype=means.dtype, device=means.device),
            scale_modifier=1.0,
            viewmatrix=view_matrix[view],
            projmatrix=full_projection[view],
            projmatrix_raw=projection[view],
            sh_degree=0,
            campos=extrinsics_f[view, :3, 3],
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)
        means_2d = torch.zeros_like(means[view], requires_grad=True)
        image, _, _, accumulated_alpha, _ = rasterizer(
            means3D=means[view],
            means2D=means_2d,
            shs=None,
            colors_precomp=pseudo_color[view],
            opacities=opacities[view, ..., None],
            cov3D_precomp=covariances[view, :, row, col],
            theta=None,
            rho=None,
        )
        images.append(image)
        alpha_differences.append(
            (image[0] - accumulated_alpha.squeeze(0)).abs().max()
        )
    return torch.stack(images, dim=0), float(torch.stack(alpha_differences).max().item())


def _sample_local_groups(
    scene: str,
    num_context_views: int,
    height: int,
    width: int,
    tile_size: int,
    groups_per_scene: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    candidates_by_view: list[list[dict[str, Any]]] = []
    for context_view in range(num_context_views):
        view_candidates = []
        base = context_view * height * width
        for y0 in range(0, height, tile_size):
            for x0 in range(0, width, tile_size):
                y1 = min(y0 + tile_size, height)
                x1 = min(x0 + tile_size, width)
                indices = [
                    base + y * width + x
                    for y in range(y0, y1)
                    for x in range(x0, x1)
                ]
                view_candidates.append(
                    {
                        "group_id": f"v{context_view}_y{y0}_x{x0}",
                        "context_view": context_view,
                        "tile_y": y0,
                        "tile_x": x0,
                        "tile_height": y1 - y0,
                        "tile_width": x1 - x0,
                        "indices": indices,
                    }
                )
        candidates_by_view.append(view_candidates)

    rng = random.Random(seed ^ zlib.crc32(scene.encode("utf-8")))
    base_count, remainder = divmod(groups_per_scene, num_context_views)
    selected = []
    for view, candidates in enumerate(candidates_by_view):
        requested = base_count + int(view < remainder)
        if requested > len(candidates):
            raise ValueError(
                f"Requested {requested} groups from context view {view}, but only "
                f"{len(candidates)} tiles exist."
            )
        selected.extend(rng.sample(candidates, requested))
    selected.sort(key=lambda group: group["group_id"])
    for group in selected:
        group["indices_tensor"] = torch.tensor(
            group.pop("indices"), dtype=torch.long, device=device
        )
    return selected


def _rotation_matrix_about_first_axis(
    angle: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=dtype,
        device=device,
    )


@torch.no_grad()
def _perturb_gaussians(
    gaussians: Gaussians,
    indices: torch.Tensor,
    mode: str,
    direction: int,
    log_epsilon: float,
    rotation_epsilon: float,
) -> tuple[Gaussians, float]:
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or +1")
    means = gaussians.means
    covariances = gaussians.covariances
    opacities = gaussians.opacities

    if mode == "opacity":
        opacities = opacities.clone()
        selected = opacities[:, indices]
        eps = torch.finfo(selected.dtype).eps
        thickness = -torch.log1p(-selected.clamp(0.0, 1.0 - eps))
        thickness *= math.exp(direction * log_epsilon)
        opacities[:, indices] = -torch.expm1(-thickness)
        denominator_epsilon = log_epsilon

    elif mode == "uniform_scale":
        covariances = covariances.clone()
        covariances[:, indices] *= math.exp(2.0 * direction * log_epsilon)
        denominator_epsilon = log_epsilon

    elif mode == "anisotropy":
        covariances = covariances.clone()
        selected = covariances[:, indices]
        eigenvalues, eigenvectors = torch.linalg.eigh(selected)
        scale_log_change = torch.tensor(
            [-direction * log_epsilon, 0.0, direction * log_epsilon],
            dtype=selected.dtype,
            device=selected.device,
        )
        eigenvalues = eigenvalues * torch.exp(2.0 * scale_log_change)
        covariances[:, indices] = eigenvectors @ torch.diag_embed(eigenvalues) @ (
            eigenvectors.transpose(-1, -2)
        )
        denominator_epsilon = log_epsilon

    elif mode == "rotation":
        covariances = covariances.clone()
        selected = covariances[:, indices]
        eigenvalues, eigenvectors = torch.linalg.eigh(selected)
        local_rotation = _rotation_matrix_about_first_axis(
            direction * rotation_epsilon, selected.dtype, selected.device
        )
        rotated_local = (
            local_rotation
            @ torch.diag_embed(eigenvalues)
            @ local_rotation.transpose(-1, -2)
        )
        covariances[:, indices] = (
            eigenvectors @ rotated_local @ eigenvectors.transpose(-1, -2)
        )
        denominator_epsilon = rotation_epsilon

    elif mode == "radial_center":
        means = means.clone()
        means[:, indices] *= math.exp(direction * log_epsilon)
        denominator_epsilon = log_epsilon

    else:
        raise ValueError(f"Unknown perturbation mode: {mode}")

    return (
        Gaussians(
            means=means,
            covariances=covariances,
            harmonics=gaussians.harmonics,
            opacities=opacities,
        ),
        denominator_epsilon,
    )


@torch.no_grad()
def _finite_difference(
    decoder: nn.Module,
    gaussians: Gaussians,
    indices: torch.Tensor,
    mode: str,
    cfg: CouplingAnalysisCfg,
    batch: dict[str, Any],
    aligned_extrinsics: torch.Tensor,
) -> torch.Tensor:
    measures = []
    denominator = None
    for direction in (1, -1):
        perturbed, denominator_epsilon = _perturb_gaussians(
            gaussians,
            indices,
            mode,
            direction,
            cfg.log_perturbation,
            cfg.rotation_perturbation_radians,
        )
        measure, _ = _render_contribution_measure(
            decoder,
            perturbed,
            aligned_extrinsics,
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            tuple(batch["target"]["image"].shape[-2:]),
        )
        measures.append(measure)
        denominator = denominator_epsilon
    return (measures[0] - measures[1]) / (2.0 * denominator)


def _interaction_statistics(
    geometry_derivative: torch.Tensor,
    opacity_derivative: torch.Tensor,
    channels: tuple[int, ...],
    minimum_rms: float,
) -> dict[str, Any]:
    geometry = geometry_derivative[list(channels)].reshape(-1).double()
    opacity = opacity_derivative[list(channels)].reshape(-1).double()
    geometry_energy = float(geometry.square().mean().item())
    opacity_energy = float(opacity.square().mean().item())
    cross_energy = float((geometry * opacity).mean().item())
    geometry_rms = math.sqrt(max(geometry_energy, 0.0))
    opacity_rms = math.sqrt(max(opacity_energy, 0.0))
    valid = geometry_rms >= minimum_rms and opacity_rms >= minimum_rms
    if not valid:
        return {
            "valid": False,
            "geometry_energy": geometry_energy,
            "opacity_energy": opacity_energy,
            "cross_energy": cross_energy,
            "geometry_rms": geometry_rms,
            "opacity_rms": opacity_rms,
            "cosine": None,
            "optimal_beta": None,
            "optimal_recovery": None,
            "inverse_sign": None,
        }

    cosine = cross_energy / math.sqrt(geometry_energy * opacity_energy)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    optimal_beta = -cross_energy / opacity_energy
    recovery = float(np.clip(cosine * cosine, 0.0, 1.0))
    return {
        "valid": True,
        "geometry_energy": geometry_energy,
        "opacity_energy": opacity_energy,
        "cross_energy": cross_energy,
        "geometry_rms": geometry_rms,
        "opacity_rms": opacity_rms,
        "cosine": cosine,
        "optimal_beta": optimal_beta,
        "optimal_recovery": recovery,
        "inverse_sign": bool(optimal_beta < 0.0),
    }


def _group_pose_free_features(
    group: dict[str, Any],
    scales: torch.Tensor,
    opacities: torch.Tensor,
    means: torch.Tensor,
    dpt_features: torch.Tensor | None,
) -> dict[str, Any]:
    indices = group["indices_tensor"]
    selected_scales = scales[indices].float().clamp_min(1e-12)
    selected_opacity = opacities[indices].float().clamp(1e-7, 1.0 - 1e-7)
    selected_means = means[indices].float()
    squared = selected_scales.square()
    support = torch.sqrt(
        (
            squared[:, 0] * squared[:, 1]
            + squared[:, 0] * squared[:, 2]
            + squared[:, 1] * squared[:, 2]
        )
        / 3.0
    ).clamp_min(1e-12)
    anisotropy = (
        selected_scales.max(dim=-1).values
        / selected_scales.min(dim=-1).values.clamp_min(1e-12)
    )
    scale_geometric_mean = selected_scales.log().mean(dim=-1)
    center_radius = selected_means.norm(dim=-1).clamp_min(1e-12).log()
    opacity_logit = torch.logit(selected_opacity)

    def mean_std(values: torch.Tensor) -> tuple[float, float]:
        return float(values.mean().item()), float(values.std(unbiased=False).item())

    log_support_mean, log_support_std = mean_std(support.log())
    anisotropy_mean, anisotropy_std = mean_std(anisotropy.log())
    log_scale_mean, log_scale_std = mean_std(scale_geometric_mean)
    radius_mean, radius_std = mean_std(center_radius)
    opacity_mean, opacity_std = mean_std(opacity_logit)
    result: dict[str, Any] = {
        "group_id": group["group_id"],
        "context_view": group["context_view"],
        "tile_y": group["tile_y"],
        "tile_x": group["tile_x"],
        "num_gaussians": int(indices.numel()),
        "log_support_mean": log_support_mean,
        "log_support_std": log_support_std,
        "log_anisotropy_mean": anisotropy_mean,
        "log_anisotropy_std": anisotropy_std,
        "log_scale_mean": log_scale_mean,
        "log_scale_std": log_scale_std,
        "log_center_radius_mean": radius_mean,
        "log_center_radius_std": radius_std,
        "opacity_logit_mean": opacity_mean,
        "opacity_logit_std": opacity_std,
    }
    if dpt_features is not None:
        view = group["context_view"]
        y0, x0 = group["tile_y"], group["tile_x"]
        y1 = y0 + group["tile_height"]
        x1 = x0 + group["tile_width"]
        pooled = dpt_features[view, :, y0:y1, x0:x1].mean(dim=(1, 2))
        result["dpt_feature"] = pooled.float().cpu().tolist()
    return result


def _annotate_importance_percentiles(rows: list[dict[str, Any]]) -> None:
    reference: dict[tuple[int, str], float] = {}
    for row in rows:
        if row["geometry_mode"] == "uniform_scale" and row["measure_type"] == "joint":
            reference[(row["target_position"], row["group_id"])] = float(
                row["geometry_energy"]
            )
    percentiles: dict[tuple[int, str], float] = {}
    target_positions = sorted({key[0] for key in reference})
    for target_position in target_positions:
        items = [
            (group_id, importance)
            for (position, group_id), importance in reference.items()
            if position == target_position
        ]
        items.sort(key=lambda item: item[1])
        count = len(items)
        for rank, (group_id, _) in enumerate(items):
            percentiles[(target_position, group_id)] = (rank + 1) / count
    for row in rows:
        row["importance_percentile"] = percentiles[
            (row["target_position"], row["group_id"])
        ]


def _analyze_scene(
    scene: str,
    batch: dict[str, Any],
    encoder: nn.Module,
    decoder: nn.Module,
    losses: Iterable[nn.Module],
    feature_capture: DPTFeatureCapture,
    cfg: CouplingAnalysisCfg,
    test_cfg: Any,
    global_step: int,
) -> dict[str, Any]:
    batch_size, context_views, _, height, width = batch["context"]["image"].shape
    if batch_size != 1:
        raise ValueError(
            "Local coupling analysis requires data_loader.test.batch_size=1"
        )
    expected_gaussians = context_views * height * width
    visualization_dump: dict[str, torch.Tensor] = {}
    feature_capture.clear()
    with torch.no_grad():
        gaussians = encoder(
            batch["context"],
            global_step,
            visualization_dump=visualization_dump,
        )
    if gaussians.means.shape[1] != expected_gaussians:
        raise ValueError(
            "Local pixel groups assume one Gaussian per context pixel. "
            f"Expected {expected_gaussians}, got {gaussians.means.shape[1]}."
        )
    dpt_features = feature_capture.stacked(context_views)
    aligned_extrinsics = _align_target_extrinsics(
        decoder=decoder,
        losses=losses,
        batch=batch,
        gaussians=gaussians,
        global_step=global_step,
        enabled=test_cfg.align_pose,
        steps=test_cfg.pose_align_steps,
        rotation_lr=test_cfg.rot_opt_lr,
        translation_lr=test_cfg.trans_opt_lr,
    )
    _, alpha_identity_error = _render_contribution_measure(
        decoder,
        gaussians,
        aligned_extrinsics,
        batch["target"]["intrinsics"],
        batch["target"]["near"],
        batch["target"]["far"],
        tuple(batch["target"]["image"].shape[-2:]),
    )
    if alpha_identity_error > cfg.alpha_identity_tolerance:
        raise RuntimeError(
            "Pseudo-color M0 does not match the rasterizer's accumulated alpha: "
            f"max error {alpha_identity_error:.3e} exceeds "
            f"{cfg.alpha_identity_tolerance:.3e}."
        )

    groups = _sample_local_groups(
        scene,
        context_views,
        height,
        width,
        cfg.tile_size,
        cfg.groups_per_scene,
        cfg.seed,
        gaussians.means.device,
    )
    scales = visualization_dump["scales"][0].reshape(-1, 3)
    opacities = gaussians.opacities[0]
    means = gaussians.means[0]
    group_features = [
        _group_pose_free_features(
            group, scales, opacities, means, dpt_features
        )
        for group in groups
    ]

    overlap = float(batch["context"]["overlap"].reshape(-1)[0].item())
    overlap_tag = get_overlap_tag(overlap)
    target_indices = _index_list(batch["target"]["index"])
    rows: list[dict[str, Any]] = []
    for group in groups:
        opacity_derivative = _finite_difference(
            decoder,
            gaussians,
            group["indices_tensor"],
            "opacity",
            cfg,
            batch,
            aligned_extrinsics,
        )
        for geometry_mode in cfg.geometry_modes:
            geometry_derivative = _finite_difference(
                decoder,
                gaussians,
                group["indices_tensor"],
                geometry_mode,
                cfg,
                batch,
                aligned_extrinsics,
            )
            for target_position, target_index in enumerate(target_indices):
                for measure_type, channels in MEASURE_CHANNELS.items():
                    statistics = _interaction_statistics(
                        geometry_derivative[target_position],
                        opacity_derivative[target_position],
                        channels,
                        cfg.minimum_rms_sensitivity,
                    )
                    rows.append(
                        {
                            "scene": scene,
                            "overlap": overlap,
                            "overlap_tag": overlap_tag,
                            "group_id": group["group_id"],
                            "context_view": group["context_view"],
                            "tile_y": group["tile_y"],
                            "tile_x": group["tile_x"],
                            "num_gaussians": int(group["indices_tensor"].numel()),
                            "target_position": target_position,
                            "target_index": target_index,
                            "geometry_mode": geometry_mode,
                            "measure_type": measure_type,
                            **statistics,
                        }
                    )
    _annotate_importance_percentiles(rows)
    for group in groups:
        group.pop("indices_tensor", None)
    return {
        "scene": scene,
        "overlap": overlap,
        "overlap_tag": overlap_tag,
        "context_indices": _index_list(batch["context"]["index"]),
        "target_indices": target_indices,
        "alpha_measure_identity_max_error": alpha_identity_error,
        "rows": rows,
        "group_features": group_features,
    }


def _bootstrap_scene_mean(
    scene_values: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if len(scene_values) == 0:
        return float("nan"), float("nan")
    if len(scene_values) == 1:
        return float(scene_values[0]), float(scene_values[0])
    indices = rng.integers(0, len(scene_values), size=(samples, len(scene_values)))
    means = scene_values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _weighted_mean(rows: list[dict[str, Any]], field: str, weight: str) -> float:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    weights = np.asarray([float(row[weight]) for row in rows], dtype=np.float64)
    if weights.sum() <= 0:
        return float(values.mean())
    return float(np.sum(values * weights) / np.sum(weights))


def _summarize_interactions(
    rows: list[dict[str, Any]],
    cfg: CouplingAnalysisCfg,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(cfg.seed + 100)
    summaries = []
    overlap_scopes: list[str | None] = [None, *OVERLAP_TAGS]
    for overlap_tag in overlap_scopes:
        for geometry_mode in cfg.geometry_modes:
            for measure_type in MEASURE_CHANNELS:
                base = [
                    row
                    for row in rows
                    if row["geometry_mode"] == geometry_mode
                    and row["measure_type"] == measure_type
                    and (overlap_tag is None or row["overlap_tag"] == overlap_tag)
                ]
                for subset_name, threshold in IMPORTANCE_SUBSETS.items():
                    subset = [
                        row
                        for row in base
                        if row["importance_percentile"] >= threshold
                    ]
                    valid = [row for row in subset if row["valid"]]
                    if not subset:
                        continue
                    scene_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
                    for row in valid:
                        scene_groups[row["scene"]].append(row)
                    scene_recoveries = np.asarray(
                        [
                            _weighted_mean(items, "optimal_recovery", "geometry_energy")
                            for items in scene_groups.values()
                        ],
                        dtype=np.float64,
                    )
                    ci_low, ci_high = _bootstrap_scene_mean(
                        scene_recoveries, cfg.bootstrap_samples, rng
                    )
                    if valid:
                        beta_values = np.asarray(
                            [float(row["optimal_beta"]) for row in valid]
                        )
                        cosine_values = np.asarray(
                            [float(row["cosine"]) for row in valid]
                        )
                        weighted_recovery = _weighted_mean(
                            valid, "optimal_recovery", "geometry_energy"
                        )
                        inverse_rate = float(
                            np.mean([bool(row["inverse_sign"]) for row in valid])
                        )
                    else:
                        beta_values = np.asarray([], dtype=np.float64)
                        cosine_values = np.asarray([], dtype=np.float64)
                        weighted_recovery = float("nan")
                        inverse_rate = float("nan")
                    summaries.append(
                        {
                            "overlap_tag": overlap_tag or "all",
                            "geometry_mode": geometry_mode,
                            "measure_type": measure_type,
                            "importance_subset": subset_name,
                            "num_rows": len(subset),
                            "num_valid_rows": len(valid),
                            "valid_fraction": len(valid) / len(subset),
                            "num_scenes": len(scene_groups),
                            "weighted_optimal_recovery": weighted_recovery,
                            "scene_recovery_ci95_low": ci_low,
                            "scene_recovery_ci95_high": ci_high,
                            "mean_cosine": (
                                float(cosine_values.mean())
                                if len(cosine_values)
                                else None
                            ),
                            "median_optimal_beta": (
                                float(np.median(beta_values))
                                if len(beta_values)
                                else None
                            ),
                            "mean_optimal_beta": (
                                float(beta_values.mean()) if len(beta_values) else None
                            ),
                            "inverse_sign_rate": inverse_rate,
                        }
                    )
    return summaries


def _compute_view_consistency(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["geometry_mode"] == "uniform_scale"
        and row["measure_type"] == "joint"
        and row["valid"]
        and row["importance_percentile"] >= IMPORTANCE_SUBSETS["top50"]
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[(row["scene"], row["group_id"])].append(row)

    results = []
    for (scene, group_id), items in grouped.items():
        by_target = {int(row["target_position"]): row for row in items}
        items = list(by_target.values())
        if len(items) < 2:
            continue
        geometry_energy = sum(float(row["geometry_energy"]) for row in items)
        opacity_energy = sum(float(row["opacity_energy"]) for row in items)
        cross_energy = sum(float(row["cross_energy"]) for row in items)
        shared_beta = -cross_energy / max(opacity_energy, 1e-30)
        shared_residual = (
            geometry_energy
            + 2.0 * shared_beta * cross_energy
            + shared_beta * shared_beta * opacity_energy
        )
        shared_recovery = 1.0 - shared_residual / max(geometry_energy, 1e-30)
        oracle_recovery = sum(
            float(row["optimal_recovery"]) * float(row["geometry_energy"])
            for row in items
        ) / max(geometry_energy, 1e-30)
        betas = np.asarray([float(row["optimal_beta"]) for row in items])
        results.append(
            {
                "scene": scene,
                "overlap_tag": items[0]["overlap_tag"],
                "group_id": group_id,
                "context_view": items[0]["context_view"],
                "num_visible_target_views": len(items),
                "shared_beta": shared_beta,
                "shared_recovery": shared_recovery,
                "oracle_per_view_recovery": oracle_recovery,
                "view_adaptation_gap": oracle_recovery - shared_recovery,
                "all_views_inverse_sign": bool(np.all(betas < 0)),
                "same_sign_across_views": bool(np.all(betas < 0) or np.all(betas > 0)),
                "beta_mean": float(betas.mean()),
                "beta_std": float(betas.std()),
                "geometry_energy": geometry_energy,
                "opacity_energy": opacity_energy,
                "cross_energy": cross_energy,
            }
        )
    return results


def _summarize_view_consistency(
    rows: list[dict[str, Any]],
    cfg: CouplingAnalysisCfg,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(cfg.seed + 200)
    summaries = []
    for overlap_tag in [None, *OVERLAP_TAGS]:
        subset = [
            row
            for row in rows
            if overlap_tag is None or row["overlap_tag"] == overlap_tag
        ]
        if not subset:
            continue
        scene_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in subset:
            scene_groups[row["scene"]].append(row)
        scene_shared = np.asarray(
            [
                _weighted_mean(items, "shared_recovery", "geometry_energy")
                for items in scene_groups.values()
            ]
        )
        low, high = _bootstrap_scene_mean(
            scene_shared, cfg.bootstrap_samples, rng
        )
        summaries.append(
            {
                "overlap_tag": overlap_tag or "all",
                "num_groups": len(subset),
                "num_scenes": len(scene_groups),
                "weighted_shared_recovery": _weighted_mean(
                    subset, "shared_recovery", "geometry_energy"
                ),
                "shared_recovery_ci95_low": low,
                "shared_recovery_ci95_high": high,
                "weighted_oracle_recovery": _weighted_mean(
                    subset, "oracle_per_view_recovery", "geometry_energy"
                ),
                "weighted_view_adaptation_gap": _weighted_mean(
                    subset, "view_adaptation_gap", "geometry_energy"
                ),
                "same_sign_rate": float(
                    np.mean([row["same_sign_across_views"] for row in subset])
                ),
                "all_views_inverse_sign_rate": float(
                    np.mean([row["all_views_inverse_sign"] for row in subset])
                ),
                "median_beta_std": float(
                    np.median([row["beta_std"] for row in subset])
                ),
            }
        )
    return summaries


def _weighted_r2(y: np.ndarray, prediction: np.ndarray, weight: np.ndarray) -> float:
    weight = weight / max(weight.sum(), 1e-30)
    mean = float(np.sum(weight * y))
    total = float(np.sum(weight * (y - mean) ** 2))
    residual = float(np.sum(weight * (y - prediction) ** 2))
    return 1.0 - residual / max(total, 1e-30)


def _fit_weighted_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weight: np.ndarray,
    test_x: np.ndarray,
    ridge_lambda: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-8] = 1.0
    train = (train_x - mean) / std
    test = (test_x - mean) / std
    train = np.concatenate([np.ones((len(train), 1)), train], axis=1)
    test = np.concatenate([np.ones((len(test), 1)), test], axis=1)
    weight = train_weight / max(train_weight.mean(), 1e-30)
    xtw = train.T * weight[None, :]
    regularizer = np.eye(train.shape[1]) * ridge_lambda
    regularizer[0, 0] = 0.0
    system = xtw @ train + regularizer
    target = xtw @ train_y
    try:
        coefficients = np.linalg.solve(system, target)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(system, target, rcond=None)[0]
    return test @ coefficients, {
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "coefficients": coefficients.tolist(),
    }


def _predictability_analysis(
    consistency_rows: list[dict[str, Any]],
    group_features: list[dict[str, Any]],
    cfg: CouplingAnalysisCfg,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    feature_lookup = {
        (row["scene"], row["group_id"]): row for row in group_features
    }
    samples = []
    dropped_beta = 0
    for row in consistency_rows:
        if abs(float(row["shared_beta"])) > cfg.beta_abs_max_for_regression:
            dropped_beta += 1
            continue
        feature = feature_lookup.get((row["scene"], row["group_id"]))
        if feature is None:
            continue
        samples.append((row, feature))
    scenes = sorted({row[0]["scene"] for row in samples})
    rng = random.Random(cfg.seed + 300)
    rng.shuffle(scenes)
    test_count = max(1, round(len(scenes) * cfg.regression_test_fraction))
    test_scenes = set(scenes[:test_count])
    train_samples = [sample for sample in samples if sample[0]["scene"] not in test_scenes]
    test_samples = [sample for sample in samples if sample[0]["scene"] in test_scenes]
    if not train_samples or not test_samples:
        return {"error": "Insufficient train/test samples"}, []

    scalar_feature_sets = {
        "P0_constant": [],
        "P1_support": ["log_support_mean", "log_support_std"],
        "P2_geometry": [
            "log_support_mean",
            "log_support_std",
            "log_anisotropy_mean",
            "log_anisotropy_std",
            "log_scale_mean",
            "log_scale_std",
            "log_center_radius_mean",
            "log_center_radius_std",
        ],
    }
    if all("dpt_feature" in feature for _, feature in samples):
        scalar_feature_sets["P3_geometry_dpt"] = scalar_feature_sets[
            "P2_geometry"
        ]

    def matrix(sample_list, model_name, scalar_names):
        matrix_rows = []
        for _, feature in sample_list:
            values = [float(feature[name]) for name in scalar_names]
            if model_name == "P3_geometry_dpt":
                values.extend(float(value) for value in feature["dpt_feature"])
            matrix_rows.append(values)
        return np.asarray(matrix_rows, dtype=np.float64)

    train_y = np.asarray([float(row["shared_beta"]) for row, _ in train_samples])
    test_y = np.asarray([float(row["shared_beta"]) for row, _ in test_samples])
    train_weight = np.asarray(
        [float(row["geometry_energy"]) for row, _ in train_samples]
    )
    test_weight = np.asarray(
        [float(row["geometry_energy"]) for row, _ in test_samples]
    )
    results: dict[str, Any] = {
        "num_scenes": len(scenes),
        "num_train_scenes": len(scenes) - len(test_scenes),
        "num_test_scenes": len(test_scenes),
        "num_train_groups": len(train_samples),
        "num_test_groups": len(test_samples),
        "dropped_abs_beta_outliers": dropped_beta,
        "test_scenes": sorted(test_scenes),
        "models": {},
    }
    prediction_rows = []
    for model_name, scalar_names in scalar_feature_sets.items():
        if model_name == "P0_constant":
            constant = float(np.average(train_y, weights=train_weight))
            prediction = np.full_like(test_y, constant)
            fit_info = {"constant": constant}
            feature_count = 0
        else:
            train_x = matrix(train_samples, model_name, scalar_names)
            test_x = matrix(test_samples, model_name, scalar_names)
            prediction, fit_info = _fit_weighted_ridge(
                train_x,
                train_y,
                train_weight,
                test_x,
                cfg.ridge_lambda,
            )
            feature_count = train_x.shape[1]

        sign_accuracy = float(np.mean((prediction < 0) == (test_y < 0)))
        majority_inverse = float(np.mean(test_y < 0))
        predicted_recoveries = []
        for (row, _), beta_prediction in zip(test_samples, prediction):
            geometry_energy = float(row["geometry_energy"])
            opacity_energy = float(row["opacity_energy"])
            cross_energy = float(row["cross_energy"])
            residual = (
                geometry_energy
                + 2.0 * beta_prediction * cross_energy
                + beta_prediction * beta_prediction * opacity_energy
            )
            predicted_recoveries.append(
                1.0 - residual / max(geometry_energy, 1e-30)
            )
        predicted_recoveries = np.asarray(predicted_recoveries)
        weighted_recovery = float(
            np.sum(predicted_recoveries * test_weight) / max(test_weight.sum(), 1e-30)
        )
        results["models"][model_name] = {
            "num_features": feature_count,
            "weighted_r2": _weighted_r2(test_y, prediction, test_weight),
            "unweighted_r2": _weighted_r2(
                test_y, prediction, np.ones_like(test_weight)
            ),
            "inverse_sign_accuracy": sign_accuracy,
            "test_inverse_majority_rate": majority_inverse,
            "weighted_predicted_recovery": weighted_recovery,
            "weighted_clipped_predicted_recovery": float(
                np.sum(np.clip(predicted_recoveries, 0.0, 1.0) * test_weight)
                / max(test_weight.sum(), 1e-30)
            ),
            "fit": fit_info,
        }
        for (row, _), target, predicted, recovery in zip(
            test_samples, test_y, prediction, predicted_recoveries
        ):
            prediction_rows.append(
                {
                    "model": model_name,
                    "scene": row["scene"],
                    "group_id": row["group_id"],
                    "target_shared_beta": float(target),
                    "predicted_beta": float(predicted),
                    "predicted_recovery": float(recovery),
                    "geometry_energy": float(row["geometry_energy"]),
                }
            )
    return results, prediction_rows


def _build_decision_summary(
    interaction: list[dict[str, Any]],
    view_summary: list[dict[str, Any]],
    predictability: dict[str, Any],
) -> dict[str, Any]:
    local = next(
        (
            row
            for row in interaction
            if row["overlap_tag"] == "all"
            and row["geometry_mode"] == "uniform_scale"
            and row["measure_type"] == "joint"
            and row["importance_subset"] == "top50"
        ),
        None,
    )
    shared = next(
        (row for row in view_summary if row["overlap_tag"] == "all"), None
    )
    model_summary = {
        name: {
            "weighted_r2": metrics["weighted_r2"],
            "inverse_sign_accuracy": metrics["inverse_sign_accuracy"],
            "weighted_predicted_recovery": metrics[
                "weighted_predicted_recovery"
            ],
        }
        for name, metrics in predictability.get("models", {}).items()
    }
    return {
        "question_1_local_opacity_can_explain_scale_effect": (
            None
            if local is None
            else {
                "weighted_oracle_recovery": local["weighted_optimal_recovery"],
                "ci95_low": local["scene_recovery_ci95_low"],
                "ci95_high": local["scene_recovery_ci95_high"],
                "inverse_sign_rate": local["inverse_sign_rate"],
                "valid_fraction": local["valid_fraction"],
            }
        ),
        "question_2_one_pose_free_rule_is_consistent_across_views": (
            None
            if shared is None
            else {
                "weighted_shared_recovery": shared["weighted_shared_recovery"],
                "weighted_per_view_oracle_recovery": shared[
                    "weighted_oracle_recovery"
                ],
                "weighted_view_adaptation_gap": shared[
                    "weighted_view_adaptation_gap"
                ],
                "same_sign_rate": shared["same_sign_rate"],
            }
        ),
        "question_3_pose_free_inputs_predict_the_shared_rule": model_summary,
        "reading_order": [
            "First require non-trivial local oracle recovery (question 1).",
            "Then require a small shared-vs-oracle gap (question 2).",
            "Finally compare P1/P2/P3 against P0 to choose Stage-3 inputs (question 3).",
        ],
    }


def _plot_summaries(
    output_dir: Path,
    interaction: list[dict[str, Any]],
    view_summary: list[dict[str, Any]],
    predictability: dict[str, Any],
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    selected = [
        row
        for row in interaction
        if row["overlap_tag"] == "all"
        and row["measure_type"] == "joint"
        and row["importance_subset"] == "top50"
        and all(
            row[key] is not None and math.isfinite(float(row[key]))
            for key in (
                "weighted_optimal_recovery",
                "scene_recovery_ci95_low",
                "scene_recovery_ci95_high",
                "median_optimal_beta",
                "inverse_sign_rate",
            )
        )
    ]
    if selected:
        modes = [row["geometry_mode"] for row in selected]
        recovery = [row["weighted_optimal_recovery"] for row in selected]
        low = [row["scene_recovery_ci95_low"] for row in selected]
        high = [row["scene_recovery_ci95_high"] for row in selected]
        yerr = np.vstack(
            [
                np.maximum(np.asarray(recovery) - np.asarray(low), 0),
                np.maximum(np.asarray(high) - np.asarray(recovery), 0),
            ]
        )
        plt.figure(figsize=(7.2, 4.3))
        plt.bar(modes, recovery, yerr=yerr, capsize=4)
        plt.ylabel("Opacity-explainable fraction (recovery)")
        plt.ylim(bottom=0)
        plt.xticks(rotation=15)
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(figure_dir / "recovery_by_geometry.png", dpi=180)
        plt.close()

        beta = [row["median_optimal_beta"] for row in selected]
        inverse = [row["inverse_sign_rate"] for row in selected]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        axes[0].bar(modes, beta)
        axes[0].axhline(0, color="black", linewidth=1)
        axes[0].set_ylabel("Median optimal beta")
        axes[0].tick_params(axis="x", rotation=15)
        axes[1].bar(modes, inverse)
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("Inverse-sign rate")
        axes[1].tick_params(axis="x", rotation=15)
        fig.tight_layout()
        fig.savefig(figure_dir / "beta_and_sign_by_geometry.png", dpi=180)
        plt.close(fig)

    overall_view = next(
        (row for row in view_summary if row["overlap_tag"] == "all"), None
    )
    if overall_view is not None:
        plt.figure(figsize=(5.5, 4.2))
        plt.bar(
            ["Shared beta", "Per-view oracle"],
            [
                overall_view["weighted_shared_recovery"],
                overall_view["weighted_oracle_recovery"],
            ],
        )
        plt.ylabel("Recovery")
        plt.ylim(bottom=0)
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(figure_dir / "shared_vs_view_specific_recovery.png", dpi=180)
        plt.close()

    models = predictability.get("models", {})
    if models:
        names = list(models)
        recoveries = [models[name]["weighted_predicted_recovery"] for name in names]
        plt.figure(figsize=(7.0, 4.2))
        plt.bar(names, recoveries)
        plt.axhline(0, color="black", linewidth=1)
        plt.ylabel("Held-out predicted recovery")
        plt.xticks(rotation=15)
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(figure_dir / "pose_free_predictability.png", dpi=180)
        plt.close()


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg_dict: DictConfig) -> None:
    analysis = _coupling_cfg(cfg_dict)
    output_dir = _absolute_path(analysis.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = output_dir / "scene_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    dataset_name, sampler_cfg = _find_evaluation_sampler(cfg_dict)
    source_index_path = _absolute_path(sampler_cfg.index_path)
    selected_index_path = output_dir / "selected_scenes.json"
    selected_index, _ = _make_stratified_index(
        source_index_path,
        selected_index_path,
        analysis.scenes_per_overlap,
        analysis.seed,
    )
    sampler_cfg.index_path = str(selected_index_path)

    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    OmegaConf.save(config=cfg_dict, f=str(output_dir / "resolved_config.yaml"))
    if cfg.checkpointing.load is None:
        raise ValueError("checkpointing.load must point to the baseline checkpoint")
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)
    checkpoint_path = _absolute_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA Gaussian rasterization is required")
    device = torch.device("cuda")

    run_spec = {
        "analysis": asdict(analysis),
        "dataset": dataset_name,
        "source_index": str(source_index_path),
        "checkpoint": str(checkpoint_path),
        "pose_alignment": {
            "enabled": cfg.test.align_pose,
            "steps": cfg.test.pose_align_steps,
            "rotation_lr": cfg.test.rot_opt_lr,
            "translation_lr": cfg.test.trans_opt_lr,
        },
    }
    run_spec_path = output_dir / "run_spec.json"
    if run_spec_path.exists():
        existing = json.loads(run_spec_path.read_text(encoding="utf-8"))
        if existing != _json_safe(run_spec):
            raise RuntimeError(
                f"Existing run configuration differs: {run_spec_path}. "
                "Use a new coupling_analysis.output_dir."
            )
    else:
        _save_json(run_spec_path, run_spec)

    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder)
    encoder = _freeze(encoder.to(device))
    decoder = _freeze(decoder.to(device))
    losses = nn.ModuleList(get_losses(cfg.loss)).to(device)
    _freeze(losses)
    global_step, missing_keys, unexpected_keys = _load_encoder_checkpoint(
        encoder, checkpoint_path, analysis.allow_checkpoint_mismatch
    )
    feature_capture = DPTFeatureCapture(encoder, analysis.capture_dpt_features)

    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        StepTracker(),
        global_rank=0,
    )
    data_module.setup("test")
    test_loader = data_module.test_dataloader()
    if isinstance(test_loader, list):
        if len(test_loader) != 1:
            raise ValueError("Only one evaluation dataset is supported")
        test_loader = test_loader[0]
    data_shim = encoder.get_data_shim()

    completed_before_start = {
        path.stem.replace(".json", "") for path in chunk_dir.glob("*.json.gz")
    }
    if completed_before_start and not analysis.resume:
        raise RuntimeError(
            f"Found {len(completed_before_start)} existing chunks but resume=false. "
            "Use a new output directory."
        )
    progress = tqdm(total=len(selected_index), desc="Local coupling analysis")
    progress.update(len(completed_before_start & set(selected_index)))
    try:
        for batch in test_loader:
            scene = _scene_name(batch)
            if scene not in selected_index:
                continue
            chunk_path = chunk_dir / f"{scene}.json.gz"
            if chunk_path.exists() and analysis.resume:
                continue
            batch = _to_device(batch, device)
            batch = data_shim(batch)
            payload = _analyze_scene(
                scene,
                batch,
                encoder,
                decoder,
                losses,
                feature_capture,
                analysis,
                cfg.test,
                global_step,
            )
            _save_chunk(chunk_path, payload)
            progress.update(1)
            progress.set_postfix(
                {
                    "scene": scene[:8],
                    "overlap": payload["overlap_tag"],
                    "rows": len(payload["rows"]),
                }
            )
    finally:
        feature_capture.close()
        progress.close()

    missing_scenes = [
        scene
        for scene in selected_index
        if not (chunk_dir / f"{scene}.json.gz").exists()
    ]
    if missing_scenes:
        raise RuntimeError(
            f"Analysis ended with {len(missing_scenes)} missing scenes. Re-run the same "
            "command to resume. First missing scenes: {missing_scenes[:10]}"
        )

    all_rows: list[dict[str, Any]] = []
    all_features: list[dict[str, Any]] = []
    processed_scenes = []
    alpha_errors = []
    for scene in selected_index:
        payload = _load_chunk(chunk_dir / f"{scene}.json.gz")
        all_rows.extend(payload["rows"])
        for feature in payload["group_features"]:
            all_features.append({"scene": scene, **feature})
        processed_scenes.append(
            {
                "scene": scene,
                "overlap": payload["overlap"],
                "overlap_tag": payload["overlap_tag"],
                "context_indices": payload["context_indices"],
                "target_indices": payload["target_indices"],
            }
        )
        alpha_errors.append(float(payload["alpha_measure_identity_max_error"]))

    interaction_summary = _summarize_interactions(all_rows, analysis)
    view_consistency = _compute_view_consistency(all_rows)
    view_summary = _summarize_view_consistency(view_consistency, analysis)
    predictability, prediction_rows = _predictability_analysis(
        view_consistency, all_features, analysis
    )
    decision_summary = _build_decision_summary(
        interaction_summary, view_summary, predictability
    )

    _write_csv(output_dir / "interaction_records.csv", all_rows)
    _write_csv(output_dir / "interaction_summary.csv", interaction_summary)
    _write_csv(output_dir / "view_consistency.csv", view_consistency)
    _write_csv(output_dir / "view_consistency_summary.csv", view_summary)
    scalar_features = [
        {key: value for key, value in row.items() if key != "dpt_feature"}
        for row in all_features
    ]
    _write_csv(output_dir / "group_features.csv", scalar_features)
    if analysis.capture_dpt_features:
        dpt = np.asarray([row["dpt_feature"] for row in all_features], dtype=np.float32)
        np.savez_compressed(
            output_dir / "dpt_group_features.npz",
            scene=np.asarray([row["scene"] for row in all_features]),
            group_id=np.asarray([row["group_id"] for row in all_features]),
            feature=dpt,
        )
    _write_csv(output_dir / "predictability_predictions.csv", prediction_rows)
    _save_json(output_dir / "predictability.json", predictability)
    _save_json(output_dir / "decision_summary.json", decision_summary)
    _save_json(output_dir / "processed_scenes.json", processed_scenes)
    _save_json(
        output_dir / "summary.json",
        {
            "interaction": interaction_summary,
            "view_consistency": view_summary,
            "predictability": predictability,
            "decision_summary": decision_summary,
        },
    )

    processed_counts = {
        tag: sum(scene["overlap_tag"] == tag for scene in processed_scenes)
        for tag in OVERLAP_TAGS
    }
    manifest = {
        **run_spec,
        "checkpoint_global_step": global_step,
        "checkpoint_missing_encoder_keys": missing_keys,
        "checkpoint_unexpected_encoder_keys": unexpected_keys,
        "processed_scene_counts": processed_counts,
        "num_local_groups": len(all_features),
        "num_group_view_interaction_rows": len(all_rows),
        "contribution_measure": {
            "M0": "sum_i w_i",
            "M1": "sum_i w_i * normalized_camera_depth_i",
            "M2": "sum_i w_i * normalized_camera_depth_i^2",
            "includes_projection_visibility_ordering_transmittance": True,
            "uses_actual_gaussian_color": False,
        },
        "analysis_protocol": {
            "gaussian_decoder_inputs": "context views only",
            "target_information_use": (
                "post-decoding evaluation probe and oracle-label construction only"
            ),
            "local_group_sampling": (
                "deterministic context-image tiles, balanced across context views"
            ),
            "importance_reference": (
                "target-view percentile of uniform-scale joint geometry energy"
            ),
            "view_consistency_scope": "uniform-scale, joint measure, top-50% groups",
            "predictability_split": "scene-disjoint 80/20 split",
        },
        "alpha_measure_identity_max_error": max(alpha_errors),
        "git": _git_info(),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device),
    }
    _save_json(output_dir / "manifest.json", manifest)
    _plot_summaries(
        output_dir, interaction_summary, view_summary, predictability
    )

    print(f"\nAnalysis complete: {output_dir}")
    print(f"Processed scenes: {processed_counts}")
    print(f"Local groups: {len(all_features)}")
    print(f"Interaction rows: {len(all_rows)}")
    print(f"Alpha-measure identity max error: {max(alpha_errors):.3e}")
    print("Primary outputs:")
    print(f"  {output_dir / 'decision_summary.json'}")
    print(f"  {output_dir / 'interaction_summary.csv'}")
    print(f"  {output_dir / 'view_consistency_summary.csv'}")
    print(f"  {output_dir / 'predictability.json'}")


if __name__ == "__main__":
    main()
