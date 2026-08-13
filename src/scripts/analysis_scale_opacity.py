"""Analyze the scale--opacity coupling of a pretrained NoPoSplat model.

The script does not train or modify the encoder.  For each selected RE10K scene it:

1. Decodes the context views into the original Gaussian scene.
2. Runs the standard pose-free target-pose alignment once with that original scene.
3. Freezes the aligned cameras and renders two post-decoding interventions:

   scale-only:
       covariance_k = k^2 * covariance, alpha_k = alpha

   compensated:
       covariance_k = k^2 * covariance
       tau           = -log(1 - alpha)
       tau_k         = tau / k^2
       alpha_k       = 1 - exp(-tau_k)

Only final Gaussian covariances and opacities are changed.  Means and harmonics
(color) remain fixed.  Target images and poses are used only by the ordinary
evaluation renderer and pose-alignment procedure, never by the Gaussian decoder.

Recommended full run:

    python -m src.scripts.analysis_scale_opacity \
        +experiment=re10k mode=test wandb.mode=disabled \
        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
        checkpointing.load=./pretrained_weights/re10k.ckpt \
        test.save_image=false

Quick smoke test (two scenes per overlap group, fewer bootstrap samples):

    python -m src.scripts.analysis_scale_opacity \
        +experiment=re10k mode=test wandb.mode=disabled \
        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
        checkpointing.load=./pretrained_weights/re10k.ckpt \
        test.save_image=false \
        +analysis.scenes_per_overlap=2 \
        +analysis.bootstrap_samples=100

Outputs are written to ``outputs/scale_opacity_analysis/re10k`` by default.
"""

from __future__ import annotations

import csv
import json
import math
import random
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import hydra
import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
from einops import rearrange
from hydra.utils import to_absolute_path
from jaxtyping import install_import_hook
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm

with install_import_hook(("src",), ("beartype", "beartype")):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.evaluation.metrics import compute_psnr
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.cam_utils import update_pose
    from src.misc.step_tracker import StepTracker
    from src.misc.utils import get_overlap_tag
    from src.misc.wandb_tools import update_checkpoint_path
    from src.misc.weight_modify import checkpoint_filter_fn
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.types import Gaussians


@dataclass(frozen=True)
class AnalysisCfg:
    output_dir: str = "outputs/scale_opacity_analysis/re10k"
    scenes_per_overlap: int = 100
    seed: int = 20250308
    scale_factors: tuple[float, ...] = (0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2)
    bootstrap_samples: int = 2000
    attribute_samples_per_scene: int = 1024
    examples_per_overlap: int = 2
    example_scale_factors: tuple[float, ...] = (0.9, 1.1)
    error_visualization_gain: float = 5.0
    identity_tolerance: float = 5e-5
    save_every_scenes: int = 10
    allow_checkpoint_mismatch: bool = False


OVERLAP_TAGS = ("small", "medium", "large")
RAW_FIELDS = (
    "scene",
    "context_indices",
    "target_index",
    "target_position",
    "overlap",
    "overlap_tag",
    "scale_factor",
    "mode",
    "psnr_gt",
    "delta_psnr",
    "mse_to_original",
    "mae_to_original",
)


def _analysis_cfg(cfg_dict: DictConfig) -> AnalysisCfg:
    """Read optional ``+analysis.*`` Hydra overrides without changing RootCfg."""
    analysis_node = cfg_dict.get("analysis")
    if analysis_node is None:
        raw: dict[str, Any] = {}
    elif OmegaConf.is_config(analysis_node):
        container = OmegaConf.to_container(analysis_node, resolve=True)
        raw = {} if container is None else dict(container)
    else:
        raw = dict(analysis_node)
    if "scale_factors" in raw:
        raw["scale_factors"] = tuple(float(x) for x in raw["scale_factors"])
    if "example_scale_factors" in raw:
        raw["example_scale_factors"] = tuple(
            float(x) for x in raw["example_scale_factors"]
        )
    cfg = AnalysisCfg(**raw)
    if cfg.scenes_per_overlap <= 0:
        raise ValueError("analysis.scenes_per_overlap must be positive")
    if cfg.bootstrap_samples <= 0:
        raise ValueError("analysis.bootstrap_samples must be positive")
    if cfg.attribute_samples_per_scene <= 0:
        raise ValueError("analysis.attribute_samples_per_scene must be positive")
    if cfg.examples_per_overlap < 0:
        raise ValueError("analysis.examples_per_overlap cannot be negative")
    if cfg.save_every_scenes <= 0:
        raise ValueError("analysis.save_every_scenes must be positive")
    if not cfg.scale_factors or any(k <= 0 for k in cfg.scale_factors):
        raise ValueError("analysis.scale_factors must contain positive values")
    if not any(math.isclose(k, 1.0) for k in cfg.scale_factors):
        raise ValueError("analysis.scale_factors must include 1.0")
    missing_examples = [
        k
        for k in cfg.example_scale_factors
        if not any(math.isclose(k, scale_k) for scale_k in cfg.scale_factors)
    ]
    if missing_examples:
        raise ValueError(
            "Every analysis.example_scale_factors value must also be in "
            f"analysis.scale_factors. Missing: {missing_examples}"
        )
    return cfg


def _absolute_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path(to_absolute_path(str(path)))


def _find_evaluation_sampler(cfg_dict: DictConfig) -> tuple[str, DictConfig]:
    for name, dataset_cfg in cfg_dict.dataset.items():
        sampler = dataset_cfg.get("view_sampler")
        if sampler is not None and sampler.get("name") == "evaluation":
            return str(name), sampler
    raise ValueError(
        "No evaluation view sampler found. Use "
        "dataset/view_sampler@dataset.re10k.view_sampler=evaluation."
    )


def _overlap_tag_from_json(value: Any) -> str:
    if isinstance(value, str):
        if value in OVERLAP_TAGS:
            return value
        raise ValueError(f"Unknown overlap label: {value}")
    return get_overlap_tag(float(value))


def _make_stratified_index(
    source_path: Path,
    output_path: Path,
    scenes_per_overlap: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    with source_path.open("r", encoding="utf-8") as f:
        full_index = json.load(f)

    grouped: dict[str, list[str]] = {tag: [] for tag in OVERLAP_TAGS}
    for scene, entry in full_index.items():
        if entry is None:
            continue
        tag = _overlap_tag_from_json(entry["overlap"])
        if tag in grouped:
            grouped[tag].append(scene)

    rng = random.Random(seed)
    selected_by_tag: dict[str, list[str]] = {}
    for tag in OVERLAP_TAGS:
        candidates = sorted(grouped[tag])
        if len(candidates) < scenes_per_overlap:
            raise ValueError(
                f"Only {len(candidates)} {tag} scenes are available, but "
                f"{scenes_per_overlap} were requested."
            )
        selected_by_tag[tag] = rng.sample(candidates, scenes_per_overlap)

    # Preserve a deterministic group/order in the saved index.
    selected_index = {
        scene: full_index[scene]
        for tag in OVERLAP_TAGS
        for scene in selected_by_tag[tag]
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(selected_index, f, indent=2)

    return selected_index, selected_by_tag


def _to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def _freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def _load_encoder_checkpoint(
    encoder: nn.Module,
    checkpoint_path: Path,
    allow_mismatch: bool,
) -> tuple[int, list[str], list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    global_step = int(checkpoint.get("global_step", 0)) if isinstance(checkpoint, dict) else 0

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        encoder_state = {
            key[len("encoder.") :]: value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith("encoder.")
        }
        if not encoder_state:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has a state_dict but no encoder.* keys."
            )
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        encoder_state = checkpoint_filter_fn(checkpoint["model"], encoder)
    elif isinstance(checkpoint, dict):
        encoder_state = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if missing or unexpected:
        message = "Checkpoint does not exactly match the baseline encoder."
        print(message)
        if missing:
            print(f"  Missing encoder keys ({len(missing)}): {missing[:20]}")
        if unexpected:
            print(f"  Unexpected encoder keys ({len(unexpected)}): {unexpected[:20]}")
        if not allow_mismatch:
            raise RuntimeError(
                f"{message} Set +analysis.allow_checkpoint_mismatch=true only if "
                "this mismatch is understood and intentional."
            )
    return global_step, list(missing), list(unexpected)


def _align_target_extrinsics(
    decoder: nn.Module,
    losses: Iterable[nn.Module],
    batch: dict[str, Any],
    gaussians: Gaussians,
    global_step: int,
    enabled: bool,
    steps: int,
    rotation_lr: float,
    translation_lr: float,
) -> torch.Tensor:
    """Run the same iterative target-camera alignment used by ModelWrapper."""
    extrinsics = batch["target"]["extrinsics"].detach().clone()
    if not enabled:
        return extrinsics

    b, v = extrinsics.shape[:2]
    h, w = batch["target"]["image"].shape[-2:]
    with torch.enable_grad():
        rotation_delta = nn.Parameter(torch.zeros(b, v, 3, device=extrinsics.device))
        translation_delta = nn.Parameter(torch.zeros(b, v, 3, device=extrinsics.device))
        optimizer = torch.optim.Adam(
            [
                {"params": [rotation_delta], "lr": rotation_lr},
                {"params": [translation_delta], "lr": translation_lr},
            ]
        )

        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            output = decoder.forward(
                gaussians,
                extrinsics,
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                cam_rot_delta=rotation_delta,
                cam_trans_delta=translation_delta,
            )
            total_loss = sum(
                loss_fn.forward(output, batch, gaussians, global_step)
                for loss_fn in losses
            )
            if not torch.isfinite(total_loss):
                raise FloatingPointError("Non-finite loss during target-pose alignment")
            total_loss.backward()
            with torch.no_grad():
                optimizer.step()
                updated = update_pose(
                    cam_rot_delta=rearrange(rotation_delta, "b v i -> (b v) i"),
                    cam_trans_delta=rearrange(translation_delta, "b v i -> (b v) i"),
                    extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j"),
                )
                extrinsics = rearrange(updated, "(b v) i j -> b v i j", b=b, v=v)
                rotation_delta.zero_()
                translation_delta.zero_()

    return extrinsics.detach()


def _make_variant(gaussians: Gaussians, scale_factor: float, mode: str) -> Gaussians:
    if mode not in ("scale_only", "compensated"):
        raise ValueError(f"Unknown intervention mode: {mode}")

    covariance = gaussians.covariances * (scale_factor**2)
    opacity = gaussians.opacities
    if mode == "compensated" and not math.isclose(scale_factor, 1.0):
        # NoPoSplat opacities originate from a sigmoid and should be in (0, 1).
        # Clamping only protects log1p from rare float32 saturation at exactly 1.
        eps = torch.finfo(opacity.dtype).eps
        safe_opacity = opacity.clamp(min=0.0, max=1.0 - eps)
        optical_thickness = -torch.log1p(-safe_opacity)
        opacity = -torch.expm1(-optical_thickness / (scale_factor**2))

    return Gaussians(
        means=gaussians.means,
        covariances=covariance,
        harmonics=gaussians.harmonics,
        opacities=opacity,
    )


@torch.no_grad()
def _render(
    decoder: nn.Module,
    gaussians: Gaussians,
    batch: dict[str, Any],
    extrinsics: torch.Tensor,
) -> torch.Tensor:
    h, w = batch["target"]["image"].shape[-2:]
    return decoder.forward(
        gaussians,
        extrinsics,
        batch["target"]["intrinsics"],
        batch["target"]["near"],
        batch["target"]["far"],
        (h, w),
    ).color.clamp(0.0, 1.0)


def _metric_vectors(
    prediction: torch.Tensor,
    original: torch.Tensor,
    ground_truth: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b, v = prediction.shape[:2]
    assert b == 1
    pred_flat = rearrange(prediction, "b v c h w -> (b v) c h w")
    gt_flat = rearrange(ground_truth, "b v c h w -> (b v) c h w")
    psnr = compute_psnr(gt_flat, pred_flat)
    mse_self = ((prediction - original) ** 2).mean(dim=(0, 2, 3, 4))
    mae_self = (prediction - original).abs().mean(dim=(0, 2, 3, 4))
    assert len(psnr) == v
    return (
        psnr.detach().cpu().numpy(),
        mse_self.detach().cpu().numpy(),
        mae_self.detach().cpu().numpy(),
    )


def _sample_attributes(
    visualization_dump: dict[str, torch.Tensor],
    gaussians: Gaussians,
    sample_count: int,
    rng: np.random.Generator,
    overlap_tag: str,
) -> dict[str, np.ndarray]:
    scales = visualization_dump.get("scales")
    if scales is None:
        # Fallback for encoders that do not expose scale in visualization_dump.
        eigenvalues = torch.linalg.eigvalsh(gaussians.covariances).clamp_min(0)
        scales = eigenvalues.sqrt()
    scales = scales.reshape(-1, 3)
    opacity = gaussians.opacities.reshape(-1)
    count = min(sample_count, scales.shape[0])
    indices = rng.choice(scales.shape[0], size=count, replace=False)
    indices_t = torch.as_tensor(indices, device=scales.device, dtype=torch.long)
    sampled_scales = scales[indices_t].float()
    sampled_opacity = opacity[indices_t].float()

    squared = sampled_scales.square()
    coverage = torch.sqrt(
        (
            squared[:, 0] * squared[:, 1]
            + squared[:, 0] * squared[:, 2]
            + squared[:, 1] * squared[:, 2]
        )
        / 3.0
    )
    eps = torch.finfo(sampled_opacity.dtype).eps
    thickness = -torch.log1p(-sampled_opacity.clamp(0.0, 1.0 - eps))
    tag_id = OVERLAP_TAGS.index(overlap_tag)
    return {
        "scales": sampled_scales.cpu().numpy(),
        "coverage": coverage.cpu().numpy(),
        "opacity": sampled_opacity.cpu().numpy(),
        "optical_thickness": thickness.cpu().numpy(),
        "overlap_tag_id": np.full(count, tag_id, dtype=np.int8),
    }


def _scene_name(batch: dict[str, Any]) -> str:
    scene = batch["scene"]
    if isinstance(scene, (list, tuple)):
        if len(scene) != 1:
            raise ValueError("Analysis requires test batch_size=1")
        return str(scene[0])
    return str(scene)


def _index_list(tensor: torch.Tensor) -> list[int]:
    return [int(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def _append_records(
    records: list[dict[str, Any]],
    scene: str,
    context_indices: list[int],
    target_indices: list[int],
    overlap: float,
    overlap_tag: str,
    scale_factor: float,
    mode: str,
    psnr: np.ndarray,
    original_psnr: np.ndarray,
    mse_self: np.ndarray,
    mae_self: np.ndarray,
) -> None:
    for target_position, target_index in enumerate(target_indices):
        records.append(
            {
                "scene": scene,
                "context_indices": "-".join(map(str, context_indices)),
                "target_index": target_index,
                "target_position": target_position,
                "overlap": overlap,
                "overlap_tag": overlap_tag,
                "scale_factor": scale_factor,
                "mode": mode,
                "psnr_gt": float(psnr[target_position]),
                "delta_psnr": float(psnr[target_position] - original_psnr[target_position]),
                "mse_to_original": float(mse_self[target_position]),
                "mae_to_original": float(mae_self[target_position]),
            }
        )


def _bootstrap_ci(
    values: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value
    indices = rng.integers(0, values.size, size=(samples, values.size))
    bootstrap_means = values[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(low), float(high)


def _mean_by_scene(
    records: list[dict[str, Any]],
    scale_factor: float,
    mode: str,
    field: str,
    overlap_tag: str | None = None,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        if row["mode"] != mode or not math.isclose(row["scale_factor"], scale_factor):
            continue
        if overlap_tag is not None and row["overlap_tag"] != overlap_tag:
            continue
        grouped[row["scene"]].append(float(row[field]))
    return {scene: float(np.mean(values)) for scene, values in grouped.items()}


def _summarize(
    records: list[dict[str, Any]],
    scale_factors: Iterable[float],
    bootstrap_samples: int,
    seed: int,
    overlap_tag: str | None = None,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    summary: list[dict[str, Any]] = []
    for k in sorted(float(x) for x in scale_factors):
        scale_mse = _mean_by_scene(records, k, "scale_only", "mse_to_original", overlap_tag)
        comp_mse = _mean_by_scene(records, k, "compensated", "mse_to_original", overlap_tag)
        scale_delta = _mean_by_scene(records, k, "scale_only", "delta_psnr", overlap_tag)
        comp_delta = _mean_by_scene(records, k, "compensated", "delta_psnr", overlap_tag)
        scale_psnr = _mean_by_scene(records, k, "scale_only", "psnr_gt", overlap_tag)
        comp_psnr = _mean_by_scene(records, k, "compensated", "psnr_gt", overlap_tag)
        scenes = sorted(
            set(scale_mse)
            & set(comp_mse)
            & set(scale_delta)
            & set(comp_delta)
            & set(scale_psnr)
            & set(comp_psnr)
        )
        if not scenes:
            continue

        scale_mse_values = np.asarray([scale_mse[s] for s in scenes])
        comp_mse_values = np.asarray([comp_mse[s] for s in scenes])
        scale_delta_values = np.asarray([scale_delta[s] for s in scenes])
        comp_delta_values = np.asarray([comp_delta[s] for s in scenes])
        gain_values = np.asarray([comp_psnr[s] - scale_psnr[s] for s in scenes])
        drift_reduction = scale_mse_values - comp_mse_values

        gain_ci = _bootstrap_ci(gain_values, bootstrap_samples, rng)
        drift_ci = _bootstrap_ci(drift_reduction, bootstrap_samples, rng)
        mean_scale_mse = float(scale_mse_values.mean())
        mean_comp_mse = float(comp_mse_values.mean())
        drift_ratio = (
            mean_comp_mse / mean_scale_mse if mean_scale_mse > 0 else None
        )
        summary.append(
            {
                "overlap_tag": overlap_tag or "all",
                "scale_factor": k,
                "num_scenes": len(scenes),
                "scale_only_self_mse": mean_scale_mse,
                "compensated_self_mse": mean_comp_mse,
                "self_mse_ratio_comp_over_scale": drift_ratio,
                "self_mse_reduction": float(drift_reduction.mean()),
                "self_mse_reduction_ci95_low": drift_ci[0],
                "self_mse_reduction_ci95_high": drift_ci[1],
                "scale_only_delta_psnr": float(scale_delta_values.mean()),
                "compensated_delta_psnr": float(comp_delta_values.mean()),
                "compensation_gain_psnr": float(gain_values.mean()),
                "compensation_gain_ci95_low": gain_ci[0],
                "compensation_gain_ci95_high": gain_ci[1],
                "paired_win_rate": (
                    float((gain_values > 0).mean())
                    if not math.isclose(k, 1.0)
                    else None
                ),
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _save_raw_npz(path: Path, records: list[dict[str, Any]]) -> None:
    arrays = {field: np.asarray([row[field] for row in records]) for field in RAW_FIELDS}
    np.savez_compressed(path, **arrays)


def _json_safe(value: Any) -> Any:
    """Convert numpy values and non-finite floats into strict JSON values."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _attribute_summary(samples: dict[str, np.ndarray]) -> dict[str, Any]:
    quantile_points = np.asarray([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])

    def describe(values: np.ndarray) -> dict[str, Any]:
        values = np.asarray(values, dtype=np.float64)
        quantiles = np.quantile(values, quantile_points)
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "quantiles": {
                f"p{int(q * 100):02d}": float(value)
                for q, value in zip(quantile_points, quantiles)
            },
        }

    coverage = samples["coverage"]
    opacity = samples["opacity"]
    thickness = samples["optical_thickness"]
    valid = (coverage > 0) & (thickness > 0) & np.isfinite(coverage) & np.isfinite(thickness)
    correlation = (
        float(np.corrcoef(np.log(coverage[valid]), np.log(thickness[valid]))[0, 1])
        if valid.sum() > 1
        else float("nan")
    )
    return {
        "num_samples": int(len(coverage)),
        "scale_axis_0": describe(samples["scales"][:, 0]),
        "scale_axis_1": describe(samples["scales"][:, 1]),
        "scale_axis_2": describe(samples["scales"][:, 2]),
        "coverage": describe(coverage),
        "opacity": describe(opacity),
        "optical_thickness": describe(thickness),
        "opacity_below_0.01_fraction": float((opacity < 0.01).mean()),
        "opacity_above_0.99_fraction": float((opacity > 0.99).mean()),
        "log_coverage_log_thickness_correlation": correlation,
    }


def _save_example(
    output_path: Path,
    scene: str,
    target_index: int,
    gt: torch.Tensor,
    original: torch.Tensor,
    variants: dict[tuple[str, float], torch.Tensor],
    factors: Iterable[float],
    error_gain: float,
) -> None:
    factors = list(factors)
    figure, axes = plt.subplots(
        len(factors),
        6,
        figsize=(18, 3.2 * len(factors)),
        squeeze=False,
    )
    for row_index, k in enumerate(factors):
        scale = variants[("scale_only", float(k))]
        compensated = variants[("compensated", float(k))]
        scale_error = (scale - original).abs().mean(dim=0, keepdim=True).repeat(3, 1, 1)
        comp_error = (
            (compensated - original).abs().mean(dim=0, keepdim=True).repeat(3, 1, 1)
        )
        images = (
            gt,
            original,
            scale,
            compensated,
            (scale_error * error_gain).clamp(0, 1),
            (comp_error * error_gain).clamp(0, 1),
        )
        titles = (
            "GT",
            "Original",
            f"Scale-only k={k:g}",
            f"Compensated k={k:g}",
            f"Scale error x{error_gain:g}",
            f"Comp. error x{error_gain:g}",
        )
        for axis, image, title in zip(axes[row_index], images, titles):
            array = image.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            axis.imshow(array)
            axis.set_title(title)
            axis.axis("off")

    output_path.mkdir(parents=True, exist_ok=True)
    figure.suptitle(f"Scene {scene} | target {target_index}")
    figure.tight_layout()
    figure.savefig(
        output_path / f"{scene}_target_{target_index:06d}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_results(output_dir: Path, summary: list[dict[str, Any]]) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    overall = sorted(
        [row for row in summary if row["overlap_tag"] == "all"],
        key=lambda row: row["scale_factor"],
    )
    if not overall:
        return
    k = np.asarray([row["scale_factor"] for row in overall])

    plt.figure(figsize=(6.2, 4.2))
    scale_mse = np.asarray([row["scale_only_self_mse"] for row in overall])
    comp_mse = np.asarray([row["compensated_self_mse"] for row in overall])
    non_identity = ~np.isclose(k, 1.0)
    plt.semilogy(
        k[non_identity],
        np.maximum(scale_mse[non_identity], 1e-16),
        "o-",
        label="Scale-only",
    )
    plt.semilogy(
        k[non_identity],
        np.maximum(comp_mse[non_identity], 1e-16),
        "o-",
        label="Compensated",
    )
    plt.xlabel("Scale factor k")
    plt.ylabel("MSE to original rendering")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "rendering_drift_vs_scale.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6.2, 4.2))
    scale_delta = np.asarray([row["scale_only_delta_psnr"] for row in overall])
    comp_delta = np.asarray([row["compensated_delta_psnr"] for row in overall])
    plt.plot(k, scale_delta, "o-", label="Scale-only")
    plt.plot(k, comp_delta, "o-", label="Compensated")
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Scale factor k")
    plt.ylabel("PSNR change from original (dB)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "delta_psnr_vs_scale.png", dpi=180)
    plt.close()

    gain = np.asarray([row["compensation_gain_psnr"] for row in overall])
    low = np.asarray([row["compensation_gain_ci95_low"] for row in overall])
    high = np.asarray([row["compensation_gain_ci95_high"] for row in overall])
    plt.figure(figsize=(6.2, 4.2))
    yerr = np.vstack(
        [np.maximum(gain - low, 0.0), np.maximum(high - gain, 0.0)]
    )
    plt.errorbar(k, gain, yerr=yerr, fmt="o-")
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Scale factor k")
    plt.ylabel("Compensated PSNR - scale-only PSNR (dB)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures / "compensation_gain_vs_scale.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6.2, 4.2))
    for tag in OVERLAP_TAGS:
        rows = sorted(
            [row for row in summary if row["overlap_tag"] == tag],
            key=lambda row: row["scale_factor"],
        )
        if rows:
            plt.plot(
                [row["scale_factor"] for row in rows],
                [row["compensation_gain_psnr"] for row in rows],
                "o-",
                label=tag,
            )
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Scale factor k")
    plt.ylabel("Compensation gain (dB)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "gain_by_overlap.png", dpi=180)
    plt.close()


def _git_info() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status and status != "unknown"),
        "status": status,
    }


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg_dict: DictConfig) -> None:
    analysis = _analysis_cfg(cfg_dict)
    output_dir = _absolute_path(analysis.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name, sampler_cfg = _find_evaluation_sampler(cfg_dict)
    source_index_path = _absolute_path(sampler_cfg.index_path)
    selected_index_path = output_dir / "selected_scenes.json"
    selected_index, selected_by_tag = _make_stratified_index(
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
        raise ValueError("checkpointing.load must point to a pretrained NoPoSplat checkpoint")
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)
    checkpoint_path = _absolute_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not torch.cuda.is_available():
        raise RuntimeError("This analysis requires CUDA Gaussian rasterization")
    device = torch.device("cuda")

    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder)
    encoder = _freeze(encoder.to(device))
    decoder = _freeze(decoder.to(device))
    losses = nn.ModuleList(get_losses(cfg.loss)).to(device)
    _freeze(losses)
    global_step, missing_keys, unexpected_keys = _load_encoder_checkpoint(
        encoder,
        checkpoint_path,
        analysis.allow_checkpoint_mismatch,
    )

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
            raise ValueError("Scale-opacity analysis currently supports one test dataset")
        test_loader = test_loader[0]
    data_shim = encoder.get_data_shim()

    example_scenes = {
        scene
        for tag in OVERLAP_TAGS
        for scene in selected_by_tag[tag][: analysis.examples_per_overlap]
    }
    records: list[dict[str, Any]] = []
    processed_scenes: list[dict[str, Any]] = []
    attribute_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    attribute_rng = np.random.default_rng(analysis.seed + 1)
    identity_max_abs = float("nan")

    progress = tqdm(total=len(selected_index), desc="Scale-opacity analysis")
    for batch in test_loader:
        batch = _to_device(batch, device)
        batch = data_shim(batch)
        scene = _scene_name(batch)
        if scene not in selected_index:
            continue
        if batch["target"]["image"].shape[0] != 1:
            raise ValueError("Analysis requires data_loader.test.batch_size=1")

        overlap = float(batch["context"]["overlap"].reshape(-1)[0].item())
        overlap_tag = get_overlap_tag(overlap)
        context_indices = _index_list(batch["context"]["index"])
        target_indices = _index_list(batch["target"]["index"])

        visualization_dump: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            gaussians = encoder(
                batch["context"],
                global_step,
                visualization_dump=visualization_dump,
            )
        aligned_extrinsics = _align_target_extrinsics(
            decoder=decoder,
            losses=losses,
            batch=batch,
            gaussians=gaussians,
            global_step=global_step,
            enabled=cfg.test.align_pose,
            steps=cfg.test.pose_align_steps,
            rotation_lr=cfg.test.rot_opt_lr,
            translation_lr=cfg.test.trans_opt_lr,
        )
        original = _render(decoder, gaussians, batch, aligned_extrinsics)
        ground_truth = batch["target"]["image"].clamp(0.0, 1.0)
        original_psnr, _, _ = _metric_vectors(original, original, ground_truth)

        zeros = np.zeros_like(original_psnr)
        _append_records(
            records,
            scene,
            context_indices,
            target_indices,
            overlap,
            overlap_tag,
            1.0,
            "original",
            original_psnr,
            original_psnr,
            zeros,
            zeros,
        )

        attributes = _sample_attributes(
            visualization_dump,
            gaussians,
            analysis.attribute_samples_per_scene,
            attribute_rng,
            overlap_tag,
        )
        for key, values in attributes.items():
            attribute_chunks[key].append(values)

        example_variants: dict[tuple[str, float], torch.Tensor] = {}
        for scale_factor in analysis.scale_factors:
            for mode in ("scale_only", "compensated"):
                if math.isclose(scale_factor, 1.0):
                    rendered = original
                else:
                    variant = _make_variant(gaussians, scale_factor, mode)
                    rendered = _render(decoder, variant, batch, aligned_extrinsics)
                psnr, mse_self, mae_self = _metric_vectors(
                    rendered, original, ground_truth
                )
                _append_records(
                    records,
                    scene,
                    context_indices,
                    target_indices,
                    overlap,
                    overlap_tag,
                    float(scale_factor),
                    mode,
                    psnr,
                    original_psnr,
                    mse_self,
                    mae_self,
                )
                if (
                    scene in example_scenes
                    and any(
                        math.isclose(scale_factor, example_k)
                        for example_k in analysis.example_scale_factors
                    )
                ):
                    example_variants[(mode, float(scale_factor))] = rendered[0, 0].cpu()

        if not processed_scenes:
            scale_identity = _render(
                decoder,
                _make_variant(gaussians, 1.0, "scale_only"),
                batch,
                aligned_extrinsics,
            )
            compensated_identity = _render(
                decoder,
                _make_variant(gaussians, 1.0, "compensated"),
                batch,
                aligned_extrinsics,
            )
            identity_max_abs = float(
                torch.stack(
                    [
                        (scale_identity - original).abs().max(),
                        (compensated_identity - original).abs().max(),
                    ]
                )
                .max()
                .item()
            )
            if identity_max_abs > analysis.identity_tolerance:
                raise AssertionError(
                    f"k=1 identity check failed: max abs error={identity_max_abs:.3e}"
                )

        if scene in example_scenes:
            expected_example_keys = {
                (mode, float(k))
                for k in analysis.example_scale_factors
                for mode in ("scale_only", "compensated")
            }
            missing_example_keys = expected_example_keys - set(example_variants)
            if missing_example_keys:
                raise ValueError(
                    "Every analysis.example_scale_factors value must also appear in "
                    f"analysis.scale_factors. Missing: {sorted(missing_example_keys)}"
                )
            _save_example(
                output_dir / "examples" / overlap_tag,
                scene,
                target_indices[0],
                ground_truth[0, 0].cpu(),
                original[0, 0].cpu(),
                example_variants,
                analysis.example_scale_factors,
                analysis.error_visualization_gain,
            )

        processed_scenes.append(
            {
                "scene": scene,
                "overlap": overlap,
                "overlap_tag": overlap_tag,
                "context": context_indices,
                "target": target_indices,
            }
        )
        if len(processed_scenes) % analysis.save_every_scenes == 0:
            _write_csv(output_dir / "records.partial.csv", records, RAW_FIELDS)
            with (output_dir / "processed_scenes.partial.json").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(processed_scenes, f, indent=2)
        progress.update(1)
        progress.set_postfix(
            {
                "scene": scene[:8],
                "overlap": overlap_tag,
                "psnr": f"{original_psnr.mean():.2f}",
            }
        )
    progress.close()

    if not processed_scenes:
        raise RuntimeError(
            "No selected evaluation scenes were found in the configured RE10K dataset."
        )

    processed_counts = {
        tag: sum(scene["overlap_tag"] == tag for scene in processed_scenes)
        for tag in OVERLAP_TAGS
    }
    for tag, count in processed_counts.items():
        if count != analysis.scenes_per_overlap:
            print(
                f"Warning: processed {count}/{analysis.scenes_per_overlap} {tag} scenes. "
                "Some indexed scenes may be absent or invalid in this dataset copy."
            )

    attributes = {
        key: np.concatenate(chunks, axis=0) for key, chunks in attribute_chunks.items()
    }
    attribute_stats = _attribute_summary(attributes)
    np.savez_compressed(output_dir / "attribute_stats.npz", **attributes)

    overall_summary = _summarize(
        records,
        analysis.scale_factors,
        analysis.bootstrap_samples,
        analysis.seed + 2,
    )
    overlap_summary = [
        row
        for tag_index, tag in enumerate(OVERLAP_TAGS)
        for row in _summarize(
            records,
            analysis.scale_factors,
            analysis.bootstrap_samples,
            analysis.seed + 10 + tag_index,
            overlap_tag=tag,
        )
    ]
    summary = overall_summary + overlap_summary

    _write_csv(output_dir / "records.csv", records, RAW_FIELDS)
    _save_raw_npz(output_dir / "raw_results.npz", records)
    summary_fields = list(summary[0].keys()) if summary else []
    if summary_fields:
        _write_csv(output_dir / "summary.csv", summary, summary_fields)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            _json_safe(
                {
                    "aggregate": summary,
                    "attribute_statistics": attribute_stats,
                }
            ),
            f,
            indent=2,
            allow_nan=False,
        )
    with (output_dir / "processed_scenes.json").open("w", encoding="utf-8") as f:
        json.dump(processed_scenes, f, indent=2)

    manifest = {
        "analysis": asdict(analysis),
        "dataset": dataset_name,
        "source_evaluation_index": str(source_index_path),
        "selected_evaluation_index": str(selected_index_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_global_step": global_step,
        "checkpoint_missing_encoder_keys": missing_keys,
        "checkpoint_unexpected_encoder_keys": unexpected_keys,
        "processed_scene_counts": processed_counts,
        "num_target_records": len(records),
        "identity_max_abs_error": identity_max_abs,
        "pose_alignment": {
            "enabled": cfg.test.align_pose,
            "steps": cfg.test.pose_align_steps,
            "rotation_lr": cfg.test.rot_opt_lr,
            "translation_lr": cfg.test.trans_opt_lr,
            "optimized_once_on_original_gaussians": True,
            "frozen_for_all_interventions": True,
        },
        "interventions": {
            "scale_only": "Sigma_k = k^2 Sigma; alpha_k = alpha",
            "compensated": (
                "Sigma_k = k^2 Sigma; tau=-log(1-alpha); "
                "tau_k=tau/k^2; alpha_k=1-exp(-tau_k)"
            ),
            "means_fixed": True,
            "harmonics_fixed": True,
        },
        "git": _git_info(),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(manifest), f, indent=2, allow_nan=False)

    _plot_results(output_dir, summary)
    print(f"\nAnalysis complete: {output_dir}")
    print(f"Processed scenes: {processed_counts}")
    print(f"Identity max abs error: {identity_max_abs:.3e}")
    print("Primary table: summary.csv")


if __name__ == "__main__":
    main()
