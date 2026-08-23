"""Audit attribute-wise photometric gradients of a frozen NoPoSplat decoder.

The encoder receives context images only.  Held-out target images and poses are
used after Gaussian decoding, solely to render and differentiate an analysis MSE.
No model parameter is trainable and no optimizer step is performed.

The Gaussian-head contribution is measured at the common feature entering its
final 1x1 prediction convolution.  Scale gradients are decomposed in physical
log-scale coordinates into an isotropic ``size`` component and a zero-mean
``shape`` component.  Together with rotation, opacity, SH DC, and remaining SH,
these groups exactly partition the gradient produced by the Gaussian head.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import hydra
import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import install_import_hook
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm

with install_import_hook(("src",), ("beartype", "beartype")):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.step_tracker import StepTracker
    from src.misc.utils import get_overlap_tag
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.scripts.analysis_scale_opacity import (
        OVERLAP_TAGS,
        _absolute_path,
        _align_target_extrinsics,
        _find_evaluation_sampler,
        _freeze,
        _index_list,
        _load_encoder_checkpoint,
        _scene_name,
        _to_device,
    )


GROUPS = (
    "size",
    "shape",
    "rotation",
    "opacity",
    "base_color",
    "view_color",
)


@dataclass(frozen=True)
class GradientAnalysisCfg:
    output_dir: str = "outputs/gaussian_decoder_analysis/gradient_audit/run"
    scene_split_path: str = (
        "outputs/gaussian_decoder_analysis/splits/gradient_900.json"
    )
    bootstrap_samples: int = 2000
    seed: int = 20260823
    identity_relative_tolerance: float = 1e-4
    minimum_gradient_norm: float = 1e-20
    allow_checkpoint_mismatch: bool = False
    resume: bool = True
    fail_fast: bool = True


def _gradient_cfg(cfg_dict: DictConfig) -> GradientAnalysisCfg:
    node = cfg_dict.get("gradient_analysis")
    if node is None:
        raw: dict[str, Any] = {}
    elif OmegaConf.is_config(node):
        container = OmegaConf.to_container(node, resolve=True)
        raw = {} if container is None else dict(container)
    else:
        raw = dict(node)
    cfg = GradientAnalysisCfg(**raw)
    if cfg.bootstrap_samples <= 0:
        raise ValueError("gradient_analysis.bootstrap_samples must be positive")
    if cfg.identity_relative_tolerance <= 0:
        raise ValueError(
            "gradient_analysis.identity_relative_tolerance must be positive"
        )
    if cfg.minimum_gradient_norm <= 0:
        raise ValueError("gradient_analysis.minimum_gradient_norm must be positive")
    return cfg


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_info() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args], check=True, capture_output=True, text=True
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


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(value), file, indent=2, allow_nan=False)
    temporary.replace(path)


def _save_chunk(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as file:
        json.dump(_json_safe(value), file, separators=(",", ":"), allow_nan=False)
    temporary.replace(path)


def _load_chunk(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return json.load(file)


def _append_failure(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_json_safe(row), allow_nan=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_split(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        index = json.load(file)
    if not isinstance(index, dict) or not index:
        raise ValueError(f"Scene split must be a non-empty JSON object: {path}")
    for scene, entry in index.items():
        if entry is None or "overlap" not in entry:
            raise ValueError(f"Invalid selected scene {scene} in {path}")
        if get_overlap_tag(float(entry["overlap"])) not in OVERLAP_TAGS:
            raise ValueError(f"Selected scene {scene} has unsupported overlap")
    return index


class GaussianHeadFeatureCapture:
    """Replace final-conv inputs by leaf tensors and retain its raw outputs."""

    def __init__(self, encoder: nn.Module) -> None:
        self.enabled = False
        self.features: dict[int, torch.Tensor] = {}
        self.raw_outputs: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []

        heads = [encoder.gaussian_param_head, encoder.gaussian_param_head2]
        for view_index, head in enumerate(heads):
            try:
                final_layer = head.dpt.head[-1]
            except (AttributeError, IndexError, TypeError) as error:
                raise RuntimeError(
                    "Could not locate the baseline DPT Gaussian head's final "
                    "prediction convolution."
                ) from error
            if not isinstance(final_layer, nn.Conv2d):
                raise TypeError(
                    "Expected the final Gaussian prediction layer to be Conv2d, "
                    f"got {type(final_layer).__name__}."
                )

            def pre_hook(_module, inputs, vi=view_index):
                if not self.enabled:
                    return None
                if len(inputs) != 1:
                    raise RuntimeError("Unexpected final-convolution input signature")
                feature = inputs[0].detach().requires_grad_(True)
                self.features[vi] = feature
                return (feature,)

            def output_hook(_module, _inputs, output, vi=view_index):
                if self.enabled:
                    self.raw_outputs[vi] = output

            self.handles.append(final_layer.register_forward_pre_hook(pre_hook))
            self.handles.append(final_layer.register_forward_hook(output_hook))

    def start(self) -> None:
        self.clear()
        self.enabled = True

    def stop(self) -> None:
        self.enabled = False

    def clear(self) -> None:
        self.features.clear()
        self.raw_outputs.clear()

    def ordered(
        self, expected_views: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        missing_features = [i for i in range(expected_views) if i not in self.features]
        missing_outputs = [i for i in range(expected_views) if i not in self.raw_outputs]
        if missing_features or missing_outputs:
            raise RuntimeError(
                "Gaussian-head capture is incomplete: "
                f"missing features={missing_features}, outputs={missing_outputs}"
            )
        return (
            [self.features[i] for i in range(expected_views)],
            [self.raw_outputs[i] for i in range(expected_views)],
        )

    def close(self) -> None:
        self.stop()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()


def _stable_dlogsoftplus(raw_scale: torch.Tensor) -> torch.Tensor:
    """Derivative d log(softplus(q)) / dq, evaluated without low-tail 0/0."""
    ordinary = raw_scale.sigmoid() / F.softplus(raw_scale).clamp_min(1e-30)
    return torch.where(raw_scale < -20.0, torch.ones_like(ordinary), ordinary)


def _partition_one_raw_gradient(
    raw: torch.Tensor,
    raw_gradient: torch.Tensor,
    d_sh: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    expected_channels = 1 + 3 + 4 + 3 * d_sh
    if raw.shape[1] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} Gaussian channels, got {raw.shape[1]}"
        )

    pieces = {group: torch.zeros_like(raw_gradient) for group in GROUPS}
    coordinates: dict[str, torch.Tensor] = {}

    # Opacity is the first baseline output logit.
    pieces["opacity"][:, 0:1] = raw_gradient[:, 0:1]
    coordinates["opacity"] = raw_gradient[:, 0:1]

    # Baseline raw scales q map to s = 0.001 softplus(q).  Transform the
    # gradient into log-scale coordinates, project it into span{(1,1,1)} and
    # its zero-mean complement, then map both pieces back to q coordinates.
    raw_scale = raw[:, 1:4].detach()
    raw_scale_gradient = raw_gradient[:, 1:4]
    dlogscale_draw = _stable_dlogsoftplus(raw_scale)
    logscale_gradient = raw_scale_gradient / dlogscale_draw.clamp_min(1e-12)
    size_log_gradient = logscale_gradient.mean(dim=1, keepdim=True).expand_as(
        logscale_gradient
    )
    shape_log_gradient = logscale_gradient - size_log_gradient
    pieces["size"][:, 1:4] = size_log_gradient * dlogscale_draw
    pieces["shape"][:, 1:4] = shape_log_gradient * dlogscale_draw
    coordinates["size"] = size_log_gradient
    coordinates["shape"] = shape_log_gradient

    pieces["rotation"][:, 4:8] = raw_gradient[:, 4:8]
    coordinates["rotation"] = raw_gradient[:, 4:8]

    sh_start = 8
    base_indices = [sh_start + color * d_sh for color in range(3)]
    view_indices = [
        sh_start + color * d_sh + coefficient
        for color in range(3)
        for coefficient in range(1, d_sh)
    ]
    pieces["base_color"][:, base_indices] = raw_gradient[:, base_indices]
    pieces["view_color"][:, view_indices] = raw_gradient[:, view_indices]
    coordinates["base_color"] = raw_gradient[:, base_indices]
    coordinates["view_color"] = raw_gradient[:, view_indices]
    return pieces, coordinates


def _tensor_list_dot(
    left: list[torch.Tensor], right: list[torch.Tensor]
) -> torch.Tensor:
    # torch.dot avoids materializing a full element-wise product or float64 copy
    # of the high-resolution DPT feature maps.  Only the scalar accumulator is
    # promoted to float64.
    total = torch.zeros((), dtype=torch.float64, device=left[0].device)
    for a, b in zip(left, right):
        total = total + torch.dot(a.reshape(-1), b.reshape(-1)).double()
    return total


def _tensor_list_numel(values: list[torch.Tensor]) -> int:
    return sum(value.numel() for value in values)


def _norm_metrics(values: list[torch.Tensor]) -> tuple[float, float]:
    squared = _tensor_list_dot(values, values)
    l2 = squared.clamp_min(0).sqrt()
    rms = (squared / max(_tensor_list_numel(values), 1)).clamp_min(0).sqrt()
    return float(l2.item()), float(rms.item())


def _relative_error(
    actual: list[torch.Tensor], expected: list[torch.Tensor], epsilon: float
) -> tuple[float, float]:
    differences = [a - b for a, b in zip(actual, expected)]
    diff_l2, _ = _norm_metrics(differences)
    expected_l2, _ = _norm_metrics(expected)
    max_abs = max(float(value.abs().max().item()) for value in differences)
    return diff_l2 / max(expected_l2, epsilon), max_abs


def _analyze_scene(
    batch: dict[str, Any],
    encoder: nn.Module,
    decoder: nn.Module,
    alignment_losses: nn.ModuleList,
    capture: GaussianHeadFeatureCapture,
    cfg: GradientAnalysisCfg,
    test_cfg: Any,
    global_step: int,
) -> dict[str, Any]:
    batch_size, context_views = batch["context"]["image"].shape[:2]
    if batch_size != 1:
        raise ValueError("Gradient audit requires data_loader.test.batch_size=1")
    if context_views != 2:
        raise ValueError(
            "The current baseline gradient capture expects exactly two context heads"
        )

    # Target-pose alignment is an evaluation-only operation.  Decode once without
    # gradient tracking, align, and then decode the identical context again for the
    # gradient audit while treating the aligned cameras as constants.
    capture.stop()
    with torch.no_grad():
        alignment_gaussians = encoder(batch["context"], global_step)
    aligned_extrinsics = _align_target_extrinsics(
        decoder=decoder,
        losses=alignment_losses,
        batch=batch,
        gaussians=alignment_gaussians,
        global_step=global_step,
        enabled=test_cfg.align_pose,
        steps=test_cfg.pose_align_steps,
        rotation_lr=test_cfg.rot_opt_lr,
        translation_lr=test_cfg.trans_opt_lr,
    )
    del alignment_gaussians

    capture.start()
    try:
        with torch.enable_grad():
            gaussians = encoder(batch["context"], global_step)
            features, raw_outputs = capture.ordered(context_views)
            height, width = batch["target"]["image"].shape[-2:]
            prediction = decoder.forward(
                gaussians,
                aligned_extrinsics,
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (height, width),
            )
            delta = prediction.color - batch["target"]["image"]
            loss = delta.square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite photometric MSE")

            raw_gradients = list(
                torch.autograd.grad(
                    loss,
                    raw_outputs,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )
            )
            if not all(torch.isfinite(value).all() for value in raw_gradients):
                raise FloatingPointError("Non-finite gradient at Gaussian output")

            d_sh = int(encoder.gaussian_adapter.d_sh)
            grouped_raw = {group: [] for group in GROUPS}
            grouped_coordinates = {group: [] for group in GROUPS}
            for raw, raw_gradient in zip(raw_outputs, raw_gradients):
                pieces, coordinates = _partition_one_raw_gradient(
                    raw, raw_gradient, d_sh
                )
                for group in GROUPS:
                    grouped_raw[group].append(pieces[group])
                    grouped_coordinates[group].append(coordinates[group])

            raw_sum = [
                sum(grouped_raw[group][view] for group in GROUPS)
                for view in range(context_views)
            ]
            raw_identity_relative, raw_identity_max_abs = _relative_error(
                raw_sum, raw_gradients, cfg.minimum_gradient_norm
            )

            feature_gradients: dict[str, list[torch.Tensor]] = {}
            for group in GROUPS:
                feature_gradients[group] = list(
                    torch.autograd.grad(
                        raw_outputs,
                        features,
                        grad_outputs=grouped_raw[group],
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=False,
                    )
                )

            full_feature_gradient = list(
                torch.autograd.grad(
                    raw_outputs,
                    features,
                    grad_outputs=raw_gradients,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )
            )
            feature_sum = [
                sum(feature_gradients[group][view] for group in GROUPS)
                for view in range(context_views)
            ]
            feature_identity_relative, feature_identity_max_abs = _relative_error(
                feature_sum, full_feature_gradient, cfg.minimum_gradient_norm
            )

            if not all(
                torch.isfinite(value).all()
                for group in GROUPS
                for value in feature_gradients[group]
            ):
                raise FloatingPointError("Non-finite shared-feature gradient")

            mse_by_view = delta.square().mean(dim=(0, 2, 3, 4))
            psnr_by_view = -10.0 * torch.log10(mse_by_view.clamp_min(1e-12))

            group_rows: list[dict[str, Any]] = []
            norms: dict[str, float] = {}
            for group in GROUPS:
                feature_l2, feature_rms = _norm_metrics(feature_gradients[group])
                output_l2, output_rms = _norm_metrics(grouped_coordinates[group])
                norms[group] = feature_l2
                group_rows.append(
                    {
                        "group": group,
                        "feature_gradient_l2": feature_l2,
                        "feature_gradient_rms": feature_rms,
                        "output_gradient_l2": output_l2,
                        "output_gradient_rms": output_rms,
                        "feature_gradient_finite": True,
                    }
                )

            pair_rows: list[dict[str, Any]] = []
            for first_index, group_a in enumerate(GROUPS):
                for group_b in GROUPS[first_index + 1 :]:
                    denominator = norms[group_a] * norms[group_b]
                    valid = denominator > cfg.minimum_gradient_norm
                    cosine = (
                        float(
                            (
                                _tensor_list_dot(
                                    feature_gradients[group_a],
                                    feature_gradients[group_b],
                                )
                                / denominator
                            ).item()
                        )
                        if valid
                        else float("nan")
                    )
                    pair_rows.append(
                        {
                            "group_a": group_a,
                            "group_b": group_b,
                            "cosine": max(-1.0, min(1.0, cosine))
                            if math.isfinite(cosine)
                            else cosine,
                            "negative": bool(cosine < 0) if valid else False,
                            "valid": valid,
                        }
                    )

            full_l2, full_rms = _norm_metrics(full_feature_gradient)
            identity_pass = (
                raw_identity_relative <= cfg.identity_relative_tolerance
                and feature_identity_relative <= cfg.identity_relative_tolerance
            )
            scene = _scene_name(batch)
            overlap = float(batch["context"]["overlap"].reshape(-1)[0].item())
            scene_row = {
                "scene": scene,
                "overlap": overlap,
                "overlap_tag": get_overlap_tag(overlap),
                "context_indices": _index_list(batch["context"]["index"]),
                "target_indices": _index_list(batch["target"]["index"]),
                "target_views": int(delta.shape[1]),
                "mse": float(loss.item()),
                "psnr_mean": float(psnr_by_view.mean().item()),
                "psnr_by_view": [float(value) for value in psnr_by_view.tolist()],
                "full_feature_gradient_l2": full_l2,
                "full_feature_gradient_rms": full_rms,
                "raw_identity_relative_error": raw_identity_relative,
                "raw_identity_max_abs": raw_identity_max_abs,
                "feature_identity_relative_error": feature_identity_relative,
                "feature_identity_max_abs": feature_identity_max_abs,
                "identity_pass": identity_pass,
            }
            if not identity_pass:
                raise AssertionError(
                    "Attribute gradient partition identity failed: "
                    f"raw={raw_identity_relative:.3e}, "
                    f"feature={feature_identity_relative:.3e}, tolerance="
                    f"{cfg.identity_relative_tolerance:.3e}"
                )
            return {
                "scene": scene_row,
                "groups": group_rows,
                "pairs": pair_rows,
            }
    finally:
        capture.stop()
        # Do not retain the per-scene autograd graph while the next scene runs its
        # camera-alignment pass.
        capture.clear()


def _bootstrap_mean_ci(
    values: list[float], samples: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    array = np.asarray([value for value in values if math.isfinite(value)])
    if array.size == 0:
        return float("nan"), float("nan"), float("nan")
    if array.size == 1:
        value = float(array[0])
        return value, value, value
    draw = rng.integers(0, array.size, size=(samples, array.size))
    means = array[draw].mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _summaries(
    payloads: list[dict[str, Any]], cfg: GradientAnalysisCfg
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    group_summary: list[dict[str, Any]] = []
    pair_summary: list[dict[str, Any]] = []
    rng = np.random.default_rng(cfg.seed + 1)
    subsets = (*OVERLAP_TAGS, "overall")

    for subset in subsets:
        selected = [
            payload
            for payload in payloads
            if subset == "overall" or payload["scene"]["overlap_tag"] == subset
        ]
        for group in GROUPS:
            rows = [
                row
                for payload in selected
                for row in payload["groups"]
                if row["group"] == group
            ]
            rms_values = [float(row["feature_gradient_rms"]) for row in rows]
            mean, ci_low, ci_high = _bootstrap_mean_ci(
                rms_values, cfg.bootstrap_samples, rng
            )
            group_summary.append(
                {
                    "overlap_tag": subset,
                    "group": group,
                    "count": len(rms_values),
                    "feature_gradient_rms_mean": mean,
                    "feature_gradient_rms_median": float(np.median(rms_values))
                    if rms_values
                    else float("nan"),
                    "feature_gradient_rms_ci_low": ci_low,
                    "feature_gradient_rms_ci_high": ci_high,
                }
            )

        for first_index, group_a in enumerate(GROUPS):
            for group_b in GROUPS[first_index + 1 :]:
                rows = [
                    row
                    for payload in selected
                    for row in payload["pairs"]
                    if row["group_a"] == group_a
                    and row["group_b"] == group_b
                    and row["valid"]
                    and row["cosine"] is not None
                ]
                cosine_values = [float(row["cosine"]) for row in rows]
                mean, ci_low, ci_high = _bootstrap_mean_ci(
                    cosine_values, cfg.bootstrap_samples, rng
                )
                pair_summary.append(
                    {
                        "overlap_tag": subset,
                        "group_a": group_a,
                        "group_b": group_b,
                        "count": len(cosine_values),
                        "cosine_mean": mean,
                        "cosine_median": float(np.median(cosine_values))
                        if cosine_values
                        else float("nan"),
                        "cosine_ci_low": ci_low,
                        "cosine_ci_high": ci_high,
                        "negative_rate": float(
                            np.mean([value < 0 for value in cosine_values])
                        )
                        if cosine_values
                        else float("nan"),
                    }
                )
    return group_summary, pair_summary


def _plot_summaries(
    output_dir: Path,
    group_summary: list[dict[str, Any]],
    pair_summary: list[dict[str, Any]],
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    overall_groups = {
        row["group"]: row
        for row in group_summary
        if row["overlap_tag"] == "overall"
    }
    values = [overall_groups[group]["feature_gradient_rms_mean"] for group in GROUPS]
    lows = [
        overall_groups[group]["feature_gradient_rms_mean"]
        - overall_groups[group]["feature_gradient_rms_ci_low"]
        for group in GROUPS
    ]
    highs = [
        overall_groups[group]["feature_gradient_rms_ci_high"]
        - overall_groups[group]["feature_gradient_rms_mean"]
        for group in GROUPS
    ]
    plt.figure(figsize=(10, 5))
    plt.bar(GROUPS, values, yerr=[lows, highs], capsize=4)
    plt.yscale("log")
    plt.ylabel("Shared-feature gradient RMS")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(plot_dir / "gradient_norm_by_group.png", dpi=180)
    plt.close()

    for subset in (*OVERLAP_TAGS, "overall"):
        matrix = np.eye(len(GROUPS), dtype=np.float64)
        for row in pair_summary:
            if row["overlap_tag"] != subset:
                continue
            i = GROUPS.index(row["group_a"])
            j = GROUPS.index(row["group_b"])
            matrix[i, j] = matrix[j, i] = row["cosine_mean"]
        plt.figure(figsize=(7, 6))
        image = plt.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        plt.colorbar(image, label="Mean gradient cosine")
        plt.xticks(range(len(GROUPS)), GROUPS, rotation=35, ha="right")
        plt.yticks(range(len(GROUPS)), GROUPS)
        plt.title(f"Gradient cosine: {subset}")
        plt.tight_layout()
        plt.savefig(plot_dir / f"gradient_cosine_{subset}.png", dpi=180)
        plt.close()


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg_dict: DictConfig) -> None:
    analysis = _gradient_cfg(cfg_dict)
    output_dir = _absolute_path(analysis.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    failure_path = output_dir / "failures.jsonl"

    _, sampler_cfg = _find_evaluation_sampler(cfg_dict)
    source_index_path = _absolute_path(sampler_cfg.index_path)
    split_path = _absolute_path(analysis.scene_split_path)
    if not split_path.exists():
        raise FileNotFoundError(
            f"Analysis scene split not found: {split_path}. Run "
            "python -m src.scripts.prepare_gaussian_analysis_splits first."
        )
    selected_index = _read_split(split_path)
    sampler_cfg.index_path = str(split_path)

    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    OmegaConf.save(config=cfg_dict, f=str(output_dir / "resolved_config.yaml"))
    if cfg.checkpointing.load is None:
        raise ValueError("checkpointing.load must point to a baseline checkpoint")
    checkpoint_path = _absolute_path(
        update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not torch.cuda.is_available():
        raise RuntimeError("Gradient audit requires CUDA Gaussian rasterization")
    if cfg.data_loader.test.batch_size != 1:
        raise ValueError("Set data_loader.test.batch_size=1 for gradient analysis")
    device = torch.device("cuda")

    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder)
    encoder = _freeze(encoder.to(device))
    decoder = _freeze(decoder.to(device))
    alignment_losses = nn.ModuleList(get_losses(cfg.loss)).to(device)
    _freeze(alignment_losses)
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise AssertionError(
            "The gradient audit must keep every encoder weight frozen"
        )
    global_step, missing_keys, unexpected_keys = _load_encoder_checkpoint(
        encoder, checkpoint_path, analysis.allow_checkpoint_mismatch
    )
    capture = GaussianHeadFeatureCapture(encoder)

    manifest = {
        "status": "running",
        "analysis": asdict(analysis),
        "groups": list(GROUPS),
        "group_definition": {
            "size": "isotropic component of gradient in physical log-scale coordinates",
            "shape": "zero-mean component of gradient in physical log-scale coordinates",
            "rotation": "four raw quaternion channels",
            "opacity": "baseline opacity-logit channel",
            "base_color": "SH coefficient 0 for RGB",
            "view_color": "remaining SH coefficients for RGB",
        },
        "loss": "mean squared RGB error over every held-out target view",
        "target_information_policy": (
            "Target poses/images are used only for post-decoding alignment, rendering, "
            "and MSE. The encoder is called with batch['context'] only."
        ),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "global_step": global_step,
            "missing_encoder_keys": missing_keys,
            "unexpected_encoder_keys": unexpected_keys,
        },
        "source_index": {
            "path": str(source_index_path),
            "sha256": _sha256(source_index_path),
        },
        "scene_split": {
            "path": str(split_path),
            "sha256": _sha256(split_path),
            "scenes": len(selected_index),
        },
        "git": _git_info(),
    }
    _save_json(output_dir / "manifest.json", manifest)

    data_module = DataModule(
        cfg.dataset, cfg.data_loader, StepTracker(), global_rank=0
    )
    data_module.setup("test")
    test_loader = data_module.test_dataloader()
    if isinstance(test_loader, list):
        if len(test_loader) != 1:
            raise ValueError("Gradient audit supports exactly one test dataset")
        test_loader = test_loader[0]
    data_shim = encoder.get_data_shim()

    progress = tqdm(total=len(selected_index), desc="Attribute-gradient audit")
    completed = 0
    try:
        for batch in test_loader:
            scene = _scene_name(batch)
            if scene not in selected_index:
                continue
            chunk_path = chunk_dir / f"{scene}.json.gz"
            if analysis.resume and chunk_path.exists():
                completed += 1
                progress.update(1)
                continue
            try:
                batch = _to_device(batch, device)
                batch = data_shim(batch)
                payload = _analyze_scene(
                    batch=batch,
                    encoder=encoder,
                    decoder=decoder,
                    alignment_losses=alignment_losses,
                    capture=capture,
                    cfg=analysis,
                    test_cfg=cfg.test,
                    global_step=global_step,
                )
                _save_chunk(chunk_path, payload)
                completed += 1
                progress.update(1)
                progress.set_postfix(
                    scene=scene[:8],
                    psnr=f"{payload['scene']['psnr_mean']:.2f}",
                    identity=f"{payload['scene']['feature_identity_relative_error']:.1e}",
                )
            except Exception as error:
                _append_failure(
                    failure_path,
                    {
                        "scene": scene,
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                )
                if analysis.fail_fast:
                    raise
                progress.update(1)
    finally:
        capture.close()
        progress.close()

    missing_scenes = [
        scene
        for scene in selected_index
        if not (chunk_dir / f"{scene}.json.gz").exists()
    ]
    if missing_scenes:
        raise RuntimeError(
            f"Gradient audit did not produce chunks for {len(missing_scenes)} scenes. "
            f"First missing scenes: {missing_scenes[:10]}"
        )

    payloads = [
        _load_chunk(chunk_dir / f"{scene}.json.gz") for scene in selected_index
    ]
    scene_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for payload in payloads:
        scene_row = payload["scene"]
        scene_rows.append(scene_row)
        common = {
            "scene": scene_row["scene"],
            "overlap": scene_row["overlap"],
            "overlap_tag": scene_row["overlap_tag"],
        }
        group_rows.extend({**common, **row} for row in payload["groups"])
        pair_rows.extend({**common, **row} for row in payload["pairs"])

    group_summary, pair_summary = _summaries(payloads, analysis)
    _write_csv(output_dir / "per_scene.csv", scene_rows)
    _write_csv(output_dir / "per_scene_group.csv", group_rows)
    _write_csv(output_dir / "per_scene_pair.csv", pair_rows)
    _write_csv(output_dir / "gradient_norm_summary.csv", group_summary)
    _write_csv(output_dir / "gradient_cosine_summary.csv", pair_summary)
    _plot_summaries(output_dir, group_summary, pair_summary)

    overlap_counts = {
        tag: sum(row["overlap_tag"] == tag for row in scene_rows)
        for tag in OVERLAP_TAGS
    }
    max_raw_identity = max(
        float(row["raw_identity_relative_error"]) for row in scene_rows
    )
    max_feature_identity = max(
        float(row["feature_identity_relative_error"]) for row in scene_rows
    )
    summary = {
        "processed_scenes": len(scene_rows),
        "overlap_counts": overlap_counts,
        "global_step": global_step,
        "all_identity_checks_pass": all(row["identity_pass"] for row in scene_rows),
        "max_raw_identity_relative_error": max_raw_identity,
        "max_feature_identity_relative_error": max_feature_identity,
        "encoder_trainable_parameters": sum(
            parameter.numel()
            for parameter in encoder.parameters()
            if parameter.requires_grad
        ),
        "encoder_parameters_with_stored_grad": sum(
            parameter.grad is not None for parameter in encoder.parameters()
        ),
        "group_summary_file": "gradient_norm_summary.csv",
        "pair_summary_file": "gradient_cosine_summary.csv",
    }
    _save_json(output_dir / "summary.json", summary)
    manifest["status"] = "complete"
    manifest["summary"] = summary
    _save_json(output_dir / "manifest.json", manifest)

    print(f"Gradient audit complete: {output_dir}")
    print(f"Processed scenes: {overlap_counts}")
    print(f"Checkpoint step: {global_step}")
    print(f"Max raw identity relative error: {max_raw_identity:.3e}")
    print(f"Max feature identity relative error: {max_feature_identity:.3e}")


if __name__ == "__main__":
    main()
