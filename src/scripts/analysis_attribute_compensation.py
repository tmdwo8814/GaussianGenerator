"""Measure renderer-level compensation between Gaussian attribute modes.

The frozen encoder sees context views only.  Held-out target cameras are used
after decoding to render counterfactual Gaussian scenes; target RGB is never an
input to Gaussian construction or to the primary compensation objective.

The analysis has five resumable phases:

1. ``calibrate``: choose a bounded perturbation amplitude for each mode.
2. ``response``: render the six isolated perturbation responses for one fold.
3. ``merge_responses``: estimate cross-fitted global compensation coefficients.
4. ``recovery``: render every ordered perturbation/recovery pair for one fold.
5. ``finalize``: aggregate CSV/JSON summaries and plots.

The primary recovery score for perturbation A and recovery mode B is

    1 - MSE(render(A then B), original_render) / MSE(render(A), original_render).

One scalar beta per ordered pair is estimated from other folds and then fixed
for every Gaussian and target view in the held-out fold.  This deliberately
prevents per-Gaussian or per-view oracle fitting.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
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
    from src.model.types import Gaussians
    from src.scripts.analysis_attribute_gradients import (
        _append_failure,
        _git_info,
        _load_chunk,
        _read_split,
        _save_chunk,
        _save_json,
        _sha256,
        _write_csv,
    )
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


MODES = (
    "size",
    "shape",
    "rotation",
    "opacity",
    "base_color",
    "view_color",
)
PHASES = (
    "calibrate",
    "response",
    "merge_responses",
    "recovery",
    "finalize",
)


@dataclass(frozen=True)
class CompensationAnalysisCfg:
    output_dir: str = "outputs/gaussian_decoder_analysis/compensation/step_80000"
    scene_split_path: str = (
        "outputs/gaussian_decoder_analysis/splits/gradient_900.json"
    )
    phase: str = "calibrate"
    fold_id: int = 0
    num_folds: int = 3
    calibration_scenes_per_overlap: int = 30
    target_perturbation_mse: float = 1e-4
    calibration_fractions: tuple[float, ...] = (
        0.125,
        0.25,
        0.5,
        0.75,
        1.0,
    )
    size_cap: float = math.log(1.25)
    shape_cap: float = 0.20
    rotation_cap_radians: float = math.radians(10.0)
    opacity_cap: float = 0.50
    base_color_cap: float = math.log(1.25)
    view_color_cap: float = math.log(2.0)
    beta_abs_max: float = 2.0
    minimum_damage: float = 1e-12
    bootstrap_samples: int = 2000
    seed: int = 20260823
    diagonal_recovery_warning: float = 0.99
    allow_checkpoint_mismatch: bool = False
    resume: bool = True
    fail_fast: bool = True


def _compensation_cfg(cfg_dict: DictConfig) -> CompensationAnalysisCfg:
    node = cfg_dict.get("compensation_analysis")
    if node is None:
        raw: dict[str, Any] = {}
    elif OmegaConf.is_config(node):
        container = OmegaConf.to_container(node, resolve=True)
        raw = {} if container is None else dict(container)
    else:
        raw = dict(node)
    if "calibration_fractions" in raw:
        raw["calibration_fractions"] = tuple(
            float(value) for value in raw["calibration_fractions"]
        )
    cfg = CompensationAnalysisCfg(**raw)
    if cfg.phase not in PHASES:
        raise ValueError(f"Unknown compensation phase: {cfg.phase}")
    if cfg.num_folds < 2:
        raise ValueError("compensation_analysis.num_folds must be at least 2")
    if not 0 <= cfg.fold_id < cfg.num_folds:
        raise ValueError("compensation_analysis.fold_id is outside num_folds")
    if cfg.calibration_scenes_per_overlap <= 0:
        raise ValueError("calibration_scenes_per_overlap must be positive")
    if cfg.target_perturbation_mse <= 0:
        raise ValueError("target_perturbation_mse must be positive")
    if not cfg.calibration_fractions or any(
        value <= 0 or value > 1 for value in cfg.calibration_fractions
    ):
        raise ValueError("calibration_fractions must lie in (0, 1]")
    if cfg.beta_abs_max <= 0 or cfg.minimum_damage <= 0:
        raise ValueError("beta_abs_max and minimum_damage must be positive")
    if cfg.bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    for mode, cap in _mode_caps(cfg).items():
        if cap <= 0:
            raise ValueError(f"Perturbation cap for {mode} must be positive")
    return cfg


def _mode_caps(cfg: CompensationAnalysisCfg) -> dict[str, float]:
    return {
        "size": cfg.size_cap,
        "shape": cfg.shape_cap,
        "rotation": cfg.rotation_cap_radians,
        "opacity": cfg.opacity_cap,
        "base_color": cfg.base_color_cap,
        "view_color": cfg.view_color_cap,
    }


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _check_resumable_chunk(path: Path, signature: str) -> bool:
    if not path.exists():
        return False
    payload = _load_chunk(path)
    if payload.get("analysis_signature") != signature:
        raise RuntimeError(
            f"Existing chunk was produced by a different analysis configuration: "
            f"{path}. Use a new output_dir or remove only this analysis run."
        )
    return True


def _fold_map(
    selected_index: dict[str, Any], num_folds: int
) -> dict[str, int]:
    grouped: dict[str, list[str]] = {tag: [] for tag in OVERLAP_TAGS}
    for scene, entry in selected_index.items():
        grouped[get_overlap_tag(float(entry["overlap"]))].append(scene)
    result: dict[str, int] = {}
    for tag in OVERLAP_TAGS:
        for index, scene in enumerate(sorted(grouped[tag])):
            result[scene] = index % num_folds
    return result


def _calibration_scenes(
    selected_index: dict[str, Any], scenes_per_overlap: int
) -> set[str]:
    grouped: dict[str, list[str]] = {tag: [] for tag in OVERLAP_TAGS}
    for scene, entry in selected_index.items():
        grouped[get_overlap_tag(float(entry["overlap"]))].append(scene)
    selected: set[str] = set()
    for tag, scenes in grouped.items():
        scenes = sorted(scenes)
        if len(scenes) < scenes_per_overlap:
            raise ValueError(
                f"Only {len(scenes)} {tag} scenes are available for calibration"
            )
        selected.update(scenes[:scenes_per_overlap])
    return selected


def _low_frequency_field(
    scene: str,
    context_views: int,
    height: int,
    width: int,
    gaussians: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    pixels = context_views * height * width
    if gaussians % pixels != 0:
        raise ValueError(
            "Compensation field requires a fixed number of Gaussians per context "
            f"pixel, got {gaussians} Gaussians for {pixels} pixels"
        )
    repetitions = gaussians // pixels
    checksum = zlib.crc32(scene.encode("utf-8")) ^ seed
    phase_a = 2.0 * math.pi * ((checksum & 0xFFFF) / 65536.0)
    phase_b = 2.0 * math.pi * (((checksum >> 16) & 0xFFFF) / 65536.0)
    y = torch.linspace(-1.0, 1.0, height, dtype=dtype, device=device)
    x = torch.linspace(-1.0, 1.0, width, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    views = []
    for view in range(context_views):
        field = torch.sin(math.pi * xx + phase_a + 0.71 * view)
        field += 0.65 * torch.cos(math.pi * yy + phase_b - 0.43 * view)
        field += 0.35 * torch.sin(math.pi * (xx + yy) + 0.5 * phase_a)
        field -= field.mean()
        field /= field.abs().max().clamp_min(1e-8)
        views.append(field.reshape(-1))
    result = torch.cat(views)
    if repetitions > 1:
        result = result.repeat_interleave(repetitions)
    return result[None]


def _axis_angle_matrix(angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation around a fixed target-independent world axis."""
    axis = torch.tensor(
        [1.0, 2.0, 3.0], dtype=angle.dtype, device=angle.device
    )
    axis = axis / axis.norm()
    x, y, z = axis.unbind()
    zero = torch.zeros((), dtype=angle.dtype, device=angle.device)
    skew = torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero]
    ).reshape(3, 3)
    identity = torch.eye(3, dtype=angle.dtype, device=angle.device)
    outer = axis[:, None] * axis[None, :]
    cosine = angle.cos()[..., None, None]
    sine = angle.sin()[..., None, None]
    return cosine * identity + (1.0 - cosine) * outer + sine * skew


@torch.no_grad()
def _apply_mode(
    gaussians: Gaussians,
    mode: str,
    field: torch.Tensor,
    amplitude: float,
) -> Gaussians:
    if mode not in MODES:
        raise ValueError(f"Unknown compensation mode: {mode}")
    if abs(float(amplitude)) < 1e-15:
        return gaussians
    delta = field.to(
        dtype=gaussians.covariances.dtype,
        device=gaussians.covariances.device,
    ) * float(amplitude)
    means = gaussians.means
    covariances = gaussians.covariances
    harmonics = gaussians.harmonics
    opacities = gaussians.opacities

    if mode == "size":
        covariances = covariances * torch.exp(2.0 * delta)[..., None, None]
    elif mode == "shape":
        eigenvalues, eigenvectors = torch.linalg.eigh(covariances)
        pattern = torch.tensor(
            [-1.0, 0.0, 1.0],
            dtype=eigenvalues.dtype,
            device=eigenvalues.device,
        )
        log_scale_change = delta[..., None] * pattern
        changed = eigenvalues.clamp_min(1e-20) * torch.exp(2.0 * log_scale_change)
        covariances = (
            eigenvectors
            @ torch.diag_embed(changed)
            @ eigenvectors.transpose(-1, -2)
        )
    elif mode == "rotation":
        rotation = _axis_angle_matrix(delta)
        covariances = rotation @ covariances @ rotation.transpose(-1, -2)
    elif mode == "opacity":
        epsilon = torch.finfo(opacities.dtype).eps
        logits = torch.logit(opacities.clamp(epsilon, 1.0 - epsilon))
        opacities = torch.sigmoid(logits + delta)
    elif mode == "base_color":
        harmonics = harmonics.clone()
        harmonics[..., 0] *= torch.exp(delta)[..., None]
    elif mode == "view_color":
        if harmonics.shape[-1] <= 1:
            raise ValueError("view_color requires SH degree greater than zero")
        harmonics = harmonics.clone()
        harmonics[..., 1:] *= torch.exp(delta)[..., None, None]

    return Gaussians(
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=opacities,
    )


@torch.no_grad()
def _render(
    decoder: nn.Module,
    gaussians: Gaussians,
    batch: dict[str, Any],
    aligned_extrinsics: torch.Tensor,
) -> torch.Tensor:
    height, width = batch["target"]["image"].shape[-2:]
    output = decoder.forward(
        gaussians,
        aligned_extrinsics,
        batch["target"]["intrinsics"],
        batch["target"]["near"],
        batch["target"]["far"],
        (height, width),
    )
    return output.color[0].float()


def _view_and_all_statistics(
    left: torch.Tensor, right: torch.Tensor | None = None
) -> dict[int, float]:
    if right is None:
        values = left.double().square()
    else:
        values = left.double() * right.double()
    result = {
        index: float(values[index].mean().item())
        for index in range(values.shape[0])
    }
    result[-1] = float(values.mean().item())
    return result


def _psnr_by_view(image: torch.Tensor, target: torch.Tensor) -> list[float]:
    mse = (image.double() - target.double()).square().mean(dim=(1, 2, 3))
    return [float(value) for value in (-10.0 * torch.log10(mse.clamp_min(1e-12)))]


def _prepare_model_run(
    cfg_dict: DictConfig,
    analysis: CompensationAnalysisCfg,
    phase_dir: Path,
) -> tuple[
    Any,
    nn.Module,
    nn.Module,
    nn.ModuleList,
    Iterable,
    Any,
    dict[str, Any],
    Path,
    int,
    dict[str, Any],
]:
    _, sampler_cfg = _find_evaluation_sampler(cfg_dict)
    source_index_path = _absolute_path(sampler_cfg.index_path)
    split_path = _absolute_path(analysis.scene_split_path)
    if not split_path.exists():
        raise FileNotFoundError(
            f"Analysis split not found: {split_path}. Run "
            "python -m src.scripts.prepare_gaussian_analysis_splits first."
        )
    selected_index = _read_split(split_path)
    sampler_cfg.index_path = str(split_path)
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    phase_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg_dict, f=str(phase_dir / "resolved_config.yaml"))
    if cfg.checkpointing.load is None:
        raise ValueError("checkpointing.load must point to a baseline checkpoint")
    checkpoint_path = _absolute_path(
        update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not torch.cuda.is_available():
        raise RuntimeError("Compensation analysis requires CUDA rasterization")
    if cfg.data_loader.test.batch_size != 1:
        raise ValueError("Set data_loader.test.batch_size=1")
    device = torch.device("cuda")
    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder)
    encoder = _freeze(encoder.to(device))
    decoder = _freeze(decoder.to(device))
    losses = nn.ModuleList(get_losses(cfg.loss)).to(device)
    _freeze(losses)
    global_step, missing, unexpected = _load_encoder_checkpoint(
        encoder, checkpoint_path, analysis.allow_checkpoint_mismatch
    )
    data_module = DataModule(
        cfg.dataset, cfg.data_loader, StepTracker(), global_rank=0
    )
    data_module.setup("test")
    loader = data_module.test_dataloader()
    if isinstance(loader, list):
        if len(loader) != 1:
            raise ValueError("Compensation analysis supports one test dataset")
        loader = loader[0]
    manifest_base = {
        "analysis": asdict(analysis),
        "modes": list(MODES),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "global_step": global_step,
            "missing_encoder_keys": missing,
            "unexpected_encoder_keys": unexpected,
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
        "target_information_policy": (
            "The encoder receives batch['context'] only. Target cameras are used "
            "after decoding for alignment and counterfactual rendering. Original "
            "baseline renders, not target RGB, define compensation recovery."
        ),
    }
    return (
        cfg,
        encoder,
        decoder,
        losses,
        loader,
        encoder.get_data_shim(),
        selected_index,
        checkpoint_path,
        global_step,
        manifest_base,
    )


@torch.no_grad()
def _decode_scene(
    batch: dict[str, Any],
    encoder: nn.Module,
    decoder: nn.Module,
    losses: nn.ModuleList,
    cfg: Any,
    global_step: int,
    field_seed: int,
) -> tuple[Gaussians, torch.Tensor, torch.Tensor, torch.Tensor]:
    gaussians = encoder(batch["context"], global_step)
    aligned = _align_target_extrinsics(
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
    anchor = _render(decoder, gaussians, batch, aligned)
    b, context_views, _, height, width = batch["context"]["image"].shape
    if b != 1:
        raise ValueError("Compensation analysis requires batch_size=1")
    field = _low_frequency_field(
        _scene_name(batch),
        context_views,
        height,
        width,
        gaussians.means.shape[1],
        field_seed,
        gaussians.covariances.dtype,
        gaussians.covariances.device,
    )
    return gaussians, aligned, anchor, field


def _scene_common(batch: dict[str, Any], fold: int) -> dict[str, Any]:
    overlap = float(batch["context"]["overlap"].reshape(-1)[0].item())
    return {
        "scene": _scene_name(batch),
        "fold": fold,
        "overlap": overlap,
        "overlap_tag": get_overlap_tag(overlap),
        "context_indices": _index_list(batch["context"]["index"]),
        "target_indices": _index_list(batch["target"]["index"]),
    }


def _run_calibration(
    cfg_dict: DictConfig, analysis: CompensationAnalysisCfg, root: Path
) -> None:
    phase_dir = root / "calibration_run"
    (
        cfg,
        encoder,
        decoder,
        losses,
        loader,
        data_shim,
        selected_index,
        _,
        global_step,
        manifest_base,
    ) = _prepare_model_run(cfg_dict, analysis, phase_dir)
    selected = _calibration_scenes(
        selected_index, analysis.calibration_scenes_per_overlap
    )
    chunks = phase_dir / "scene_chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    failure_path = phase_dir / "failures.jsonl"
    caps = _mode_caps(analysis)
    signature = _json_sha256(
        {
            "phase": "calibrate",
            "checkpoint": manifest_base["checkpoint"]["sha256"],
            "scene_split": manifest_base["scene_split"]["sha256"],
            "seed": analysis.seed,
            "caps": caps,
            "fractions": analysis.calibration_fractions,
        }
    )
    progress = tqdm(total=len(selected), desc="Compensation calibration")
    for batch in loader:
        scene = _scene_name(batch)
        if scene not in selected:
            continue
        path = chunks / f"{scene}.json.gz"
        if analysis.resume and _check_resumable_chunk(path, signature):
            progress.update(1)
            continue
        try:
            batch = data_shim(_to_device(batch, torch.device("cuda")))
            gaussians, aligned, anchor, field = _decode_scene(
                batch, encoder, decoder, losses, cfg, global_step, analysis.seed
            )
            rows = []
            for mode in MODES:
                for fraction in analysis.calibration_fractions:
                    amplitude = caps[mode] * fraction
                    image = _render(
                        decoder,
                        _apply_mode(gaussians, mode, field, amplitude),
                        batch,
                        aligned,
                    )
                    damage = float((image.double() - anchor.double()).square().mean())
                    rows.append(
                        {
                            "mode": mode,
                            "fraction": fraction,
                            "amplitude": amplitude,
                            "damage": damage,
                        }
                    )
            _save_chunk(
                path,
                {
                    **_scene_common(batch, -1),
                    "analysis_signature": signature,
                    "rows": rows,
                },
            )
            progress.update(1)
            progress.set_postfix(scene=scene[:8])
        except Exception as error:
            _append_failure(
                failure_path,
                {"scene": scene, "type": type(error).__name__, "message": str(error)},
            )
            if analysis.fail_fast:
                progress.close()
                raise
            progress.update(1)
    progress.close()
    missing = [scene for scene in selected if not (chunks / f"{scene}.json.gz").exists()]
    if missing:
        raise RuntimeError(f"Calibration is missing {len(missing)} scenes: {missing[:5]}")
    payloads = [_load_chunk(chunks / f"{scene}.json.gz") for scene in sorted(selected)]
    curve_rows: list[dict[str, Any]] = []
    amplitudes: dict[str, float] = {}
    choices: dict[str, Any] = {}
    for mode in MODES:
        candidates = []
        for fraction in analysis.calibration_fractions:
            values = [
                float(row["damage"])
                for payload in payloads
                for row in payload["rows"]
                if row["mode"] == mode and math.isclose(float(row["fraction"]), fraction)
            ]
            median = float(np.median(values))
            mean = float(np.mean(values))
            amplitude = caps[mode] * fraction
            row = {
                "mode": mode,
                "fraction": fraction,
                "amplitude": amplitude,
                "damage_mean": mean,
                "damage_median": median,
                "count": len(values),
            }
            curve_rows.append(row)
            candidates.append(row)
        usable = [row for row in candidates if row["damage_median"] > 0]
        if not usable:
            raise RuntimeError(f"Every calibration response is zero for {mode}")
        chosen = min(
            usable,
            key=lambda row: abs(
                math.log(row["damage_median"])
                - math.log(analysis.target_perturbation_mse)
            ),
        )
        amplitudes[mode] = float(chosen["amplitude"])
        choices[mode] = chosen
    calibration = {
        "status": "complete",
        "target_perturbation_mse": analysis.target_perturbation_mse,
        "calibration_scenes": len(selected),
        "amplitudes": amplitudes,
        "choices": choices,
        "caps": caps,
        "mode_definition": _mode_definition(),
        "checkpoint": manifest_base["checkpoint"],
        "scene_split": manifest_base["scene_split"],
    }
    calibration["sha256"] = _json_sha256(calibration)
    _write_csv(root / "calibration_curve.csv", curve_rows)
    _save_json(root / "calibration.json", calibration)
    manifest = {**manifest_base, "status": "complete", "calibration": calibration}
    _save_json(phase_dir / "manifest.json", manifest)
    print(f"Calibration complete: {root / 'calibration.json'}")
    print(f"Selected amplitudes: {amplitudes}")


def _mode_definition() -> dict[str, str]:
    return {
        "size": "additive isotropic log-scale field",
        "shape": "zero-sum [-1,0,1] log-eigenscale field",
        "rotation": "axis-angle covariance rotation around fixed [1,2,3] axis",
        "opacity": "additive final-opacity logit field",
        "base_color": "multiplicative SH-DC log-gain field",
        "view_color": "multiplicative non-DC SH log-gain field",
    }


def _load_calibration(root: Path) -> dict[str, Any]:
    path = root / "calibration.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing calibration: {path}")
    with path.open("r", encoding="utf-8") as file:
        calibration = json.load(file)
    if calibration.get("status") != "complete":
        raise RuntimeError("Calibration is not complete")
    if set(calibration.get("amplitudes", {})) != set(MODES):
        raise ValueError("Calibration does not contain all compensation modes")
    stored_hash = calibration.get("sha256")
    unhashed = {key: value for key, value in calibration.items() if key != "sha256"}
    if stored_hash != _json_sha256(unhashed):
        raise RuntimeError("Calibration JSON checksum does not match its contents")
    return calibration


def _run_response(
    cfg_dict: DictConfig, analysis: CompensationAnalysisCfg, root: Path
) -> None:
    calibration = _load_calibration(root)
    fold = analysis.fold_id
    phase_dir = root / "responses" / f"fold_{fold}"
    (
        cfg,
        encoder,
        decoder,
        losses,
        loader,
        data_shim,
        selected_index,
        _,
        global_step,
        manifest_base,
    ) = _prepare_model_run(cfg_dict, analysis, phase_dir)
    if calibration["checkpoint"]["sha256"] != manifest_base["checkpoint"]["sha256"]:
        raise RuntimeError("Calibration and response checkpoint hashes differ")
    fold_by_scene = _fold_map(selected_index, analysis.num_folds)
    scenes = {scene for scene, value in fold_by_scene.items() if value == fold}
    chunks = phase_dir / "scene_chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    failure_path = phase_dir / "failures.jsonl"
    amplitudes = {key: float(value) for key, value in calibration["amplitudes"].items()}
    signature = _json_sha256(
        {
            "phase": "response",
            "fold": fold,
            "num_folds": analysis.num_folds,
            "checkpoint": manifest_base["checkpoint"]["sha256"],
            "scene_split": manifest_base["scene_split"]["sha256"],
            "calibration": calibration["sha256"],
            "seed": analysis.seed,
        }
    )
    progress = tqdm(total=len(scenes), desc=f"Compensation responses fold {fold}")
    for batch in loader:
        scene = _scene_name(batch)
        if scene not in scenes:
            continue
        path = chunks / f"{scene}.json.gz"
        if analysis.resume and _check_resumable_chunk(path, signature):
            progress.update(1)
            continue
        try:
            batch = data_shim(_to_device(batch, torch.device("cuda")))
            gaussians, aligned, anchor, field = _decode_scene(
                batch, encoder, decoder, losses, cfg, global_step, analysis.seed
            )
            field = _low_frequency_field(
                scene,
                batch["context"]["image"].shape[1],
                batch["context"]["image"].shape[-2],
                batch["context"]["image"].shape[-1],
                gaussians.means.shape[1],
                analysis.seed,
                gaussians.covariances.dtype,
                gaussians.covariances.device,
            )
            responses: dict[str, torch.Tensor] = {}
            for mode in MODES:
                image = _render(
                    decoder,
                    _apply_mode(gaussians, mode, field, amplitudes[mode]),
                    batch,
                    aligned,
                )
                responses[mode] = image - anchor
            energy_rows = []
            for mode in MODES:
                for target_position, energy in _view_and_all_statistics(
                    responses[mode]
                ).items():
                    energy_rows.append(
                        {
                            "mode": mode,
                            "target_position": target_position,
                            "energy": energy,
                        }
                    )
            cross_rows = []
            for index, mode_a in enumerate(MODES):
                for mode_b in MODES[index + 1 :]:
                    values = _view_and_all_statistics(
                        responses[mode_a], responses[mode_b]
                    )
                    for target_position, cross in values.items():
                        cross_rows.append(
                            {
                                "mode_a": mode_a,
                                "mode_b": mode_b,
                                "target_position": target_position,
                                "cross": cross,
                            }
                        )
            gt = batch["target"]["image"][0].float()
            payload = {
                **_scene_common(batch, fold),
                "analysis_signature": signature,
                "anchor_psnr_by_view": _psnr_by_view(anchor, gt),
                "energy": energy_rows,
                "cross": cross_rows,
            }
            _save_chunk(path, payload)
            progress.update(1)
            progress.set_postfix(scene=scene[:8])
        except Exception as error:
            _append_failure(
                failure_path,
                {"scene": scene, "type": type(error).__name__, "message": str(error)},
            )
            if analysis.fail_fast:
                progress.close()
                raise
            progress.update(1)
    progress.close()
    missing = [scene for scene in scenes if not (chunks / f"{scene}.json.gz").exists()]
    if missing:
        raise RuntimeError(f"Response fold {fold} is missing {len(missing)} scenes")
    manifest = {
        **manifest_base,
        "status": "complete",
        "phase": "response",
        "fold": fold,
        "fold_scenes": len(scenes),
        "calibration_sha256": calibration["sha256"],
    }
    _save_json(phase_dir / "manifest.json", manifest)
    print(f"Response fold {fold} complete: {phase_dir}")


def _load_response_payloads(
    root: Path, num_folds: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payloads = []
    manifests = []
    for fold in range(num_folds):
        fold_dir = root / "responses" / f"fold_{fold}"
        manifest_path = fold_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing response manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        if manifest.get("status") != "complete":
            raise RuntimeError(f"Response fold {fold} is incomplete")
        manifests.append(manifest)
        chunks = sorted((fold_dir / "scene_chunks").glob("*.json.gz"))
        payloads.extend(_load_chunk(path) for path in chunks)
    if len({payload["scene"] for payload in payloads}) != len(payloads):
        raise RuntimeError("Response payloads contain duplicate scenes")
    expected = sum(int(manifest["fold_scenes"]) for manifest in manifests)
    if len(payloads) != expected:
        raise RuntimeError(
            f"Expected {expected} response chunks from manifests, found {len(payloads)}"
        )
    return payloads, manifests


def _response_lookups(
    payload: dict[str, Any], target_position: int
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    energy = {
        row["mode"]: float(row["energy"])
        for row in payload["energy"]
        if int(row["target_position"]) == target_position
    }
    cross: dict[tuple[str, str], float] = {}
    for row in payload["cross"]:
        if int(row["target_position"]) != target_position:
            continue
        key = tuple(sorted((row["mode_a"], row["mode_b"])))
        cross[key] = float(row["cross"])
    return energy, cross


def _cross_value(
    mode_a: str,
    mode_b: str,
    energy: dict[str, float],
    cross: dict[tuple[str, str], float],
) -> float:
    return energy[mode_a] if mode_a == mode_b else cross[tuple(sorted((mode_a, mode_b)))]


def _clip_beta(value: float, limit: float) -> float:
    return float(np.clip(value, -limit, limit))


def _run_merge_responses(analysis: CompensationAnalysisCfg, root: Path) -> None:
    calibration = _load_calibration(root)
    payloads, manifests = _load_response_payloads(root, analysis.num_folds)
    checkpoint_hashes = {manifest["checkpoint"]["sha256"] for manifest in manifests}
    split_hashes = {manifest["scene_split"]["sha256"] for manifest in manifests}
    calibration_hashes = {manifest["calibration_sha256"] for manifest in manifests}
    if len(checkpoint_hashes) != 1 or len(split_hashes) != 1:
        raise RuntimeError("Response folds used different checkpoints or scene splits")
    if calibration_hashes != {calibration["sha256"]}:
        raise RuntimeError("Response folds used different calibrations")
    beta_rows: list[dict[str, Any]] = []
    beta_lookup: dict[tuple[int, str, str], float] = {}
    for eval_fold in range(analysis.num_folds):
        train = [payload for payload in payloads if int(payload["fold"]) != eval_fold]
        for mode_a in MODES:
            for mode_b in MODES:
                numerator = 0.0
                denominator = 0.0
                for payload in train:
                    energy, cross = _response_lookups(payload, -1)
                    numerator += _cross_value(mode_a, mode_b, energy, cross)
                    denominator += energy[mode_b]
                valid = denominator > analysis.minimum_damage
                beta_unclipped = -numerator / denominator if valid else 0.0
                beta = _clip_beta(beta_unclipped, analysis.beta_abs_max)
                beta_lookup[(eval_fold, mode_a, mode_b)] = beta
                beta_rows.append(
                    {
                        "eval_fold": eval_fold,
                        "perturbed_mode": mode_a,
                        "recovery_mode": mode_b,
                        "beta": beta,
                        "beta_unclipped": beta_unclipped,
                        "clipped": not math.isclose(beta, beta_unclipped),
                        "train_scenes": len(train),
                        "cross_sum": numerator,
                        "recovery_energy_sum": denominator,
                        "valid": valid,
                    }
                )
    linear_rows: list[dict[str, Any]] = []
    for payload in payloads:
        fold = int(payload["fold"])
        target_positions = sorted(
            {int(row["target_position"]) for row in payload["energy"]}
        )
        for target_position in target_positions:
            energy, cross = _response_lookups(payload, target_position)
            for mode_a in MODES:
                for mode_b in MODES:
                    damage = energy[mode_a]
                    recovery_energy = energy[mode_b]
                    cross_value = _cross_value(mode_a, mode_b, energy, cross)
                    beta = beta_lookup[(fold, mode_a, mode_b)]
                    residual = damage + 2.0 * beta * cross_value + beta * beta * recovery_energy
                    valid = damage > analysis.minimum_damage and recovery_energy > analysis.minimum_damage
                    recovery = 1.0 - residual / damage if valid else None
                    oracle_beta_unclipped = (
                        -cross_value / recovery_energy if valid else 0.0
                    )
                    oracle_beta = _clip_beta(
                        oracle_beta_unclipped, analysis.beta_abs_max
                    )
                    oracle_residual = (
                        damage
                        + 2.0 * oracle_beta * cross_value
                        + oracle_beta * oracle_beta * recovery_energy
                    )
                    oracle_recovery = (
                        1.0 - oracle_residual / damage if valid else None
                    )
                    linear_rows.append(
                        {
                            "scene": payload["scene"],
                            "fold": fold,
                            "overlap": payload["overlap"],
                            "overlap_tag": payload["overlap_tag"],
                            "target_position": target_position,
                            "perturbed_mode": mode_a,
                            "recovery_mode": mode_b,
                            "damage": damage,
                            "cross": cross_value,
                            "recovery_energy": recovery_energy,
                            "global_beta": beta,
                            "global_linear_recovery": recovery,
                            "oracle_beta": oracle_beta,
                            "oracle_beta_unclipped": oracle_beta_unclipped,
                            "oracle_linear_recovery": oracle_recovery,
                            "oracle_gap": (
                                oracle_recovery - recovery
                                if recovery is not None and oracle_recovery is not None
                                else None
                            ),
                            "valid": valid,
                        }
                    )
    beta_dir = root / "betas"
    beta_dir.mkdir(parents=True, exist_ok=True)
    response_energy_rows = []
    response_cross_rows = []
    for payload in payloads:
        common = {
            key: payload[key]
            for key in ("scene", "fold", "overlap", "overlap_tag")
        }
        response_energy_rows.extend(
            {**common, **row} for row in payload["energy"]
        )
        response_cross_rows.extend(
            {**common, **row} for row in payload["cross"]
        )
    _write_csv(beta_dir / "response_energy_per_scene.csv", response_energy_rows)
    _write_csv(beta_dir / "response_cross_per_scene.csv", response_cross_rows)
    _write_csv(beta_dir / "global_beta.csv", beta_rows)
    _write_csv(beta_dir / "linear_compensation_per_scene.csv", linear_rows)
    summary = _summarize_rows(
        [row for row in linear_rows if int(row["target_position"]) == -1],
        value_keys=(
            "global_linear_recovery",
            "oracle_linear_recovery",
            "oracle_gap",
        ),
        bootstrap_samples=analysis.bootstrap_samples,
        seed=analysis.seed,
    )
    _write_csv(beta_dir / "linear_compensation_summary.csv", summary)
    merge_manifest = {
        "status": "complete",
        "phase": "merge_responses",
        "analysis": asdict(analysis),
        "response_scenes": len(payloads),
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "scene_split_sha256": next(iter(split_hashes)),
        "calibration_sha256": calibration["sha256"],
        "beta_file": "global_beta.csv",
    }
    _save_json(beta_dir / "manifest.json", merge_manifest)
    print(f"Response merge complete: {beta_dir / 'global_beta.csv'}")


def _beta_for_fold(root: Path, fold: int) -> dict[tuple[str, str], float]:
    path = root / "betas" / "global_beta.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing beta file: {path}")
    rows = _read_csv(path)
    result = {
        (row["perturbed_mode"], row["recovery_mode"]): float(row["beta"])
        for row in rows
        if int(row["eval_fold"]) == fold
    }
    expected = {(mode_a, mode_b) for mode_a in MODES for mode_b in MODES}
    if set(result) != expected:
        raise RuntimeError(f"Beta file is incomplete for fold {fold}")
    return result


def _run_recovery(
    cfg_dict: DictConfig, analysis: CompensationAnalysisCfg, root: Path
) -> None:
    calibration = _load_calibration(root)
    fold = analysis.fold_id
    beta_lookup = _beta_for_fold(root, fold)
    phase_dir = root / "recovery" / f"fold_{fold}"
    (
        cfg,
        encoder,
        decoder,
        losses,
        loader,
        data_shim,
        selected_index,
        _,
        global_step,
        manifest_base,
    ) = _prepare_model_run(cfg_dict, analysis, phase_dir)
    if calibration["checkpoint"]["sha256"] != manifest_base["checkpoint"]["sha256"]:
        raise RuntimeError("Calibration and recovery checkpoint hashes differ")
    fold_by_scene = _fold_map(selected_index, analysis.num_folds)
    scenes = {scene for scene, value in fold_by_scene.items() if value == fold}
    chunks = phase_dir / "scene_chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    failure_path = phase_dir / "failures.jsonl"
    amplitudes = {key: float(value) for key, value in calibration["amplitudes"].items()}
    beta_file_sha = _sha256(root / "betas" / "global_beta.csv")
    signature = _json_sha256(
        {
            "phase": "recovery",
            "fold": fold,
            "num_folds": analysis.num_folds,
            "checkpoint": manifest_base["checkpoint"]["sha256"],
            "scene_split": manifest_base["scene_split"]["sha256"],
            "calibration": calibration["sha256"],
            "beta_file": beta_file_sha,
            "seed": analysis.seed,
        }
    )
    progress = tqdm(total=len(scenes), desc=f"Exact compensation fold {fold}")
    for batch in loader:
        scene = _scene_name(batch)
        if scene not in scenes:
            continue
        path = chunks / f"{scene}.json.gz"
        if analysis.resume and _check_resumable_chunk(path, signature):
            progress.update(1)
            continue
        try:
            batch = data_shim(_to_device(batch, torch.device("cuda")))
            gaussians, aligned, anchor, _ = _decode_scene(
                batch, encoder, decoder, losses, cfg, global_step, analysis.seed
            )
            field = _low_frequency_field(
                scene,
                batch["context"]["image"].shape[1],
                batch["context"]["image"].shape[-2],
                batch["context"]["image"].shape[-1],
                gaussians.means.shape[1],
                analysis.seed,
                gaussians.covariances.dtype,
                gaussians.covariances.device,
            )
            gt = batch["target"]["image"][0].float()
            anchor_psnr = _psnr_by_view(anchor, gt)
            rows = []
            for mode_a in MODES:
                perturbed_gaussians = _apply_mode(
                    gaussians, mode_a, field, amplitudes[mode_a]
                )
                perturbed_image = _render(
                    decoder, perturbed_gaussians, batch, aligned
                )
                perturbed_delta = perturbed_image - anchor
                damage_by_view = _view_and_all_statistics(perturbed_delta)
                perturbed_psnr = _psnr_by_view(perturbed_image, gt)
                for mode_b in MODES:
                    beta = beta_lookup[(mode_a, mode_b)]
                    if mode_b == mode_a:
                        # All semantic modes are additive in their analysis
                        # coordinate.  Compose the diagonal in one operation so
                        # beta=-1 is an exact identity even when covariance
                        # eigenspaces contain nearly repeated eigenvalues.
                        recovered_gaussians = _apply_mode(
                            gaussians,
                            mode_a,
                            field,
                            amplitudes[mode_a] + beta * amplitudes[mode_b],
                        )
                    else:
                        recovered_gaussians = _apply_mode(
                            perturbed_gaussians,
                            mode_b,
                            field,
                            beta * amplitudes[mode_b],
                        )
                    recovered_image = _render(
                        decoder, recovered_gaussians, batch, aligned
                    )
                    recovered_delta = recovered_image - anchor
                    recovered_by_view = _view_and_all_statistics(recovered_delta)
                    recovered_psnr = _psnr_by_view(recovered_image, gt)
                    for target_position in sorted(damage_by_view):
                        damage = damage_by_view[target_position]
                        residual = recovered_by_view[target_position]
                        valid = damage > analysis.minimum_damage
                        recovery = 1.0 - residual / damage if valid else None
                        if target_position == -1:
                            original_gt = float(np.mean(anchor_psnr))
                            perturbed_gt = float(np.mean(perturbed_psnr))
                            recovered_gt = float(np.mean(recovered_psnr))
                        else:
                            original_gt = anchor_psnr[target_position]
                            perturbed_gt = perturbed_psnr[target_position]
                            recovered_gt = recovered_psnr[target_position]
                        rows.append(
                            {
                                "target_position": target_position,
                                "perturbed_mode": mode_a,
                                "recovery_mode": mode_b,
                                "beta": beta,
                                "perturbation_mse": damage,
                                "recovered_mse": residual,
                                "recovery": recovery,
                                "original_psnr_gt": original_gt,
                                "perturbed_psnr_gt": perturbed_gt,
                                "recovered_psnr_gt": recovered_gt,
                                "recovery_delta_psnr_gt": recovered_gt - perturbed_gt,
                                "valid": valid,
                            }
                        )
            _save_chunk(
                path,
                {
                    **_scene_common(batch, fold),
                    "analysis_signature": signature,
                    "rows": rows,
                },
            )
            progress.update(1)
            progress.set_postfix(scene=scene[:8])
        except Exception as error:
            _append_failure(
                failure_path,
                {"scene": scene, "type": type(error).__name__, "message": str(error)},
            )
            if analysis.fail_fast:
                progress.close()
                raise
            progress.update(1)
    progress.close()
    missing = [scene for scene in scenes if not (chunks / f"{scene}.json.gz").exists()]
    if missing:
        raise RuntimeError(f"Recovery fold {fold} is missing {len(missing)} scenes")
    manifest = {
        **manifest_base,
        "status": "complete",
        "phase": "recovery",
        "fold": fold,
        "fold_scenes": len(scenes),
        "calibration_sha256": calibration["sha256"],
        "beta_file_sha256": beta_file_sha,
    }
    _save_json(phase_dir / "manifest.json", manifest)
    print(f"Recovery fold {fold} complete: {phase_dir}")


def _bootstrap_ci(
    values: list[float], samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), float(array[0])
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _summarize_rows(
    rows: list[dict[str, Any]],
    value_keys: tuple[str, ...],
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    result = []
    rng = np.random.default_rng(seed)
    for overlap_tag in (*OVERLAP_TAGS, "all"):
        subset = rows if overlap_tag == "all" else [
            row for row in rows if row["overlap_tag"] == overlap_tag
        ]
        for mode_a in MODES:
            for mode_b in MODES:
                pair = [
                    row
                    for row in subset
                    if row["perturbed_mode"] == mode_a
                    and row["recovery_mode"] == mode_b
                    and bool(row.get("valid", True))
                ]
                summary: dict[str, Any] = {
                    "overlap_tag": overlap_tag,
                    "perturbed_mode": mode_a,
                    "recovery_mode": mode_b,
                    "count": len(pair),
                }
                for key in value_keys:
                    values = [
                        float(row[key])
                        for row in pair
                        if row.get(key) is not None and math.isfinite(float(row[key]))
                    ]
                    if values:
                        low, high = _bootstrap_ci(values, bootstrap_samples, rng)
                        summary.update(
                            {
                                f"{key}_mean": float(np.mean(values)),
                                f"{key}_median": float(np.median(values)),
                                f"{key}_ci_low": low,
                                f"{key}_ci_high": high,
                                f"{key}_negative_rate": float(
                                    np.mean(np.asarray(values) < 0)
                                ),
                            }
                        )
                result.append(summary)
    return result


def _plot_matrix(
    path: Path,
    summary: list[dict[str, Any]],
    value_key: str,
    title: str,
    overlap_tag: str = "all",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    lookup = {
        (row["perturbed_mode"], row["recovery_mode"]): row.get(value_key)
        for row in summary
        if row["overlap_tag"] == overlap_tag
    }
    matrix = np.asarray(
        [
            [float(lookup.get((mode_a, mode_b), np.nan)) for mode_b in MODES]
            for mode_a in MODES
        ]
    )
    plt.figure(figsize=(8, 7))
    image = plt.imshow(matrix, cmap="coolwarm", vmin=vmin, vmax=vmax)
    plt.colorbar(image, label=value_key)
    plt.xticks(range(len(MODES)), MODES, rotation=35, ha="right")
    plt.yticks(range(len(MODES)), MODES)
    plt.xlabel("Recovery mode B")
    plt.ylabel("Perturbed mode A")
    plt.title(title)
    for row in range(len(MODES)):
        for column in range(len(MODES)):
            if math.isfinite(matrix[row, column]):
                plt.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _summarize_perturbations(
    rows: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    result = []
    for overlap_tag in (*OVERLAP_TAGS, "all"):
        subset = rows if overlap_tag == "all" else [
            row for row in rows if row["overlap_tag"] == overlap_tag
        ]
        for mode in MODES:
            selected = [row for row in subset if row["perturbed_mode"] == mode]
            damage = [float(row["perturbation_mse"]) for row in selected]
            psnr_change = [
                float(row["perturbed_psnr_gt"])
                - float(row["original_psnr_gt"])
                for row in selected
            ]
            damage_low, damage_high = _bootstrap_ci(
                damage, bootstrap_samples, rng
            )
            psnr_low, psnr_high = _bootstrap_ci(
                psnr_change, bootstrap_samples, rng
            )
            result.append(
                {
                    "overlap_tag": overlap_tag,
                    "perturbed_mode": mode,
                    "count": len(selected),
                    "perturbation_mse_mean": float(np.mean(damage)),
                    "perturbation_mse_median": float(np.median(damage)),
                    "perturbation_mse_ci_low": damage_low,
                    "perturbation_mse_ci_high": damage_high,
                    "perturbation_delta_psnr_gt_mean": float(np.mean(psnr_change)),
                    "perturbation_delta_psnr_gt_median": float(np.median(psnr_change)),
                    "perturbation_delta_psnr_gt_ci_low": psnr_low,
                    "perturbation_delta_psnr_gt_ci_high": psnr_high,
                }
            )
    return result


def _plot_perturbation_damage(
    path: Path, summary: list[dict[str, Any]]
) -> None:
    rows = [row for row in summary if row["overlap_tag"] == "all"]
    lookup = {row["perturbed_mode"]: row for row in rows}
    values = [lookup[mode]["perturbation_mse_median"] for mode in MODES]
    plt.figure(figsize=(9, 5))
    plt.bar(MODES, values)
    plt.yscale("log")
    plt.ylabel("Median MSE to original rendering")
    plt.xticks(rotation=30, ha="right")
    plt.title("Calibrated perturbation damage")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _run_finalize(analysis: CompensationAnalysisCfg, root: Path) -> None:
    calibration = _load_calibration(root)
    manifests = []
    payloads = []
    for fold in range(analysis.num_folds):
        fold_dir = root / "recovery" / f"fold_{fold}"
        manifest_path = fold_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing recovery manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        if manifest.get("status") != "complete":
            raise RuntimeError(f"Recovery fold {fold} is incomplete")
        manifests.append(manifest)
        payloads.extend(
            _load_chunk(path)
            for path in sorted((fold_dir / "scene_chunks").glob("*.json.gz"))
        )
    if len({payload["scene"] for payload in payloads}) != len(payloads):
        raise RuntimeError("Recovery payloads contain duplicate scenes")
    expected = sum(int(manifest["fold_scenes"]) for manifest in manifests)
    if len(payloads) != expected:
        raise RuntimeError(
            f"Expected {expected} recovery chunks from manifests, found {len(payloads)}"
        )
    checkpoint_hashes = {manifest["checkpoint"]["sha256"] for manifest in manifests}
    split_hashes = {manifest["scene_split"]["sha256"] for manifest in manifests}
    calibration_hashes = {manifest["calibration_sha256"] for manifest in manifests}
    beta_hashes = {manifest["beta_file_sha256"] for manifest in manifests}
    if len(checkpoint_hashes) != 1 or len(split_hashes) != 1 or len(beta_hashes) != 1:
        raise RuntimeError("Recovery folds used inconsistent inputs")
    if calibration_hashes != {calibration["sha256"]}:
        raise RuntimeError("Recovery folds used inconsistent calibration")
    rows = []
    for payload in payloads:
        common = {
            key: payload[key]
            for key in ("scene", "fold", "overlap", "overlap_tag")
        }
        rows.extend({**common, **row} for row in payload["rows"])
    all_view_rows = [row for row in rows if int(row["target_position"]) == -1]
    summary = _summarize_rows(
        all_view_rows,
        value_keys=("recovery", "recovery_delta_psnr_gt"),
        bootstrap_samples=analysis.bootstrap_samples,
        seed=analysis.seed,
    )
    view_rows = [row for row in rows if int(row["target_position"]) >= 0]
    position_summary = []
    for target_position in sorted({int(row["target_position"]) for row in view_rows}):
        current = _summarize_rows(
            [row for row in view_rows if int(row["target_position"]) == target_position],
            value_keys=("recovery",),
            bootstrap_samples=analysis.bootstrap_samples,
            seed=analysis.seed + target_position + 1,
        )
        for row in current:
            row["target_position"] = target_position
        position_summary.extend(current)
    consistency_rows = []
    grouped_views: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in view_rows:
        grouped_views[
            (row["scene"], row["perturbed_mode"], row["recovery_mode"])
        ].append(row)
    for (scene, mode_a, mode_b), selected in grouped_views.items():
        values = [
            float(row["recovery"])
            for row in selected
            if bool(row["valid"]) and row["recovery"] is not None
        ]
        reference = selected[0]
        if values:
            consistency_rows.append(
                {
                    "scene": scene,
                    "fold": reference["fold"],
                    "overlap": reference["overlap"],
                    "overlap_tag": reference["overlap_tag"],
                    "perturbed_mode": mode_a,
                    "recovery_mode": mode_b,
                    "recovery_view_mean": float(np.mean(values)),
                    "recovery_view_std": float(np.std(values)),
                    "recovery_view_range": float(np.max(values) - np.min(values)),
                    "valid_views": len(values),
                    "valid": True,
                }
            )
    consistency_summary = _summarize_rows(
        consistency_rows,
        value_keys=("recovery_view_mean", "recovery_view_std", "recovery_view_range"),
        bootstrap_samples=analysis.bootstrap_samples,
        seed=analysis.seed + 100,
    )
    perturbation_rows = []
    seen_perturbations = set()
    for row in all_view_rows:
        key = (row["scene"], row["perturbed_mode"])
        if key in seen_perturbations:
            continue
        seen_perturbations.add(key)
        perturbation_rows.append(
            {
                "scene": row["scene"],
                "fold": row["fold"],
                "overlap": row["overlap"],
                "overlap_tag": row["overlap_tag"],
                "perturbed_mode": row["perturbed_mode"],
                "perturbation_mse": row["perturbation_mse"],
                "original_psnr_gt": row["original_psnr_gt"],
                "perturbed_psnr_gt": row["perturbed_psnr_gt"],
            }
        )
    output = root / "results"
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "compensation_per_scene.csv", all_view_rows)
    _write_csv(output / "compensation_summary.csv", summary)
    _write_csv(output / "recovery_by_target_position.csv", position_summary)
    _write_csv(output / "view_consistency_per_scene.csv", consistency_rows)
    _write_csv(output / "view_consistency_summary.csv", consistency_summary)
    _write_csv(output / "perturbation_per_scene.csv", perturbation_rows)
    perturbation_summary = _summarize_perturbations(
        perturbation_rows, analysis.bootstrap_samples, analysis.seed
    )
    _write_csv(output / "perturbation_summary.csv", perturbation_summary)
    linear_summary = _read_csv(root / "betas" / "linear_compensation_summary.csv")
    _plot_matrix(
        plots / "compensation_global.png",
        summary,
        "recovery_mean",
        "Cross-fitted exact compensation",
        vmin=-0.25,
        vmax=1.0,
    )
    _plot_matrix(
        plots / "compensation_oracle_linear.png",
        linear_summary,
        "oracle_linear_recovery_mean",
        "Per-scene linear oracle compensation",
        vmin=0.0,
        vmax=1.0,
    )
    _plot_matrix(
        plots / "global_vs_oracle_gap.png",
        linear_summary,
        "oracle_gap_mean",
        "Linear oracle minus cross-fitted global recovery",
        vmin=0.0,
        vmax=1.0,
    )
    beta_rows = _read_csv(root / "betas" / "global_beta.csv")
    beta_plot_summary = []
    for mode_a in MODES:
        for mode_b in MODES:
            values = [
                float(row["beta"])
                for row in beta_rows
                if row["perturbed_mode"] == mode_a
                and row["recovery_mode"] == mode_b
            ]
            beta_plot_summary.append(
                {
                    "overlap_tag": "all",
                    "perturbed_mode": mode_a,
                    "recovery_mode": mode_b,
                    "beta_mean": float(np.mean(values)),
                }
            )
    _plot_matrix(
        plots / "beta_by_pair.png",
        beta_plot_summary,
        "beta_mean",
        "Cross-fitted compensation coefficient",
        vmin=-analysis.beta_abs_max,
        vmax=analysis.beta_abs_max,
    )
    _plot_perturbation_damage(
        plots / "perturbation_damage.png", perturbation_summary
    )
    for tag in OVERLAP_TAGS:
        _plot_matrix(
            plots / f"compensation_{tag}.png",
            summary,
            "recovery_mean",
            f"Exact compensation: {tag}",
            overlap_tag=tag,
            vmin=-0.25,
            vmax=1.0,
        )
    diagonal = {
        mode: next(
            row["recovery_median"]
            for row in summary
            if row["overlap_tag"] == "all"
            and row["perturbed_mode"] == mode
            and row["recovery_mode"] == mode
        )
        for mode in MODES
    }
    warnings = [
        mode
        for mode, value in diagonal.items()
        if float(value) < analysis.diagonal_recovery_warning
    ]
    final_summary = {
        "status": "complete",
        "processed_scenes": len(payloads),
        "overlap_counts": {
            tag: sum(payload["overlap_tag"] == tag for payload in payloads)
            for tag in OVERLAP_TAGS
        },
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "scene_split_sha256": next(iter(split_hashes)),
        "calibration_sha256": calibration["sha256"],
        "beta_file_sha256": next(iter(beta_hashes)),
        "diagonal_recovery_median": diagonal,
        "diagonal_warnings": warnings,
        "primary_table": "compensation_summary.csv",
    }
    _save_json(output / "summary.json", final_summary)
    _save_json(
        output / "manifest.json",
        {
            "status": "complete",
            "analysis": asdict(analysis),
            "mode_definition": _mode_definition(),
            "calibration": calibration,
            "summary": final_summary,
            "git": _git_info(),
        },
    )
    print(f"Compensation analysis complete: {output}")
    print(f"Processed scenes: {final_summary['overlap_counts']}")
    print(f"Diagonal recovery medians: {diagonal}")
    if warnings:
        print(f"WARNING: low diagonal recovery for {warnings}")


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg_dict: DictConfig) -> None:
    analysis = _compensation_cfg(cfg_dict)
    root = _absolute_path(analysis.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if analysis.phase == "calibrate":
        _run_calibration(cfg_dict, analysis, root)
    elif analysis.phase == "response":
        _run_response(cfg_dict, analysis, root)
    elif analysis.phase == "merge_responses":
        _run_merge_responses(analysis, root)
    elif analysis.phase == "recovery":
        _run_recovery(cfg_dict, analysis, root)
    elif analysis.phase == "finalize":
        _run_finalize(analysis, root)
    else:
        raise AssertionError(analysis.phase)


if __name__ == "__main__":
    main()
