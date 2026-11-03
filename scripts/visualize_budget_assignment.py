r"""Run real checkpoint inference and visualize selected RE10K Gaussians.

Run from the repository root, in the existing CUDA evaluation environment:

    python scripts/visualize_budget_assignment.py

The five selections and server checkpoint paths are already configured in
assets/budget_analysis_samples.json. Default execution strictly loads each model,
infers Gaussians from the original .torch context frames, then rasterizes plots.
Missing inputs or mismatched weights stop execution; no synthetic-data fallback
exists. --dry-run only validates selections and never generates plots.

Each comparison has two image columns: the full selected context and the same
20x20 crop around the selected pixel. Red crosses mark Gaussian centers; blue
points mark input feature locations. Ours opacity uses q=budget*softmax, on one
shared scale for both Ours conditions. Baseline shows k*k feature locations and
uses the kernel L2 norm over input/output channels at each offset. That static
kernel strength is not a sample-specific attribution or comparable in units to q.
All Ours 3D supports, including those from the other view, are projected into the
selected context. No surface purity is inferred from surface names alone.

The Ours control reuses the exact points/features, kNN graph, and assignment,
replacing every point budget by its scene-wide mean. All downstream attributes,
including appearance, are recomputed. Display cameras are aligned once with
the original model, then frozen for its constant-budget control.

The default budget_max=2 matches this branch's original Final++ experiment.
Use --ours-config TRAIN_RUN/.hydra/config.yaml or --budget-max VALUE when the
checkpoint's training settings differ; budget_max is not a checkpoint tensor.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import re
import sys
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("means", "covariances", "harmonics", "opacities")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def encoder_weights(checkpoint: dict) -> dict:
    """Extract encoder weights from Lightning, model, or plain state dictionaries."""
    state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint does not contain an encoder state dictionary")
    encoder = {
        key[len("encoder."):]: value
        for key, value in state.items() if key.startswith("encoder.")
    }
    return encoder or state


def load_encoder_checkpoint(encoder: Any, path: Path, require_budget: bool = False):
    """Load every encoder parameter strictly before any scene inference."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    state = encoder_weights(checkpoint)
    if require_budget and not all(
        f"gaussian_decoder.budget_head.{name}" in state for name in ("weight", "bias")
    ):
        raise ValueError(
            "Ours checkpoint has no learned budget head. Supply the original "
            "budget-trained Final++ checkpoint for this analysis."
        )
    encoder.load_state_dict(state, strict=True)
    return int(checkpoint.get("global_step", 0)), len(state)


def checkpoint_paths(args: Any, config: dict) -> dict[str, Path]:
    return {
        "baseline": Path(args.baseline_ckpt or config["baseline_checkpoint"]),
        "ours": Path(args.ours_ckpt or config["ours_checkpoint"]),
    }


def require_server_files(paths) -> None:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Server input not found: {path}. Run in the server environment "
                "with the checkpoints and RE10K data mounted. No plots were generated."
            )


def selection_plan(
    config: dict[str, Any], index: dict[str, Any], manifest: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Resolve pixel slots without guessing context order or missing frames."""
    size = (manifest or {}).get("image_size_wh", [256, 256])
    width, height = size
    exported = {s["scene"]: s for s in (manifest or {}).get("samples", [])}
    planned = []
    seen = set()
    scenes = set()
    for sample in config["samples"]:
        name = sample["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in seen:
            raise ValueError(f"Invalid or duplicate sample ID: {name}")
        seen.add(name)
        image = PurePosixPath(sample["image_path"].replace("\\", "/"))
        scene = image.parent.name.rsplit("_", 1)[-1]
        if scene in scenes:
            raise ValueError("Select one Gaussian per scene in each analysis run")
        scenes.add(scene)
        match = re.fullmatch(r"context_(\d+)_frame_(\d+)\.png", image.name)
        if match is None or index.get(scene) is None:
            raise ValueError(f"Cannot resolve an evaluation scene/frame: {image}")
        view, frame = map(int, match.groups())
        entry = index[scene]
        if len(entry["context"]) != 2 or view not in (0, 1):
            raise ValueError(f"Expected two context views: {image}")
        if entry["context"][view] != frame:
            raise ValueError(f"Frame differs from evaluation context order: {image}")
        xy = sample["pixel_xy"]
        if len(xy) != 2 or any(type(value) is not int for value in xy):
            raise ValueError(f"pixel_xy must contain two integers: {name}")
        x, y = xy
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"Pixel is outside {width}x{height}: {name} {xy}")
        source = exported.get(scene)
        if manifest is not None:
            if source is None or source["context"] != entry["context"]:
                raise ValueError(
                    f"Scene/context missing or changed in manifest: {scene}"
                )
            exported_image = source["images"][view]
            if PurePosixPath(exported_image["path"]).name != image.name:
                raise ValueError(f"Image differs from manifest: {image}")
        targets = source["target"] if source else entry["target"]
        if not targets:
            raise ValueError(f"At least one target frame is required: {scene}")
        planned.append(
            {
                **sample,
                "scene": scene,
                "context_view": view,
                "frame_index": frame,
                "context": entry["context"],
                "target": targets,
                "slot": view * height * width + y * width + x,
                "image_size_wh": size,
                "chunk_path": source["chunk_path"] if source else None,
            }
        )
    if not planned:
        raise ValueError("No samples were selected")
    return planned


@contextmanager
def capture_decoder(decoder: Any):
    """Observe the production decoder without changing its calculations."""
    import torch

    trace: dict[str, Any] = {"allocation_chunks": []}
    aggregate = decoder.aggregate_moments

    def inputs(module, args, kwargs):
        trace["args"], trace["kwargs"] = args, dict(kwargs)

    def budget(module, args, output):
        trace["budget"] = (
            decoder.cfg.budget_max * output.detach().sigmoid()
        ).squeeze(-1)

    def allocation(module, args, output):
        trace["allocation_chunks"].append(output.detach().squeeze(-1).softmax(-1))

    def moments(points, features, neighbors, q, return_weights=False):
        if "neighbors" in trace:
            raise ValueError("Budget analysis expects batch size one")
        result = aggregate(points, features, neighbors, q, return_weights=True)
        trace.update(neighbors=neighbors, q=q, weights=result[-1], mass=result[0])
        return result if return_weights else result[:-1]

    handles = [
        decoder.register_forward_pre_hook(inputs, with_kwargs=True),
        decoder.budget_head.register_forward_hook(budget),
        decoder.allocation_head[2].register_forward_hook(allocation),
    ]
    try:
        with patch.object(decoder, "aggregate_moments", side_effect=moments):
            yield trace
        trace["allocation"] = torch.cat(trace.pop("allocation_chunks"))
    finally:
        for handle in handles:
            handle.remove()


def budget_intervention(encoder: Any, context: dict, step: int):
    """Keep decoder input, directed graph and A fixed; replace only b by mean(b)."""
    import torch

    decoder = encoder.gaussian_decoder
    with torch.no_grad(), capture_decoder(decoder) as learned_trace:
        learned = encoder(context, step)
    mean_budget = learned_trace["budget"].mean()
    constant_logit = torch.logit(mean_budget / decoder.cfg.budget_max)
    module = importlib.import_module(type(decoder).__module__)

    def constant_budget(module, args, output):
        return constant_logit.expand_as(output)

    handle = decoder.budget_head.register_forward_hook(constant_budget)
    try:
        with torch.no_grad(), patch.object(
            module, "build_knn", return_value=learned_trace["neighbors"]
        ), capture_decoder(decoder) as constant_trace:
            constant = decoder(*learned_trace["args"], **learned_trace["kwargs"])
    finally:
        handle.remove()
    torch.testing.assert_close(
        learned_trace["allocation"], constant_trace["allocation"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        learned_trace["mass"].sum(), constant_trace["mass"].sum(),
        rtol=2e-6, atol=1e-6,
    )
    for trace in (learned_trace, constant_trace):
        torch.testing.assert_close(
            trace["q"], trace["allocation"] * trace["budget"][:, None]
        )
    return learned, constant, learned_trace, constant_trace


def incoming_evidence(learned: dict, constant: dict, slot: int) -> dict[str, Any]:
    """Select incoming edges, which need not equal the selected slot's kNN row."""
    source, edge = (learned["neighbors"] == slot).nonzero(as_tuple=True)
    values = {
        "source_index": source,
        "edge_index": edge,
        "point_xyz": learned["args"][0][0, source],
        "assignment": learned["allocation"][source, edge],
    }
    for name, trace in (("learned", learned), ("constant", constant)):
        values[f"budget_{name}"] = trace["budget"][source]
        values[f"q_{name}"] = trace["q"][source, edge]
        values[f"weight_{name}"] = trace["weights"][source, edge]
    return {key: value.detach().cpu().numpy() for key, value in values.items()}


def project_means(means: Any, cameras: Any, intrinsics: Any, shape: tuple[int, int]):
    """Match the existing renderer's centered FOV projection, including -0.5."""
    import torch
    from src.geometry.projection import get_fov

    height, width = shape
    camera_points = (
        cameras.inverse()[:, :3, :3] @ means[:, :, None]
    ).squeeze(-1) + cameras.inverse()[:, :3, 3]
    tan = (0.5 * get_fov(intrinsics)).tan()
    scale = means.new_tensor([width, height]) / (2 * tan)
    xy = scale * camera_points[:, :2] / camera_points[:, 2:3]
    xy += means.new_tensor([(width - 1) / 2, (height - 1) / 2])
    xy = torch.where(camera_points[:, 2:3] > 0, xy, torch.nan)
    return xy.detach().cpu().numpy()


def fixed_crop_bounds(pixel_xy, size=20):
    """Exact pixel-cell bounds, shared across conditions; do not recenter on mu."""
    x, y = pixel_xy
    left, top = x - size // 2, y - size // 2
    return left - .5, left + size - .5, top - .5, top + size - .5


def kernel_input_mapping(
    weights, pixel_xy, shape, stride=(1, 1), padding=(1, 1), dilation=(1, 1),
):
    """One static, nonnegative kernel norm for each spatial feature offset."""
    import numpy as np

    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 4:
        raise ValueError("Expected Conv2d weights [out, in, kernel_h, kernel_w]")
    height, width = shape
    kh, kw = weights.shape[-2:]
    ky, kx = np.mgrid[:kh, :kw]
    x, y = pixel_xy
    source_x = x * stride[1] - padding[1] + kx * dilation[1]
    source_y = y * stride[0] - padding[0] + ky * dilation[0]
    valid = ((source_x >= 0) & (source_x < width)
             & (source_y >= 0) & (source_y < height))
    norms = np.sqrt(np.square(weights).sum(axis=(0, 1)))
    return {
        "mapped_x": source_x.ravel(), "mapped_y": source_y.ravel(),
        "kernel_norm": norms.ravel(), "valid": valid.ravel(),
        "kernel_size_hw": np.array([kh, kw]),
    }


def baseline_input_mapping(encoder, sample, shape):
    """Read the loaded spatial CNN kernel for the selected view's Gaussian head."""
    head = (encoder.gaussian_param_head if sample["context_view"] == 0
            else encoder.gaussian_param_head2)
    conv = head.dpt.head[0]
    if not hasattr(conv, "kernel_size"):
        raise ValueError("Expected the baseline Gaussian head's first Conv2d")
    return kernel_input_mapping(
        conv.weight.detach().float().cpu().numpy(), sample["pixel_xy"], shape,
        stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
    )


def project_input_mapping(evidence, cameras, intrinsics, view, shape):
    """Project every incoming 3D support into the selected display image."""
    import torch

    points = torch.as_tensor(evidence["point_xyz"], device=cameras.device,
                             dtype=cameras.dtype)
    count = len(points)
    coordinates = project_means(
        points, cameras[0, view].expand(count, -1, -1),
        intrinsics[0, view].expand(count, -1, -1), shape,
    )
    evidence["mapped_x"], evidence["mapped_y"] = coordinates.T


def comparison_rows(results, evidence):
    """Raw measures and alpha scales; never normalize the two Ours rows separately."""
    import numpy as np

    baseline = results["baseline"]["input_mapping"]
    kh, kw = baseline["kernel_size_hw"]
    kernel_max = float(np.max(baseline["kernel_norm"])) or 1.
    q_max = max(float(np.max(evidence[f"q_{name}"]))
                for name in ("constant", "learned")) or 1.
    rows = [{
        "name": "baseline", "label": f"Baseline ({kh} x {kw} kernel)",
        "xy": np.column_stack((baseline["mapped_x"], baseline["mapped_y"])),
        "values": baseline["kernel_norm"], "vmax": kernel_max,
        "valid": baseline["valid"], "measure": "Kernel L2 norm",
    }]
    for name in ("constant", "learned"):
        rows.append({
            "name": name, "label": f"Ours: {name} budget",
            "xy": np.column_stack((evidence["mapped_x"], evidence["mapped_y"])),
            "values": evidence[f"q_{name}"], "vmax": q_max,
            "valid": np.isfinite(evidence["mapped_x"])
            & np.isfinite(evidence["mapped_y"]),
            "measure": "q = budget x softmax",
        })
    for row in rows:
        row["alpha"] = np.where(
            row["valid"], np.clip(row["values"] / row["vmax"], 0, 1), 0,
        )
    return rows


def save_evidence(directory: Path, evidence: dict, shape: tuple[int, int]) -> None:
    import numpy as np

    height, width = shape
    sources = evidence["source_index"]
    evidence["source_view"] = sources // (height * width)
    evidence["source_y"] = sources % (height * width) // width
    evidence["source_x"] = sources % width
    np.savez_compressed(directory / "evidence.npz", **evidence)
    columns = [key for key, value in evidence.items() if value.ndim == 1]
    with (directory / "evidence.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[*columns, "surface_label"])
        writer.writeheader()
        for i in range(len(sources)):
            writer.writerow({key: evidence[key][i].item() for key in columns})


def save_baseline_evidence(directory, mapping):
    import numpy as np

    np.savez_compressed(directory / "baseline_evidence.npz", **mapping)
    columns = ("mapped_x", "mapped_y", "kernel_norm", "valid", "display_alpha")
    with (directory / "baseline_evidence.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for i in range(len(mapping["kernel_norm"])):
            writer.writerow({key: mapping[key][i].item() for key in columns})


def comparison_figure(sample, results, evidence):
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    view = sample["context_view"]
    image = results["learned"]["context_rgb"][view]
    height, width = image.shape[:2]
    bounds = fixed_crop_bounds(sample["pixel_xy"])
    rows = comparison_rows(results, evidence)
    # A pure blue colormap whose alpha increases linearly with the raw value.
    rgba = np.zeros((256, 4))
    rgba[:, 2], rgba[:, 3] = 1, np.linspace(0, 1, 256)
    cmap = ListedColormap(rgba)
    fig = plt.figure(figsize=(10.5, 13))
    grid = fig.add_gridspec(3, 3, width_ratios=(1, 1, .045),
                           left=.12, right=.88, bottom=.13, top=.88,
                           wspace=.13, hspace=.24)
    axes = np.empty((3, 2), dtype=object)
    for row_index, row in enumerate(rows):
        valid = row["valid"] & np.isfinite(row["xy"]).all(-1)
        colors = np.zeros((int(valid.sum()), 4))
        colors[:, 2], colors[:, 3] = 1, row["alpha"][valid]
        center = results[row["name"]]["centers"][view]
        for column in range(2):
            axis = fig.add_subplot(grid[row_index, column])
            axes[row_index, column] = axis
            axis.set_facecolor("white")
            axis.imshow(np.clip(image, 0, 1), interpolation="nearest")
            axis.scatter(*row["xy"][valid].T, c=colors,
                         s=18 if column == 0 else 65,
                         edgecolors="none", zorder=3)
            if np.isfinite(center).all():
                axis.scatter(*center, c="red", marker="+", zorder=5,
                             s=95 if column == 0 else 155, linewidths=1.7)
            else:
                axis.text(.02, .02, "Center behind camera",
                          transform=axis.transAxes, color="red", fontsize=8)
            if column == 0:
                axis.set_xlim(-.5, width - .5)
                axis.set_ylim(height - .5, -.5)
                axis.add_patch(Rectangle(
                    (bounds[0], bounds[2]), 20, 20, fill=False,
                    edgecolor="black", linestyle="--", linewidth=.8,
                ))
                axis.set_ylabel(row["label"], fontsize=11)
            else:
                # Exactly the same 20 pixel cells for every condition.
                # At image edges the uncovered area stays white; never resize.
                axis.set_xlim(bounds[0], bounds[1])
                axis.set_ylim(bounds[3], bounds[2])
            axis.set_aspect("equal")
            axis.tick_params(labelsize=8)
        bar_axis = fig.add_subplot(grid[row_index, 2])
        bar_axis.set_facecolor("white")
        fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, row["vmax"]), cmap=cmap),
                     cax=bar_axis, label=row["measure"])
        bar_axis.tick_params(labelsize=8)
    axes[0, 0].set_title(
        f"Full image (context {view}, frame {sample['context'][view]})", fontsize=11,
    )
    axes[0, 1].set_title(f"20 x 20 crop at {sample['pixel_xy']}", fontsize=11)
    fig.suptitle(
        f"{sample['id']}: {sample['label']}\n{sample['scene']}",
        fontsize=13, y=.97,
    )
    fig.legend(handles=[
        Line2D([], [], color="red", marker="+", linestyle="none",
               markersize=10, label="Gaussian center"),
        Line2D([], [], color="blue", marker="o", linestyle="none",
               markersize=6, label="Input feature mapping"),
    ], loc="upper center", bbox_to_anchor=(.5, .935), ncol=2, frameon=False)
    fig.text(.12, .04,
             "Transparency = value / bar maximum. Ours rows share the same range.\n"
             "Baseline: kernel L2 norm over input/output channels (static weights).\n"
             "Ours: q = budget x softmax; all incoming 3D evidence is projected "
             "into this context.\n"
             "The crop is centered on the selected pixel and fixed across rows.",
             fontsize=9)
    return fig, axes


def plot_comparison(sample: dict, results: dict, evidence: dict, directory: Path):
    fig, _ = comparison_figure(sample, results, evidence)
    from matplotlib import pyplot as plt

    fig.savefig(directory / "comparison.png", dpi=180)
    fig.savefig(directory / "comparison.pdf")
    plt.close(fig)


def model_config(
    experiment: str, data_root: Path, index_path: Path, size: list[int],
    saved_config: Path | None = None, budget_max: float | None = None,
):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg

    if saved_config is not None:
        config = OmegaConf.load(saved_config)
    else:
        with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
            config = compose(config_name="main", overrides=[
                f"+experiment={experiment}",
                "dataset/view_sampler@dataset.re10k.view_sampler=evaluation",
            ])
    config.mode, config.wandb.mode = "test", "disabled"
    config.dataset.re10k.roots = [str(data_root)]
    config.dataset.re10k.view_sampler = dict(
        name="evaluation", index_path=str(index_path), num_context_views=2,
    )
    config.dataset.re10k.input_image_shape = [size[1], size[0]]
    config.data_loader.test.num_workers = 0
    config.data_loader.test.batch_size = 1
    config.model.encoder.pretrained_weights = ""
    if config.model.encoder.gs_params_head_type == "moment":
        # Expose hooks and avoid compilation warmup; this changes no parameters.
        config.model.encoder.moment_decoder.compile_kernels = False
        if budget_max is not None:
            config.model.encoder.moment_decoder.budget_max = budget_max
    set_cfg(config)
    return load_typed_root_config(config), config


def describe_gaussian(gaussians: Any, slot: int) -> dict[str, Any]:
    return {
        "mean": gaussians.means[0, slot].detach().cpu().tolist(),
        "covariance": gaussians.covariances[0, slot].detach().cpu().tolist(),
        "harmonics": gaussians.harmonics[0, slot].detach().cpu().tolist(),
        "opacity": float(gaussians.opacities[0, slot]),
    }


def render_result(decoder: Any, gaussians: Any, sample: dict, display: dict,
                  cameras: Any, context_rgb: Any) -> dict:
    import numpy as np
    import torch
    from src.model.types import Gaussians

    target = display["target"]
    height, width = target["image"].shape[-2:]
    slot, view = sample["slot"], sample["context_view"]
    selected = Gaussians(**{
        field: getattr(gaussians, field)[:, slot : slot + 1] for field in FIELDS
    })
    with torch.no_grad():
        full = decoder(gaussians, cameras, target["intrinsics"], target["near"],
                       target["far"], (height, width)).color
        args = (cameras[:, view:view + 1], target["intrinsics"][:, view:view + 1],
                target["near"][:, view:view + 1], target["far"][:, view:view + 1],
                (height, width))
        isolated = decoder(selected, *args).color[0, 0]
        white_sh = torch.zeros_like(selected.harmonics)
        white_sh[..., 0] = 0.5 / math.sqrt(1 / (4 * math.pi))
        white = Gaussians(selected.means, selected.covariances, white_sh,
                          selected.opacities)
        coverage = decoder(white, *args).color[0, 0].mean(0).clamp(0, 1)
        centers = project_means(
            selected.means[0].expand(3, -1), cameras[0], target["intrinsics"][0],
            (height, width),
        )
    rgb = context_rgb[0].permute(0, 2, 3, 1).detach().cpu().numpy()
    gray = rgb[view].mean(-1, keepdims=True)
    gray = 0.5 * gray + 0.5
    alpha = coverage.detach().cpu().numpy()
    color = isolated.permute(1, 2, 0).detach().cpu().numpy()
    composite = color + (1 - alpha[..., None]) * gray
    return {
        "attributes": describe_gaussian(gaussians, slot),
        "selected_gaussian": {
            name: getattr(selected, name).detach().cpu() for name in FIELDS
        },
        "centers": centers,
        "context_rgb": rgb,
        "isolated_composite": np.clip(composite, 0, 1),
        "isolated_alpha": alpha,
        "novel_render": full[0, 2].permute(1, 2, 0).detach().cpu().numpy(),
        "cameras": cameras[0].detach().cpu().numpy(),
    }


def execute(args: Any, config: dict, plan: list[dict], manifest: dict) -> None:
    # Validate real files before importing CUDA libraries or creating output.
    data_path = Path(args.data_root or manifest["data_path"])
    if data_path.suffix == ".torch":
        data_path = data_path.parent
    data_root = data_path.parent if data_path.name == "test" else data_path
    if not (data_root / "test").is_dir():
        raise ValueError("Cannot find dataset test/; pass --data-root /path/to/re10k")
    chunks = []
    for sample in plan:
        path = Path(sample["chunk_path"])
        if args.data_root or not path.is_file():
            path = data_root / "test" / path.name
        if not path.is_file():
            raise FileNotFoundError(path)
        chunks.append(path)
    checkpoints = checkpoint_paths(args, config)
    require_server_files([
        *checkpoints.values(), *[Path(s["image_path"]) for s in plan],
    ])

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    from src.dataset.data_module import DataModule
    from src.loss import get_losses
    from src.misc.step_tracker import StepTracker
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.scripts.analysis_scale_opacity import (
        _align_target_extrinsics, _freeze, _to_device,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Actual inference requires CUDA Gaussian rasterization")
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    index = read_json(args.index)
    selected_index = {s["scene"]: {**index[s["scene"]], "target": s["target"]}
                      for s in plan}
    index_path = output / "selected_evaluation_index.json"
    write_json(index_path, selected_index)
    write_json(output / "selection_plan.json", {"samples": plan})
    run_metadata = {
        "source": "real_checkpoint_inference",
        "comparison_layout": "full context + fixed 20x20 crop; red center, blue inputs",
        "checkpoints": {name: str(path) for name, path in checkpoints.items()},
        "data_root": str(data_root),
        "pose_align_steps": args.pose_align_steps,
        "global_steps": {},
        "models": {},
        "complete": False,
    }
    write_json(output / "run_metadata.json", run_metadata)
    by_scene = {s["scene"]: s for s in plan}
    results: dict[str, dict] = {s["id"]: {} for s in plan}
    evidence_by_id = {}
    device = torch.device("cuda")

    def restrict_chunks(dataset, stage):
        dataset.chunks = sorted(set(chunks))
        return dataset

    for model, experiment in (
        ("baseline", "re10k"), ("ours", "re10k_budget_analysis"),
    ):
        checkpoint = checkpoints[model]
        cfg, raw_cfg = model_config(
            experiment, data_root, index_path, plan[0]["image_size_wh"],
            saved_config=args.ours_config if model == "ours" else args.baseline_config,
            budget_max=args.budget_max if model == "ours" else None,
        )
        if cfg.model.encoder.num_surfaces != 1:
            raise ValueError("Analysis requires one Gaussian slot per source pixel")
        if cfg.model.encoder.gaussians_per_pixel != 1:
            raise ValueError("Analysis requires one Gaussian slot per source pixel")
        OmegaConf.save(raw_cfg, output / f"{model}_config.yaml")
        torch.manual_seed(0)
        encoder, _ = get_encoder(cfg.model.encoder)
        print(f"[Checkpoint] Strictly loading {model}: {checkpoint}", flush=True)
        step, tensor_count = load_encoder_checkpoint(
            encoder, checkpoint, require_budget=model == "ours",
        )
        run_metadata["global_steps"][model] = step
        run_metadata["models"][model] = {
            "strict_load": True, "encoder_tensors": tensor_count,
            "budget_max": cfg.model.encoder.moment_decoder.budget_max
            if model == "ours" else None,
            "budget_setting_source": (
                "command line" if args.budget_max is not None else
                "saved training config" if args.ours_config else
                "original Final++ inference config"
            ) if model == "ours" else None,
            "inferred_scenes": [],
        }
        write_json(output / "run_metadata.json", run_metadata)
        encoder = _freeze(encoder.to(device))
        decoder = _freeze(get_decoder(cfg.model.decoder).to(device))
        if decoder.background_color.abs().max().item() != 0:
            raise ValueError(
                "Alpha compositing expects the configured black background"
            )
        losses = torch.nn.ModuleList(get_losses(cfg.loss)).to(device)
        _freeze(losses)
        loader = DataModule(
            cfg.dataset, cfg.data_loader, StepTracker(),
            dataset_shim=restrict_chunks, global_rank=0,
        ).test_dataloader()
        done = set()
        for batch in loader:
            scene = batch["scene"][0]
            if scene not in by_scene or scene in done:
                continue
            sample = by_scene[scene]
            print(f"[Inference] {model}: {sample['id']} {scene}", flush=True)
            batch = _to_device(batch, device)
            context_rgb = batch["context"]["image"]
            if list(context_rgb.shape[-2:]) != sample["image_size_wh"][::-1]:
                raise ValueError(f"Model input size changed: {scene}")
            if batch["context"]["index"][0].tolist() != sample["context"]:
                raise ValueError(f"Context order changed: {scene}")
            # Verify the chosen PNG is exactly the same frame and preprocessing.
            with Image.open(sample["image_path"]) as image:
                chosen = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
            actual = context_rgb[0, sample["context_view"]].permute(1, 2, 0)
            if (
                chosen.shape != tuple(actual.shape)
                or not np.allclose(chosen, actual.cpu().numpy(), atol=1 / 255 + 1e-7)
            ):
                raise ValueError(
                    f"Exported PNG differs from the loaded frame: {scene}"
                )
            batch = encoder.get_data_shim()(batch)
            if model == "ours":
                original, constant, learned_trace, constant_trace = budget_intervention(
                    encoder, batch["context"], step
                )
                evidence = incoming_evidence(
                    learned_trace, constant_trace, sample["slot"]
                )
                evidence_by_id[sample["id"]] = evidence
            else:
                with torch.no_grad():
                    original = encoder(batch["context"], step)
                kernel_mapping = baseline_input_mapping(
                    encoder, sample, tuple(context_rgb.shape[-2:])
                )
            display = {**batch, "target": {}}
            for key in ("image", "extrinsics", "intrinsics", "near", "far", "index"):
                context_value = (
                    context_rgb if key == "image" else batch["context"][key]
                )
                display["target"][key] = torch.cat(
                    (context_value, batch["target"][key][:, :1]), dim=1
                )
            # Align visualization cameras on the original full scene once.
            # Constant-budget decoding never gets independently aligned.
            cameras = _align_target_extrinsics(
                decoder, losses, display, original, step, args.pose_align_steps > 0,
                args.pose_align_steps, cfg.test.rot_opt_lr, cfg.test.trans_opt_lr,
            )
            if model == "ours":
                project_input_mapping(
                    evidence, cameras, display["target"]["intrinsics"],
                    sample["context_view"], tuple(context_rgb.shape[-2:]),
                )
            variants = {"baseline": original} if model == "baseline" else {
                "learned": original, "constant": constant,
            }
            print(f"[Render] {model}: {sample['id']}", flush=True)
            for name, gaussians in variants.items():
                results[sample["id"]][name] = render_result(
                    decoder, gaussians, sample, display, cameras, context_rgb
                )
            if model == "baseline":
                results[sample["id"]]["baseline"]["input_mapping"] = kernel_mapping
            run_metadata["models"][model]["inferred_scenes"].append(scene)
            write_json(output / "run_metadata.json", run_metadata)
            if model == "ours":
                directory = output / sample["id"]
                directory.mkdir()
                evidence = evidence_by_id[sample["id"]]
                rows = comparison_rows(results[sample["id"]], evidence)
                for row in rows[1:]:
                    evidence[f"display_alpha_{row['name']}"] = row["alpha"]
                kernel_mapping = results[sample["id"]]["baseline"]["input_mapping"]
                kernel_mapping["display_alpha"] = rows[0]["alpha"]
                save_baseline_evidence(directory, kernel_mapping)
                save_evidence(directory, evidence, tuple(context_rgb.shape[-2:]))
                total = float(learned_trace["mass"].sum())
                attributes = {
                    "source": "real_checkpoint_inference",
                    "sample": sample,
                    "conditions": {
                        name: result["attributes"]
                        for name, result in results[sample["id"]].items()
                    },
                    "scene_mean_budget": float(learned_trace["budget"].mean()),
                    "learned_total_mass": total,
                    "constant_total_mass": float(constant_trace["mass"].sum()),
                    "assignment_identical": True,
                    "incoming_supports": len(evidence["source_index"]),
                    "surface_purity_computed": False,
                    "visualization": {
                        "crop_size_wh": [20, 20],
                        "crop_center_xy": sample["pixel_xy"],
                        "baseline_kernel_size_hw": kernel_mapping[
                            "kernel_size_hw"
                        ].tolist(),
                        "baseline_weight": "L2 norm over both kernel channel axes",
                        "ours_weight": "q = budget * outgoing softmax",
                        "ours_mapping": "all 3D supports projected to selected context",
                        "alpha_rule": "raw value / bar maximum",
                        "bar_maxima": {row["name"]: row["vmax"] for row in rows},
                    },
                    "camera_alignment": (
                        "Original model only; frozen for constant budget"
                    ),
                }
                write_json(directory / "attributes.json", attributes)
                torch.save({
                    name: result["selected_gaussian"]
                    for name, result in results[sample["id"]].items()
                }, directory / "selected_gaussians.pt")
                plot_comparison(sample, results[sample["id"]], evidence, directory)
                # Save unmarked images for inspection, not only a composed figure.
                for name, result in results[sample["id"]].items():
                    for kind in ("novel_render", "isolated_composite"):
                        pixels = np.clip(result[kind] * 255, 0, 255).astype(np.uint8)
                        Image.fromarray(pixels).save(directory / f"{name}_{kind}.png")
                    np.savez_compressed(
                        directory / f"{name}_projection.npz",
                        centers=result["centers"], cameras=result["cameras"],
                        alpha=result["isolated_alpha"],
                    )
                gt = (
                    display["target"]["image"][0, 2]
                    .permute(1, 2, 0).cpu().numpy()
                )
                Image.fromarray((np.clip(gt, 0, 1) * 255).astype(np.uint8)).save(
                    directory / "novel_ground_truth.png"
                )
                print(f"[Saved] {directory / 'comparison.png'}", flush=True)
                del learned_trace, constant_trace, constant
            done.add(scene)
            del original, gaussians, variants, batch, context_rgb, display, cameras
            if done == set(by_scene):
                break
        if done != set(by_scene):
            raise RuntimeError(
                f"Selected scenes were skipped by dataset: {set(by_scene) - done}"
            )
        del encoder, decoder, losses, loader
        torch.cuda.empty_cache()
    run_metadata["complete"] = True
    write_json(output / "run_metadata.json", run_metadata)
    print(f"Real inference completed: {output}", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--samples", type=Path,
                        default=ROOT / "assets/budget_analysis_samples.json")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--index", type=Path,
                        default=ROOT / "assets/evaluation_index_re10k.json")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--baseline-checkpoint", "--baseline-ckpt",
                        dest="baseline_ckpt", type=Path)
    parser.add_argument("--ours-checkpoint", "--ours-ckpt", dest="ours_ckpt", type=Path)
    parser.add_argument("--ours-config", type=Path, help="Saved full training YAML")
    parser.add_argument("--baseline-config", type=Path)
    parser.add_argument("--budget-max", type=float)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/budget_assignment_analysis"))
    parser.add_argument("--pose-align-steps", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.pose_align_steps < 0:
        parser.error("--pose-align-steps must be nonnegative")
    if args.budget_max is not None and (
        not math.isfinite(args.budget_max) or args.budget_max <= 0
    ):
        parser.error("--budget-max must be positive and finite")
    config = read_json(args.samples)
    manifest_path = args.manifest or Path(config["manifest"])
    if not args.dry_run:
        try:
            require_server_files([
                *checkpoint_paths(args, config).values(), manifest_path,
            ])
        except FileNotFoundError as error:
            parser.exit(2, f"Error: {error}\n")
    manifest = read_json(manifest_path) if manifest_path.is_file() else None
    plan = selection_plan(config, read_json(args.index), manifest)
    for sample in plan:
        print(
            f"{sample['id']}: {sample['scene']} context={sample['context']} "
            f"view={sample['context_view']} pixel={sample['pixel_xy']} "
            f"slot={sample['slot']}"
        )
    if args.dry_run:
        print("Selection validation passed; no model or data files were loaded.")
        if manifest is None:
            print(f"Manifest unavailable in this environment: {manifest_path}")
        return
    if manifest is None:
        parser.error(f"Export manifest not found: {manifest_path}; supply --manifest")
    sys.path.insert(0, str(ROOT))
    try:
        execute(args, config, plan, manifest)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
