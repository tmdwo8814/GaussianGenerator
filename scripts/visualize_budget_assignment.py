r"""Run real checkpoint inference, then plot the five selected RE10K Gaussians.

On the CUDA server, from the repository root:

    python scripts/visualize_budget_assignment.py

Defaults are assets/budget_analysis_samples.json and the server checkpoint paths
in that file. Both encoders are loaded strictly before inference. There is no
synthetic-data or cached-image fallback. Original .torch scenes come from the
image-export manifest; exported PNGs only verify the selected pixel coordinates.

The default budget_max=2 follows the original experiment inspected for this
analysis. This non-tensor setting cannot be read from weights alone. Supply
--ours-config TRAIN_RUN/.hydra/config.yaml or --budget-max VALUE if the checkpoint
used another setting. The no-budget branch restores its checkpoint budget head
for this inference instance only.

Outputs: comparison.png/pdf, actual renders, selected_gaussians.pt,
evidence.csv/npz, attributes.json, resolved configs and run_metadata.json.
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
from types import MethodType
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("means", "covariances", "harmonics", "opacities")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def resolve_selections(config, index, manifest):
    width, height = manifest["image_size_wh"]
    exported = {entry["scene"]: entry for entry in manifest["samples"]}
    selections, seen = [], set()
    for sample in config["samples"]:
        name = sample["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"Invalid output ID: {name}")
        path = PurePosixPath(sample["image_path"].replace("\\", "/"))
        match = re.fullmatch(r"context_(\d+)_frame_(\d+)\.png", path.name)
        scene = path.parent.name.rsplit("_", 1)[-1]
        if match is None or scene not in exported or index.get(scene) is None:
            raise ValueError(f"Sample missing from manifest/evaluation index: {path}")
        view, frame = map(int, match.groups())
        entry, source = index[scene], exported[scene]
        if scene in seen or name in seen:
            raise ValueError("Use unique sample IDs and one selected pixel per scene")
        seen.update((scene, name))
        if source["context"] != entry["context"] or len(source["context"]) != 2:
            raise ValueError(f"Context pair changed: {scene}")
        if view not in (0, 1) or source["context"][view] != frame:
            raise ValueError(f"Selected view/frame changed: {scene}")
        if PurePosixPath(source["images"][view]["path"]).name != path.name:
            raise ValueError(f"Selected PNG differs from the manifest: {scene}")
        xy = sample["pixel_xy"]
        if len(xy) != 2 or any(type(value) is not int for value in xy):
            raise ValueError(f"Expected integer (x,y): {name}")
        x, y = xy
        if not (0 <= x < width and 0 <= y < height) or not source["target"]:
            raise ValueError(f"Invalid pixel or missing target frame: {name}")
        selections.append({
            **sample, "scene": scene, "view": view, "context": source["context"],
            "target": source["target"], "chunk_path": source["chunk_path"],
            "slot": view * height * width + y * width + x,
            "image_size_wh": [width, height],
        })
    if not selections:
        raise ValueError("No selected samples")
    return selections


def encoder_weights(checkpoint):
    """Accept Lightning encoder state or a plain encoder state, never partial load."""
    if "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint.get("model", checkpoint)
    prefixed = {k[8:]: v for k, v in state.items() if k.startswith("encoder.")}
    return prefixed or state


def restore_budget_path(decoder, state, budget_max):
    """Restore original checkpoint semantics only on this inference instance."""
    import torch

    key = "gaussian_decoder.budget_head.weight"
    if key not in state or "gaussian_decoder.budget_head.bias" not in state:
        raise ValueError(
            "Ours checkpoint has no learned budget head; it cannot support this "
            "budget analysis. Supply the original budget-trained Final++ checkpoint."
        )
    if not math.isfinite(budget_max) or budget_max <= 0:
        raise ValueError("budget_max must be positive and finite")
    decoder.cfg.budget_max = budget_max
    if hasattr(decoder, "budget_head"):
        return "native budget decoder"
    # The current no-budget implementation returns A from predict_allocation.
    # Add only the original b_j head and A*b_j operation, then strictly load it.
    original = decoder.predict_allocation
    decoder.budget_head = torch.nn.Linear(decoder.cfg.feature_dim, 1)

    def predict(self, points, features, neighbors, return_appearance=False):
        result = original(points, features, neighbors, return_appearance)
        budget = self.cfg.budget_max * self.budget_head(features).sigmoid()
        if return_appearance:
            allocation, scores = result
            return allocation * budget, scores
        return result * budget

    decoder.predict_allocation = MethodType(predict, decoder)
    return "checkpoint budget restored on inference instance"


@contextmanager
def trace_decoder(decoder):
    import torch

    trace = {"allocation_chunks": []}
    original = decoder.aggregate_moments

    def inputs(module, args, kwargs):
        trace["args"], trace["kwargs"] = args, dict(kwargs)

    def budgets(module, args, output):
        trace["budget"] = (
            decoder.cfg.budget_max * output.detach().sigmoid()
        ).squeeze(-1)

    def allocation(module, args, output):
        trace["allocation_chunks"].append(output.detach().squeeze(-1).softmax(-1))

    def aggregate(points, features, neighbors, q, return_weights=False):
        if "neighbors" in trace:
            raise ValueError("Use batch size one for this analysis")
        result = original(points, features, neighbors, q, return_weights=True)
        trace.update(neighbors=neighbors, q=q, weights=result[-1], mass=result[0])
        return result if return_weights else result[:-1]

    handles = [
        decoder.register_forward_pre_hook(inputs, with_kwargs=True),
        decoder.budget_head.register_forward_hook(budgets),
        decoder.allocation_head[2].register_forward_hook(allocation),
    ]
    try:
        with patch.object(decoder, "aggregate_moments", side_effect=aggregate):
            yield trace
        trace["allocation"] = torch.cat(trace.pop("allocation_chunks"))
    finally:
        for handle in handles:
            handle.remove()


def infer_budget_pair(encoder, context, step):
    import torch

    decoder = encoder.gaussian_decoder
    with torch.no_grad(), trace_decoder(decoder) as learned_trace:
        learned = encoder(context, step)
    logit = torch.logit(learned_trace["budget"].mean() / decoder.cfg.budget_max)
    handle = decoder.budget_head.register_forward_hook(
        lambda module, args, output: logit.expand_as(output)
    )
    try:
        module = importlib.import_module(type(decoder).__module__)
        with torch.no_grad(), patch.object(
            module, "build_knn", return_value=learned_trace["neighbors"]
        ), trace_decoder(decoder) as constant_trace:
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


def incoming_evidence(learned, constant, slot, shape):
    height, width = shape
    source, edge = (learned["neighbors"] == slot).nonzero(as_tuple=True)
    data = {
        "source_index": source, "edge_index": edge,
        "source_view": source // (height * width),
        "source_x": source % width,
        "source_y": source % (height * width) // width,
        "point_xyz": learned["args"][0][0, source],
        "assignment": learned["allocation"][source, edge],
    }
    for name, trace in (("learned", learned), ("constant", constant)):
        data[f"budget_{name}"] = trace["budget"][source]
        data[f"q_{name}"] = trace["q"][source, edge]
        data[f"weight_{name}"] = trace["weights"][source, edge]
    return {key: value.detach().cpu().numpy() for key, value in data.items()}


def projected_centers(mean, cameras, intrinsics, shape):
    import torch
    from src.geometry.projection import get_fov

    height, width = shape
    inverse = cameras.inverse()
    points = (inverse[:, :3, :3] @ mean[:, :, None]).squeeze(-1)
    points += inverse[:, :3, 3]
    focal = mean.new_tensor([width, height]) / (2 * (0.5 * get_fov(intrinsics)).tan())
    xy = focal * points[:, :2] / points[:, 2:3]
    xy += mean.new_tensor([(width - 1) / 2, (height - 1) / 2])
    return torch.where(points[:, 2:3] > 0, xy, torch.nan).detach().cpu().numpy()


def render_condition(decoder, gaussians, sample, display, cameras, rgb):
    import numpy as np
    import torch
    from src.model.types import Gaussians

    target = display["target"]
    shape = tuple(target["image"].shape[-2:])
    slot, view = sample["slot"], sample["view"]
    selected = Gaussians(**{
        name: getattr(gaussians, name)[:, slot:slot + 1] for name in FIELDS
    })
    with torch.no_grad():
        output = decoder(gaussians, cameras, target["intrinsics"], target["near"],
                         target["far"], shape).color
        args = (cameras[:, view:view + 1], target["intrinsics"][:, view:view + 1],
                target["near"][:, view:view + 1], target["far"][:, view:view + 1],
                shape)
        isolated = decoder(selected, *args).color[0, 0]
        white_sh = torch.zeros_like(selected.harmonics)
        white_sh[..., 0] = 0.5 / math.sqrt(1 / (4 * math.pi))
        white = Gaussians(selected.means, selected.covariances, white_sh,
                          selected.opacities)
        alpha = decoder(white, *args).color[0, 0].mean(0).clamp(0, 1)
        centers = projected_centers(selected.means[0].expand(3, -1), cameras[0],
                                    target["intrinsics"][0], shape)
    rgb = rgb[0].permute(0, 2, 3, 1).detach().cpu().numpy()
    alpha = alpha.detach().cpu().numpy()
    color = isolated.permute(1, 2, 0).detach().cpu().numpy()
    background = 0.5 + 0.5 * rgb[view].mean(-1, keepdims=True)
    return {
        "centers": centers, "context_rgb": rgb, "alpha": alpha,
        "isolated": np.clip(color + (1 - alpha[..., None]) * background, 0, 1),
        "novel": output[0, 2].permute(1, 2, 0).detach().cpu().numpy(),
        "selected": {name: getattr(selected, name).cpu() for name in FIELDS},
        "opacity": float(selected.opacities[0, 0]),
        "cameras": cameras[0].detach().cpu().numpy(),
    }


def crop_bounds(points, shape):
    import numpy as np

    height, width = shape
    points = np.asarray(points).reshape(-1, 2)
    points = points[np.isfinite(points).all(-1)]
    points = points[(points[:, 0] >= 0) & (points[:, 0] < width)
                    & (points[:, 1] >= 0) & (points[:, 1] < height)]
    if not len(points):
        points = np.array([[(width - 1) / 2, (height - 1) / 2]])
    low, high = points.min(0) - 12, points.max(0) + 12
    center, half = (low + high) / 2, np.maximum((high - low) / 2, 32)
    return (max(-.5, center[0] - half[0]), min(width - .5, center[0] + half[0]),
            max(-.5, center[1] - half[1]), min(height - .5, center[1] + half[1]))


def plot_comparison(sample, conditions, evidence, directory):
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    names = ("baseline", "constant", "learned")
    labels = ("Baseline", "Ours: constant budget", "Ours: learned budget")
    rgb = conditions["learned"]["context_rgb"]
    shape = rgb.shape[1:3]
    xy = np.column_stack((evidence["source_x"], evidence["source_y"]))
    vmax = max(evidence["weight_constant"].max(),
               evidence["weight_learned"].max(), 1e-8)
    norm, cmap = Normalize(0, vmax), plt.get_cmap("magma")
    evidence_bounds = []
    for view in range(2):
        points = list(xy[evidence["source_view"] == view])
        points.extend(conditions[name]["centers"][view] for name in names)
        if view == sample["view"]:
            points.append(sample["pixel_xy"])
        evidence_bounds.append(crop_bounds(points, shape))
    source_points = [sample["pixel_xy"]]
    for condition in conditions.values():
        source_points.append(condition["centers"][sample["view"]])
        ys, xs = np.nonzero(condition["alpha"] > 0)
        if len(xs):
            source_points.extend(([xs.min(), ys.min()], [xs.max(), ys.max()]))
    source_bounds = crop_bounds(source_points, shape)
    novel_bounds = crop_bounds([c["centers"][2] for c in conditions.values()], shape)
    fig, axes = plt.subplots(3, 4, figsize=(17, 12))

    def show(axis, image, bounds):
        axis.imshow(np.clip(image, 0, 1), interpolation="nearest")
        axis.set_xlim(bounds[:2])
        axis.set_ylim(bounds[3], bounds[2])
        axis.tick_params(labelsize=7)

    def star(axis, center):
        if np.isfinite(center).all():
            axis.scatter(*center, marker="*", s=180, c="white",
                         edgecolors="black", linewidths=1, zorder=6)

    changes = abs(evidence["weight_learned"] - evidence["weight_constant"])
    for row, (name, label) in enumerate(zip(names, labels)):
        result = conditions[name]
        for view in range(2):
            axis = axes[row, view]
            show(axis, rgb[view], evidence_bounds[view])
            if name == "baseline":
                if view == sample["view"]:
                    x, y = sample["pixel_xy"]
                    axis.add_patch(Rectangle((x - 1.5, y - 1.5), 3, 3,
                                             fill=False, edgecolor="cyan"))
                else:
                    axis.text(.02, .03, "No direct head input in this view",
                              transform=axis.transAxes, fontsize=8,
                              bbox=dict(facecolor="white", alpha=.8))
            else:
                mask = evidence["source_view"] == view
                weights = evidence[f"weight_{name}"][mask]
                axis.scatter(*xy[mask].T, s=13, c="gray", zorder=3)
                axis.scatter(*xy[mask].T, s=350 * weights / vmax,
                             c=weights, cmap=cmap, norm=norm,
                             edgecolors="white", linewidths=.3, zorder=4)
                candidates = np.flatnonzero(mask)
                for i in candidates[np.argsort(changes[mask])[-2:]]:
                    axis.annotate(f"b={evidence[f'budget_{name}'][i]:.2f}", xy[i],
                                  xytext=(4, 5), textcoords="offset points",
                                  fontsize=7, bbox=dict(facecolor="white", alpha=.8))
            star(axis, result["centers"][view])
            if view == sample["view"]:
                axis.scatter(*sample["pixel_xy"], s=80, facecolors="none",
                             edgecolors="cyan", linewidths=1.4, zorder=7)
            if row == 0:
                axis.set_title(
                    f"Evidence: context {view}, frame {sample['context'][view]}"
                )
        axes[row, 0].set_ylabel(label)
        show(axes[row, 2], result["isolated"], source_bounds)
        level = result["opacity"] * math.exp(-2)
        if result["alpha"].min() < level < result["alpha"].max():
            axes[row, 2].contour(result["alpha"], levels=[level],
                                 colors="cyan", linewidths=.8)
        star(axes[row, 2], result["centers"][sample["view"]])
        axes[row, 2].set_xlabel(f"opacity = {result['opacity']:.4f}")
        show(axes[row, 3], result["novel"], novel_bounds)
        star(axes[row, 3], result["centers"][2])
    axes[0, 2].set_title("Actual isolated Gaussian / source context")
    axes[0, 3].set_title(f"Actual novel-view render / frame {sample['target'][0]}")
    fig.suptitle(f"{sample['id']}: {sample['label']} | {sample['scene']} | "
                 f"context {sample['view']}, pixel {sample['pixel_xy']}")
    fig.text(.04, .025,
             "Star: projected center; cyan circle: selected pixel. "
             "Marker area/color: actual incoming geometry weight; gray: candidates.\n"
             "Baseline box: structural 3x3 head input, not attribution. "
             "Cyan contour: raster alpha = opacity * exp(-2).\n"
             "Assignment/cameras are fixed between Ours conditions. "
             "Color has separate appearance paths; surface names are not masks.",
             fontsize=8)
    fig.tight_layout(rect=(0, .09, .95, .96))
    bar = fig.add_axes((.96, .23, .012, .45))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=bar,
                 label="Geometry contribution")
    fig.savefig(directory / "comparison.png", dpi=180)
    fig.savefig(directory / "comparison.pdf")
    plt.close(fig)


def compose_config(model, args, data_root, index_path, size):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    saved = args.ours_config if model == "ours" else args.baseline_config
    if saved:
        config = OmegaConf.load(saved)
    else:
        experiment = "re10k_budget_analysis" if model == "ours" else "re10k"
        with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
            config = compose(config_name="main", overrides=[
                f"+experiment={experiment}",
                "dataset/view_sampler@dataset.re10k.view_sampler=evaluation",
            ])
    config.mode, config.wandb.mode = "test", "disabled"
    config.model.encoder.pretrained_weights = ""
    config.dataset.re10k.roots = [str(data_root)]
    config.dataset.re10k.view_sampler = dict(
        name="evaluation", index_path=str(index_path), num_context_views=2,
    )
    config.dataset.re10k.input_image_shape = [size[1], size[0]]
    config.data_loader.test.num_workers = 0
    config.data_loader.test.batch_size = 1
    if model == "ours":
        config.model.encoder.moment_decoder.compile_kernels = False
        if args.budget_max is not None:
            config.model.encoder.moment_decoder.budget_max = args.budget_max
    return config


def run(args):
    config = read_json(args.samples)
    manifest_path = args.manifest or Path(config["manifest"])
    checkpoints = {
        "ours": args.ours_checkpoint or Path(config["ours_checkpoint"]),
        "baseline": args.baseline_checkpoint or Path(config["baseline_checkpoint"]),
    }
    # Preflight before importing CUDA libraries or creating any result files.
    for path in [*checkpoints.values(), manifest_path]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Server input not found: {path}. Run on the server with the "
                "checkpoints and original RE10K data mounted. No plots were generated."
            )
    manifest = read_json(manifest_path)
    index = read_json(args.index)
    selections = resolve_selections(config, index, manifest)
    root = Path(args.data_root or manifest["data_path"])
    if root.suffix == ".torch":
        root = root.parent
    if root.name == "test":
        root = root.parent
    if not (root / "test").is_dir():
        raise FileNotFoundError(f"Original test/ data missing: {root}; use --data-root")
    chunks = []
    for sample in selections:
        chunk = Path(sample["chunk_path"])
        if args.data_root or not chunk.is_file():
            chunk = root / "test" / chunk.name
        for path in (chunk, Path(sample["image_path"])):
            if not path.is_file():
                raise FileNotFoundError(path)
        chunks.append(chunk)

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.step_tracker import StepTracker
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.scripts.analysis_scale_opacity import (
        _align_target_extrinsics, _freeze, _to_device,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Actual inference requires CUDA and the Gaussian rasterizer")
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    selected_index = {
        s["scene"]: {**index[s["scene"]], "target": s["target"]} for s in selections
    }
    index_path = output / "selected_evaluation_index.json"
    write_json(index_path, selected_index)
    write_json(output / "selections.json", {"samples": selections})
    metadata = {
        "source": "real_checkpoint_inference", "complete": False,
        "checkpoints": {name: str(path) for name, path in checkpoints.items()},
        "manifest": str(manifest_path), "models": {},
        "pose_align_steps": args.pose_align_steps,
    }
    write_json(output / "run_metadata.json", metadata)
    results = {s["id"]: {} for s in selections}
    by_scene = {s["scene"]: s for s in selections}
    device = torch.device("cuda")

    def selected_chunks(dataset, stage):
        dataset.chunks = sorted(set(chunks))
        return dataset

    for model in ("baseline", "ours"):
        raw = compose_config(model, args, root, index_path,
                             selections[0]["image_size_wh"])
        budget_max = (
            float(raw.model.encoder.moment_decoder.budget_max) if model == "ours"
            else None
        )
        OmegaConf.save(raw, output / f"{model}_config.yaml")
        set_cfg(raw)
        cfg = load_typed_root_config(raw)
        if (cfg.model.encoder.num_surfaces != 1
                or cfg.model.encoder.gaussians_per_pixel != 1):
            raise ValueError("Analysis requires one Gaussian slot per pixel")
        print(f"[Checkpoint] Loading {model}: {checkpoints[model]}", flush=True)
        checkpoint = torch.load(
            checkpoints[model], map_location="cpu", weights_only=False
        )
        state = encoder_weights(checkpoint)
        encoder, _ = get_encoder(cfg.model.encoder)
        compatibility = "native baseline"
        if model == "ours":
            compatibility = restore_budget_path(
                encoder.gaussian_decoder, state, budget_max
            )
        encoder.load_state_dict(state, strict=True)
        step = int(checkpoint.get("global_step", 0))
        metadata["models"][model] = {
            "global_step": step, "strict_load": True, "encoder_tensors": len(state),
            "budget_max": budget_max, "compatibility": compatibility,
            "budget_setting_source": (
                "command line" if args.budget_max is not None else
                "saved training config" if args.ours_config else
                "original experiment snapshot (budget_max=2)"
            ) if model == "ours" else None,
            "inferred_scenes": [],
        }
        write_json(output / "run_metadata.json", metadata)
        del checkpoint, state
        encoder = _freeze(encoder.to(device))
        decoder = _freeze(get_decoder(cfg.model.decoder).to(device))
        if decoder.background_color.abs().max().item() != 0:
            raise ValueError("Isolated alpha rendering expects a black background")
        losses = _freeze(torch.nn.ModuleList(get_losses(cfg.loss)).to(device))
        loader = DataModule(cfg.dataset, cfg.data_loader, StepTracker(),
                            dataset_shim=selected_chunks).test_dataloader()
        done = set()
        for batch in loader:
            scene = batch["scene"][0]
            if scene not in by_scene or scene in done:
                continue
            sample = by_scene[scene]
            print(f"[Inference] {model}: {sample['id']} / {scene}", flush=True)
            batch = _to_device(batch, device)
            rgb = batch["context"]["image"]
            if batch["context"]["index"][0].tolist() != sample["context"]:
                raise ValueError(f"Context order changed: {scene}")
            with Image.open(sample["image_path"]) as image:
                png = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
            actual = rgb[0, sample["view"]].permute(1, 2, 0).cpu().numpy()
            if (png.shape != actual.shape
                    or not np.allclose(png, actual, atol=1/255 + 1e-7)):
                raise ValueError(f"Chosen PNG differs from the model input: {scene}")
            batch = encoder.get_data_shim()(batch)
            if model == "ours":
                original, constant, lt, ct = infer_budget_pair(
                    encoder, batch["context"], step
                )
                evidence = incoming_evidence(lt, ct, sample["slot"], rgb.shape[-2:])
            else:
                with torch.no_grad():
                    original = encoder(batch["context"], step)
            display = {**batch, "target": {}}
            for key in ("image", "intrinsics", "extrinsics", "near", "far", "index"):
                context_value = rgb if key == "image" else batch["context"][key]
                display["target"][key] = torch.cat(
                    (context_value, batch["target"][key][:, :1]), dim=1
                )
            cameras = _align_target_extrinsics(
                decoder, losses, display, original, step, args.pose_align_steps > 0,
                args.pose_align_steps, cfg.test.rot_opt_lr, cfg.test.trans_opt_lr,
            )
            variants = ({"baseline": original} if model == "baseline" else
                        {"learned": original, "constant": constant})
            print(f"[Render] {model}: {sample['id']}", flush=True)
            for name, gaussians in variants.items():
                results[sample["id"]][name] = render_condition(
                    decoder, gaussians, sample, display, cameras, rgb
                )
            metadata["models"][model]["inferred_scenes"].append(scene)
            write_json(output / "run_metadata.json", metadata)
            if model == "ours":
                directory = output / sample["id"]
                directory.mkdir()
                conditions = results[sample["id"]]
                np.savez_compressed(directory / "evidence.npz", **evidence)
                columns = [key for key, value in evidence.items() if value.ndim == 1]
                with (directory / "evidence.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as file:
                    writer = csv.DictWriter(file, fieldnames=columns)
                    writer.writeheader()
                    for i in range(len(evidence["source_index"])):
                        writer.writerow({
                            key: evidence[key][i].item() for key in columns
                        })
                torch.save(
                    {name: value["selected"] for name, value in conditions.items()},
                    directory / "selected_gaussians.pt",
                )
                attributes = {
                    "source": "real_checkpoint_inference", "sample": sample,
                    "assignment_identical": True,
                    "mean_budget": float(lt["budget"].mean()),
                    "learned_total_mass": float(lt["mass"].sum()),
                    "constant_total_mass": float(ct["mass"].sum()),
                    "conditions": {
                        name: {key: value.tolist()
                               for key, value in result["selected"].items()}
                        for name, result in conditions.items()
                    },
                    "surface_purity_computed": False,
                }
                write_json(directory / "attributes.json", attributes)
                for name, result in conditions.items():
                    for kind in ("isolated", "novel"):
                        pixels = np.clip(result[kind] * 255, 0, 255).astype(np.uint8)
                        Image.fromarray(pixels).save(directory / f"{name}_{kind}.png")
                    np.savez_compressed(
                        directory / f"{name}_projection.npz",
                        centers=result["centers"], cameras=result["cameras"],
                        alpha=result["alpha"],
                    )
                gt = display["target"]["image"][0, 2].permute(1, 2, 0).cpu().numpy()
                Image.fromarray((np.clip(gt, 0, 1) * 255).astype(np.uint8)).save(
                    directory / "novel_ground_truth.png"
                )
                plot_comparison(sample, conditions, evidence, directory)
                print(f"[Saved] {directory / 'comparison.png'}", flush=True)
                del lt, ct, constant
            done.add(scene)
            del original, variants, gaussians, batch, display, rgb, cameras
            if done == set(by_scene):
                break
        if done != set(by_scene):
            raise RuntimeError(
                f"Dataset skipped selected scenes: {set(by_scene) - done}"
            )
        del encoder, decoder, losses, loader
        torch.cuda.empty_cache()
    metadata["complete"] = True
    write_json(output / "run_metadata.json", metadata)
    print(f"Real inference completed: {output}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--samples", type=Path,
                        default=ROOT / "assets/budget_analysis_samples.json")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--ours-checkpoint", type=Path)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--ours-config", type=Path, help="Saved full training YAML")
    parser.add_argument("--baseline-config", type=Path)
    parser.add_argument("--budget-max", type=float)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--index", type=Path,
                        default=ROOT / "assets/evaluation_index_re10k.json")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/budget_assignment_analysis"))
    parser.add_argument("--pose-align-steps", type=int, default=100)
    args = parser.parse_args(argv)
    if args.pose_align_steps < 0:
        parser.error("--pose-align-steps must be nonnegative")
    sys.path.insert(0, str(ROOT))
    try:
        run(args)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
