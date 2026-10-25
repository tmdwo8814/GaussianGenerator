"""Generate two frozen Gaussian scenes per index entry and plot scale vs opacity.

Run from the repository root: python -m exp3.plot_scale_opacity --help
Heavy repository/CUDA imports happen only when extracting, not in --plot-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from exp3.plot import make_figure
from exp3.scale_opacity import PrioritySample, log_max_scale


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ("/nas2/data/tmdwo8814/research/GaussianGenerator/outputs/exp_noposplat/"
            "2026-10-12_16-46-59/checkpoints/epoch_0-step_60000.ckpt")
OURS = ("/nas2/data/tmdwo8814/research/GaussianGenerator/outputs/exp_final_plus/"
        "2026-10-16_18-35-06/checkpoints/epoch_2-step_140000.ckpt")


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
                    + "\n", encoding="utf-8")


def overlap_group(value):
    if isinstance(value, str) and value in ("small", "medium", "large"):
        return value
    value = float(value)
    if 0.05 <= value <= 0.3:
        return "small"
    if 0.3 < value <= 0.55:
        return "medium"
    if 0.55 < value <= 0.8:
        return "large"
    raise ValueError(f"Unexpected overlap: {value}")


def select_scenes(index, count, seed):
    """Full valid index or fixed, equally stratified subset without silent drops."""
    valid = {scene: entry for scene, entry in index.items() if entry is not None}
    groups = {name: [] for name in ("small", "medium", "large")}
    for scene, entry in valid.items():
        if len(entry["context"]) != 2:
            raise ValueError(f"{scene}: these experiments require two context views")
        if not entry["target"]:
            raise ValueError(f"{scene}: repository dataset requires a target entry")
        groups[overlap_group(entry["overlap"])].append(scene)
    if not valid:
        raise ValueError("No valid scenes in evaluation index")
    if count < 0:
        raise ValueError("num-scenes must be nonnegative; 0 means all scenes")
    if count == 0 or count >= len(valid):
        return dict(sorted(valid.items()))
    rng = random.Random(seed)
    for values in groups.values():
        values.sort()
        rng.shuffle(values)
    selected = []
    while len(selected) < count:
        for values in groups.values():
            if values and len(selected) < count:
                selected.append(values.pop())
    return {scene: valid[scene] for scene in sorted(selected)}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--original-checkpoint", default=ORIGINAL, type=Path)
    result.add_argument("--ours-checkpoint", default=OURS, type=Path)
    result.add_argument("--data-root", default="/local_datasets/3dgs/re10k/re10k", type=Path)
    result.add_argument("--index", default=ROOT / "assets/evaluation_index_re10k.json", type=Path)
    result.add_argument("--output-dir", type=Path,
                        default=ROOT / "exp3/output" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    result.add_argument("--num-scenes", default=500, type=int, help="0: entire non-null index")
    result.add_argument("--sample-size", default=60000, type=int)
    result.add_argument("--seed", default=0, type=int)
    result.add_argument("--num-workers", default=4, type=int)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--include-inactive", action="store_true",
                        help="Default: use final opacity >= 1/255 only")
    result.add_argument("--save-gaussians", action="store_true",
                        help="Also save final means/covariance/opacity per scene (float32)")
    result.add_argument("--plot-only", action="store_true", help="Replot cached output-dir samples")
    result.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                        help="Additional Hydra override shared by both models; repeat as needed")
    result.add_argument("--original-override", action="append", default=[], metavar="KEY=VALUE")
    result.add_argument("--ours-override", action="append", default=[], metavar="KEY=VALUE")
    return result


def compose_config(args, name, selected_index):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    experiment = "re10k" if name == "original" else "re10k_moment"
    extra = args.original_override if name == "original" else args.ours_override
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        cfg = compose(config_name="main", overrides=[f"+experiment={experiment}",
                                                     *args.override, *extra])
    cfg.mode = "test"
    cfg.wandb.mode = "disabled"
    cfg.seed = args.seed
    cfg.model.encoder.pretrained_weights = ""
    # torch.compile is unnecessary for this inference-only visualization.
    if cfg.model.encoder.gs_params_head_type == "moment":
        cfg.model.encoder.moment_decoder.compile_kernels = False
    if len(cfg.dataset) != 1 or "re10k" not in cfg.dataset:
        raise ValueError("This entrypoint supports the supplied NoPoSplat RE10K experiments")
    cfg.dataset.re10k.roots = [str(args.data_root.resolve())]
    cfg.dataset.re10k.view_sampler = {
        "name": "evaluation", "index_path": str(selected_index), "num_context_views": 2,
    }
    cfg.data_loader.test.batch_size = 1
    cfg.data_loader.test.num_workers = args.num_workers
    cfg.data_loader.test.persistent_workers = False
    cfg.data_loader.test.seed = args.seed
    return cfg, OmegaConf


def restore_encoder(encoder, checkpoint_path):
    """Strictly restore all encoder weights; never use MASt3R initialization."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    state = {key[len("encoder."):]: value for key, value in state.items()
             if key.startswith("encoder.")}
    if not state:
        raise ValueError(f"Checkpoint has no Lightning encoder.* state: {checkpoint_path}")
    encoder.load_state_dict(state, strict=True)
    return int(checkpoint.get("global_step", 0))


def move_tensors(value, device):
    import torch
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensors(item, device) for item in value]
    return value


def append_failure(directory, name, scene, error):
    with (directory / "failures.txt").open("a", encoding="utf-8") as file:
        file.write(f"{name}\t{scene}\t{error}\n")


def extract(args, name, scenes, directory):
    import torch
    from lightning.pytorch import seed_everything
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule, get_data_shim
    from src.global_cfg import set_cfg
    from src.model.encoder import get_encoder

    seed_everything(args.seed, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg_dict, omega = compose_config(args, name, directory / "scenes.json")
    if cfg_dict.model.encoder.name != "noposplat" or not cfg_dict.model.encoder.pose_free:
        raise ValueError("Expected the supplied pose-free noposplat encoder")
    expected = "dpt_gs" if name == "original" else "moment"
    if cfg_dict.model.encoder.gs_params_head_type != expected:
        raise ValueError(f"{name} must use {expected}")
    omega.save(cfg_dict, directory / f"config_{name}.yaml")
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    encoder, _ = get_encoder(cfg.model.encoder)
    step = restore_encoder(encoder, getattr(args, f"{name}_checkpoint"))
    encoder = encoder.to(args.device).eval().requires_grad_(False)
    shim = get_data_shim(encoder)
    loader = DataModule(cfg.dataset, cfg.data_loader).test_dataloader()
    sampler = PrioritySample(args.sample_size)
    seen = set()
    ordered = {scene: i for i, scene in enumerate(scenes)}
    active_cutoff = 1 / 255
    counts = dict.fromkeys(("N_total", "N_active", "N_selected"), 0)
    scene = "__loader__"
    try:
        for batch in loader:
            (scene,) = batch["scene"]
            if scene not in scenes or scene in seen:
                raise ValueError(f"Unexpected/duplicate scene: {scene}")
            seen.add(scene)
            if tuple(batch["context"]["image"].shape[-2:]) != (256, 256):
                raise ValueError("Both supplied experiments must use 256x256 images")
            # Only context tensors go to GPU; no target rendering or pose fitting.
            batch["context"] = move_tensors(batch["context"], args.device)
            with torch.no_grad():
                gaussians = encoder(shim(batch)["context"], step)
            opacity = gaussians.opacities[0].detach().float().cpu().numpy()
            if not np.isfinite(opacity).all() or np.any((opacity < 0) | (opacity > 1)):
                raise ValueError("Non-finite or out-of-range final opacity")
            if len(opacity) != 2 * 256 * 256:
                raise ValueError("Expected one Gaussian slot per context pixel")
            active = opacity >= active_cutoff
            selected = np.ones(len(opacity), dtype=bool) if args.include_inactive else active
            covariance = gaussians.covariances[0].detach().float().cpu().numpy()
            x = log_max_scale(covariance[selected])
            digest = hashlib.sha256(f"{args.seed}:{scene}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            priorities = rng.random(len(opacity))[selected]
            sampler.add(priorities, log_s_max=x, opacity=opacity[selected],
                        scene_index=np.full(len(x), ordered[scene]),
                        gaussian_index=np.flatnonzero(selected))
            row = {"decoder": name, "scene": scene,
                   "overlap": overlap_group(scenes[scene]["overlap"]),
                   "N_total": len(opacity), "N_active": int(active.sum()),
                   "N_selected": len(x),
                   "median_log_s_max": float(np.median(x)) if len(x) else ""}
            with (directory / "per_scene.csv").open("a", newline="", encoding="utf-8") as file:
                csv.DictWriter(file, fieldnames=row).writerow(row)
            for key in counts:
                counts[key] += row[key]
            if args.save_gaussians:
                dump = directory / "dumps" / name
                dump.mkdir(parents=True, exist_ok=True)
                # Use a numeric filename; scene identity lives in scenes.json.
                np.savez_compressed(dump / f"{ordered[scene]:06d}.npz",
                                    means=gaussians.means[0].detach().float().cpu().numpy(),
                                    cov=covariance, opacity=opacity)
            del gaussians, covariance
            if len(seen) % 10 == 0 or len(seen) == len(scenes):
                print(f"[{name}] {len(seen)}/{len(scenes)} scenes; "
                      f"{counts['N_selected']:,} selected Gaussians", flush=True)
    except Exception:
        append_failure(directory, name, scene, traceback.format_exc())
        raise
    missing = sorted(set(scenes) - seen)
    for scene in missing:
        append_failure(directory, name, scene, "Selected scene was not yielded by the dataset")
    if missing:
        raise RuntimeError(f"{name}: {len(missing)} selected scenes missing; see failures.txt")
    np.savez_compressed(directory / "samples" / f"{name}.npz", **sampler.arrays())
    del encoder, loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {**counts, "checkpoint_step": step, "scenes": len(seen)}


def main(argv=None):
    args = parser().parse_args(argv)
    directory = args.output_dir.resolve()
    if args.plot_only:
        make_figure(directory)
        print(directory / "figs/figC_scale_opacity.png")
        return
    if args.num_workers < 0 or args.sample_size < 1 or args.num_scenes < 0 or args.seed < 0:
        raise ValueError("Invalid worker count, sample size, scene count, or seed")
    for path in (args.original_checkpoint, args.ours_checkpoint, args.index):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "samples").mkdir()
    index = json.loads(args.index.read_text(encoding="utf-8"))
    scenes = select_scenes(index, args.num_scenes, args.seed)
    write_json(directory / "scenes.json", scenes)
    write_json(directory / "index_exclusions.json", {
        "null_entries": sorted(scene for scene, entry in index.items() if entry is None),
        "not_selected_by_fixed_subset": sorted(set(index) - set(scenes)
                                               - {s for s, e in index.items() if e is None}),
    })
    (directory / "failures.txt").touch()
    (directory / "status.txt").write_text("RUNNING\n", encoding="utf-8")
    fields = ("decoder", "scene", "overlap", "N_total", "N_active", "N_selected", "median_log_s_max")
    with (directory / "per_scene.csv").open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(fields)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, check=False).stdout.strip()
    versions = sorted(f"{item.metadata['Name']}=={item.version}"
                      for item in importlib.metadata.distributions())
    (directory / "env.txt").write_text(
        f"git_commit={commit}\npython={platform.python_version()}\nseed={args.seed}\n"
        + "\n".join(versions) + "\n", encoding="utf-8")
    metadata = {
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "index_sha256": hashlib.sha256(args.index.read_bytes()).hexdigest(),
        "git_commit": commit, "num_scenes": len(scenes),
        "scale_definition": "log10(max_k sqrt(eig_k(final covariance)))",
        "coordinate_system": "encoder canonical world frame, before renderer near rescaling",
        "opacity_definition": "final encoder output passed unchanged to the rasterizer",
        "selection": "all slots" if args.include_inactive else "final opacity >= 1/255",
        "sampling": "uniform pooled Gaussian priorities, fixed seed and scene identity",
        "comparison": "user-specified converged checkpoints; unequal steps accepted",
        "counts": {},
    }
    write_json(directory / "metadata.json", metadata)
    try:
        for name in ("original", "ours"):
            metadata["counts"][name] = extract(args, name, scenes, directory)
            write_json(directory / "metadata.json", metadata)
        make_figure(directory)
    except Exception:
        append_failure(directory, "run", "__run__", traceback.format_exc())
        (directory / "status.txt").write_text("FAILED\n", encoding="utf-8")
        raise
    (directory / "status.txt").write_text("COMPLETE\n", encoding="utf-8")
    print(f"Saved figure and samples to {directory}", flush=True)


if __name__ == "__main__":
    main()
