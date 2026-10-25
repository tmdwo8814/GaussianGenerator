"""Visualize final Gaussian scale and opacity from two complete frozen encoders.

Run at the repository root: python -m exp3.plot_scale_opacity --help
"""

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
FIELDS = ("decoder", "scene", "overlap", "N_total", "N_active", "N_selected", "median_log_s_max")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
                    + "\n", encoding="utf-8")


def overlap_group(value):
    if isinstance(value, str) and value in ("small", "medium", "large"):
        return value
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Overlap must be finite and within [0,1]: {value}")
    return "small" if value <= 0.3 else "medium" if value <= 0.55 else "large"


def select_scenes(index, count, seed):
    valid = {scene: entry for scene, entry in index.items() if entry is not None}
    groups = {tag: [] for tag in ("small", "medium", "large")}
    for scene, entry in valid.items():
        if len(entry["context"]) != 2 or not entry["target"]:
            raise ValueError(f"{scene}: expected two context views and target entries")
        groups[overlap_group(entry["overlap"])].append(scene)
    if not valid or count < 0:
        raise ValueError("Index must contain valid scenes; num-scenes must be >=0")
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
    result.add_argument("--num-scenes", type=int, default=500, help="0: entire valid index")
    result.add_argument("--sample-size", type=int, default=60000)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--num-workers", type=int, default=4)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--include-inactive", action="store_true")
    result.add_argument("--save-gaussians", action="store_true")
    modes = result.add_mutually_exclusive_group()
    modes.add_argument("--plot-only", action="store_true")
    modes.add_argument("--resume", action="store_true")
    for key in ("override", "original-override", "ours-override"):
        result.add_argument(f"--{key}", action="append", default=[], metavar="KEY=VALUE")
    return result


def compose_config(args, name, selected_index):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    experiment = "re10k" if name == "original" else "re10k_moment"
    extra = args.original_override if name == "original" else args.ours_override
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        cfg = compose(config_name="main", overrides=[f"+experiment={experiment}",
                                                     *args.override, *extra])
    cfg.mode, cfg.wandb.mode, cfg.seed = "test", "disabled", args.seed
    cfg.model.encoder.pretrained_weights = ""
    if cfg.model.encoder.name != "noposplat" or not cfg.model.encoder.pose_free:
        raise ValueError("Expected the supplied pose-free noposplat experiments")
    expected = "dpt_gs" if name == "original" else "moment"
    if cfg.model.encoder.gs_params_head_type != expected:
        raise ValueError(f"{name} must use {expected}")
    if name == "ours":
        # Skip compilation startup, retaining the complete trained architecture.
        cfg.model.encoder.moment_decoder.compile_kernels = False
    if len(cfg.dataset) != 1 or "re10k" not in cfg.dataset:
        raise ValueError("Expected one RE10K dataset")
    if list(cfg.dataset.re10k.input_image_shape) != [256, 256]:
        raise ValueError("Both models must use 256x256 input images")
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
    """Restore EVERY encoder parameter, including the full appearance/SH path."""
    import torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {key[len("encoder."):]: value for key, value in checkpoint["state_dict"].items()
             if key.startswith("encoder.")}
    if not state:
        raise ValueError(f"No Lightning encoder.* weights in {checkpoint_path}")
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
    omega.save(cfg_dict, directory / f"config_{name}.yaml")
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    encoder, _ = get_encoder(cfg.model.encoder)
    step = restore_encoder(encoder, getattr(args, f"{name}_checkpoint"))
    write_json(directory / f"checkpoint_load_{name}.json", {
        "strict": True, "all_encoder_weights": True, "checkpoint_step": step,
        "omitted_appearance_keys": [],
    })
    encoder = encoder.to(args.device).eval().requires_grad_(False)
    shim = get_data_shim(encoder)
    loader = DataModule(cfg.dataset, cfg.data_loader).test_dataloader()
    sampler = PrioritySample(args.sample_size)
    counts = dict.fromkeys(("N_total", "N_active", "N_selected"), 0)
    ordered = {scene: i for i, scene in enumerate(scenes)}
    seen, scene = set(), "__loader__"
    try:
        for batch in loader:
            (scene,) = batch["scene"]
            if scene not in scenes or scene in seen:
                raise ValueError(f"Unexpected/duplicate scene: {scene}")
            seen.add(scene)
            batch["context"] = move_tensors(batch["context"], args.device)
            with torch.no_grad():
                gaussians = encoder(shim(batch)["context"], step)
            opacity = gaussians.opacities[0].detach().float().cpu().numpy()
            if len(opacity) != 2 * 256 * 256:
                raise ValueError("Expected one Gaussian per context pixel")
            if not np.isfinite(opacity).all() or np.any((opacity < 0) | (opacity > 1)):
                raise ValueError("Invalid final opacity")
            active = opacity >= 1 / 255
            selected = np.ones(len(opacity), dtype=bool) if args.include_inactive else active
            covariance = gaussians.covariances[0].detach().float().cpu().numpy()
            x = log_max_scale(covariance[selected])
            digest = hashlib.sha256(f"{args.seed}:{scene}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            sampler.add(rng.random(len(opacity))[selected], log_s_max=x,
                        opacity=opacity[selected], scene_index=np.full(len(x), ordered[scene]),
                        gaussian_index=np.flatnonzero(selected))
            row = {"decoder": name, "scene": scene,
                   "overlap": overlap_group(scenes[scene]["overlap"]),
                   "N_total": len(opacity), "N_active": int(active.sum()),
                   "N_selected": len(x),
                   "median_log_s_max": float(np.median(x)) if len(x) else ""}
            with (directory / "per_scene.csv").open("a", newline="", encoding="utf-8") as file:
                csv.DictWriter(file, fieldnames=FIELDS).writerow(row)
            for key in counts:
                counts[key] += row[key]
            if args.save_gaussians:
                dump = directory / "dumps" / name
                dump.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dump / f"{ordered[scene]:06d}.npz", means=gaussians.means[0].cpu().numpy(),
                                    cov=covariance, opacity=opacity,
                                    harmonics=gaussians.harmonics[0].cpu().numpy())
            del gaussians, covariance
            if len(seen) % 10 == 0 or len(seen) == len(scenes):
                print(f"[{name}] {len(seen)}/{len(scenes)} scenes; "
                      f"{counts['N_selected']:,} selected Gaussians", flush=True)
    except Exception:
        append_failure(directory, name, scene, traceback.format_exc())
        raise
    missing = sorted(set(scenes) - seen)
    for scene in missing:
        append_failure(directory, name, scene, "Selected scene not yielded by dataset")
    if missing:
        raise RuntimeError(f"{name}: {len(missing)} selected scenes missing; see failures.txt")
    np.savez_compressed(directory / "samples" / f"{name}.npz", **sampler.arrays())
    del encoder, loader, shim
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {**counts, "checkpoint_step": step, "scenes": len(seen)}


def resume_metadata(args, directory, scenes):
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    operational = {"output_dir", "resume", "plot_only", "num_workers", "device"}
    for key, value in vars(args).items():
        value = str(value) if isinstance(value, Path) else value
        if key not in operational and metadata["arguments"].get(key) != value:
            raise ValueError(f"Cannot resume with changed {key}")
    if metadata["index_sha256"] != hashlib.sha256(args.index.read_bytes()).hexdigest():
        raise ValueError("Evaluation index changed")
    saved = json.loads((directory / "scenes.json").read_text(encoding="utf-8"))
    if list(saved.items()) != list(scenes.items()):
        raise ValueError("Saved scene/view list differs")
    with (directory / "per_scene.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    completed = metadata["counts"]
    if not set(completed).issubset({"original", "ours"}):
        raise ValueError("Unknown cached decoder")
    for name, counts in completed.items():
        if name == "ours":
            info = json.loads((directory / "checkpoint_load_ours.json").read_text(encoding="utf-8"))
            if not info.get("all_encoder_weights") or not info.get("strict"):
                raise ValueError("Cached ours used a different restore policy; use a fresh output directory")
        cached = [row for row in rows if row["decoder"] == name]
        if (counts["scenes"] != len(scenes) or len(cached) != len(scenes)
                or {row["scene"] for row in cached} != set(scenes)):
            raise ValueError(f"Incomplete cached scenes for {name}")
        for key in ("N_total", "N_active", "N_selected"):
            if sum(int(row[key]) for row in cached) != counts[key]:
                raise ValueError(f"Invalid cached {key} for {name}")
        with np.load(directory / "samples" / f"{name}.npz", allow_pickle=False) as data:
            for key in ("log_s_max", "opacity", "scene_index", "gaussian_index"):
                if (data[key].shape != (min(args.sample_size, counts["N_selected"]),)
                        or not np.isfinite(data[key]).all()):
                    raise ValueError(f"Invalid cached {key} for {name}")
    temporary = directory / "per_scene.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(row for row in rows if row["decoder"] in completed)
    temporary.replace(directory / "per_scene.csv")
    return metadata


def main(argv=None):
    args = parser().parse_args(argv)
    directory = args.output_dir.resolve()
    if args.plot_only:
        make_figure(directory)
        return
    if args.num_scenes < 0 or args.sample_size < 1 or args.num_workers < 0 or args.seed < 0:
        raise ValueError("Invalid scene/sample/worker count or seed")
    for path in (args.original_checkpoint, args.ours_checkpoint, args.index):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)
    index = json.loads(args.index.read_text(encoding="utf-8"))
    scenes = select_scenes(index, args.num_scenes, args.seed)
    if args.resume:
        metadata = resume_metadata(args, directory, scenes)
    else:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Use an empty output directory: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "samples").mkdir()
        write_json(directory / "scenes.json", scenes)
        write_json(directory / "index_exclusions.json", {
            "null_entries": sorted(s for s, e in index.items() if e is None),
            "not_selected_by_fixed_subset": sorted(s for s, e in index.items() if e is not None and s not in scenes),
        })
        (directory / "failures.txt").touch()
        with (directory / "per_scene.csv").open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(FIELDS)
        metadata = {
            "arguments": {key: str(v) if isinstance(v, Path) else v for key, v in vars(args).items()},
            "index_sha256": hashlib.sha256(args.index.read_bytes()).hexdigest(),
            "num_scenes": len(scenes), "counts": {},
            "scale_definition": "log10(max_k sqrt(eig_k(final covariance)))",
            "opacity_definition": "final encoder opacity, unchanged by rasterizer wrapper",
            "coordinate_system": "encoder canonical world frame, before renderer near rescaling",
            "selection": "all slots" if args.include_inactive else "final opacity >= 1/255",
            "comparison": "user-specified converged checkpoints; unequal steps accepted",
        }
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, check=False).stdout.strip()
    versions = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    (directory / ("env_resume.txt" if args.resume else "env.txt")).write_text(
        f"git_commit={commit}\npython={platform.python_version()}\nseed={args.seed}\n"
        + "\n".join(versions) + "\n", encoding="utf-8")
    metadata.setdefault("git_commit", commit)
    metadata.pop("ours_extraction", None)  # Remove an older geometry-only policy label on resume.
    metadata["encoder_restore"] = "all encoder weights, including appearance/SH, strict=True"
    if args.resume:
        metadata.setdefault("resumes", []).append({"git_commit": commit})
    write_json(directory / "metadata.json", metadata)
    print(f"[exp3] output={directory}; scenes={len(scenes)}; resume={args.resume}", flush=True)
    (directory / "status.txt").write_text("RUNNING\n", encoding="utf-8")
    try:
        for name in ("original", "ours"):
            if name in metadata["counts"]:
                print(f"[{name}] reusing completed samples ({metadata['counts'][name]['scenes']} scenes)", flush=True)
                continue
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
