"""Evaluate RE10K-trained Final++ and NoPoSplat on identical DL3DV test views.

Run from the repository root: python -m scripts.eval_re10k2dl3dv --gpus 0 1
The launcher and --dry-run do not import the CUDA/model stack.
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {"ours": "re10k_moment", "baseline": "re10k"}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def fixed_views(num_frames: int, context_gap: int, num_targets: int):
    """Match DL3DV's bounded-test context [0, gap], with novel targets only."""
    if context_gap < 2 or not 1 <= num_targets < context_gap:
        raise ValueError("Require context_gap >= 2 and 1 <= num_targets < context_gap")
    if num_frames <= context_gap:
        return None
    return {
        "context": [0, context_gap],
        "target": [i * context_gap // (num_targets + 1)
                   for i in range(1, num_targets + 1)],
    }


def read_view_index(path: Path, dataset_root: Path) -> dict:
    index = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(index, dict) or not index:
        raise ValueError(f"Empty or invalid evaluation index: {path}")
    scene_index = json.loads((dataset_root / "test/index.json").read_text(encoding="utf-8"))
    for scene, entry in index.items():
        if scene not in scene_index:
            raise ValueError(f"Evaluation scene {scene!r} is absent from DL3DV test/index.json")
        if entry is None:
            continue
        if not isinstance(entry, dict):
            raise ValueError(
                f"{path} must contain scene -> {{context: [...], target: [...]}}. "
                "DL3DV's test/index.json maps scenes to chunks; it is not a view index."
            )
        context, target = entry.get("context"), entry.get("target")
        if (not isinstance(context, list) or len(context) != 2
                or not isinstance(target, list) or not target):
            raise ValueError(f"Scene {scene}: require exactly two context views and nonempty targets")
        indices = context + target
        if any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != len(indices):
            raise ValueError(f"Scene {scene}: view indices must be distinct nonnegative integers")
    if not any(entry is not None for entry in index.values()):
        raise ValueError(f"No evaluable scenes in {path}")
    return index


def generate_view_index(dataset_root: Path, path: Path, context_gap: int, num_targets: int):
    # Read one chunk at a time on CPU; no image decoding or GPU is needed.
    import torch

    scene_index = json.loads((dataset_root / "test/index.json").read_text(encoding="utf-8"))
    chunks = sorted((dataset_root / "test").glob("*.torch"))
    if not chunks:
        raise FileNotFoundError(f"No .torch chunks in {dataset_root / 'test'}")
    index = {}
    for number, chunk_path in enumerate(chunks, 1):
        chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)
        for example in chunk:
            scene = example["key"]
            if scene in index:
                raise ValueError(f"Duplicate scene in DL3DV chunks: {scene}")
            if len(example["images"]) != len(example["cameras"]):
                raise ValueError(f"Scene {scene}: image/camera counts differ")
            index[scene] = fixed_views(len(example["cameras"]), context_gap, num_targets)
        del chunk
        print(f"[CrossDomain] Indexed chunk {number}/{len(chunks)}", flush=True)
    if set(index) != set(scene_index):
        raise ValueError("DL3DV test/index.json and chunk scene keys disagree")
    if not any(entry is not None for entry in index.values()):
        raise ValueError("No scene has enough frames; reduce --context-gap")
    write_json(path, index)
    write_json(path.with_suffix(".protocol.json"), {
        "dataset_root": str(dataset_root), "protocol": "fixed_interval_novel_views",
        "context_gap": context_gap, "num_targets": num_targets,
        "num_scenes": len(index),
        "too_short_scenes": sorted(scene for scene, entry in index.items() if entry is None),
        "note": "Custom deterministic protocol; not an official DL3DV benchmark index.",
    })
    return read_view_index(path, dataset_root)


def compose_config(label, checkpoint, dataset_root, index_path, output_dir, args):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "config")):
        # Keep the complete RE10K model/loss configuration (including scale
        # invariance); remove the RE10K dataset and add only the DL3DV loader.
        config = compose(config_name="main", overrides=[
            f"+experiment={EXPERIMENTS[label]}",
            "~dataset.re10k",
            "+dataset@dataset.dl3dv=dl3dv",
            "dataset/view_sampler@dataset.dl3dv.view_sampler=evaluation",
        ])
    config.mode = "test"
    config.dataset.dl3dv.roots = [str(dataset_root)]
    # DL3DV's optional bounded-specific defaults also merge sampler keys.
    # Replace the whole node so no training sampler settings survive.
    config.dataset.dl3dv.view_sampler = {
        "name": "evaluation", "index_path": str(index_path), "num_context_views": 2,
    }
    config.checkpointing.load = str(checkpoint)
    config.model.encoder.pretrained_weights = ""
    config.trainer.num_nodes = 1
    config.trainer.auto_eval = False
    config.wandb.mode = "disabled"
    config.wandb.name = label
    config.seed = args.seed
    config.data_loader.test.batch_size = 1
    config.data_loader.test.num_workers = args.num_workers
    config.test.output_path = str(output_dir / "renders")
    config.test.compute_scores = True
    config.test.align_pose = not args.no_align_pose
    config.test.save_image = args.save_image
    config.test.save_compare = args.save_image
    config.test.save_video = False
    OmegaConf.resolve(config)
    return config


def evaluate_worker(config_path: Path) -> None:
    from omegaconf import OmegaConf
    # Use exactly the same factories, type checking, and test_step as src.main.
    from src.main import (
        DataModule, ModelWrapper, Trainer, get_decoder, get_encoder, get_losses,
        load_typed_root_config, set_cfg, torch,
    )
    from lightning.pytorch import Callback

    config = OmegaConf.load(config_path)
    set_cfg(config)
    cfg = load_typed_root_config(config)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each evaluation worker requires exactly one visible CUDA GPU")
    torch.manual_seed(cfg.seed)
    index_path = Path(config.dataset.dl3dv.view_sampler.index_path)
    index = read_view_index(index_path, Path(config.dataset.dl3dv.roots[0]))
    expected = {scene: entry for scene, entry in index.items() if entry is not None}
    evaluated = {}

    class CoverageCallback(Callback):
        def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            scene = batch["scene"][0]
            if scene in evaluated:
                raise RuntimeError(f"Scene evaluated more than once: {scene}")
            evaluated[scene] = {
                "context": batch["context"]["index"][0].tolist(),
                "target": batch["target"]["index"][0].tolist(),
            }

    trainer = Trainer(
        accelerator="gpu", devices=1, num_nodes=1, strategy="auto",
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        inference_mode=not cfg.test.align_pose, callbacks=[CoverageCallback()],
    )
    encoder, visualizer = get_encoder(cfg.model.encoder)
    model = ModelWrapper(
        cfg.optimizer, cfg.test, cfg.train, encoder, visualizer,
        get_decoder(cfg.model.decoder), get_losses(cfg.loss), step_tracker=None,
    )
    model.strict_loading = True
    # Generated indices have no measured overlap; avoid misleading subgroups.
    if any(entry.get("overlap") is None for entry in expected.values()):
        model._test_overlap_tags = ()
    data_module = DataModule(cfg.dataset, cfg.data_loader, step_tracker=None)
    trainer.test(model, datamodule=data_module, ckpt_path=cfg.checkpointing.load)
    output_dir = config_path.parent
    write_json(output_dir / "evaluated_scenes.json", evaluated)
    missing = sorted(set(expected) - set(evaluated))
    write_json(output_dir / "coverage.json", {
        "requested_scenes": len(expected), "evaluated_scenes": len(evaluated),
        "skipped_scenes": missing,
    })
    if missing:
        raise RuntimeError(
            f"{len(missing)} requested scenes were skipped by the dataset loader. "
            f"Check {output_dir / 'coverage.json'} and console.log for image shape, "
            "FOV, baseline, or bad-image filtering; incomplete scores are not accepted."
        )
    if any(evaluated[scene] != {k: expected[scene][k] for k in ("context", "target")}
           for scene in evaluated):
        raise RuntimeError("Evaluated views differ from the shared evaluation index")
    count = int(model._test_metric_count.item())
    if count == 0 or count != len(evaluated):
        raise RuntimeError("Evaluation produced zero or inconsistent scored scene counts")
    metrics = {name.removesuffix("_ours"): (total / count).item()
               for name, total in model._test_metric_sums.items()}
    write_json(output_dir / "metrics.json", {
        "model": config.wandb.name, "checkpoint": cfg.checkpointing.load,
        "evaluation_index": str(index_path),
        "evaluation_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "align_pose": cfg.test.align_pose, "pose_align_steps": cfg.test.pose_align_steps,
        "num_scenes": count,
        "num_target_views": sum(len(entry["target"]) for entry in evaluated.values()),
        "metrics": metrics,
    })
    print(f"[CrossDomain] Saved {output_dir / 'metrics.json'}", flush=True)


def summarize(output_dir: Path, labels: list[str]) -> None:
    results, coverage = {}, []
    for label in labels:
        results[label] = json.loads((output_dir / label / "metrics.json").read_text(encoding="utf-8"))
        coverage_path = output_dir / label / "evaluated_scenes.json"
        coverage.append(json.loads(coverage_path.read_text(encoding="utf-8")))
    if any(scenes != coverage[0] for scenes in coverage[1:]):
        raise RuntimeError("Models evaluated different scenes/views; comparison rejected")
    write_json(output_dir / "summary.json", results)
    lines = ["model,num_scenes,num_target_views,psnr,ssim,lpips"]
    print("\nModel       Scenes   PSNR       SSIM       LPIPS", flush=True)
    for label, result in results.items():
        metric = result["metrics"]
        print(f"{label:10}  {result['num_scenes']:6d}   {metric['psnr']:.4f}    "
              f"{metric['ssim']:.4f}     {metric['lpips']:.4f}", flush=True)
        lines.append(f"{label},{result['num_scenes']},{result['num_target_views']},"
                     f"{metric['psnr']},{metric['ssim']},{metric['lpips']}")
    (output_dir / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    parser.add_argument(
        "--gpus", nargs="+", default=["0", "1"],
        help="One physical GPU ID per model, in model order",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("/workspace/datasets/dl3dv"))
    parser.add_argument(
        "--ours-checkpoint", type=Path,
        default=REPO_ROOT / "pretrained_weights/Final++_re10k.ckpt",
    )
    parser.add_argument(
        "--baseline-checkpoint", type=Path,
        default=REPO_ROOT / "pretrained_weights/re10k.ckpt",
    )
    parser.add_argument(
        "--index", type=Path,
        help="Existing scene -> context/target JSON; otherwise generate once for both models",
    )
    parser.add_argument("--context-gap", type=int, default=22)
    parser.add_argument("--num-targets", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4, help="Data-loader workers per model")
    parser.add_argument("--seed", type=int, default=111123)
    parser.add_argument(
        "--save-image", action="store_true",
        help="Save target predictions and comparison panels",
    )
    parser.add_argument(
        "--no-align-pose", action="store_true",
        help="Disable the standard test-time target pose alignment for both models",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate paths and compose both configs without loading torch or writing outputs",
    )
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_config is not None:
        evaluate_worker(args.worker_config)
        return
    if len(set(args.models)) != len(args.models):
        parser.error("--models must be distinct")
    gpus = args.gpus[:len(args.models)]
    if (len(gpus) != len(args.models) or len(set(gpus)) != len(gpus)
            or any(not gpu.isdecimal() for gpu in gpus)):
        parser.error("Supply a distinct numeric --gpus ID for each model")
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    fixed_views(1000000, args.context_gap, args.num_targets)
    dataset_root = args.dataset_root.expanduser().resolve()
    if not (dataset_root / "test/index.json").is_file():
        raise FileNotFoundError(dataset_root / "test/index.json")
    checkpoints = {
        label: getattr(args, f"{label}_checkpoint").expanduser().resolve()
        for label in args.models
    }
    for path in checkpoints.values():
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}; set --ours-checkpoint / --baseline-checkpoint"
            )
    output_dir = (
        args.output_dir or REPO_ROOT / "outputs/cross_domain/re10k2dl3dv"
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    ).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Use a new or empty --output-dir to preserve existing results: {output_dir}"
        )
    index_path = (
        args.index.expanduser().resolve() if args.index
        else output_dir / "evaluation_index_dl3dv.json"
    )
    if args.index:
        read_view_index(index_path, dataset_root)
    configs = {
        label: compose_config(
            label, checkpoints[label], dataset_root, index_path, output_dir, args
        ) for label in args.models
    }
    for label, gpu in zip(args.models, gpus):
        print(
            f"[CrossDomain] {label}: experiment={EXPERIMENTS[label]}, "
            f"GPU={gpu}, checkpoint={checkpoints[label]}", flush=True,
        )
    print(f"[CrossDomain] DL3DV={dataset_root}; index={index_path}; output={output_dir}", flush=True)
    print(
        f"[CrossDomain] align_pose={not args.no_align_pose}; "
        f"context gap={args.context_gap}, targets={args.num_targets} "
        "(used only when generating an index)", flush=True,
    )
    if args.dry_run:
        print("[CrossDomain] Dry run OK; no GPU/model initialization or output writes.", flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.index:
        generate_view_index(dataset_root, index_path, args.context_gap, args.num_targets)
    # Freeze even a supplied index into this run so both workers read identical bytes.
    frozen_index = output_dir / "evaluation_index_dl3dv.json"
    if index_path != frozen_index:
        frozen_index.write_bytes(index_path.read_bytes())
    from omegaconf import OmegaConf

    jobs = []
    try:
        for label, gpu in zip(args.models, gpus):
            model_dir = output_dir / label
            model_dir.mkdir()
            config = configs[label]
            config.dataset.dl3dv.view_sampler.index_path = str(frozen_index)
            config_path = model_dir / "config.yaml"
            OmegaConf.save(config, config_path)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=gpu, WANDB_MODE="disabled", PYTHONUNBUFFERED="1",
                       OMP_NUM_THREADS="1", HYDRA_FULL_ERROR="1")
            # A worker needs no experiment tracker and never resumes a training run.
            for name in ("WANDB_RUN_ID", "WANDB_RESUME"):
                env.pop(name, None)
            log = (model_dir / "console.log").open("w", encoding="utf-8")
            try:
                process = subprocess.Popen(
                    [sys.executable, "-u", "-m", "scripts.eval_re10k2dl3dv",
                     "--worker-config", str(config_path)],
                    cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                )
            finally:
                log.close()
            jobs.append((label, process))
            print(
                f"[CrossDomain] Started {label} PID={process.pid}; "
                f"log={model_dir / 'console.log'}", flush=True,
            )
        failures = [(label, code) for label, process in jobs if (code := process.wait()) != 0]
        if failures:
            raise RuntimeError(
                f"Evaluation failed: {failures}. "
                f"Inspect each model's console.log in {output_dir}"
            )
    finally:
        for _, process in jobs:
            if process.poll() is None:
                process.terminate()
        for _, process in jobs:
            process.wait()
    summarize(output_dir, args.models)
    print(f"[CrossDomain] Results: {output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
