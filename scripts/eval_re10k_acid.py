"""Run RE10K -> ACID and ACID -> RE10K, each with Final++ and NoPoSplat.

Usage: python -m scripts.eval_re10k_acid --gpus 0 1 2 3
"""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

from scripts import eval_re10k2dl3dv as evaluation


DIRECTIONS = (("re10k", "acid"), ("acid", "re10k"))
MODELS = ("ours", "baseline")


def summarize(output_dir: Path) -> None:
    results, rows = {}, []
    for source, target in DIRECTIONS:
        direction = f"{source}2{target}"
        directory = output_dir / direction
        print(f"\n[CrossDomain] {source.upper()} -> {target.upper()}", flush=True)
        # Coverage must agree within each destination's model pair.
        evaluation.summarize(directory, list(MODELS))
        pair = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        for model in MODELS:
            result = pair[model]
            results[f"{direction}/{model}"] = result
            rows.append({
                "direction": direction, "model": model,
                "training_dataset": source, "evaluation_dataset": target,
                "num_scenes": result["num_scenes"],
                "num_target_views": result["num_target_views"], **result["metrics"],
            })
    evaluation.write_json(output_dir / "summary.json", results)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--gpus", nargs=4, default=["0", "1", "2", "3"],
        help="GPUs for RE10K->ACID ours/baseline, then ACID->RE10K ours/baseline",
    )
    for dataset in ("re10k", "acid"):
        parser.add_argument(
            f"--{dataset}-root", type=Path,
            default=Path("/workspace/datasets") / dataset,
            help=f"{dataset.upper()} dataset root (contains test/)",
        )
        parser.add_argument(
            f"--{dataset}-index", type=Path,
            default=evaluation.REPO_ROOT / f"assets/evaluation_index_{dataset}.json",
            help=f"Existing {dataset.upper()} benchmark view index",
        )
        for model in MODELS:
            parser.add_argument(
                f"--{dataset}-{model}-checkpoint", type=Path,
                default=evaluation.REPO_ROOT / "pretrained_weights"
                / evaluation.CHECKPOINT_NAMES[dataset][model],
                help=f"{dataset.upper()}-trained {model} checkpoint",
            )
    parser.add_argument("--num-workers", type=int, default=4, help="Data-loader workers per GPU")
    parser.add_argument("--seed", type=int, default=111123)
    parser.add_argument("--save-image", action="store_true")
    parser.add_argument("--no-align-pose", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Check all four configs without GPUs or writes")
    args = parser.parse_args(argv)
    if len(set(args.gpus)) != 4 or any(not gpu.isdecimal() for gpu in args.gpus):
        parser.error("--gpus requires four distinct numeric GPU IDs")
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")

    roots, indices, checkpoints = {}, {}, {}
    # Validate every experiment before launching any expensive worker.
    for dataset in ("re10k", "acid"):
        roots[dataset] = getattr(args, f"{dataset}_root").expanduser().resolve()
        indices[dataset] = getattr(args, f"{dataset}_index").expanduser().resolve()
        for model in MODELS:
            path = getattr(args, f"{dataset}_{model}_checkpoint").expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(
                    f"Checkpoint not found: {path}; set --{dataset}-{model}-checkpoint"
                )
            checkpoints[dataset, model] = path
        evaluation.read_view_index(
            indices[dataset], roots[dataset], evaluation_dataset=dataset
        )
    output_dir = (
        args.output_dir or evaluation.REPO_ROOT / "outputs/cross_domain/re10k_acid"
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    ).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use a new or empty --output-dir: {output_dir}")

    configs = {}
    for source, target in DIRECTIONS:
        direction = f"{source}2{target}"
        for model in MODELS:
            job = f"{direction}/{model}"
            config = evaluation.compose_config(
                model, checkpoints[source, model], roots[target], indices[target],
                output_dir / direction, args,
                training_dataset=source, evaluation_dataset=target,
            )
            configs[job] = config
            gpu = args.gpus[len(configs) - 1]
            print(
                f"[CrossDomain] {job}: GPU={gpu}, "
                f"experiment={config.cross_domain.experiment}, "
                f"checkpoint={checkpoints[source, model]}", flush=True,
            )
    print(f"[CrossDomain] Output: {output_dir}", flush=True)
    if args.dry_run:
        print("[CrossDomain] Dry run OK: all four configs validated; no GPU use or writes.", flush=True)
        return

    for source, target in DIRECTIONS:
        directory = output_dir / f"{source}2{target}"
        directory.mkdir(parents=True)
        frozen_index = directory / f"evaluation_index_{target}.json"
        frozen_index.write_bytes(indices[target].read_bytes())
        for model in MODELS:
            config = configs[f"{source}2{target}/{model}"]
            loader = evaluation.EVALUATION_LOADERS[target]
            config.dataset[loader].view_sampler.index_path = str(frozen_index)
    evaluation.run_evaluations(configs, args.gpus, output_dir)
    summarize(output_dir)
    print(f"[CrossDomain] All four results: {output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
