"""Create deterministic, nested RE10K splits for Gaussian-decoder analyses.

This script is deliberately independent of CUDA and model checkpoints.  It reads
the ordinary evaluation index, stratifies valid scenes by overlap, and writes:

* ``gradient_900.json``: 300 scenes per overlap group;
* ``compensation_300.json``: a nested 100-scene-per-group subset; and
* ``smoke_3.json``: one nested scene per group for implementation validation.

The counts and file names are configurable.  Existing files are never silently
overwritten unless ``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from src.misc.utils import get_overlap_tag


OVERLAP_TAGS = ("small", "medium", "large")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_index(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Evaluation index must be a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {path}. Pass --overwrite intentionally."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, allow_nan=False)
    temporary.replace(path)


def _group_valid_scenes(index: dict[str, Any]) -> dict[str, list[str]]:
    grouped = {tag: [] for tag in OVERLAP_TAGS}
    for scene, entry in index.items():
        if entry is None:
            continue
        if not isinstance(entry, dict) or "overlap" not in entry:
            raise ValueError(f"Malformed evaluation-index entry for scene {scene}")
        tag = get_overlap_tag(float(entry["overlap"]))
        if tag in grouped:
            grouped[tag].append(str(scene))
    return {tag: sorted(scenes) for tag, scenes in grouped.items()}


def _stratified_sample(
    grouped: dict[str, list[str]],
    scenes_per_overlap: int,
    seed: int,
) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for offset, tag in enumerate(OVERLAP_TAGS):
        candidates = grouped[tag]
        if len(candidates) < scenes_per_overlap:
            raise ValueError(
                f"Requested {scenes_per_overlap} {tag} scenes, but only "
                f"{len(candidates)} are available."
            )
        rng = random.Random(seed + offset)
        selected[tag] = rng.sample(candidates, scenes_per_overlap)
    return selected


def _nested_subset(
    parent: dict[str, list[str]],
    scenes_per_overlap: int,
    seed: int,
) -> dict[str, list[str]]:
    subset: dict[str, list[str]] = {}
    for offset, tag in enumerate(OVERLAP_TAGS):
        candidates = list(parent[tag])
        if len(candidates) < scenes_per_overlap:
            raise ValueError(
                f"Cannot take {scenes_per_overlap} {tag} scenes from a parent "
                f"split containing {len(candidates)}."
            )
        rng = random.Random(seed + offset)
        subset[tag] = rng.sample(candidates, scenes_per_overlap)
    return subset


def _materialize(
    full_index: dict[str, Any], selected: dict[str, list[str]]
) -> dict[str, Any]:
    # Keep an explicit small -> medium -> large order for easy human inspection.
    return {
        scene: full_index[scene]
        for tag in OVERLAP_TAGS
        for scene in selected[tag]
    }


def _membership(selected: dict[str, list[str]]) -> dict[str, str]:
    return {
        scene: tag
        for tag in OVERLAP_TAGS
        for scene in selected[tag]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("assets/evaluation_index_re10k.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/gaussian_decoder_analysis/splits"),
    )
    parser.add_argument("--gradient-per-overlap", type=int, default=300)
    parser.add_argument("--compensation-per-overlap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.gradient_per_overlap <= 0:
        raise ValueError("--gradient-per-overlap must be positive")
    if not 0 < args.compensation_per_overlap <= args.gradient_per_overlap:
        raise ValueError(
            "--compensation-per-overlap must be positive and no larger than "
            "--gradient-per-overlap"
        )

    index_path = args.index.resolve()
    output_dir = args.output_dir.resolve()
    full_index = _read_index(index_path)
    grouped = _group_valid_scenes(full_index)

    gradient = _stratified_sample(
        grouped, args.gradient_per_overlap, args.seed
    )
    compensation = _nested_subset(
        gradient, args.compensation_per_overlap, args.seed + 10_000
    )
    smoke = _nested_subset(compensation, 1, args.seed + 20_000)

    paths = {
        "gradient": output_dir / "gradient_900.json",
        "compensation": output_dir / "compensation_300.json",
        "smoke": output_dir / "smoke_3.json",
    }
    _write_json(
        paths["gradient"], _materialize(full_index, gradient), args.overwrite
    )
    _write_json(
        paths["compensation"],
        _materialize(full_index, compensation),
        args.overwrite,
    )
    _write_json(paths["smoke"], _materialize(full_index, smoke), args.overwrite)

    gradient_membership = _membership(gradient)
    if any(scene not in gradient_membership for scene in _membership(compensation)):
        raise AssertionError("Compensation split is not nested in gradient split")
    if any(scene not in _membership(compensation) for scene in _membership(smoke)):
        raise AssertionError("Smoke split is not nested in compensation split")

    manifest = {
        "source_index": str(index_path),
        "source_index_sha256": _sha256(index_path),
        "seed": args.seed,
        "overlap_definition": {
            "small": "0.05 <= overlap <= 0.30",
            "medium": "0.30 < overlap <= 0.55",
            "large": "0.55 < overlap <= 0.80",
        },
        "available_valid_scenes": {
            tag: len(grouped[tag]) for tag in OVERLAP_TAGS
        },
        "splits": {
            "gradient": {
                "path": str(paths["gradient"]),
                "sha256": _sha256(paths["gradient"]),
                "counts": {tag: len(gradient[tag]) for tag in OVERLAP_TAGS},
            },
            "compensation": {
                "path": str(paths["compensation"]),
                "sha256": _sha256(paths["compensation"]),
                "counts": {
                    tag: len(compensation[tag]) for tag in OVERLAP_TAGS
                },
                "nested_in": "gradient",
            },
            "smoke": {
                "path": str(paths["smoke"]),
                "sha256": _sha256(paths["smoke"]),
                "counts": {tag: len(smoke[tag]) for tag in OVERLAP_TAGS},
                "nested_in": "compensation",
            },
        },
    }
    _write_json(output_dir / "manifest.json", manifest, args.overwrite)

    print(f"Analysis splits written to: {output_dir}")
    for name, info in manifest["splits"].items():
        print(f"  {name}: {info['counts']} ({info['sha256'][:12]})")


if __name__ == "__main__":
    main()
