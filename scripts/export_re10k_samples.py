"""Export RE10K evaluation context pairs for selecting Gaussian-analysis pixels.

Run from the repository root (CPU only; requires torch, numpy and Pillow):

    python scripts/export_re10k_samples.py --data-path /path/to/re10k \
        --output-dir outputs/re10k_samples --num-scenes 100

--data-path also accepts the test directory or a single .torch chunk. The
evaluation index supplies the exact two context frames and target frame IDs.
The default export is 100 scenes / 200 context PNGs, plus contact sheets,
manifest.json, and selected_evaluation_index.json. No checkpoints are needed.

Choose (x, y) on an individual scenes/.../context_*.png, not on a contact sheet.
Coordinates are zero-based, with the origin at the top-left. PNGs use the same
aspect-preserving LANCZOS resize and center crop as the dataset's crop shim.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Any


DEFAULT_INDEX = (
    Path(__file__).resolve().parents[1] / "assets/evaluation_index_re10k.json"
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def find_chunks(data_path: Path, evaluation_index: dict[str, Any]) -> list[Path]:
    if data_path.is_file():
        if data_path.suffix != ".torch":
            raise ValueError("--data-path must be a .torch file or dataset directory")
        return [data_path]
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    test_dir = data_path / "test"
    directory = test_dir if test_dir.is_dir() else data_path
    chunk_index_path = directory / "index.json"
    if chunk_index_path.is_file():
        # Avoid opening chunks that contain no scenes in the evaluation split.
        chunk_index = read_json(chunk_index_path)
        chunks = {
            (directory / chunk_index[scene]).resolve()
            for scene, entry in evaluation_index.items()
            if entry is not None and scene in chunk_index
        }
        missing = [path for path in chunks if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Chunk listed in index.json is missing: {missing[0]}"
            )
    else:
        chunks = set(directory.glob("*.torch"))
    if not chunks:
        raise ValueError(f"No evaluation .torch chunks found in {directory}")
    return sorted(chunks)


def decode_contexts(
    example: dict[str, Any], frames: list[int], image_size: int
) -> list[tuple[Any, dict[str, Any]]]:
    from PIL import Image

    if len(frames) != 2:
        raise ValueError("Expected exactly two context frames for these experiments")
    decoded = []
    for frame in frames:
        if not isinstance(frame, int) or not 0 <= frame < len(example["images"]):
            raise ValueError(f"Invalid context frame index: {frame}")
        encoded = example["images"][frame]
        payload = (
            bytes(encoded)
            if isinstance(encoded, (bytes, bytearray))
            else encoded.numpy().tobytes()
        )
        with Image.open(BytesIO(payload)) as source:
            image = source.convert("RGB")
        width, height = image.size
        if image_size > min(width, height):
            raise ValueError(
                f"Frame {frame} is {width}x{height}, smaller than {image_size}x"
                f"{image_size}; the dataset crop shim does not upscale"
            )
        # Match src/dataset/shims/crop_shim.py:rescale_and_crop. Keeping this
        # small PIL operation here avoids importing the training/dataset stack.
        factor = max(image_size / height, image_size / width)
        resized = (round(width * factor), round(height * factor))
        image = image.resize(resized, Image.Resampling.LANCZOS)
        left = (resized[0] - image_size) // 2
        top = (resized[1] - image_size) // 2
        image = image.crop((left, top, left + image_size, top + image_size))
        decoded.append(
            (
                image,
                {
                    "original_size_wh": [width, height],
                    "resized_size_wh": list(resized),
                    "crop_origin_xy": [left, top],
                },
            )
        )
    return decoded


def save_contact_sheets(output_dir: Path, samples: list[dict[str, Any]]) -> None:
    from PIL import Image, ImageDraw

    directory = output_dir / "contact_sheets"
    directory.mkdir()
    # Two pairs per row, five rows per page. Individual PNGs remain untouched.
    thumbnail_size = 256
    tile_width, tile_height = 2 * thumbnail_size, thumbnail_size + 28
    for start in range(0, len(samples), 10):
        page = samples[start : start + 10]
        rows = (len(page) + 1) // 2
        sheet = Image.new("RGB", (2 * tile_width, rows * tile_height), "white")
        draw = ImageDraw.Draw(sheet)
        for slot, sample in enumerate(page):
            x, y = (slot % 2) * tile_width, (slot // 2) * tile_height
            draw.text(
                (x + 4, y + 6),
                f"{sample['sample_id']} {sample['scene']} "
                f"| c0={sample['context'][0]}, c1={sample['context'][1]}",
                fill="black",
            )
            for view, info in enumerate(sample["images"]):
                with Image.open(output_dir / info["path"]) as source:
                    image = source.convert("RGB")
                    image.thumbnail((thumbnail_size, thumbnail_size))
                    sheet.paste(image, (x + view * thumbnail_size, y + 28))
            draw.rectangle(
                (x, y, x + tile_width - 1, y + tile_height - 1), outline="gray"
            )
        sheet.save(directory / f"page_{start // 10 + 1:03d}.png")


def export_samples(
    data_path: Path,
    output_dir: Path,
    index_path: Path = DEFAULT_INDEX,
    num_scenes: int = 100,
    image_size: int = 256,
    seed: int = 0,
) -> dict[str, Any]:
    import torch

    if num_scenes <= 0 or image_size <= 0:
        raise ValueError("--num-scenes and --image-size must be positive")
    data_path, output_dir, index_path = (
        path.resolve() for path in (data_path, output_dir, index_path)
    )
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise ValueError(
            f"Output directory must be empty; choose a new one: {output_dir}"
        )
    evaluation_index = read_json(index_path)
    chunks = find_chunks(data_path, evaluation_index)
    rng = random.Random(seed)
    rng.shuffle(chunks)
    samples: list[dict[str, Any]] = []
    skipped = []
    seen: set[str] = set()
    for chunk_path in chunks:
        print(f"Reading {chunk_path}", flush=True)
        chunk = torch.load(chunk_path, map_location="cpu", weights_only=True)
        if not isinstance(chunk, list):
            raise ValueError(f"Expected a list of scenes in {chunk_path}")
        rng.shuffle(chunk)
        for example in chunk:
            scene = example["key"]
            entry = evaluation_index.get(scene)
            if entry is None or scene in seen:
                continue
            seen.add(scene)
            try:
                frames = list(entry["context"])
                decoded = decode_contexts(example, frames, image_size)
            except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
                skipped.append({"scene": scene, "reason": str(error)})
                print(f"Skipping {scene}: {error}", file=sys.stderr, flush=True)
                continue
            sample_id = f"{len(samples) + 1:03d}"
            safe_scene = re.sub(r"[^A-Za-z0-9_.-]", "_", scene)
            scene_dir = output_dir / "scenes" / f"{sample_id}_{safe_scene}"
            scene_dir.mkdir(parents=True)
            images = []
            for view, (image, transform) in enumerate(decoded):
                path = scene_dir / f"context_{view}_frame_{frames[view]:06d}.png"
                image.save(path)
                images.append(
                    {
                        "context_view": view,
                        "frame_index": frames[view],
                        "path": path.relative_to(output_dir).as_posix(),
                        "size_wh": list(image.size),
                        **transform,
                    }
                )
            samples.append(
                {
                    "sample_id": sample_id,
                    "scene": scene,
                    "chunk_path": str(chunk_path),
                    "context": frames,
                    "target": entry.get("target", []),
                    "images": images,
                }
            )
            print(
                f"[{len(samples)}/{num_scenes}] {scene}: context={frames}", flush=True
            )
            if len(samples) == num_scenes:
                break
        if len(samples) == num_scenes:
            break
    if not samples:
        raise ValueError(
            "No scenes exported. Check that --data-path and --index refer to the "
            "same RE10K split; see any skipped-scene messages above."
        )
    manifest = {
        "data_path": str(data_path),
        "evaluation_index": str(index_path),
        "seed": seed,
        "requested_scenes": num_scenes,
        "exported_scenes": len(samples),
        "image_size_wh": [image_size, image_size],
        "pixel_coordinates": (
            "(x, y), zero-based, origin at top-left; use individual PNGs"
        ),
        "context_order": "context_0 then context_1; original evaluation-index order",
        "samples": samples,
        "skipped": skipped,
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(
        output_dir / "selected_evaluation_index.json",
        {sample["scene"]: evaluation_index[sample["scene"]] for sample in samples},
    )
    save_contact_sheets(output_dir, samples)
    if len(samples) < num_scenes:
        print(
            f"Only {len(samples)} usable evaluation scenes were found "
            f"(requested {num_scenes}).",
            file=sys.stderr,
        )
    print(
        f"Saved {len(samples)} scenes / {2 * len(samples)} context PNGs to {output_dir}"
    )
    print("Browse contact_sheets/, then choose pixels on individual scenes/... PNGs.")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/re10k_samples")
    )
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--num-scenes", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        export_samples(
            args.data_path, args.output_dir, args.index,
            args.num_scenes, args.image_size, args.seed,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
