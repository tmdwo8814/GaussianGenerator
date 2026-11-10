"""Frame-weighted throughput from synchronized inference, independent of scores."""

from math import isfinite


def summarize_inference(records: list[dict], warmup_steps: int) -> dict:
    if warmup_steps < 0 or len(records) <= warmup_steps:
        raise ValueError(
            f"FPS needs more than {warmup_steps} scenes; got {len(records)}"
        )
    measured = records[warmup_steps:]
    frames = sum(row["num_frames"] for row in measured)
    encoder_seconds = sum(row["encoder_seconds"] for row in measured)
    render_seconds = sum(row["render_seconds"] for row in measured)
    if frames <= 0 or any(not isfinite(t) or t <= 0
                          for t in (encoder_seconds, render_seconds)):
        raise ValueError("FPS requires positive finite inference times and frame counts")
    return {
        "fps": frames / (encoder_seconds + render_seconds),
        "render_fps": frames / render_seconds,
        "num_timed_scenes": len(measured),
        "num_timed_frames": frames,
        "warmup_scenes": warmup_steps,
        "encoder_seconds": encoder_seconds,
        "render_seconds": render_seconds,
        "definition": (
            "total target frames / (scene encoding + final rasterization seconds); "
            "scene encoding includes backbone, point/DPT heads, kNN and Gaussian decoder; "
            "encoding is amortized over each scene's target frames; CUDA synchronized; "
            "excludes warmup scenes, data loading/transfer, pose optimization, metrics and I/O"
        ),
    }
