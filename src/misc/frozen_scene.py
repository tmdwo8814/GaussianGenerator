"""Strict scene-only initialization for decoder retraining experiments."""

from pathlib import Path

import torch


def load_scene_encoder(encoder, checkpoint_path: str) -> dict:
    """Load all scene providers, leaving the newly initialized decoder intact.

    Final++ Lightning checkpoints store the point heads and the two DPT feature
    heads under encoder.*, alongside the CroCo backbone. A partial scene load
    would freeze random weights, so missing/unexpected keys and shapes are fatal.
    """
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint:
        raise ValueError("Scene initialization requires a Final++ Lightning state_dict")
    prefix = "gaussian_decoder."
    scene_state = {
        key.removeprefix("encoder."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("encoder.") and not key.startswith("encoder." + prefix)
    }
    expected = {key: value for key, value in encoder.state_dict().items()
                if not key.startswith(prefix)}
    missing = sorted(expected.keys() - scene_state.keys())
    unexpected = sorted(scene_state.keys() - expected.keys())
    mismatched = [key for key in expected.keys() & scene_state.keys()
                  if expected[key].shape != scene_state[key].shape]
    if missing or unexpected or mismatched:
        raise ValueError(
            f"Incompatible scene checkpoint: missing={missing}, "
            f"unexpected={unexpected}, shape_mismatch={sorted(mismatched)}"
        )
    if not expected:
        raise ValueError("The encoder has no scene weights to initialize")
    encoder.load_state_dict(scene_state, strict=False)
    report = {
        "checkpoint": str(checkpoint_path),
        "scene_tensors_loaded": len(scene_state),
        "decoder_initialization": "random (shared seed across K)",
        "trainable_parameters": sum(p.numel() for p in encoder.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in encoder.parameters() if not p.requires_grad),
    }
    print(f"[FrozenScene] {report}", flush=True)
    return report
