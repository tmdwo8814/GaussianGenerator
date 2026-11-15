"""Strict Final++ initialization for frozen-evidence geometry experiments."""

from pathlib import Path

import torch


def freeze_geometry_readout(encoder):
    encoder.requires_grad_(False)
    encoder.gaussian_decoder.enable_geometry_readout_training()
    encoder.train(encoder.training)


def initialize_geometry_readout(encoder, weight_path: str, seed: int):
    """Load weights only, validate every shared key, then reset ONLY readout heads.

    Optimizer, scheduler and global step are deliberately not restored. Saved
    experiment checkpoints are resumed separately through Trainer.fit().
    """
    if not weight_path:
        raise ValueError('A Final++ checkpoint is required for geometry_readout_only training')
    checkpoint = torch.load(Path(weight_path), map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
        raise ValueError('Expected a Final++ Lightning checkpoint containing state_dict')
    weights = {name.removeprefix('encoder.'): value
               for name, value in checkpoint['state_dict'].items() if name.startswith('encoder.')}
    expected = encoder.state_dict()
    new_prefix = 'gaussian_decoder.regression_head.'
    missing = sorted(name for name in expected if name not in weights and not name.startswith(new_prefix))
    unexpected = sorted(set(weights) - set(expected))
    mismatched = sorted(name for name in weights.keys() & expected.keys()
                        if weights[name].shape != expected[name].shape)
    if missing or unexpected or mismatched:
        raise ValueError(f'Final++ checkpoint does not match the shared model: '
                         f'missing={missing}, unexpected={unexpected}, shape_mismatch={mismatched}')
    encoder.load_state_dict(weights, strict=False)  # Only the NEW regression head may be missing.
    encoder.gaussian_decoder.reset_geometry_readout(seed)
    freeze_geometry_readout(encoder)
    trainable = [(name, parameter.numel()) for name, parameter in encoder.named_parameters()
                 if parameter.requires_grad]
    print(f'[GeometryReadout] Loaded frozen Final++: {weight_path}')
    print(f'[GeometryReadout] mode={encoder.gaussian_decoder.cfg.geometry_mode}, '
          f'trainable_parameters={sum(count for _, count in trainable)}, reset_seed={seed}')
    for name, count in trainable:
        print(f'[GeometryReadout] trainable: {name} ({count})')
