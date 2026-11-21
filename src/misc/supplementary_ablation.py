"""Frozen scene evidence, fresh jointly trained BEAM/appearance decoder."""

from pathlib import Path

import torch


def freeze_scene_encoder(encoder):
    """The Gaussian decoder lives inside encoder, but must remain trainable."""
    encoder.requires_grad_(False)
    encoder.gaussian_decoder.enable_supplementary_decoder_training()
    encoder.train(encoder.training)


def initialize_supplementary_ablation(encoder, weight_path: str, seed: int):
    if not weight_path:
        raise ValueError('Specify the common pretrained NoPoSplat checkpoint for supplementary ablations')
    checkpoint = torch.load(Path(weight_path), map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
        raise ValueError('Expected a NoPoSplat/EviFit Lightning checkpoint with encoder state_dict entries')
    weights = {name.removeprefix('encoder.'): value
               for name, value in checkpoint['state_dict'].items() if name.startswith('encoder.')}
    expected = {name: value for name, value in encoder.state_dict().items()
                if not name.startswith('gaussian_decoder.')}
    # Original NoPoSplat has attribute-output convolutions at these keys. The
    # feature extractor uses the same DPT/RGB blocks and an identity output.
    obsolete_outputs = ('gaussian_param_head.dpt.head.', 'gaussian_param_head2.dpt.head.')
    missing = sorted(set(expected) - set(weights))
    mismatched = sorted(name for name in weights.keys() & expected.keys()
                        if weights[name].shape != expected[name].shape)
    unexpected = sorted(name for name in weights if name not in expected
                        and not name.startswith(('gaussian_decoder.', *obsolete_outputs)))
    if missing or mismatched or unexpected:
        raise ValueError(f'Frozen encoder checkpoint mismatch: missing={missing}, '
                         f'shape_mismatch={mismatched}, unexpected={unexpected}')
    encoder.load_state_dict({name: weights[name] for name in expected}, strict=False)
    decoder = encoder.gaussian_decoder
    if decoder.regression_head is not None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            decoder.regression_head.reset_parameters()
    freeze_scene_encoder(encoder)
    trainable = [(name, parameter.numel()) for name, parameter in encoder.named_parameters()
                 if parameter.requires_grad]
    if any(not name.startswith('gaussian_decoder.') for name, _ in trainable):
        raise RuntimeError('Supplementary experiment has trainable scene-encoder parameters')
    print(f'[Supplementary] Loaded frozen scene encoder: {weight_path}')
    print(f'[Supplementary] assignment={decoder.cfg.assignment_mode}, '
          f'attributes={decoder.cfg.attribute_modes}, coverage={decoder.cfg.covariance_coverage}, '
          f'trainable_parameters={sum(count for _, count in trainable)}')
    print('[Supplementary] BEAM budget/appearance and Gaussian decoder train jointly; '
          'no pretrained Gaussian-decoder weights were loaded.')
