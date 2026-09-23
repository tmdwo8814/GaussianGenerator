"""Cache weights/DINOv3 code and run the training matcher on one CUDA GPU.

Defaults to a generated texture matched with itself; no example files required.
This checks execution only. Use real overlapping images to check match quality.
Run once before DDP: python -m scripts.check_romav2
Optional: --image-a a.png --image-b b.png
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from src.model.auxiliary.roma_matching import RoMaMatcher, RoMaMatcherCfg


def load_pair(paths):
    if paths is None:
        # A deterministic self-pair exercises inference without bundled assets.
        # Synthetic input is not evidence of real-scene matching/pose quality.
        generator = torch.Generator().manual_seed(0)
        texture = torch.rand(1, 3, 64, 64, generator=generator)
        texture = torch.nn.functional.interpolate(
            texture, size=(256, 256), mode='bilinear', align_corners=False)
        return texture.repeat(2, 1, 1, 1)
    images = []
    for path in paths:
        with Image.open(path) as source:
            rgb = ImageOps.exif_transpose(source).convert('RGB').resize((256, 256), Image.Resampling.BILINEAR)
            images.append(torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float() / 255)
    return torch.stack(images)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--setting', choices=['turbo', 'fast', 'base', 'precise'], default='fast')
    parser.add_argument('--image-a', type=Path)
    parser.add_argument('--image-b', type=Path)
    args = parser.parse_args()
    if (args.image_a is None) != (args.image_b is None):
        parser.error('Supply both --image-a and --image-b')
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        parser.error('Run this check inside a CUDA GPU allocation')
    synthetic = args.image_a is None
    paths = None if synthetic else [args.image_a, args.image_b]
    if paths is not None:
        for path in paths:
            if not path.is_file():
                parser.error(f'Missing image {path}')
    images = load_pair(paths)
    print('Input: generated self-pair (execution check only)' if synthetic
          else f'Input: {args.image_a}, {args.image_b}', flush=True)

    import torchvision

    print(f'PyTorch {torch.__version__}; torchvision {torchvision.__version__}; CUDA {torch.version.cuda}', flush=True)
    print(f'GPU: {torch.cuda.get_device_name(device)}; torch.hub cache: {torch.hub.get_dir()}', flush=True)
    matcher = RoMaMatcher(RoMaMatcherCfg(setting=args.setting))
    matcher.initialize(device)
    from romav2.local_correlation import local_corr
    print(f'Local correlation: {"fused CUDA" if local_corr is not None else "native PyTorch"}', flush=True)
    previous = torch.get_float32_matmul_precision()
    try:
        # Reproduce the backbone's TF32 setting and exercise the production guard.
        torch.set_float32_matmul_precision('high')
        matches = matcher.match(images.to(device))
        if torch.get_float32_matmul_precision() != 'high':
            raise RuntimeError('Matching changed the training matmul precision')
        torch.cuda.synchronize(device)
        count = len(matches.confidence)
        if count == 0 and not synthetic:
            raise RuntimeError('No reliable matches. Check input overlap and the RoMaV2 installation.')
        if count == 0:
            print('No reliable synthetic matches; correspondence sampling may have been skipped.', flush=True)
        print(f'Execution completed: {count} matches with confidence >= 0.5; precision restored.', flush=True)
        print(f'Peak allocated CUDA memory: {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB')
        print('RoMaV2 execution check completed. RE10K match/pose quality and training backward still need a training run.')
    finally:
        torch.set_float32_matmul_precision(previous)
        matcher.release()


if __name__ == '__main__':
    main()
