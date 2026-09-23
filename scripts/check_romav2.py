"""Cache weights/DINOv3 code and run the training matcher on one CUDA GPU.

Defaults to the local RoMaV2 Toronto example pair, resized to 256px as in RE10K.
Run once before DDP: python -m scripts.check_romav2
Optional: --image-a a.png --image-b b.png
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from src.model.auxiliary.roma_matching import RoMaMatcher, RoMaMatcherCfg


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
    assets = Path(__file__).resolve().parents[1] / 'RoMaV2/assets'
    paths = [args.image_a, args.image_b] if args.image_a else [assets / 'toronto_A.jpg', assets / 'toronto_B.jpg']
    images = []
    for path in paths:
        if not path.is_file():
            parser.error(f'Missing image {path}; supply --image-a and --image-b')
        with Image.open(path) as source:
            rgb = ImageOps.exif_transpose(source).convert('RGB').resize((256, 256), Image.Resampling.BILINEAR)
            images.append(torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float() / 255)

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
        matches = matcher.match(torch.stack(images).to(device))
        if torch.get_float32_matmul_precision() != 'high':
            raise RuntimeError('Matching changed the training matmul precision')
        torch.cuda.synchronize(device)
        count = len(matches.confidence)
        if count == 0:
            raise RuntimeError('No reliable matches. Check input overlap and the RoMaV2 installation.')
        print(f'PASS: {count} matches with confidence >= 0.5; precision restored.', flush=True)
        print(f'Peak allocated CUDA memory: {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB')
        print('Matching smoke check passed. RE10K pose acceptance and training backward still need a training run.')
    finally:
        torch.set_float32_matmul_precision(previous)
        matcher.release()


if __name__ == '__main__':
    main()
