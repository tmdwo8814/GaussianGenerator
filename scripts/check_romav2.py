"""Prefetch RoMaV2/DINOv3 code and optionally smoke-test matching on one GPU.

Run from the repository root: python -m scripts.check_romav2
Optional: --image-a a.png --image-b b.png
"""

import argparse
import importlib.util
from pathlib import Path

import torch


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
        parser.error('This training integration requires a CUDA GPU')

    import torchvision
    from romav2 import RoMaV2

    print(f'PyTorch {torch.__version__}; torchvision {torchvision.__version__}; CUDA {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(device)}; torch.hub cache: {torch.hub.get_dir()}')
    print(f'Fused local correlation installed: {importlib.util.find_spec("local_corr") is not None}')
    with torch.cuda.device(device):
        model = RoMaV2(RoMaV2.Cfg(setting=args.setting, compile=False)).to(device).eval().requires_grad_(False)
        torch.cuda.synchronize(device)
        print(f'RoMaV2 initialized with setting={args.setting}. Weights and DINOv3 code are cached.')
        if args.image_a is not None:
            with torch.inference_mode():
                prediction = model.match(str(args.image_a), str(args.image_b))
            warp = prediction['warp_AB']
            confidence = prediction['overlap_AB']
            valid = torch.isfinite(warp).all(-1) & (warp.abs().amax(-1) <= 1 - 1 / warp.shape[1])
            count = min(2048, int(((confidence[..., 0] > 0) & valid).sum()) // 4)
            if count:
                with torch.no_grad():
                    matches, overlaps, _, _ = model.sample(prediction, count)
                print(f'Sampled {len(matches)} matches; confidence >= 0.5: {int((overlaps >= .5).sum())}')
            else:
                print('No valid correspondence candidates; check image overlap.')
        print(f'Peak allocated CUDA memory: {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB')


if __name__ == '__main__':
    main()
