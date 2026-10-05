"""Compare eager/optimized moment-shape forward+backward on identical inputs.

Standalone decoder benchmark: includes exact kNN, excludes backbone, renderer,
DDP and data loading. Run in a single-GPU process, not under torchrun.
"""

import argparse
import gc
import importlib
import json
from pathlib import Path
import sys
import time
from types import ModuleType

import torch


def decoder_types():
    # Avoid importing Lightning, the backbone and CUDA rasterizer just to time
    # the decoder. Production source files are loaded unchanged.
    root = Path(__file__).resolve().parents[1] / 'src/model'
    for name, path in {
        '_moment_speed': root,
        '_moment_speed.encoder': root / 'encoder',
        '_moment_speed.encoder.heads': root / 'encoder/heads',
        '_moment_speed.encoder.common': root / 'encoder/common',
    }.items():
        package = ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    module = importlib.import_module('_moment_speed.encoder.heads.moment_gaussian_decoder')
    return module.MomentDecoderCfg, module.MomentGaussianDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=1, help='Local batch size on one GPU')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if min(args.batch_size, args.height, args.width, args.steps, args.warmup) < 1:
        parser.error('Batch size, resolution, steps and warmup must be positive')
    if not torch.cuda.is_available() or torch.device(args.device).type != 'cuda':
        parser.error('A CUDA GPU is required; CPU time is not a GPU speed estimate')
    torch.cuda.set_device(args.device)
    # Matches the existing CroCo import in the training application.
    torch.backends.cuda.matmul.allow_tf32 = True
    Cfg, Decoder = decoder_types()
    options = dict(feature_dim=256, hidden_dim=64, num_neighbors=16, chunk_size=32768,
                   checkpoint_chunks=True, separate_appearance=True, appearance_2d=True,
                   appearance_dim=32, moment_shape=True, shape_hidden_dim=128)
    torch.manual_seed(42)
    initial = Decoder(Cfg(**options))
    for name in ('appearance_head.weight', 'appearance_sh_head.weight', 'shape_head.mlp.2.weight'):
        torch.nn.init.normal_(dict(initial.named_parameters())[name], std=.01)
    state = initial.state_dict()
    count = 2 * args.height * args.width
    # The same synthetic scene/features are replayed by both implementations.
    # This measures decoder execution, not reconstruction quality or renderer cost.
    points = torch.randn(args.batch_size, count, 3, device=args.device, requires_grad=True)
    features = torch.randn(args.batch_size, count, 256, device=args.device, requires_grad=True)
    rgb = torch.rand(args.batch_size, count, 3, device=args.device)
    baseline = None
    results = {}
    for name, optimized in (('eager', False), ('optimized', True)):
        model = Decoder(Cfg(**options, compile_kernels=optimized,
                            cache_image_neighbors=optimized,
                            shape_chunk_size=65536 if optimized else 0)).to(args.device).train()
        model.load_state_dict(state, strict=True)

        def step(capture=False):
            model.zero_grad(set_to_none=True)
            for tensor in (points, features):
                tensor.grad = None
            output = model(points, features, rgb=rgb, image_shape=(2, args.height, args.width))
            fields = (output.means, output.covariances, output.harmonics, output.opacities)
            sum(field.square().mean() for field in fields).backward()
            if capture:
                # Small deterministic samples; exhaustive small-scene gradients
                # and attributes are checked in test_moment_shape_speed.py.
                sample = lambda value: value.detach().reshape(-1)[::max(value.numel() // 4096, 1)].cpu()
                return ([sample(field) for field in fields],
                        [sample(tensor.grad) for tensor in (points, features)],
                        {key: sample(value.grad) for key, value in model.named_parameters()})

        print(f'{name}: warming up (optimized first run includes compilation)...', flush=True)
        for _ in range(args.warmup):
            step()
        captured = step(capture=True)
        if baseline is None:
            baseline = captured
        else:
            for left, right in zip(captured[0], baseline[0]):
                torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-5)
            for left, right in zip(captured[1], baseline[1]):
                torch.testing.assert_close(left, right, rtol=2e-3, atol=1e-7)
            for key in captured[2]:
                torch.testing.assert_close(captured[2][key], baseline[2][key], rtol=2e-3, atol=1e-7)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(args.steps):
            step()
        torch.cuda.synchronize()
        results[name] = dict(seconds=(time.perf_counter() - start) / args.steps,
                             peak_torch_gib=torch.cuda.max_memory_allocated() / 1024**3)
        print(f'{name}: {results[name]}', flush=True)
        del step, model
        for tensor in (points, features):
            tensor.grad = None
        gc.collect()
        torch.cuda.empty_cache()
    results['optimized_over_eager'] = results['optimized']['seconds'] / results['eager']['seconds']
    results['context_shape'] = [args.batch_size, 2, 3, args.height, args.width]
    results['scope'] = 'Synthetic decoder forward+backward including kNN; not full training'
    results['memory_scope'] = 'PyTorch allocated only; excludes CuPy, driver and compiler processes'
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
