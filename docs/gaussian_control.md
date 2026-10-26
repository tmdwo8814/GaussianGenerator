# Frozen Final++ Gaussian-count control

This experiment caches one complete Final++ prediction per RE10K scene and
applies three independent count controls to the same original prediction.
No model parameters are optimized. Target cameras are aligned once using the
full prediction and the existing evaluation losses, then shared by all budgets
and methods. Camera alignment is separate from count control and its timing.

## Run

Activate the existing training environment, then submit from the repository root:

```bash
sbatch scripts/eval_gaussian_control.sh
```

The default checkpoint is
`/workspace/research/GaussianGenerator/pretrained_weights/Final++_RE10k.ckpt`.
The Slurm request uses one `high_perf` GPU and 60 GB host memory. It inherits
the activated environment; no packages are installed by the launcher.

Override server paths through environment variables:

```bash
CHECKPOINT=/path/to/Final++_RE10k.ckpt \
DATASET_ROOT=/path/to/re10k \
sbatch scripts/eval_gaussian_control.sh
```

`EVALUATION_INDEX`, `CACHE_DIR`, `OUTPUT_DIR`, `STAGE`, and `PERCENTAGES` are also
configurable. The default ratios are retained percentages
`[100,75,50,25,10,5,1]`, with exact budget `max(1, floor(N * p / 100))`.
100% is evaluated once and shared as an identity across all three methods.

The same command resumes interrupted inference and completed scene/condition
evaluations. Cache identity checks include the checkpoint SHA256, model/source
configuration, evaluation index, selected frames, seed and camera protocol.
Use a new cache directory for another checkpoint, scene subset or pose protocol.
Changed timing/source settings require a new results directory.

Stages can also be run independently:

```bash
STAGE=cache sbatch scripts/eval_gaussian_control.sh
STAGE=evaluate sbatch scripts/eval_gaussian_control.sh
```

`evaluate` reads only the cache and the saved frozen moment decoder; it needs
neither the checkpoint nor the dataset/backbone. To rerun timings against the
same cache, change only the results directory:

```bash
STAGE=evaluate OUTPUT_DIR=outputs/gaussian_control/repeated_timings \
sbatch scripts/eval_gaussian_control.sh control.timing_repeats=3
```

For a small first run, choose separate cache/results paths:

```bash
CACHE_DIR=outputs/gaussian_control/smoke_cache \
OUTPUT_DIR=outputs/gaussian_control/smoke_results \
sbatch scripts/eval_gaussian_control.sh control.max_scenes=2
```

Direct execution is also supported: `bash scripts/eval_gaussian_control.sh`.
Hydra overrides appended to the command take precedence over launcher defaults.

## Methods

* **Pruning:** use exact per-view quotas on the original pixel raster. Stratify
  both image axes with midpoint sampling; retain selected Gaussian attributes
  unchanged. Selection is deterministic and independent of opacity/mass.
* **Merging (NanoGS-style):** start from all decoded Gaussians. Use the official
  NanoGS NumPy/SciPy code, pinned to commit
  `cc64a752b5def8d17f9992216bfbb9e533f8364f`. Current Gaussian-center 3D KNN=16,
  native Monte Carlo mixture-to-Gaussian distortion (one sample, seed 0), SH
  squared L2, unit cost weights, greedy disjoint pair matching and native
  opacity-volume-weighted moment matching are retained. Refresh KNN after each
  pass. Disable opacity pre-pruning and stop at the shared integer budget.
  Covariance is converted to scales/quaternion for the official kernels and
  restored afterward; every SH coefficient is preserved. See the original
  [paper](https://arxiv.org/abs/2603.16103) and
  [source](https://github.com/saliteta/NanoGS). Unmodified upstream files,
  attribution and CC BY-NC 4.0 license are in `src/third_party/nanogs`.
* **Redistribute:** select the largest original incoming masses, with original
  slot ID as stable tie break. Keep all source supports/features/RGB. Intersect
  each source's original KNN with open slots and normalize its cached original
  allocation logits over that set. Sources with no surviving neighbor send their
  full budget to the exact nearest open original support. Recompute geometry,
  independent appearance weights, feature mean/std, 2D reader context and SH with
  the frozen Final++ decoder. New fallback appearance scores are recomputed.
  Epsilon priors apply only to actual open self edges. Geometry and image-reader
  edges into closed slots are masked. The original scene scale is retained.

## Cache and timing

`cache_dir/scenes/<scene-and-frame-hash>.pt` stores the full Gaussian, normalized
support points, FP32 DPT features, original KNN, allocation logits, budgets,
appearance scores, original incoming masses, context RGB/raster layout, scene
scale, held-out RGB, and shared aligned target cameras. `decoder.pt` holds the
small frozen decoder's weights and its exact configuration. Rendered Gaussians
alone cannot reproduce support redistribution. Expect roughly 0.2 GB per scene
at two 256x256 context views; the job prints the actual size as it caches.

Postprocess latency is synchronized wall time from resident original inputs to
the final device-resident Gaussian tensor. It includes selection, graph work,
merging/reallocation, decoding, attribute conversion and transfers performed by
that method. Cache I/O, encoder inference, model loading, camera alignment,
rendering, validation and image metrics are outside the timer. Each ratio starts
from the full cached scene, not another ratio's output.

NanoGS uses its official CPU implementation; pruning/redistribution use the GPU.
Reported wall times include NanoGS's required CPU/GPU transfers and the exact CPU
nearest-open fallback in redistribution. These are implementation latencies on
the recorded machine, not a comparison of GPU kernels alone.

One untimed warmup is performed once per method/budget per process, using the
first scene requiring that condition. Then `control.timing_repeats` (default 1)
wall-time samples are collected per scene. 100% has zero identity adaptation
time. Compilation is disabled by default to exclude compilation effects. A
separate rendering latency is measured with identical per-view chunking and an
untimed warm render. It excludes PSNR/SSIM/LPIPS computation.

## Results

Default results directory: `outputs/gaussian_control/final_plus_results`.

* `summary.txt` / `summary.json`: every method, budget and overlap group.
* `prune_075.txt`, `merge_001.txt`, `redistribute_005.txt`, etc.: one TXT for each
  method/percentage; `final_100.txt` contains the shared full-model reference.
* `per_scene.jsonl`: metrics, per-view scores, actual count, timing samples and
  fallback/merge diagnostics.
* `scenes/*.json`: atomic scene checkpoints for interrupted-job recovery.
* `metadata.json`, `analysis_config.yaml`, `run.log`, `run_status.txt`:
  reproducibility metadata and completion status.

PSNR/SSIM/LPIPS are averaged over target views first and then equally over scenes.
Times use equal scene means, with additional scene-level p50/p95. Partial runs
also write reports; check the reported scene count and `run_status.txt` before
using a table. Set `control.save_variants=true` to additionally store each reduced
Gaussian set outside the measured interval.

Suggested paper sentence: “Merging applies NanoGS-style training-free
simplification to our predicted Gaussians at matched primitive budgets.” State
in implementation details that opacity pre-pruning is disabled.

## Verification

Read current cache/evaluation progress without interrupting the experiment:

```bash
python scripts/gaussian_control_status.py
```

The helper uses only the Python standard library and reads manifests and small
per-scene result JSON files. It counts fully completed requested conditions,
reports partial scenes, and estimates remaining time from the most recent 50
completed-scene file timestamps, including rendering, metrics and reporting.
It also reports merging-only time from completed scenes. Custom paths and a
different recent window are supported with `--cache-dir`, `--output-dir`, and
`--window`. `--json` emits machine-readable output. Displayed finish times use
Asia/Seoul. ETA is a projection, not a process-liveness check; pauses/restarts
or parallel jobs writing the same directory can invalidate the estimate.
This helper does not change any inference, timing or cache-signature source.

```bash
python -m unittest discover -s tests -p 'test_gaussian_control.py' -v
python -m unittest discover -s tests -p 'test_moment*.py' -v
bash -n scripts/eval_gaussian_control.sh
```

CPU tests cover cached/full decoding identity, all-open redistribution identity,
budget conservation, fallback and epsilon behavior, image-edge masking, exact
two-dimensional pixel sampling, covariance/SH conversion, native NanoGS moments,
target counts, disk cache identity and scene-equal report aggregation. Full
PSNR/SSIM/LPIPS evaluation requires the server's checkpoint, RE10K and CUDA
rasterizer.
