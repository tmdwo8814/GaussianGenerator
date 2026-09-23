# RoMaV2 observed-color auxiliary training

This branch adds **training-only** photometric supervision to the existing moment
decoder. The backbone, point heads and moment decoder still train end to end.
The decoder's inputs, parameters, Gaussian attributes and inference path do not
change. The initial NoPoSplat integration supports two context views and one
view-major slot per pixel, matching `re10k_moment`.

## Read the implementation in this order

1. `src/model/auxiliary/roma_matching.py`: frozen RoMaV2, context RGB -> matches.
2. `src/model/auxiliary/matching_prior.py`: calibrated relative pose, triangulation,
   and robust translation-scale alignment to detached raw first-view points.
3. `src/model/auxiliary/observed_color_loss.py`: fixed observed colors,
   source-only Gaussian rendering, and confidence-weighted photometric L1.
4. `config/experiment/re10k_moment_aux.yaml`: all auxiliary hyperparameters.
5. `ModelWrapper.training_step`: match before encoding, then add the auxiliary loss.

Only small hooks were added to `model_wrapper.py` and `encoder_noposplat.py`.
The existing CUDA renderer is reused via `use_sh=False` (precomputed RGB).

## Forward and backward flow

For context images `I_a, I_b`, RoMaV2 returns `(u_a, u_b, confidence)`. Images are
the actual augmented RGB in `[0,1]`, before backbone normalization. Coordinates
use pixel centers with `grid_sample(..., align_corners=False)` throughout.

Known **context intrinsics**, RANSAC essential-matrix estimation and cheirality
give `R, t`, where `X_b = R X_a + t` and `||t|| = 1`. These are estimated poses,
not dataset extrinsics. Low-confidence, low-parallax, high-reprojection-error
matches and unreliable pairs are discarded.

Triangulation yields `Q_j` in this unit-baseline first-camera frame. Sample the
**raw first-view point map before moment aggregation**, `X_j`, at `u_a`. Estimate
a positive scale after robust residual filtering:

```text
s = sum_j confidence_j <Q_j, stopgrad(X_j)>
    / sum_j confidence_j ||Q_j||^2
C_a = I
C_b = [ R^T | -R^T (s t) ]      # camera-to-world
```

This uses NoPoSplat's canonical first-camera frame; it does not assume a metric
baseline from feature matching. Scale and both cameras are detached. Pose/scale
errors remain possible, so warm-up, rejection and diagnostics matter.

For each Gaussian slot owned by source view `a`, sample source RGB at its
**current** projected center and detach the complete color-sampling path:

```text
c_i^a = stopgrad(bilinear(I_a, project(K_a, C_a, mu_i)))
Ihat_(a->b) = Render({mu_i, Sigma_i, opacity_i, c_i^a}_{i owned by a}, C_b, K_b)
L_(a->b) = sum_j confidence_j |Ihat_(a->b)(u_b,j) - I_b(u_b,j)|_1
           / (3 sum_j confidence_j)
L_aux = mean(L_(a->b), L_(b->a)) over accepted directions/pairs
L_total = L_main + lambda(step) L_aux
```

Only source-owned slots enter each auxiliary render; target-owned slots cannot
trivially paint their own target RGB. Ownership follows the original view-major
slot index, even when allocation aggregates cross-view neighbors. Invalid source
projections are excluded; loss locations are the reliable matched target pixels.
This is **sparse overlap supervision**, not a fabricated intermediate-view image
or a dense visibility label. No auxiliary SSIM, geometry/attribute GT, matching
loss, virtual camera target, or pose-prediction head is introduced.

Auxiliary gradients reach Gaussian centers, covariances and opacities through
the renderer, and then allocation/support prediction. There is no auxiliary
gradient through RoMaV2, OpenCV, scale alignment, observed RGB sampling, or the SH
output head. Shared upstream features still receive geometry-path gradients.
The main reconstruction loss continues to train appearance normally.

Neither context GT extrinsics nor target GT extrinsics enter the auxiliary API.
Clipping planes come from detached predicted support scale, not dataset near/far
values normalized by a GT camera baseline. Target GT pose remains used by the
existing main reconstruction renderer as before.

## Defaults and logging

- RoMaV2 `fast` (512px, no high-resolution refinement), up to 2048 matches.
- One scene pair per local DDP batch, rotating through its elements; two
  auxiliary renders when both directions are valid.
- Steps 0-100: no auxiliary matching/loss; linear weight ramp to `0.05` at step
  300. The main schedule now matches the optimized decoder-only branch:
  `trainer.max_steps=80001`, evaluation/checkpoint intervals of 20000 steps.
  Auxiliary warm-up/ramp and thresholds are unchanged experimental defaults.
- Matching runs before the encoder so its temporary activations do not overlap
  the encoder backward graph. Matcher weights stay on each rank's GPU during
  training, are excluded from the optimizer/checkpoint/DDP, and are released at
  train end. Validation and standalone testing do not run matching.
- RoMaV2 requires `highest` FP32 matmul precision, whereas the backbone enables
  TF32. The adapter temporarily uses `highest` and disables outer autocast during
  matching, then restores the training settings even on early return or error.
  RoMaV2 still controls its own internal mixed precision.
- Watch `aux/attempted_pairs`, `aux/valid_pairs`, `aux/inliers`,
  `aux/alignment_error`, `aux/translation_scale`, `aux/directions`,
  `aux/raw_loss`, `aux/weight`, and `loss/observed_color_aux`. These are per-rank
  training metrics, not a globally synchronized acceptance rate. A zero valid
  count means the auxiliary path is skipping, not successfully supervising.

These thresholds and the weight are initial experimental settings, not tuned
results. Compare training wall time as well as iterations; frozen matching and
two extra renders are additional compute. Increase `every_n_steps` or use
`roma.setting=turbo` to reduce cost; increase `max_pairs_per_batch` for coverage.
On skipped steps the auxiliary loss is zero (no inverse-frequency reweighting).

## Server setup (PRO 6000, torch 2.11.0, CUDA 12.8)

Run from the repository root in the **same environment used by the Slurm script**.
The existing NoPoSplat dependencies and CUDA rasterizer must already work.

The clone is already present; do not clone it again. The integration was checked
against commit `95c9968145c8906b7b59383258e9f73b02853d89` (RoMaV2 2.0.1).
Use Python >=3.10; upstream reports testing Linux/Python 3.12.

```bash
conda activate ragaussian
# Keep the working torch 2.11.0+cu128 environment. This also installs the exact
# CuPy 14.2.0 requirement used by the optimized decoder.
python -m pip install -r requirements-aux.txt

# Run ON A GPU allocation, once before multi-GPU training:
python -m scripts.check_romav2 --device cuda:0
# Optional: use real overlapping images instead of the generated self-pair.
python -m scripts.check_romav2 --image-a /path/a.png --image-b /path/b.png
```

`requirements-aux.txt` pins torch 2.11.0 and torchvision 0.26.0 to protect the
working stack. Check `python -m pip show torch torchvision` first if torchvision
has not been installed. To explicitly obtain those CUDA 12.8 wheels when needed:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
```

The check now performs actual matching and sampling through the training adapter,
using a deterministic generated texture matched with itself by default; no
bundled example images are required. It exercises the loaded CUDA correlation
backend, checks TF32 restoration, reports reliable matches and peak torch memory,
and downloads the model/cache on one process. Zero reliable synthetic matches
do not fail this execution check (sampling may then have been skipped); explicit
real image inputs still fail if no reliable matches are found. This check does
not establish RE10K match quality, pose acceptance, or convergence.
The baseline `opencv-python` dependency is also required by the auxiliary pose path.

RoMaV2 initialization downloads its `romav2.0.1.pt` checkpoint and the pinned
`facebookresearch/dinov3:adc254450203739c8149213a7a69d8d905b4fcfa` torch.hub
**source code**. The default upstream constructor does not separately download
DINOv3 pretrained weights; they are populated from the RoMaV2 checkpoint.
Use a persistent writable `TORCH_HOME` if the cluster's home cache is ephemeral.
For offline jobs, prepopulate that same cache under the training account before
submitting. Every node must see the cache or have its own populated copy.

### Fused local correlation installation

The README calls the fused CUDA extension optional, but this clone's
`pyproject.toml` declares `fused-local-corr` as a Linux dependency. Normal
installation therefore attempts to install it. Building it may need a CUDA
toolkit/compiler compatible with the installed torch wheel and GPU.

If that dependency cannot build, upstream has a native PyTorch fallback. Install
the same runtime dependencies and the clone without resolving the fused package:

```bash
python -m pip install -r requirements-fast.txt 'einops>=0.8.1' 'pillow>=12.0.0' 'rich>=14.2.0' 'tqdm>=4.67.1'
# torch/torchvision and baseline opencv-python must already be installed.
python -m pip install --no-deps -e ./RoMaV2
python -m scripts.check_romav2
```

With this fallback `pip check` may report the missing declared Linux dependency;
the upstream implementation uses native correlation when importing `local_corr`
raises `ImportError`. It may be slower. An installed extension can still fail
at runtime on the actual GPU; the matching smoke check exercises this path.

`RoMaV2/` is ignored by this repository to avoid committing an embedded Git repo
without a submodule configuration. Its contents are unchanged. Clone/install it
on each server checkout; the new integration files are tracked by the parent repo.

## Run and compare

```bash
# This branch's script now defaults to the auxiliary experiment:
bash scripts/train_re10k.sh
# First GPU run: prefetch/execution check in one process, then exercise auxiliary training.
# No separate interactive GPU allocation is required.
mkdir -p logs
ROMA_PREFLIGHT=1 sbatch scripts/train_re10k.sh \
  wandb.mode=disabled trainer.max_steps=20 trainer.auto_eval=false \
  data_loader.train.batch_size=1 \
  train.auxiliary.warm_up_steps=0 train.auxiliary.ramp_steps=0

# Full 80001-step experiment after a successful short run:
sbatch scripts/train_re10k.sh

# Decoder-only control with the original moment settings:
EXPERIMENT=re10k_moment bash scripts/train_re10k.sh
# Main configuration with the auxiliary code explicitly disabled:
bash scripts/train_re10k.sh train.auxiliary.enabled=false
# Change only the matching workload:
bash scripts/train_re10k.sh train.auxiliary.every_n_steps=4
```

For single-GPU environment validation, use the repository's existing checkpoint,
dataset and logger setup and start the auxiliary path immediately. This code uses
`devices="auto"`, so select the GPU with `CUDA_VISIBLE_DEVICES`; `trainer.devices`
is not a supported configuration key. Disable validation and post-training
auto-evaluation for the two-step check:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k_moment_aux wandb.mode=disabled \
  trainer.max_steps=2 trainer.val_check_interval=null trainer.auto_eval=false \
  data_loader.train.batch_size=1 \
  train.auxiliary.warm_up_steps=0 train.auxiliary.ramp_steps=0
```

The short run must log `aux/valid_pairs > 0`, finite `aux/raw_loss`, and completed
backward/optimizer steps on some batches; a run with every pair rejected has not
validated auxiliary learning. The optional `ROMA_PREFLIGHT=1` performs the execution
check before the training process starts; the default remains direct training.
The script inherits the active conda environment and `TORCH_HOME`.

Main training/testing may additionally need the existing data paths and model
checkpoints configured locally.

## Verification scope

CPU tests exercise the real pose/scale code on nonplanar synthetic geometry,
outlier/degenerate-pair rejection, pixel-center sampling, detachment boundaries,
loss gradients with a differentiable renderer double, RoMa's four-return API via
a matcher double, schedules, configuration composition and production training
step wiring. Run:

```bash
python -m unittest discover -s tests -v
```

CPU tests do not establish real RoMa matching accuracy, GPU memory/performance,
CUDA rasterizer backward compatibility, or training convergence. Run the actual
pair check and a short auxiliary-enabled training job on the target GPU before
the full experiment. RoMa has no trainable parameters in this research model,
but its persistent weights still consume GPU memory.
