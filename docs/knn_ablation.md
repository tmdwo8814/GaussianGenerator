# RE10K KNN ablation (abl/K16)

`re10k_knn_ablation` inherits the Final++ architecture from `re10k_moment`.
The default is **K=16, including self**, one GPU, train batch size 1, 10,000
optimizer steps, and a complete fixed-index evaluation after steps
2,000 / 4,000 / 6,000 / 8,000 / 10,000. Evaluation blocks training and reuses
the model on the same GPU, without allocating another backbone.

## Initialization and training protocol

- Load **only** scene-provider weights from
  `/workspace/research/GaussianGenerator/pretrained_weights/Final++_re10k.ckpt`:
  `backbone`, `downstream_head1/2` (3D points), and
  `gaussian_param_head/gaussian_param_head2` (DPT features).
- Freeze all of these parameters and keep the providers in eval mode throughout
  training. Missing scene keys or mismatched shapes stop the run immediately.
- Initialize `encoder.gaussian_decoder` from scratch. Only its parameters enter
  the optimizer; checkpoint optimizer/step state is not imported from Final++.
- Use the same seed (111123), batch size, sampling distances, architecture,
  optimizer (AdamW, LR 1e-4, 500 warmup steps), and scoring protocol for all K.
  The inherited 75k-step sampling curriculum is disabled, so the final context
  distance range (45–90) is used from step 0 during this short retraining run.
- Use the general exact CuPy KD-tree query for **all** K. The production
  K=16-specific query is deliberately disabled so it cannot favor K=16 FPS.
- Preserve the existing evaluation index, scene-averaged PSNR/SSIM/LPIPS and
  target pose alignment (100 optimization steps). The first five scenes are
  excluded only from FPS; they remain part of image-quality averages.

## Run on the server

Use the existing training environment, including `requirements-fast.txt`.
The dataset root is inherited from `config/dataset/re10k.yaml`.
W&B authentication follows the existing login/environment or the same
training-script credential fallback used by resumed evaluation.

From the repository root, prepare the Slurm log directory before submission:

```bash
mkdir -p logs
sbatch scripts/train_re10k_knn_ablation.sh
```

Slurm reserves one GPU. To run directly on physical GPU 2 of a four-GPU server:

```bash
GPU_ID=2 bash scripts/train_re10k_knn_ablation.sh
```

The script defaults to K=16. Later branches can use the same protocol by changing
only K (the output path and W&B name follow K automatically):

```bash
K=32 sbatch --job-name=GaussianGenerator-K32 scripts/train_re10k_knn_ablation.sh
```

Overrides are forwarded to Hydra. For example, a checkpoint path override:

```bash
bash scripts/train_re10k_knn_ablation.sh \
  model.encoder.pretrained_weights=/another/path/Final++_re10k.ckpt
```

Inspect the resolved configuration without opening a W&B run or loading weights:

```bash
bash scripts/train_re10k_knn_ablation.sh --cfg job --resolve
```

For a short server smoke run, use an index with at least six valid scenes:

```bash
WANDB_MODE=offline bash scripts/train_re10k_knn_ablation.sh \
  trainer.max_steps=2 trainer.eval_every_n_steps=1 \
  trainer.auto_eval_index_path=/path/to/small_re10k_index.json
```

This runs the real training, checkpoint, pose-aligned evaluation and FPS path;
the index must reference scenes present in the test dataset.

## Outputs and the paper table

Each run writes under `outputs/abl/K16/<timestamp>/`:

```text
ablation.json                              # Source checkpoint, seed, GPU, config, index.
metrics.csv                                # One row per evaluated optimizer step.
checkpoints/step_00002000.ckpt               # Model + optimizer/scheduler state.
evaluation/step_00002000/abl_K16/metrics.json  # Quality metrics + FPS + per-scene timings.
evaluation/step_00002000/abl_K16/benchmark.json
evaluation/step_00002000/abl_K16/peak_memory.json
...                                        # Through step_00010000.
```

`metrics.csv` contains `k, step, psnr, ssim, lpips, fps, render_fps, num_samples,
checkpoint`. For the paper's fixed-budget comparison, take **step=10000** for
every K. If selecting a different checkpoint, apply the same selection rule
to every K. These are decoder-retraining ablations with a fixed Final++ scene
encoder, so label them accordingly when comparing with the original full model.

W&B uses `auto_eval/ckpt_step` as the evaluation axis and records
`auto_eval/overall/{psnr,ssim,lpips,fps,render_fps}` in the same training run.

## FPS definition

The main `fps` column is

```text
total target frames / (total scene-encoding seconds + total final-render seconds)
```

Scene encoding includes the backbone, point heads, DPT feature heads, KNN search
and the learned Gaussian decoder. CUDA is synchronized before and after each
timed stage. The first five scenes of each evaluation are warmup, allowing
kernel compilation/caches to settle. Encoding is amortized over the indexed
target views for each scene. It excludes preprocessing, data loading/transfer,
pose optimization, metric calculation and file output. `render_fps` separately
reports rasterization throughput and excludes scene encoding. Timings and frame
counts are saved so the definition is reproducible.

For comparable paper FPS, use the same GPU model and software environment across
K, and avoid other work sharing the selected GPU.

## Checks

```bash
python -m unittest discover -s tests -p test_knn_ablation.py -v
python -m unittest discover -s tests -p test_auto_eval_logging.py -v
python -m unittest discover -s tests -p test_resume_eval.py -v
```

The first suite checks strict scene-only initialization, provider eval modes,
the five evaluation triggers, gradient-accumulation deduplication, state recovery
on failure, FPS weighting/synchronization and W&B checkpoint-axis logging. It
uses production methods with CPU doubles; run the smoke command above on the
server to validate the actual CUDA/Lightning path.
