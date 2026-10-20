# Frozen Final Gaussian-count analysis: 5% and 10%

Run from the repository root in the same Python/CUDA environment used for
training. This is a separate `Trainer.test` entry point; it never trains. The
entire Lightning state is restored with `strict=True`, then all model parameters
are frozen and the model is put in evaluation mode. No learned gates or new
checkpoint parameters are added.

```bash
mkdir -p logs
sbatch scripts/eval_gaussian_control.sh
```

Defaults:

- Checkpoint: `/nas2/data/tmdwo8814/research/GaussianGenerator/outputs/exp_final/2026-10-12_16-47-16/checkpoints/epoch_2-step_140000.ckpt`
- Dataset root: `/local_datasets/3dgs/re10k/re10k` (the current RE10K config).
- Index: `assets/evaluation_index_re10k.json`.
- Reference run: `/nas2/data/tmdwo8814/research/GaussianGenerator/outputs/gaussian_control/458185_2026-10-20_04-58-00/shared_final_pose`.
- Conditions: only 5%/10% pruning and redistribution, four renders per scene.
- Cameras and per-scene Final PSNR/SSIM/LPIPS are loaded from the completed
  reference run. No 100% render/metric evaluation or camera optimization runs.
- One `high_perf` GPU on `ariel-n1`, partition `batch_grad`, eight CPUs, 60 GB
  of system RAM, and six days. The cluster limits system RAM to 91,878 MB per
  allocated GPU; requesting the training script's 128 GB with one GPU is rejected.

Change inputs at submission time:

```bash
DATASET_ROOT=/path/to/re10k \
CHECKPOINT=/path/to/Final.ckpt \
INDEX_PATH=/path/to/evaluation_index.json \
REFERENCE_RUN=/path/to/previous/shared_final_pose \
EVAL_OUTPUT_DIR=outputs/my_control \
sbatch scripts/eval_gaussian_control.sh
```

Run a short server check before the full evaluation:

```bash
MAX_SCENES=2 sbatch scripts/eval_gaussian_control.sh
```

`MAX_SCENES=0` evaluates the complete loader. A positive value is a limited run
and is recorded in metadata. Standard dataset validity filters still apply, as
in the existing evaluation; inspect scene counts in the report. The script
disables W&B and does not require credentials. `logs/` must exist **before**
submission because Slurm opens the output log before starting the script.

Use a reference with the same checkpoint, checkpoint step, dataset roots, index,
resolution, seed and camera protocol. Cameras are matched by scene name plus
**both context and target frame indices**, independent of loader ordering. Each
camera's identity, intrinsics and near/far bounds are checked before reuse. A
missing or incompatible reference raises an error; it never silently reruns
Final or camera optimization. A full evaluation must cover all reference scenes;
`MAX_SCENES` explicitly permits a limited run.

For an existing reference evaluated with dataset cameras:

```bash
ALIGN_POSE=false REFERENCE_RUN=/path/to/previous/dataset_pose sbatch scripts/eval_gaussian_control.sh
```

The script appends `shared_final_pose/` or `dataset_pose/` to its output root so
the protocols have separate results. Additional arguments are Hydra overrides:

```bash
sbatch scripts/eval_gaussian_control.sh data_loader.test.num_workers=2
```

Use `ALIGN_POSE` and `EVAL_OUTPUT_DIR` for protocol/output changes. The effective
configuration and metadata are saved with the results. Use a new output directory
for each job; results from previous evaluations are never accumulated into a new
run. Regional compilation is disabled by default to avoid compilation overhead;
this changes execution strategy without changing weights or model formulas.

## Conditions and correctness

The original Gaussians are reconstructed with the unmodified Final forward pass. Its backbone/DPT output,
original kNN, raw allocation logits, outgoing budgets, incoming mass and appearance
scores are cached once per scene. Original incoming mass determines a stable
descending ranking; ties favor the smaller original slot ID. Both methods share
the same selected indices, kept in their original raster order. No targets or
target poses enter selection or redistribution. The previous result files do
not store features, allocation logits or Gaussian attributes, so this single
encoding/decoding pass is still necessary for pruning and slot selection. Only
the four reduced conditions are rendered and scored; the Final baseline metrics
and cameras come from disk.

| File | Operation | Gaussians at 2 x 256 x 256 |
|---|---|---:|
| `prune_005.txt` | Pruning at 5% | 6,553 |
| `redistribute_005.txt` | Reallocation and decoding at 5% | 6,553 |
| `prune_010.txt` | Pruning at 10% | 13,107 |
| `redistribute_010.txt` | Reallocation and decoding at 10% | 13,107 |

For any resolution, `L=max(1,floor(p*M))`. Pruning gathers all four original
attributes without any modification. Redistribution applies masked softmax to
original logits over surviving candidates. Supports with no surviving candidate
send their full budget to one exact nearest open slot using a SciPy KD-tree
(including CPU transfers in the measured decoding time). New appearance scores
are evaluated with the existing frozen allocation trunk and appearance head.
All supports/features/RGB remain available, including those whose slots closed.

The epsilon prior applies only to true open self edges. Moments, covariance scale
and floor, separate appearance weights, both 2D context heads and nonlinear SH
are recomputed with the original scene scale. Appearance filters geometry and
image edges by destination, using the original view IDs and pixel coordinates.
Intermediate scatter buffers retain M entries to reuse the existing pooling
and readers; attribute heads and the renderer receive exactly L entries. Thus
memory/decoding speed need not scale linearly with L. The numerical tests also
check that redistributing to all slots reproduces the original model.

## Results

The default output location is
`outputs/gaussian_control/<job-id>_<timestamp>/shared_final_pose/`.

- Four condition `.txt` files: PSNR/SSIM/LPIPS, deltas from the reused per-scene
  Final baseline, the existing small/medium/large overlap groups and scene counts.
  No new `final_100.txt`, 25%, 50% or 75% results are produced.
- `redistribution_minus_pruning.txt`: matched-ratio metric differences overall
  and by overlap group. Negative LPIPS differences indicate an improvement.
- `summary.txt`: the full combined report.
- `per_scene.jsonl`: four conditions' metrics, diagnostics and synchronized times
  per scene, plus `reference_final_100` containing the reused baseline values.
  Results are flushed incrementally so partial progress survives interruption.
- `cameras/000000.pt`, etc.: scene name, context/target frame indices, shared
  extrinsics, intrinsics and near/far bounds copied from the reference cameras.
  The original run is read only; these copies make the new results self-contained.
- `metadata.json`, `analysis_config.yaml`, `.hydra/`: inputs, camera protocol,
  checkpoint step, seed, retention fractions, reference directory and effective
  configuration. `final_evaluation_reused=true` records the reuse explicitly.
- `run_status.txt`: `COMPLETE` only after final reports are written. A remaining
  `RUNNING` means the job did not finish; inspect `run.log` and the Slurm log.

Metrics match the existing aggregation: average across target views within each
scene, then equally across scenes. Scenes outside existing overlap bins still
contribute to the overall result. Opacity and covariance-radius quantiles are
computed within each scene, then averaged. Radius is `sqrt(trace(covariance))`.

Diagnostics include actual Gaussian count/fraction, fallback-support fraction,
original retained mass, before/after budget, absolute/relative budget error,
opacity distribution and covariance size. Redistribution aborts if relative
budget error exceeds `1e-5`. Pruning is expected to lose budget. Its reported
`additional_decode_ms` is the tensor gather time.

`shared_encoder_ms` includes the single full Final encoding/decoding.
`shared_pose_alignment_ms` is zero when reference cameras are reused;
`camera_reuse_ms` measures camera loading, checks and transfer.
`additional_decode_ms` measures the extra work for each condition;
`render_ms` measures a separate synchronized render of all targets with shared
cameras, and `render_ms_per_target` divides by target count. Metrics, statistics,
and file I/O are outside those rendering timers. Initial allocator/kernel warmup
is included; raw per-scene times are available for further analysis.

Budget conservation does not imply conservation of rendered color or alpha.
These experiments compare regeneration against removal for the same slots and
count; they do not establish superiority over every pruning method.

## Validation

CPU numerical regression tests (no CUDA renderer required):

```bash
python -m unittest discover -s tests -p 'test_gaussian_control*.py' -v
bash -n scripts/eval_gaussian_control.sh
```

Server smoke evaluation is still required to validate the actual checkpoint,
dataset, CUDA rasterizer and saved camera reuse together. ACID's current
`acid.yaml` selects `dpt_gs`, and no `acid_moment.yaml` exists on this branch;
this script intentionally selects the verified `re10k_moment` Final structure.
