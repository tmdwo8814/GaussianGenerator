# Diagnose training speed before changing the decoder

## Exact GPU kNN breakdown (current moment decoder)

From the repository root in the training environment, run this once after
updating the code. The existing CuPy installation is sufficient:

```bash
TRAIN_MODULE=scripts.profile_training EXPERIMENT=re10k_moment \
  sbatch scripts/train_re10k.sh --warmup 5 --steps 10 \
  --output outputs/profile_knn.json
```

After **all ranks finish**, summarize the files with:

```bash
python -m scripts.summarize_speed outputs/profile_knn \
  --baseline-seconds 2.076870 --previous-seconds 3.479
```

The reference numbers above are the previous three-GPU, batch-16-per-rank,
256x256 measurements. They are historical references, not a new baseline run.
Keep GPU type/count, batch size, resolution and training settings unchanged.
Use a different output prefix for another run; do not mix old and new rank files.

The diagnostic has five warm-up steps, ten throughput steps and ten synchronized
stage steps, then stops automatically. New stage fields (seconds **summed across
scenes in one local batch**, not seconds per scene):

| Field | Region |
| --- | --- |
| `knn_validate_s` | Finite-value validation, once for the whole batch |
| `knn_prepare_s` | FP32 to FP64 coordinates and DLPack import |
| `knn_build_s` | CuPy KDTree construction |
| `knn_query_s` | Exact K-nearest query (`eps=0`, Euclidean) |
| `knn_export_s` | Candidate index conversion to PyTorch |
| `knn_self_s` | Reserve self without sorting, preserve other candidate order |

The five prepare/build/query/export/self regions are **inside `knn_s`**; never
add them to that total. Batch validation is inside the decoder but outside
`knn_s`. The remainder includes wrapper/stream/cleanup overhead and timing
instrumentation; do not label it all GPU search time. Detailed synchronization
also inflates enclosing stage times, so compare end-to-end speed using the
separate `throughput.step_wall_s` phase only. Its substage timers are disabled.
CPU SciPy search has no CuPy breakdown, reported as `n/a`, not zero.

This patch keeps FP64 search, exact neighbors, K=16, checkpointing, allocation
and moments unchanged. It removes sorting for self placement and checks the
whole batch for invalid points before any search, rather than synchronizing
that check once per scene. It does not introduce a new search algorithm yet.

Run both heads on the **same branch**, GPU allocation, batch size, input size,
data paths and weights. The profile entry point uses the normal `src.main` setup
and performs ordinary forward/backward/optimizer steps. By default it runs:

1. Two warm-up steps.
2. Three throughput steps (CUDA synchronization only at batch boundaries).
3. Three steps with synchronized timings of major stages.

No checkpoint is saved, no WandB run is created, and validation/post-training
evaluation are disabled. `checkpointing.load` is forced to null: this is a short
fresh diagnostic run. To inspect already trained Gaussian shapes, supply that
experiment's checkpoint as `model.encoder.pretrained_weights=/path/model.ckpt`;
the existing loader accepts `state_dict` encoder weights without optimizer state.
Warm-up initializes kernels and optimizer state; these short runs do not measure
converged-model speed unless you supply trained weights. Increase `--warmup` and
`--steps` if timings vary. `--steps` applies to EACH measured phase.

## First comparison: keep the current training resource settings

From the repository root, using the existing Slurm environment setup:

```bash
TRAIN_MODULE=scripts.profile_training EXPERIMENT=re10k \
  sbatch scripts/train_re10k.sh --output outputs/profile_baseline.json

TRAIN_MODULE=scripts.profile_training EXPERIMENT=re10k_moment \
  sbatch scripts/train_re10k.sh --output outputs/profile_moment.json
```

For an already allocated GPU shell, use `bash scripts/train_re10k.sh` instead of
`sbatch`. A CPU-only login node cannot run these commands. The parent training
script still defaults to `src.main` when `TRAIN_MODULE` is unset.

Results are written to `outputs/profile_baseline.rank0.json`, etc., independently
on every DDP rank. The Slurm log prints rank 0's summary. Read **all ranks** when
checking CPU contention or DDP waiting; these are not global averaged timings.

## Interpret the output

- `throughput.step_wall_s`: main comparison; includes batch processing and the
  gap since the preceding batch. The gap includes loader wait, transfer and
  framework work. Batch-boundary synchronization is still present.
- `throughput.compute_step_s`: forward, loss, backward and optimizer within the
  training batch. Warm-up, startup and checkpoint loading are excluded.
- `peak_allocated_gib.throughput`: peak live torch CUDA allocations after warm-up;
  not total GPU memory, allocator reserved memory, CuPy pools or NCCL external allocations.
- `stages.encoder_s`: whole encoder, including the moment head.
- `stages.moment_total_s`: kNN, allocation, aggregation and final attributes for
  all scenes in one local batch.
- `stages.knn_s`, `allocation_s`, `aggregation_s`, `attributes_s`: accumulated
  time for each substage across scenes. `_calls` reports call counts per batch.
- `stages.renderer_s`: the main target-view renderer forward.
- `stages.backward_s`: entire backward, including checkpoint recomputation and
  DDP communication/wait. It does not separately attribute decoder backward.
- `stages.training_step_s`: encoder, renderer, losses, metrics and logging;
  Lightning performs backward and the optimizer step after this function.

Stage timings are **inclusive/nested**, so do not sum encoder + moment + kNN.
Stage measurement synchronizes CUDA before/after each region and changes
execution overlap. Use it to locate a bottleneck, not to claim final throughput.
The two phases use successive training batches/steps, not identical frozen input.

## Change one variable after the first comparison

If allocation/aggregation/decoder backward dominate, test only a larger chunk:

```bash
TRAIN_MODULE=scripts.profile_training EXPERIMENT=re10k_moment \
  sbatch scripts/train_re10k.sh --output outputs/profile_chunk32768.json \
  model.encoder.moment_decoder.chunk_size=32768
```

Keep `checkpoint_chunks=true` for this first test. If measured memory has enough
headroom, a separate run with `checkpoint_chunks=false` tests recomputation cost;
it can materially increase activation memory, even at the original chunk size.
Never compare timings at different batch sizes without labeling the difference.

If kNN or inter-batch gaps dominate, check CPU contention: the current script
requests eight CPUs for three ranks, while the loader requests sixteen workers
per rank. A controlled worker-count test (e.g. `data_loader.train.num_workers=2`)
may help, but can also slow image loading. Keep every other setting fixed.

If the renderer dominates, investigate projected Gaussian size/overlap before
changing neighbor count or covariance scale. A scale change alters the model,
so it is not a pure implementation speedup. Likewise, reducing K changes the
research setting. Preserve K=16 for the first implementation comparisons.

The current decoder already uses exact GPU neighbor search and shared pointwise
projections. Further kernel specialization should follow the measured build vs.
query breakdown above. The profiler adds no changes to Gaussian equations,
precision, neighborhood or gradient checkpoint settings.
