# Finish Final++ checkpoint evaluation

The training run finished at 140k, but evaluation only completed 20k, 40k and
60k. The launcher evaluates **80k, 100k, 120k and 140k** in that order, using
the Hydra-based evaluation entry point from `exp/Final`. It never calls
`Trainer.fit` and does not need optimizer or scheduler checkpoint states.

Defaults:

- Checkpoints: `/nas2/data/tmdwo8814/research/GaussianGenerator/outputs/exp_final_plus/2026-10-16_18-35-06/checkpoints`
- Existing W&B run: `tmdwo8814-/noposplat/qqvj055m`
- Experiment: `re10k_moment` on `exp/Final++`.

From the repository root, in the activated `ragaussian` environment:

```bash
sbatch scripts/eval_re10k.sh
```

`sbatch scripts/eval_remaining_final_plus.sh` is an alias with the same Slurm
resources and behavior. Submit one of these commands, not both.

The launchers now compose the **current branch's** `config/main.yaml` with
`+experiment=re10k_moment`, just like the working Final launcher. They no
longer require the old run's `.hydra/config.yaml`. Keep the original Final++
model, dataset, loss and pose-alignment settings when comparing these scores
with 20k/40k/60k. Strict checkpoint loading catches architecture mismatches,
but cannot detect a changed evaluation index or pose-alignment setting.
Additional launcher arguments are Hydra overrides.

W&B authentication first uses an exported `WANDB_API_KEY`. If absent, the
entry point reads a literal key assignment (or its literal default) from
`scripts/train_re10k.sh`, matching the working training setup. It never
executes that script or prints the key. Otherwise W&B's saved login is used.
No API key is copied into the new files, and no W&B API host is changed.

Optional preflight, in the same installed training environment:

```bash
bash scripts/eval_re10k.sh --dry-run
```

This checks the composed config, evaluation index and checkpoint filenames
without constructing a Trainer or opening W&B. It still imports the model
dependencies and writes Hydra/console logs; it is not a GPU execution check.

Evaluation uses the existing ModelWrapper metric, overlap grouping and
distributed sum/count reduction. The evaluation sampler and save flags match
post-training auto-eval; pose alignment keeps its configured iterations,
learning rates and losses. DataModule rank is assigned after DDP launch so
each GPU reads its own shard.

The evaluator reconnects with `resume="must"`, without setting a new run
name/config/tags/notes. Rank zero appends:
`auto_eval/{overall,small,medium,large}/{psnr,ssim,lpips,num_samples}` with
`auto_eval/ckpt_step` as the plot x-axis. It does not rewind W&B's internal
history step. Existing 20k/40k/60k history is preserved; summary values advance
to the latest completed evaluation.

Logs and results:

- `slurm-eval-final-plus-<job_id>.out` in the submission directory. Slurm
  opens this before the script starts, so it does not depend on `logs/` existing.
- `outputs/resume_eval/qqvj055m/<job_id>/console.log`: stdout and stderr,
  with unbuffered Python output, full Hydra tracebacks and normal exit status.
- `outputs/resume_eval/qqvj055m/<job_id>/test/step_<8-digit-step>/metrics.json`:
  local global/overlap scores saved after each completed checkpoint.
- Hydra config and benchmark files are stored under the same output tree.

A forced kill or node failure may prevent the exit trap from running. If the
log ends abruptly, check scheduler status:

```bash
sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed,Timelimit,MaxRSS
```

If only 120k and 140k remain, resume with:

```bash
sbatch scripts/eval_re10k.sh --steps 120000 140000
```

Environment overrides include `CHECKPOINT_DIR`, `EVAL_STEPS` (Hydra list,
e.g. `[120000,140000]`), `EVAL_OUTPUT_DIR`, `EXPERIMENT`, `WANDB_ENTITY`,
`WANDB_PROJECT` and `WANDB_RUN_ID`. The scripts do not deduplicate W&B
history; submit only unfinished steps and use one job per run at a time.

Local checks (no external logging):

```bash
python -m unittest discover -s tests -p 'test_*eval*.py' -v
bash -n scripts/eval_re10k.sh
bash -n scripts/eval_remaining_final_plus.sh
```
