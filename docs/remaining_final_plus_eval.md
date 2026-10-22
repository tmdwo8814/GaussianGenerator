# Finish Final++ checkpoint evaluation

The training run finished at 140k, but its post-training evaluation only
completed 20k, 40k and 60k. This script evaluates **80k, 100k, 120k and 140k**
without calling `Trainer.fit` or requiring optimizer/scheduler checkpoint state.
The default destinations are:

- Training output: `/nas2/data/tmdwo8814/research/GaussianGenerator/outputs/exp_final_plus/2026-10-16_18-35-06`
- Existing W&B run: `tmdwo8814-/noposplat/qqvj055m`

From the repository root, in the same activated `ragaussian` environment:

```bash
# Only needed if this environment has no saved login or WANDB_API_KEY.
wandb login

# Checks paths and saved configuration without CUDA or W&B writes.
bash scripts/eval_remaining_final_plus.sh --dry-run

sbatch scripts/eval_remaining_final_plus.sh
```

Use `exp/Final++`, the code that trained these checkpoints. The script reads
`RUN_DIR/.hydra/config.yaml`; it does not compose today's experiment YAML.
If that file is missing, restore the saved training config and pass
`--config /path/to/config.yaml`. Do not silently substitute a different model
or dataset configuration.

Evaluation uses the existing ModelWrapper's metric, overlap grouping and
distributed sum/count reduction. It retains the saved pose alignment settings
(including its iterations, learning rates and losses) and uses the original
auto-eval index and image-save options. Strict checkpoint loading detects
architecture mismatches. DDP rank assignment is updated after launch so each
rank evaluates its own test scenes.

The evaluator reconnects with `resume="must"` and does not supply new run
name/config/tags/notes. Only rank zero writes W&B metrics, using the same keys:
`auto_eval/{overall,small,medium,large}/{psnr,ssim,lpips,num_samples}`.
`auto_eval/ckpt_step` is the plot x-axis. These evaluations append new history
rows; no attempt is made to rewind W&B's internal history counter to 80k.
The original 20k/40k/60k history remains, while summary values advance to the
latest evaluated checkpoint. See [W&B resume](https://docs.coreweave.com/products/wandb/runs/resuming).

The output log is `logs/eval-final-plus-<job_id>.out`. Each completed checkpoint
also saves a local result at `RUN_DIR/remaining_eval/step_<step>/metrics.json`
and benchmark files under that step directory.

If another time limit stops the evaluation, submit only the unfinished steps:

```bash
# Example: 80k and 100k completed; evaluate the remaining two.
sbatch scripts/eval_remaining_final_plus.sh --steps 120000 140000
```

The script does not automatically deduplicate existing W&B history; avoid
resubmitting already completed steps or running two jobs against this run
simultaneously. No API key is embedded in the new files: they use the existing
W&B login or inherited `WANDB_API_KEY`.

Local checks (no external logging):

```bash
python -m unittest discover -s tests -p 'test_eval_checkpoints.py' -v
python -m unittest discover -s tests -p 'test_auto_eval_logging.py' -v
```
