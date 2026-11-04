# Resume Final++ budget 1 evaluation

Evaluate only **100k, 120k, 140k**, appending results to the existing W&B run
`tmdwo8814-/noposplat/un2ifuj6` with `resume=must`:

https://forge.coreweave.com/wandb/tmdwo8814-/noposplat/runs/un2ifuj6

Training directory:
`/nas2/data/tmdwo8814/research/GaussianGenerator/outputs/exp_final_plus_budget1/2026-10-28_15-14-44`

Selected checkpoints under `checkpoints/`:

- `epoch_1-step_100000.ckpt`
- `epoch_2-step_120000.ckpt`
- `epoch_2-step_140000.ckpt`

From the repository root in the activated training environment:

```bash
sbatch scripts/eval_remaining_budget1.sh
```

Optional preflight (no Trainer or W&B run is opened):

```bash
bash scripts/eval_remaining_budget1.sh --dry-run
```

The launcher reads the saved `.hydra/config.yaml`, keeping the original model,
dataset, losses and pose-alignment settings. The existing evaluator switches to
the same evaluation sampler and save flags used for post-training auto-eval.
It verifies budget_max=1, learned allocation and support-wise budgets before
opening W&B. Old snapshots without mode flags are treated as the original model.
Checkpoint loading is strict. No training, optimizer or scheduler restoration is
performed. Model code must still be compatible with the trained Final++ model.

Missing checkpoints or saved config stop the job with an explicit error. There
is no silent fallback to current ablation settings. If the config was moved,
supply the original snapshot using `BUDGET1_CONFIG=/path/to/config.yaml` when
submitting. `BUDGET1_TRAIN_RUN_DIR` can relocate the entire training directory.

Logging follows the existing resumed-eval path:

- W&B: `auto_eval/{overall,small,medium,large}/{psnr,ssim,lpips,num_samples}`.
- Plot x-axis: `auto_eval/ckpt_step` (100000, 120000, 140000).
- Slurm stdout/stderr: `slurm-eval-budget1-<job_id>.out` in the submission directory.
- Console log: `outputs/resume_eval/un2ifuj6/<job_id>/console.log`.
- Local scores: `outputs/resume_eval/un2ifuj6/<job_id>/test/step_00100000/metrics.json`
  (and `step_00120000`, `step_00140000`).

Existing run name/config/tags and history are preserved. Summary metrics advance
to the last completed checkpoint. W&B's internal history step is not rewound.
Credentials use the existing environment/saved-login/training-script mechanism;
no key is embedded in this launcher and no W&B API host is changed.

Only run one evaluation job for this W&B run at a time. If interrupted after
100k completes, submit only the unfinished steps to avoid duplicate history:

```bash
sbatch scripts/eval_remaining_budget1.sh --steps 120000 140000
```

The Slurm stdout path does not require a pre-existing `logs/` directory. A hard
timeout can interrupt the console exit trap; inspect scheduler status with:

```bash
sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed,Timelimit,MaxRSS
```
