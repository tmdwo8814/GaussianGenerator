#!/usr/bin/env bash
#SBATCH -J FinalPlus-budget1-eval
#SBATCH --gres=gpu:high_perf:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -p batch_grad
#SBATCH -w ariel-n1
#SBATCH -t 6-00:00:00
#SBATCH -o slurm-eval-budget1-%j.out
#SBATCH --export=ALL

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# This launcher belongs to the existing budget_max=1 training run. Do not
# inherit another experiment's run ID or steps from the submission environment.
export WANDB_ENTITY=tmdwo8814-
export WANDB_PROJECT=noposplat
export WANDB_RUN_ID=un2ifuj6
export TRAIN_RUN_DIR="${BUDGET1_TRAIN_RUN_DIR:-/nas2/data/tmdwo8814/research/GaussianGenerator/outputs/exp_final_plus_budget1/2026-10-28_15-14-44}"
export CHECKPOINT_DIR="$TRAIN_RUN_DIR/checkpoints"
export EVAL_STEPS='[100000,120000,140000]'
export EVAL_CONFIG_PATH="${BUDGET1_CONFIG:-$TRAIN_RUN_DIR/.hydra/config.yaml}"
export EVAL_OUTPUT_DIR="${BUDGET1_EVAL_OUTPUT_DIR:-$PWD/outputs/resume_eval/$WANDB_RUN_ID/${SLURM_JOB_ID:-$(date +%Y-%m-%d_%H-%M-%S)}}"

# Reuse the working logger, auth, durable console log and per-step result path.
# Historical configs lack the mode flags; the evaluator checks their legacy
# defaults as learned allocation + support-wise budgeting.
exec bash scripts/eval_re10k.sh \
  '+resume_eval.expected_decoder.budget_max=1.0' \
  '+resume_eval.expected_decoder.allocation_mode=learned' \
  '+resume_eval.expected_decoder.budget_mode=support' \
  "$@"
