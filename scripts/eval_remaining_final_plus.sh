#!/bin/bash
#SBATCH -J FinalPlusEval
#SBATCH --gres=gpu:high_perf:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -p batch_grad
#SBATCH -w ariel-n1
#SBATCH -t 6-00:00:00
#SBATCH -o logs/eval-final-plus-%A.out
#SBATCH --export=ALL

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Submit from the activated ragaussian environment, as for train_re10k.sh.
# Use an existing `wandb login` or inherited WANDB_API_KEY; no credentials in this file.
export WANDB_MODE=online
export WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$PWD/.cache/wandb}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-/nas2/data/tmdwo8814/research/GaussianGenerator/outputs/exp_final_plus/2026-10-16_18-35-06}"
WANDB_RUN_PATH="${WANDB_RUN_PATH:-tmdwo8814-/noposplat/qqvj055m}"

python -m scripts.eval_checkpoints \
  --run-dir "$TRAIN_RUN_DIR" \
  --wandb-run "$WANDB_RUN_PATH" \
  "$@"
