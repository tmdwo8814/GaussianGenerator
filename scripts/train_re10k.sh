#!/usr/bin/env bash
#SBATCH -J GaussianGenerator
#SBATCH --gres=gpu:high_perf:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -p batch_grad
#SBATCH -w ariel-n1
#SBATCH -t 5-00:00:00
#SBATCH -o logs/slurm-%A.out
#SBATCH --export=ALL

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# If the batch job cannot find your existing wandb login, submit with:
#   WANDB_API_KEY=your_api_key sbatch scripts/train_re10k_sbatch.sh
# or uncomment the next line and fill it in on the cluster only.
export WANDB_API_KEY="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"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$PWD/.cache/wandb}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

# export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS=1

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node list: ${SLURM_JOB_NODELIST:-local}"
echo "Working directory: $PWD"
python --version
nvidia-smi
wandb status || true

python -m src.main "+experiment=${EXPERIMENT:-re10k_moment_aux}" wandb.mode=online "$@"