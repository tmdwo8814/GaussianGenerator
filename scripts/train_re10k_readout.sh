#!/usr/bin/env bash
# Run directly in an SSH shell; no Slurm commands or embedded credentials.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$PWD/.cache/wandb}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

# Authenticate beforehand with `wandb login`, or export WANDB_API_KEY.
# Respect the existing credentials and use a separate W&B run per experiment.
exec python -u -m src.main \
  "+experiment=${EXPERIMENT:-re10k_regression}" \
  wandb.mode=online "$@"
