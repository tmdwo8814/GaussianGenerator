#!/usr/bin/env bash
#SBATCH -J GaussianGenerator-K8
#SBATCH --gres=gpu:high_perf:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -p batch_grad
#SBATCH -w ariel-n1
#SBATCH -t 6-00:00:00
#SBATCH -o logs/knn-ablation-%A.out
#SBATCH --export=ALL

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# sbatch assigns one GPU. For a shell on a four-GPU server, select GPU_ID
# (defaults to physical GPU 0) or supply CUDA_VISIBLE_DEVICES explicitly.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID:-0}}"
fi
export OMP_NUM_THREADS=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$PWD/.cache/wandb}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

K="${K:-8}"
case "$K" in
    4|8|16|32) ;;
    *) echo "K must be one of 4, 8, 16, 32" >&2; exit 2 ;;
esac
echo "KNN ablation K=$K; GPU allocation: ${CUDA_VISIBLE_DEVICES:-Slurm-managed}"

# Reuse an existing login/key using the same helper as resumed evaluation.
# The helper never executes the training script or prints its credential.
python - "$K" "$@" <<'PY'
import os
import sys

from src.eval_checkpoints import inherit_training_auth
from pathlib import Path

inherit_training_auth(Path("scripts/train_re10k.sh"))
os.execv(sys.executable, [
    sys.executable, "-m", "src.main",
    "+experiment=re10k_knn_ablation",
    f"model.encoder.moment_decoder.num_neighbors={sys.argv[1]}",
    f"wandb.mode={os.environ['WANDB_MODE']}",
    *sys.argv[2:],
])
PY
