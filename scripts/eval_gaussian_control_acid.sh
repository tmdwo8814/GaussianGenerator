#!/usr/bin/env bash
#SBATCH -J FinalPlus-RE10K-to-ACID
#SBATCH --gres=gpu:high_perf:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH -p batch_grad
#SBATCH -w ariel-n1
#SBATCH -t 6-00:00:00
#SBATCH -o slurm-gaussian-control-acid-%j.out
#SBATCH --export=ALL

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  if [[ -n "${GPU_ID:-}" ]]; then
    printf 'Under sbatch, omit GPU_ID and use the allocated GPU.\n' >&2
    exit 2
  fi
elif [[ -n "${GPU_ID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MAX_JOBS=1

command=(python -u scripts/run_gaussian_control_acid.py
  --checkpoint "${CHECKPOINT:-/workspace/research/GaussianGenerator/pretrained_weights/Final++_RE10k.ckpt}"
  --dataset-root "${DATASET_ROOT:-/workspace/datasets/acid}"
  --index-path "${EVALUATION_INDEX:-$PWD/assets/evaluation_index_acid.json}"
  --output-root "${OUTPUT_ROOT:-$PWD/outputs/gaussian_control/re10k_to_acid}"
  --stage "${STAGE:-all}")
if [[ -n "${CACHE_DIR:-}" ]]; then
  command+=(--cache-dir "$CACHE_DIR")
fi
exec "${command[@]}" "$@"
