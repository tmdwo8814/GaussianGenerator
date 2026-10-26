#!/usr/bin/env bash
#SBATCH -J FinalPlus-LightGaussian
#SBATCH --gres=gpu:high_perf:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -p batch_grad
#SBATCH -w ariel-n1
#SBATCH -t 3-00:00:00
#SBATCH -o slurm-lightgaussian-%j.out
#SBATCH --export=ALL

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# With sbatch, respect the scheduler's GPU assignment. On the Docker server,
# require an explicit free GPU so the old experiment's GPU 0 is not used.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  if [[ -n "${GPU_ID:-}" ]]; then
    printf 'Under sbatch, omit GPU_ID and use the allocated GPU.\n' >&2
    exit 2
  fi
elif [[ -n "${GPU_ID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
else
  printf 'Choose the unused GPU explicitly: GPU_ID=1 bash scripts/eval_lightgaussian_control.sh\n' >&2
  exit 2
fi

export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MAX_JOBS=1
CACHE_DIR="${CACHE_DIR:-$PWD/outputs/gaussian_control/final_plus_cache}"
LIGHTGAUSSIAN_OUTPUT_DIR="${LIGHTGAUSSIAN_OUTPUT_DIR:-$PWD/outputs/gaussian_control/final_plus_lightgaussian_results}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/datasets/re10k}"

# All output validation and logging belong to the Python process, after checking
# separation from the live cache/results. Never tee into an unchecked path.
command=(python -u -m src.eval_lightgaussian_control
  --cache-dir "$CACHE_DIR" --output-dir "$LIGHTGAUSSIAN_OUTPUT_DIR"
  --dataset-root "$DATASET_ROOT" --threads 1 --timing-mode concurrent)
if [[ -n "${CPU_SET:-}" ]]; then
  command=(taskset -c "$CPU_SET" "${command[@]}")
fi
if command -v ionice >/dev/null 2>&1; then
  command=(ionice -c 3 "${command[@]}")
fi
if command -v nice >/dev/null 2>&1; then
  command=(nice -n 10 "${command[@]}")
fi
exec "${command[@]}" "$@"
