#!/usr/bin/env bash
#SBATCH -J FinalPlus-control
#SBATCH --gres=gpu:high_perf:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH -p batch_grad
#SBATCH -w ariel-n1
#SBATCH -t 6-00:00:00
#SBATCH -o slurm-gaussian-control-%j.out
#SBATCH --export=ALL

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

CHECKPOINT="${CHECKPOINT:-/workspace/research/GaussianGenerator/pretrained_weights/Final++_RE10k.ckpt}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/datasets/re10k}"
EVALUATION_INDEX="${EVALUATION_INDEX:-$PWD/assets/evaluation_index_re10k.json}"
# Stable paths allow an interrupted sbatch job to resume without inference.
CACHE_DIR="${CACHE_DIR:-$PWD/outputs/gaussian_control/final_plus_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/gaussian_control/final_plus_results}"
STAGE="${STAGE:-all}"
PERCENTAGES="${PERCENTAGES:-[100,75,50,25,10,5,1]}"

mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/run.log") 2>&1
trap 'code=$?; printf "[Control] ERROR exit=%s line=%s\n" "$code" "$LINENO" >&2' ERR
printf '[Control] stage=%s checkpoint=%s\ncache=%s\nresults=%s\n' "$STAGE" "$CHECKPOINT" "$CACHE_DIR" "$OUTPUT_DIR"

# The active conda environment supplies Python/CUDA dependencies, as in train_re10k.sh.
# CLI Hydra overrides come last and may override every default below.
python -u -m src.eval_gaussian_control \
  "checkpointing.load='$CHECKPOINT'" \
  "dataset.re10k.roots=['$DATASET_ROOT']" \
  "dataset.re10k.view_sampler.index_path='$EVALUATION_INDEX'" \
  "control.stage=$STAGE" \
  "control.cache_dir='$CACHE_DIR'" \
  "control.output_dir='$OUTPUT_DIR'" \
  "control.percentages=$PERCENTAGES" \
  "$@"
