#!/usr/bin/env bash
# SSH batch runner: M1-M7, four GPUs reused sequentially, fresh decoder per model.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$PWD/.cache/wandb}"
encoder_checkpoint="${ENCODER_CHECKPOINT:-/workspace/research/GaussianGenerator/pretrained_weights/re10k.ckpt}"
if [[ ! -f "$encoder_checkpoint" ]]; then
  printf 'NoPoSplat checkpoint not found: %s\n' "$encoder_checkpoint" >&2
  exit 2
fi

read -r -a models <<< "${MODELS:-M1 M2 M3 M4 M5 M6 M7}"
if (( ${#models[@]} == 0 )); then
  printf 'MODELS must contain at least one of M1-M7.\n' >&2
  exit 2
fi
declare -A seen_models=()
for model in "${models[@]}"; do
  case "$model" in M[1-7]) ;; *) printf 'Unknown model: %s\n' "$model" >&2; exit 2 ;; esac
  if [[ -n "${seen_models[$model]:-}" ]]; then
    printf 'Duplicate model: %s\n' "$model" >&2
    exit 2
  fi
  seen_models[$model]=1
done

batch_dir="${RUN_ROOT:-$PWD/outputs/supplementary/$(date +%Y-%m-%d_%H-%M-%S)}"
mkdir -p "$batch_dir"
batch_dir="$(cd -- "$batch_dir" && pwd)"
for model in "${models[@]}"; do
  if [[ -e "$batch_dir/$model" ]]; then
    printf 'Existing model directory: %s. Use a fresh RUN_ROOT.\n' "$batch_dir/$model" >&2
    exit 2
  fi
done
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"
printf '%s\n' "${models[@]}" > "$batch_dir/variants.txt"
printf '%s\n' "$encoder_checkpoint" > "$batch_dir/encoder_checkpoint.txt"
printf 'Supplementary batch: %s\n' "$batch_dir"

current_run=''
on_exit() {
  exit_code=$?
  if [[ -n "$current_run" && ! -e "$current_run/COMPLETED" ]]; then
    printf 'exit_code=%s\n' "$exit_code" > "$current_run/FAILED"
  fi
}
trap on_exit EXIT

for model in "${models[@]}"; do
  current_run="$batch_dir/$model"
  mkdir -p "$current_run"
  date -Is > "$current_run/RUNNING"
  printf '\nStarting %s: %s\n' "$model" "$current_run"
  python -u -m src.main "+experiment=re10k_supp_${model,,}" \
    "model.encoder.pretrained_weights=$encoder_checkpoint" \
    "hydra.run.dir=$current_run" "test.output_path=$current_run/test" \
    wandb.mode=online "$@" 2>&1 | tee "$current_run/console.log"
  python scripts/supplementary_status.py "$batch_dir" --write-csv --require-complete "$model"
  date -Is > "$current_run/COMPLETED"
  current_run=''
done

python scripts/supplementary_status.py "$batch_dir" --write-csv
