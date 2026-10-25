#!/usr/bin/env bash
#SBATCH -J GaussianGenerator-exp3
#SBATCH --gres=gpu:high_perf:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=45G
#SBATCH -p batch_grad
#SBATCH -w ariel-n1
#SBATCH -t 1-00:00:00
#SBATCH -o logs/slurm-exp3-%A.out
#SBATCH --export=ALL

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
fi
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

ORIGINAL_CHECKPOINT="${ORIGINAL_CHECKPOINT:-/nas2/data/tmdwo8814/research/GaussianGenerator/outputs/exp_noposplat/2026-10-12_16-46-59/checkpoints/epoch_2-step_140000.ckpt}"
OURS_CHECKPOINT="${OURS_CHECKPOINT:-/nas2/data/tmdwo8814/research/GaussianGenerator/outputs/exp_final_plus/2026-10-16_18-35-06/checkpoints/epoch_2-step_140000.ckpt}"
DATASET_ROOT="${DATASET_ROOT:-/local_datasets/3dgs/re10k/re10k}"
INDEX_PATH="${INDEX_PATH:-$PWD/assets/evaluation_index_re10k.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/exp3/output/$(date +%Y%m%d_%H%M%S)}"

python -m exp3.plot_scale_opacity \
    --original-checkpoint "$ORIGINAL_CHECKPOINT" \
    --ours-checkpoint "$OURS_CHECKPOINT" \
    --data-root "$DATASET_ROOT" \
    --index "$INDEX_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --num-scenes "${NUM_SCENES:-500}" \
    --sample-size "${SAMPLE_SIZE:-60000}" \
    --num-workers "${NUM_WORKERS:-4}" \
    --seed "${SEED:-0}" \
    "$@"
