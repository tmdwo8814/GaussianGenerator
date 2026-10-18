#!/usr/bin/env bash
#SBATCH -J GaussianGenerator-eval
#SBATCH --gres=gpu:high_perf:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -p batch_grad
#SBATCH -w ariel-n1
#SBATCH -t 6-00:00:00
#SBATCH -o logs/slurm-eval-%A.out
#SBATCH --export=ALL

set -euo pipefail

# Submit from the repository root, with the ragaussian environment active:
#   mkdir -p logs
#   sbatch scripts/eval_re10k.sh
# Use the existing wandb login, or export WANDB_API_KEY before submitting.
# To continue after another timeout, specify only the unfinished steps:
#   EVAL_STEPS='[120000,140000]' sbatch scripts/eval_re10k.sh

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

export WANDB_MODE=online
export WANDB_ENTITY="${WANDB_ENTITY:-tmdwo8814-}"
export WANDB_PROJECT="${WANDB_PROJECT:-noposplat}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-ob49551z}"
export WANDB_RESUME=must
export WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$PWD/.cache/wandb}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

# Preserve the SDK's existing server configuration (including WANDB_BASE_URL).
# The forge.coreweave.com/wandb browser URL is not the SDK API base URL.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/nas2/data/tmdwo8814/research/GaussianGenerator/outputs/exp_final/2026-10-12_16-47-16/checkpoints}"
EVAL_STEPS="${EVAL_STEPS:-[100000,120000,140000]}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$PWD/outputs/resume_eval/$WANDB_RUN_ID/${SLURM_JOB_ID:-$(date +%Y-%m-%d_%H-%M-%S)}}"

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Checkpoint directory does not exist: $CHECKPOINT_DIR" >&2
    exit 1
fi

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node list: ${SLURM_JOB_NODELIST:-local}"
echo "W&B run: $WANDB_ENTITY/$WANDB_PROJECT/$WANDB_RUN_ID (resume=must)"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Evaluation steps: $EVAL_STEPS"
echo "Output directory: $EVAL_OUTPUT_DIR"
python --version
nvidia-smi

# This entry point only calls Trainer.test; it never starts/resumes training.
# Extra arguments are Hydra overrides, e.g. data_loader.test.num_workers=2.
python -m src.eval_checkpoints \
    "+experiment=${EXPERIMENT:-re10k_moment}" \
    "wandb.entity=$WANDB_ENTITY" \
    "wandb.project=$WANDB_PROJECT" \
    "+wandb.id=$WANDB_RUN_ID" \
    "wandb.mode=online" \
    "+resume_eval.checkpoint_dir=$CHECKPOINT_DIR" \
    "+resume_eval.steps=$EVAL_STEPS" \
    "hydra.run.dir=$EVAL_OUTPUT_DIR" \
    "test.output_path=$EVAL_OUTPUT_DIR/test" \
    "$@"
