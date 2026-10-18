#!/usr/bin/env bash

set -euo pipefail

# On the SSH server, activate the same Python environment used for training:
#   bash scripts/eval_acid.sh
# To keep running after disconnecting SSH (submit once, from the repository root):
#   mkdir -p logs
#   nohup bash scripts/eval_acid.sh > logs/eval-acid.log 2>&1 < /dev/null &
#   tail -f logs/eval-acid.log
# To evaluate only remaining checkpoints:
#   EVAL_STEPS='[60000,80000]' bash scripts/eval_acid.sh

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export WANDB_MODE=online
export WANDB_ENTITY="${EVAL_WANDB_ENTITY:-tmdwo8814-}"
export WANDB_PROJECT="${EVAL_WANDB_PROJECT:-noposplat}"
export WANDB_RUN_ID="${EVAL_WANDB_RUN_ID:-p05s37ze}"
export WANDB_RESUME=must
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/research/GaussianGenerator/outputs/exp_noposplat_acid/2026-10-16_16-01-45/checkpoints}"
EVAL_STEPS="${EVAL_STEPS:-[20000,40000,60000,80000]}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$PWD/outputs/resume_eval/$WANDB_RUN_ID/$(date +%Y-%m-%d_%H-%M-%S)-$$}"
mkdir -p logs "$EVAL_OUTPUT_DIR"
# Capture setup messages and failures as well as Python/DDP output.
exec > >(tee -a "$EVAL_OUTPUT_DIR/console.log") 2>&1

# Retain the launcher's exit status even if Python cannot print a traceback.
trap 'eval_exit_status=$?; printf "[ResumeEval] launcher exit_code=%s\n" "$eval_exit_status"; exit "$eval_exit_status"' EXIT

echo "Working directory: $PWD"
echo "Console log: $EVAL_OUTPUT_DIR/console.log"

# Reuse only the literal key; sourcing train_re10k.sh would start training.
# An explicitly exported WANDB_API_KEY takes precedence. Never print the key.
WANDB_KEY_SOURCE=environment
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    WANDB_KEY_SOURCE=scripts/train_re10k.sh
    WANDB_API_KEY=""
    if [[ -f scripts/train_re10k.sh ]]; then
        WANDB_API_KEY="$(sed -n 's/^export WANDB_API_KEY="\([[:alnum:]_-]*\)"[[:space:]]*$/\1/p' scripts/train_re10k.sh | head -n 1)"
    fi
fi
if [[ -z "$WANDB_API_KEY" ]]; then
    echo "W&B authentication key not found. Export WANDB_API_KEY or set it in scripts/train_re10k.sh." >&2
    exit 1
fi
export WANDB_API_KEY
export WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$PWD/.cache/wandb}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"
echo "W&B authentication: key loaded from $WANDB_KEY_SOURCE (value hidden)"

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Checkpoint directory does not exist: $CHECKPOINT_DIR" >&2
    exit 1
fi

# Preserve the SDK's existing API server configuration. The browser URL
# forge.coreweave.com/wandb is not an API base URL.
echo "W&B run: $WANDB_ENTITY/$WANDB_PROJECT/$WANDB_RUN_ID (resume=must)"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Evaluation steps: $EVAL_STEPS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Python executable: $(command -v python)"
echo "Progress files: $EVAL_OUTPUT_DIR/progress-rank*.jsonl"

python -u -X faulthandler -m src.eval_checkpoints \
    +experiment=acid \
    "wandb.entity=$WANDB_ENTITY" \
    "wandb.project=$WANDB_PROJECT" \
    "+wandb.id=$WANDB_RUN_ID" \
    wandb.mode=online \
    "+resume_eval.checkpoint_dir=$CHECKPOINT_DIR" \
    "+resume_eval.steps=$EVAL_STEPS" \
    "hydra.run.dir=$EVAL_OUTPUT_DIR" \
    "test.output_path=$EVAL_OUTPUT_DIR/test" \
    "$@"
