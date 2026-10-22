#!/usr/bin/env bash
#SBATCH -J FinalPlus-eval
#SBATCH --gres=gpu:high_perf:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -p batch_grad
#SBATCH -w ariel-n1
#SBATCH -t 6-00:00:00
#SBATCH -o slurm-eval-final-plus-%j.out
#SBATCH --export=ALL

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export WANDB_MODE=online
export WANDB_ENTITY="${WANDB_ENTITY:-tmdwo8814-}"
export WANDB_PROJECT="${WANDB_PROJECT:-noposplat}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-qqvj055m}"
export WANDB_RESUME=must
export WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$PWD/.cache/wandb}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
# The Python entry point reuses training-script authentication if needed,
# without executing the training script or copying its key into this file.

TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-/nas2/data/tmdwo8814/research/GaussianGenerator/outputs/exp_final_plus/2026-10-16_18-35-06}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$TRAIN_RUN_DIR/checkpoints}"
EVAL_STEPS="${EVAL_STEPS:-[80000,100000,120000,140000]}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$PWD/outputs/resume_eval/$WANDB_RUN_ID/${SLURM_JOB_ID:-$(date +%Y-%m-%d_%H-%M-%S)}}"

# Slurm opens its stdout BEFORE this script runs. Its path above deliberately
# uses the existing submission directory rather than a possibly absent logs/.
mkdir -p "$EVAL_OUTPUT_DIR"
exec > >(tee -a "$EVAL_OUTPUT_DIR/console.log") 2>&1
trap 'code=$?; printf "[ResumeEval] ERROR: exit=%s line=%s\n" "$code" "$LINENO" >&2' ERR
trap 'code=$?; printf "[ResumeEval] EXIT: %s at %s\n" "$code" "$(date -Is)"; exit "$code"' EXIT
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node list: ${SLURM_JOB_NODELIST:-local}"
echo "Working directory: $PWD"
echo "W&B run: $WANDB_ENTITY/$WANDB_PROJECT/$WANDB_RUN_ID (resume=must)"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Evaluation output: $EVAL_OUTPUT_DIR"

# Preserve the previous launcher's --steps / --dry-run convenience options;
# all other arguments are Hydra overrides, as in train_re10k.sh.
extra=()
dry_run=false
while (($#)); do
  case "$1" in
    --steps)
      shift
      steps=()
      while (($#)) && [[ "$1" =~ ^[0-9]+$ ]]; do
        steps+=("$1")
        shift
      done
      if ((${#steps[@]} == 0)); then
        echo "--steps requires checkpoint steps, e.g. --steps 120000 140000" >&2
        exit 2
      fi
      joined=$(IFS=,; echo "${steps[*]}")
      EVAL_STEPS="[$joined]"
      ;;
    --dry-run)
      extra+=("+resume_eval.dry_run=true")
      dry_run=true
      shift
      ;;
    *) extra+=("$1"); shift ;;
  esac
done
echo "Evaluation steps: $EVAL_STEPS"
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "Checkpoint directory not found: $CHECKPOINT_DIR" >&2
  exit 1
fi
python --version
if ! $dry_run; then
  nvidia-smi
fi

python -u -m src.eval_checkpoints \
  "+experiment=${EXPERIMENT:-re10k_moment}" \
  "wandb.entity=$WANDB_ENTITY" \
  "wandb.project=$WANDB_PROJECT" \
  "+wandb.id=$WANDB_RUN_ID" \
  "wandb.mode=online" \
  "+resume_eval.checkpoint_dir=$CHECKPOINT_DIR" \
  "+resume_eval.steps=$EVAL_STEPS" \
  "hydra.run.dir=$EVAL_OUTPUT_DIR" \
  "test.output_path=$EVAL_OUTPUT_DIR/test" \
  "${extra[@]}"
