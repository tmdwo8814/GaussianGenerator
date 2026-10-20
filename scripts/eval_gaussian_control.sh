#!/usr/bin/env bash
#SBATCH -J GaussianGenerator-control
#SBATCH --gres=gpu:high_perf:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=45G
#SBATCH -p batch_grad
#SBATCH -w ariel-n1
#SBATCH -t 6-00:00:00
#SBATCH -o logs/slurm-control-%A.out
#SBATCH --export=ALL

set -euo pipefail

# Activate the training environment, then submit from the repository root:
#   mkdir -p logs
#   sbatch scripts/eval_gaussian_control.sh
# Optional environment overrides:
#   CHECKPOINT=/path/Final.ckpt DATASET_ROOT=/path/re10k sbatch scripts/eval_gaussian_control.sh
#   REFERENCE_RUN=/path/to/previous/shared_final_pose sbatch scripts/eval_gaussian_control.sh
#   MAX_SCENES=2 sbatch scripts/eval_gaussian_control.sh  # smoke test, not full results
# Extra arguments are Hydra overrides, e.g. data_loader.test.num_workers=2.
# This job uses ONE GPU, caches one scene's encoding/allocation, and evaluates
# only 5%/10% pruning and redistribution. Previous cameras and Final metrics
# are reused; there is no full-Final render/evaluation or camera optimization.

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

CHECKPOINT="${CHECKPOINT:-/nas2/data/tmdwo8814/research/GaussianGenerator/outputs/exp_final/2026-10-12_16-47-16/checkpoints/epoch_2-step_140000.ckpt}"
DATASET_ROOT="${DATASET_ROOT:-/local_datasets/3dgs/re10k/re10k}"
INDEX_PATH="${INDEX_PATH:-$PWD/assets/evaluation_index_re10k.json}"
REFERENCE_RUN="${REFERENCE_RUN:-/nas2/data/tmdwo8814/research/GaussianGenerator/outputs/gaussian_control/458185_2026-10-20_04-58-00/shared_final_pose}"
ALIGN_POSE="${ALIGN_POSE:-true}"
MAX_SCENES="${MAX_SCENES:-0}"
case "$ALIGN_POSE" in
    true) PROTOCOL=shared_final_pose ;;
    false) PROTOCOL=dataset_pose ;;
    *) echo "ALIGN_POSE must be true or false" >&2; exit 1 ;;
esac
if [[ ! "$MAX_SCENES" =~ ^[0-9]+$ ]]; then
    echo "MAX_SCENES must be a nonnegative integer (0 = full evaluation)" >&2
    exit 1
fi
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$PWD/outputs/gaussian_control/${SLURM_JOB_ID:-local}_$(date +%Y-%m-%d_%H-%M-%S)}/$PROTOCOL"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint does not exist: $CHECKPOINT" >&2
    exit 1
fi
if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "Dataset root does not exist: $DATASET_ROOT (set DATASET_ROOT)" >&2
    exit 1
fi
if [[ ! -f "$INDEX_PATH" ]]; then
    echo "Evaluation index does not exist: $INDEX_PATH (set INDEX_PATH)" >&2
    exit 1
fi
for required in metadata.json per_scene.jsonl run_status.txt; do
    if [[ ! -f "$REFERENCE_RUN/$required" ]]; then
        echo "Reference evaluation file does not exist: $REFERENCE_RUN/$required (set REFERENCE_RUN)" >&2
        exit 1
    fi
done
if [[ -e "$EVAL_OUTPUT_DIR/per_scene.jsonl" ]]; then
    echo "Output already contains an evaluation: $EVAL_OUTPUT_DIR (set a new EVAL_OUTPUT_DIR)" >&2
    exit 1
fi
mkdir -p "$EVAL_OUTPUT_DIR"
EVAL_OUTPUT_DIR="$(cd "$EVAL_OUTPUT_DIR" && pwd)"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node list: ${SLURM_JOB_NODELIST:-local}"
echo "Checkpoint: $CHECKPOINT"
echo "Dataset root: $DATASET_ROOT"
echo "Evaluation index: $INDEX_PATH"
echo "Camera protocol: $PROTOCOL"
echo "Retention fractions: 5%, 10% (pruning and redistribution only)"
echo "Reuse cameras and per-scene Final metrics from: $REFERENCE_RUN"
echo "Output directory: $EVAL_OUTPUT_DIR"
python --version
nvidia-smi

python -m src.eval_gaussian_control \
    +experiment=re10k_moment \
    mode=test \
    wandb.mode=disabled \
    "dataset.re10k.roots=['$DATASET_ROOT']" \
    "+gaussian_control.checkpoint=$CHECKPOINT" \
    "+gaussian_control.index_path=$INDEX_PATH" \
    "+gaussian_control.max_scenes=$MAX_SCENES" \
    "+gaussian_control.fractions=[0.05,0.10]" \
    "+gaussian_control.reference_run=$REFERENCE_RUN" \
    "test.align_pose=$ALIGN_POSE" \
    test.compute_scores=true \
    test.save_image=false \
    test.save_video=false \
    test.save_compare=false \
    model.encoder.moment_decoder.compile_kernels=false \
    "hydra.run.dir=$EVAL_OUTPUT_DIR" \
    "test.output_path=$EVAL_OUTPUT_DIR" \
    "$@" 2>&1 | tee "$EVAL_OUTPUT_DIR/run.log"

echo "Evaluation complete: $EVAL_OUTPUT_DIR/summary.txt"
