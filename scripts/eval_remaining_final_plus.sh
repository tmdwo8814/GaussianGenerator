#!/usr/bin/env bash
#SBATCH -J FinalPlus-eval
#SBATCH --gres=gpu:high_perf:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -p batch_grad
#SBATCH -w ariel-n1
#SBATCH -t 6-00:00:00
#SBATCH -o slurm-eval-final-plus-%j.out
#SBATCH --export=ALL

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Compatibility entry point: both filenames use the same evaluation code.
exec bash scripts/eval_re10k.sh "$@"
