#!/usr/bin/env bash
#SBATCH --job-name=paramest-gen-3lvl-noniid
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/t30/rab/ge48zus/paramest-nn/ParamEst-NN-main/logs/%x-%j.out
#SBATCH --error=/home/t30/rab/ge48zus/paramest-nn/ParamEst-NN-main/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="/home/t30/rab/ge48zus/paramest-nn/ParamEst-NN-main"
PYTHON_BIN="/space-nobackup/ge48zus/venvs/paramestnn/bin/python"

mkdir -p "${PROJECT_ROOT}/logs"

echo "Job started on $(hostname)"
echo "Project root: ${PROJECT_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"

export PATH="/space-nobackup/ge48zus/venvs/paramestnn/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
CPU_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" scripts/generate_training_trajectories.py \
  --output-root data/tragectories \
  --models shelving lambda v ladder \
  --n-trajectories 1000000 \
  --length 50 \
  --n-param-points 2048 \
  --grid-size 4096 \
  --tau-max-init 40.0 \
  --tau-max-limit 1024.0 \
  --tail-tol 0.0001 \
  --min-captured-area 0.995 \
  --max-redraw-attempts 256 \
  --device cpu \
  --overwrite

echo "three-level noniid trajectory generation finished."
