#!/usr/bin/env bash
#SBATCH --job-name=paramest-result-noniid
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/t30/rab/ge48zus/paramest-nn/ParamEst-NN-main/logs/%x-%j.out
#SBATCH --error=/home/t30/rab/ge48zus/paramest-nn/ParamEst-NN-main/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NOTEBOOK_DIR="${PROJECT_ROOT}/train_notebooks"
EXECUTED_NOTEBOOK_DIR="${PROJECT_ROOT}/executed_notebook"
INPUT_NOTEBOOK="${NOTEBOOK_DIR}/result-noniid.ipynb"
OUTPUT_NOTEBOOK="${EXECUTED_NOTEBOOK_DIR}/result-noniid.executed.ipynb"
PYTHON_BIN="/space-nobackup/ge48zus/venvs/paramestnn/bin/python"

mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${EXECUTED_NOTEBOOK_DIR}"

echo "Job started on $(hostname)"
echo "Project root: ${PROJECT_ROOT}"
echo "Notebook: ${INPUT_NOTEBOOK}"
echo "Executed notebook: ${OUTPUT_NOTEBOOK}"
echo "Python: ${PYTHON_BIN}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"

export PATH="/space-nobackup/ge48zus/venvs/paramestnn/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
CPU_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export TORCH_NUM_THREADS="${CPU_THREADS}"
export PYTHONUNBUFFERED=1

export PARAMEST_RESULT_MODELS="${PARAMEST_RESULT_MODELS:-v,shelving,lambda,ladder}"
export PARAMEST_RESULT_BATCH_SIZE="${PARAMEST_RESULT_BATCH_SIZE:-8192}"
export PARAMEST_RESULT_MAX_TRAJ="${PARAMEST_RESULT_MAX_TRAJ:-0}"
export PARAMEST_RESULT_SCATTER_POINTS="${PARAMEST_RESULT_SCATTER_POINTS:-5000}"
export PARAMEST_BOUND_GRID_SIZE="${PARAMEST_BOUND_GRID_SIZE:-2048}"
export PARAMEST_BOUND_DT_MAX="${PARAMEST_BOUND_DT_MAX:-0.5}"
export PARAMEST_BOUND_TAU_MAX_LIMIT="${PARAMEST_BOUND_TAU_MAX_LIMIT:-2048}"
export PARAMEST_BOUND_MAX_THETA="${PARAMEST_BOUND_MAX_THETA:-6}"
export PARAMEST_BOUND_N_TRAJ="${PARAMEST_BOUND_N_TRAJ:-384}"
export PARAMEST_BOUND_CAPTURE_TOL="${PARAMEST_BOUND_CAPTURE_TOL:-1e-3}"
export PARAMEST_BOUND_TRANSITION_TOL="${PARAMEST_BOUND_TRANSITION_TOL:-1e-3}"
export PARAMEST_BOUND_FISHER_STABILITY_TOL="${PARAMEST_BOUND_FISHER_STABILITY_TOL:-2e-2}"
export PARAMEST_BOUND_BCRB_STABILITY_TOL="${PARAMEST_BOUND_BCRB_STABILITY_TOL:-2e-2}"

cd "${NOTEBOOK_DIR}"

"${PYTHON_BIN}" -m nbconvert \
  --to notebook \
  --execute "${INPUT_NOTEBOOK}" \
  --output-dir "${EXECUTED_NOTEBOOK_DIR}" \
  --output "$(basename "${OUTPUT_NOTEBOOK}")" \
  --ExecutePreprocessor.timeout=-1

echo "Notebook execution finished."
echo "Executed notebook written to: ${OUTPUT_NOTEBOOK}"
