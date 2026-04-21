#!/usr/bin/env bash
#SBATCH --job-name=paramest-train-shelving
#SBATCH --account=gpu_rabl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/t30/rab/ge48zus/paramest-nn/ParamEst-NN-main/logs/%x-%j.out
#SBATCH --error=/home/t30/rab/ge48zus/paramest-nn/ParamEst-NN-main/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="/home/t30/rab/ge48zus/paramest-nn/ParamEst-NN-main"
NOTEBOOK_DIR="${PROJECT_ROOT}/train_notebooks"
EXECUTED_NOTEBOOK_DIR="${PROJECT_ROOT}/executed_notebook"
INPUT_NOTEBOOK="${NOTEBOOK_DIR}/train-noniid.ipynb"
OUTPUT_NOTEBOOK="${EXECUTED_NOTEBOOK_DIR}/train_shelving.executed.ipynb"
PYTHON_BIN="/space-nobackup/ge48zus/venvs/paramestnn/bin/python"

mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/data/models/shelving"
mkdir -p "${EXECUTED_NOTEBOOK_DIR}"

echo "Job started on $(hostname)"
echo "Project root: ${PROJECT_ROOT}"
echo "Notebook: ${INPUT_NOTEBOOK}"
echo "Python: ${PYTHON_BIN}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
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
export PARAMEST_MODEL_NAME="shelving"

LOCAL_SCRATCH_ROOT="${TMPDIR:-/tmp/${USER}/paramest-${SLURM_JOB_ID}}"
LOCAL_DATA_ROOT="${LOCAL_SCRATCH_ROOT}/data"
mkdir -p "${LOCAL_DATA_ROOT}/tragectories/shelving"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete "${PROJECT_ROOT}/data/tragectories/shelving/" "${LOCAL_DATA_ROOT}/tragectories/shelving/"
else
  cp -a "${PROJECT_ROOT}/data/tragectories/shelving/." "${LOCAL_DATA_ROOT}/tragectories/shelving/"
fi
export PARAMEST_DATA_ROOT="${LOCAL_DATA_ROOT}"
echo "Staged shelving data to: ${PARAMEST_DATA_ROOT}/tragectories/shelving"

cd "${NOTEBOOK_DIR}"

"${PYTHON_BIN}" -m nbconvert \
  --to notebook \
  --execute "${INPUT_NOTEBOOK}" \
  --output-dir "${EXECUTED_NOTEBOOK_DIR}" \
  --output "$(basename "${OUTPUT_NOTEBOOK}")" \
  --ExecutePreprocessor.timeout=-1

echo "Notebook execution finished."
echo "Executed notebook written to: ${OUTPUT_NOTEBOOK}"
