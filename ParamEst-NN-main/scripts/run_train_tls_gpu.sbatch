#!/usr/bin/env bash
#SBATCH --job-name=paramest-train-tls
#SBATCH --account=gpu_rabl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/t30/rab/ge48zus/paramest-nn/ParamEst-NN-main/logs/%x-%j.out
#SBATCH --error=/home/t30/rab/ge48zus/paramest-nn/ParamEst-NN-main/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="/home/t30/rab/ge48zus/paramest-nn/ParamEst-NN-main"
NOTEBOOK_DIR="${PROJECT_ROOT}/notebooks"
INPUT_NOTEBOOK="${NOTEBOOK_DIR}/train_tls.ipynb"
OUTPUT_NOTEBOOK="${NOTEBOOK_DIR}/train_tls.executed.ipynb"
PYTHON_BIN="/space-nobackup/ge48zus/venvs/paramestnn/bin/python"

mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/data/models/tls"

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

LOCAL_SCRATCH_ROOT="${TMPDIR:-/tmp/${USER}/paramest-${SLURM_JOB_ID}}"
LOCAL_DATA_ROOT="${LOCAL_SCRATCH_ROOT}/data"
mkdir -p "${LOCAL_DATA_ROOT}/tragectories/tls"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete "${PROJECT_ROOT}/data/tragectories/tls/" "${LOCAL_DATA_ROOT}/tragectories/tls/"
else
  cp -a "${PROJECT_ROOT}/data/tragectories/tls/." "${LOCAL_DATA_ROOT}/tragectories/tls/"
fi
export PARAMEST_DATA_ROOT="${LOCAL_DATA_ROOT}"
echo "Staged TLS data to: ${PARAMEST_DATA_ROOT}/tragectories/tls"

cd "${NOTEBOOK_DIR}"

"${PYTHON_BIN}" -m nbconvert \
  --to notebook \
  --execute "${INPUT_NOTEBOOK}" \
  --output "$(basename "${OUTPUT_NOTEBOOK}")" \
  --ExecutePreprocessor.timeout=-1

echo "Notebook execution finished."
echo "Executed notebook written to: ${OUTPUT_NOTEBOOK}"
