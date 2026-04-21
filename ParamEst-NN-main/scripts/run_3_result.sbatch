#!/usr/bin/env bash
#SBATCH --job-name=paramest-3result
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/t30/all/ge48zus/paramest-nn/ParamEst-NN-main/logs/%x-%j.out
#SBATCH --error=/home/t30/all/ge48zus/paramest-nn/ParamEst-NN-main/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NOTEBOOK_DIR="${PROJECT_ROOT}/notebooks"
INPUT_NOTEBOOK="${NOTEBOOK_DIR}/3-result.ipynb"
PYTHON_BIN="/space-nobackup/ge48zus/venvs/paramestnn/bin/python"

mkdir -p "${PROJECT_ROOT}/logs"

echo "Job started on $(hostname)"
echo "Project root: ${PROJECT_ROOT}"
echo "Notebook: ${INPUT_NOTEBOOK}"
echo "Python: ${PYTHON_BIN}"
export PATH="/space-nobackup/ge48zus/venvs/paramestnn/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
CPU_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export TORCH_NUM_THREADS="${CPU_THREADS}"
export PYTHONUNBUFFERED=1
export PARAMEST_MODEL_NAME="${PARAMEST_MODEL_NAME:-v}"
MODEL_TAG="$(printf '%s' "${PARAMEST_MODEL_NAME}" | tr -c '[:alnum:]_-' '_')"
OUTPUT_NOTEBOOK="${NOTEBOOK_DIR}/3-result.${MODEL_TAG}.executed.ipynb"
# 关闭逐行扫描大表打印，减少 notebook 输出 I/O 开销（需要时可改为 1）
export PARAMEST_PRINT_SCAN_TABLE="${PARAMEST_PRINT_SCAN_TABLE:-0}"
# 扫描精度与积分点抽样，可按需覆盖
export PARAMEST_N_DELTA="${PARAMEST_N_DELTA:-11}"
export PARAMEST_N_OMEGA="${PARAMEST_N_OMEGA:-11}"
export PARAMEST_TAU_STRIDE="${PARAMEST_TAU_STRIDE:-1}"
export PARAMEST_PROGRESS_EVERY="${PARAMEST_PROGRESS_EVERY:-12}"
export PARAMEST_SCORE_FD_STEP="${PARAMEST_SCORE_FD_STEP:-1e-3}"
export PARAMEST_JAC_FD_STEP="${PARAMEST_JAC_FD_STEP:-1e-3}"

cd "${NOTEBOOK_DIR}"

"${PYTHON_BIN}" -m nbconvert \
  --to notebook \
  --execute "${INPUT_NOTEBOOK}" \
  --output "$(basename "${OUTPUT_NOTEBOOK}")" \
  --ExecutePreprocessor.timeout=-1

echo "Notebook execution finished."
echo "Executed notebook written to: ${OUTPUT_NOTEBOOK}"
