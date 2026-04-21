#!/usr/bin/env bash
#SBATCH --job-name=paramest-benchmark
#SBATCH --account=gpu_rabl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/t30/rab/ge48zus/paramest-nn/ParamEst-NN-main/logs/%x-%j.out
#SBATCH --error=/home/t30/rab/ge48zus/paramest-nn/ParamEst-NN-main/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="/home/t30/rab/ge48zus/paramest-nn/ParamEst-NN-main"
PYTHON_BIN="/space-nobackup/ge48zus/venvs/paramestnn/bin/python"
SCRIPT_PATH="${PROJECT_ROOT}/scripts/build_representative_benchmarks.py"
PLOT_SCRIPT_PATH="${PROJECT_ROOT}/scripts/plot_projection_diagnostics.py"

mkdir -p "${PROJECT_ROOT}/logs"

echo "Job started on $(hostname)"
echo "Project root: ${PROJECT_ROOT}"
echo "Script: ${SCRIPT_PATH}"
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

# -------- Benchmark configuration (override via sbatch --export=ALL,VAR=...) --------
BENCH_MODELS="${PARAMEST_BENCH_MODELS:-tls}"
BENCH_DELTA="${PARAMEST_BENCH_DELTA:-1.55}"
BENCH_OMEGA="${PARAMEST_BENCH_OMEGA:-2.75}"
BENCH_N="${PARAMEST_BENCH_N:-50}"
BENCH_DEVICE="${PARAMEST_BENCH_DEVICE:-cuda}"
BENCH_BATCH_SIZE="${PARAMEST_BENCH_BATCH_SIZE:-256}"
BENCH_WARMUP_ITERS="${PARAMEST_BENCH_WARMUP_ITERS:-4}"
BENCH_TIMED_ITERS="${PARAMEST_BENCH_TIMED_ITERS:-20}"
BENCH_MLP_HIDDEN="${PARAMEST_BENCH_MLP_HIDDEN:-256,256}"
BENCH_TRAIN_MLP_EPOCHS="${PARAMEST_BENCH_TRAIN_MLP_EPOCHS:-10}"
BENCH_MLP_LR="${PARAMEST_BENCH_MLP_LR:-2e-3}"
BENCH_MLP_WEIGHT_DECAY="${PARAMEST_BENCH_MLP_WEIGHT_DECAY:-1e-4}"
BENCH_MLP_BATCH_SIZE="${PARAMEST_BENCH_MLP_BATCH_SIZE:-4096}"
BENCH_MLP_SAMPLE_CAP="${PARAMEST_BENCH_MLP_SAMPLE_CAP:-120000}"
BENCH_SEED="${PARAMEST_BENCH_SEED:-42}"
BENCH_PLOT_HEATMAP_M="${PARAMEST_BENCH_PLOT_HEATMAP_M:-1}"
BENCH_PLOT_DPI="${PARAMEST_BENCH_PLOT_DPI:-220}"

cd "${PROJECT_ROOT}"

echo "Benchmark config:"
echo "  models=${BENCH_MODELS}"
echo "  representative point: delta=${BENCH_DELTA}, omega=${BENCH_OMEGA}, N=${BENCH_N}"
echo "  device=${BENCH_DEVICE}, batch_size=${BENCH_BATCH_SIZE}"
echo "  warmup_iters=${BENCH_WARMUP_ITERS}, timed_iters=${BENCH_TIMED_ITERS}"
echo "  mlp_hidden=${BENCH_MLP_HIDDEN}, mlp_epochs=${BENCH_TRAIN_MLP_EPOCHS}"
echo "  mlp_batch_size=${BENCH_MLP_BATCH_SIZE}, mlp_sample_cap=${BENCH_MLP_SAMPLE_CAP}"
echo "  plot_heatmap_m=${BENCH_PLOT_HEATMAP_M}, plot_dpi=${BENCH_PLOT_DPI}"

"${PYTHON_BIN}" "${SCRIPT_PATH}" \
  --models ${BENCH_MODELS} \
  --delta "${BENCH_DELTA}" \
  --omega "${BENCH_OMEGA}" \
  --n "${BENCH_N}" \
  --device "${BENCH_DEVICE}" \
  --batch-size "${BENCH_BATCH_SIZE}" \
  --warmup-iters "${BENCH_WARMUP_ITERS}" \
  --timed-iters "${BENCH_TIMED_ITERS}" \
  --mlp-hidden "${BENCH_MLP_HIDDEN}" \
  --train-mlp-epochs "${BENCH_TRAIN_MLP_EPOCHS}" \
  --mlp-lr "${BENCH_MLP_LR}" \
  --mlp-weight-decay "${BENCH_MLP_WEIGHT_DECAY}" \
  --mlp-batch-size "${BENCH_MLP_BATCH_SIZE}" \
  --mlp-sample-cap "${BENCH_MLP_SAMPLE_CAP}" \
  --seed "${BENCH_SEED}"

"${PYTHON_BIN}" "${PLOT_SCRIPT_PATH}" \
  --input-dir "${PROJECT_ROOT}/notebooks/scan_cache/benchmark_tables" \
  --output-dir "${PROJECT_ROOT}/notebooks/scan_cache/benchmark_tables/figures" \
  --heatmap-m "${BENCH_PLOT_HEATMAP_M}" \
  --dpi "${BENCH_PLOT_DPI}"

echo "Benchmark table + projection plot build finished."
echo "Outputs:"
echo "  ${PROJECT_ROOT}/notebooks/scan_cache/benchmark_tables/representative_benchmark_five_plus_multi_runtime.csv"
echo "  ${PROJECT_ROOT}/notebooks/scan_cache/benchmark_tables/representative_benchmark_five_plus_multi_runtime.tex"
echo "  ${PROJECT_ROOT}/notebooks/scan_cache/benchmark_tables/representative_benchmark_projection_local_diagnostics.csv"
echo "  ${PROJECT_ROOT}/notebooks/scan_cache/benchmark_tables/representative_benchmark_projection_local_stats.csv"
echo "  ${PROJECT_ROOT}/notebooks/scan_cache/benchmark_tables/figures/projection_retention_stats_by_m.png"
