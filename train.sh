#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET="${1:-weibo}"
NUM_GPUS="${NUM_GPUS:-1}"
VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/configs/${DATASET}_qwen3vl_lora.yaml}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}"
  exit 1
fi

echo "[train] dataset=${DATASET}"
echo "[train] config=${CONFIG_PATH}"
echo "[train] CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}"
echo "[train] NUM_GPUS=${NUM_GPUS}"

export CUDA_VISIBLE_DEVICES="${VISIBLE_DEVICES}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "${SCRIPT_DIR}"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NUM_GPUS}" train_classifier.py --config "${CONFIG_PATH}"
else
  python train_classifier.py --config "${CONFIG_PATH}"
fi
