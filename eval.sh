#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET="${1:-weibo}"
CHECKPOINT_DIR="${2:-}"
SPLIT="${SPLIT:-test}"
VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/configs/${DATASET}_qwen3vl_lora.yaml}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}"
  exit 1
fi

if [[ -z "${CHECKPOINT_DIR}" ]]; then
  export CONFIG_PATH_FOR_EVAL="${CONFIG_PATH}"
  CHECKPOINT_DIR="$(python - <<'PY'
import os
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path(os.environ['CONFIG_PATH_FOR_EVAL']).read_text(encoding='utf-8'))
print(cfg['paths']['output_dir'])
PY
)"
fi

echo "[eval] dataset=${DATASET}"
echo "[eval] config=${CONFIG_PATH}"
echo "[eval] checkpoint_dir=${CHECKPOINT_DIR}"
echo "[eval] split=${SPLIT}"
echo "[eval] CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}"

export CUDA_VISIBLE_DEVICES="${VISIBLE_DEVICES}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "${SCRIPT_DIR}"
python evaluate_classifier.py --config "${CONFIG_PATH}" --checkpoint-dir "${CHECKPOINT_DIR}" --split "${SPLIT}"
