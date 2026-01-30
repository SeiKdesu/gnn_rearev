#!/usr/bin/env bash
set -euo pipefail

DATA_FOLDER="${1:-}"
MODEL="${2:-}"
CKPT_NAME="${3:-}"
shift 3 || true

if [[ -z "${DATA_FOLDER}" || -z "${MODEL}" || -z "${CKPT_NAME}" ]]; then
  echo "Usage: $0 <data_folder/> <ReaRev|NSM|GraftNet|NuTrea> <ckpt_filename> [extra args...]"
  echo "Example:"
  echo "  $0 data/CWQ/ ReaRev ReaRev_cwq.ckpt --lm sbert --test_batch_size 20"
  exit 2
fi

if [[ "${DATA_FOLDER}" != */ ]]; then
  DATA_FOLDER="${DATA_FOLDER}/"
fi

DATASET_NAME="$(basename "${DATA_FOLDER%/}" | tr '[:upper:]' '[:lower:]')"

python3 main.py "${MODEL}" \
  --data_folder "${DATA_FOLDER}" \
  --name "${DATASET_NAME}" \
  --use_shared_subgraph true \
  --is_eval \
  --load_experiment "${CKPT_NAME}" \
  "$@"

