#!/usr/bin/env bash
set -euo pipefail

DATA_FOLDER="${1:-}"
MODEL="${2:-}"
shift 2 || true

if [[ -z "${DATA_FOLDER}" || -z "${MODEL}" ]]; then
  echo "Usage: $0 <data_folder/> <ReaRev|NSM|GraftNet|NuTrea> [extra args...]"
  echo "Example:"
  echo "  $0 data/CWQ/ ReaRev --lm sbert --batch_size 8 --num_epoch 5"
  exit 2
fi

if [[ "${DATA_FOLDER}" != */ ]]; then
  DATA_FOLDER="${DATA_FOLDER}/"
fi

DATASET_NAME="$(basename "${DATA_FOLDER%/}" | tr '[:upper:]' '[:lower:]')"

# NOTE:
# - `--use_shared_subgraph true` makes the loader ignore per-line subgraphs in train/dev/test.json.
# - `--shared_subgraph_path` defaults to <data_folder>/subgraph.json if omitted.
python3 main.py "${MODEL}" \
  --data_folder "${DATA_FOLDER}" \
  --name "${DATASET_NAME}" \
  --use_shared_subgraph true \
  "$@"

