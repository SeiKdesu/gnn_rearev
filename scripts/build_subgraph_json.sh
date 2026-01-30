#!/usr/bin/env bash
set -euo pipefail

DATA_FOLDER="${1:-}"
if [[ -z "${DATA_FOLDER}" ]]; then
  echo "Usage: $0 <data_folder/>"
  echo "Example: $0 data/CWQ/"
  exit 2
fi

# Normalize trailing slash for consistency with existing configs.
if [[ "${DATA_FOLDER}" != */ ]]; then
  DATA_FOLDER="${DATA_FOLDER}/"
fi

TRAIN_JSON="${DATA_FOLDER}train.json"
ENTITIES_TXT="${DATA_FOLDER}entities.txt"
RELATIONS_TXT="${DATA_FOLDER}relations.txt"
OUT_JSON="${DATA_FOLDER}subgraph.json"

if [[ ! -f "${TRAIN_JSON}" ]]; then
  echo "ERROR: missing ${TRAIN_JSON}" >&2
  exit 1
fi

# Train-only by default (no dev/test leakage).
python3 scripts/build_shared_subgraph.py \
  --input "${TRAIN_JSON}" \
  --output "${OUT_JSON}" \
  --entities_txt "${ENTITIES_TXT}" \
  --relations_txt "${RELATIONS_TXT}"

echo "OK: wrote ${OUT_JSON}"

