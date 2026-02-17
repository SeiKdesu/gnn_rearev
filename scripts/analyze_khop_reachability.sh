#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT_DIR"

# CWQ fixed settings (no env/args needed)
DATA_DIR="data/CWQ"
MODE="undirected" # undirected|directed|both
ENTITIES_TXT="$DATA_DIR/entities.txt"
ENTITIES_SQLITE="$DATA_DIR/entities_index.sqlite"

pick_file() {
  base="$1"
  a="$DATA_DIR/${base}_khop.json"
  b="$DATA_DIR/${base}_k_hop.json"
  if [ -f "$a" ]; then
    echo "$a"
    return 0
  fi
  if [ -f "$b" ]; then
    echo "$b"
    return 0
  fi
  echo ""  # not found
  return 1
}

TRAIN_FILE="$(pick_file train 2>/dev/null || true)"
DEV_FILE="$(pick_file dev 2>/dev/null || true)"
TEST_FILE="$(pick_file test 2>/dev/null || true)"

if [ -z "${TRAIN_FILE}" ] || [ -z "${DEV_FILE}" ] || [ -z "${TEST_FILE}" ]; then
  echo "khop files not found in $DATA_DIR" >&2
  echo "expected one of: train_khop.json/train_k_hop.json (same for dev/test)" >&2
  exit 2
fi

echo "MODE=$MODE" >&2
echo "TRAIN_FILE=$TRAIN_FILE" >&2
echo "DEV_FILE=$DEV_FILE" >&2
echo "TEST_FILE=$TEST_FILE" >&2

python3 analyze_khop_seed_to_answer_reachability.py \
  --data "$TRAIN_FILE" \
  --entities-txt "$ENTITIES_TXT" \
  --entities-sqlite "$ENTITIES_SQLITE" \
  --build-entities-sqlite \
  --mode "$MODE"

python3 analyze_khop_seed_to_answer_reachability.py \
  --data "$DEV_FILE" \
  --entities-txt "$ENTITIES_TXT" \
  --entities-sqlite "$ENTITIES_SQLITE" \
  --mode "$MODE"

python3 analyze_khop_seed_to_answer_reachability.py \
  --data "$TEST_FILE" \
  --entities-txt "$ENTITIES_TXT" \
  --entities-sqlite "$ENTITIES_SQLITE" \
  --mode "$MODE"
