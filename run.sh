#!/usr/bin/env bash
# Full reproduction: precompute artifacts (offline once the model is cached;
# may exceed 5 min) then the <=5-min ranking step, then validate the output.
#
#   bash run.sh [candidates.jsonl] [submission.csv]
set -euo pipefail

CAND="${1:-./candidates.jsonl}"
OUT="${2:-./submission.csv}"

echo "[run] precompute (offline; may exceed 5 min) ..."
python scripts/precompute_embeddings.py --candidates "$CAND"

echo "[run] ranking step (CPU, no network, <= 5 min) ..."
python scripts/rank.py --candidates "$CAND" --out "$OUT"

echo "[run] validating submission ..."
python validate_submission.py "$OUT"
