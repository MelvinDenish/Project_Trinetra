#!/usr/bin/env python3
"""Ranking step — CPU only, NO network, <= 5 minutes. Produces submission.csv.

    python scripts/rank.py --candidates ./candidates.jsonl --out ./submission.csv

Offline is enforced here (before torch/transformers import) so the cross-encoder
and embedding model load from the local cache only — exactly as the Stage-3
sandbox runs it. If you have not run scripts/precompute_embeddings.py yet, do
that first (it is the only step that may use the network).
"""
from __future__ import annotations

import os

# Must be set BEFORE importing torch / transformers / sentence_transformers.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from redrob_ranker import config, pipeline  # noqa: E402


def write_csv(rows: list[dict], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in rows:
            w.writerow([
                r["candidate_id"],
                r["rank"],
                f'{r["score"]:.{config.SCORE_DECIMALS}f}',
                r["reasoning"],
            ])


def main() -> None:
    ap = argparse.ArgumentParser(description="Produce the top-100 ranking CSV.")
    ap.add_argument("--candidates", required=True, help="path to candidates.jsonl")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder")
    ap.add_argument("--top-k", type=int, default=config.TOP_K)
    args = ap.parse_args()

    t0 = time.time()
    rows = pipeline.rank_from_artifacts(
        args.candidates, top_k=args.top_k, use_rerank=not args.no_rerank
    )
    write_csv(rows, args.out)
    print(f"[rank] wrote {len(rows)} rows -> {args.out} in {time.time() - t0:.1f}s wall-clock")


if __name__ == "__main__":
    main()
