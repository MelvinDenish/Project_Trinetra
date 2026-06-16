#!/usr/bin/env python3
"""OFFLINE pre-computation (declared; may exceed the 5-minute ranking budget).

This is the ONLY step that may use the network (to download the embedding model
the first time). It writes three artifacts the offline ranking step consumes:

  artifacts/narrative_emb.npy   (N, d) float32, L2-normalized dense vectors
  artifacts/candidate_ids.npy   (N,)   candidate_id aligned to the embeddings
  artifacts/emb_meta.json       {"model", "dim", "n"}
  artifacts/bm25_tf.npz + bm25_meta.npz   the BM25 sparse index

Run:
  python scripts/precompute_embeddings.py --candidates ./candidates.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from redrob_ranker import config, embeddings, io_utils  # noqa: E402
from redrob_ranker.features import semantic  # noqa: E402
from redrob_ranker.narrative import build_narrative  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute dense embeddings + BM25 index.")
    ap.add_argument("--candidates", default=str(config.DATA_DIR / "candidates.jsonl"))
    ap.add_argument("--limit", type=int, default=None, help="cap rows (smoke tests)")
    args = ap.parse_args()

    t0 = time.time()
    ids: list[str] = []
    narratives: list[str] = []
    for i, c in enumerate(io_utils.iter_candidates(args.candidates)):
        if args.limit is not None and i >= args.limit:
            break
        ids.append(c["candidate_id"])
        narratives.append(build_narrative(c))
    print(f"[precompute] loaded {len(ids):,} narratives in {time.time() - t0:.1f}s", flush=True)

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    t1 = time.time()
    emb = embeddings.encode_passages(narratives, show_progress=True)
    print(f"[precompute] dense {emb.shape} in {time.time() - t1:.1f}s", flush=True)
    np.save(config.EMB_PATH, emb)
    np.save(config.EMB_IDS_PATH, np.array(ids))
    with open(config.EMB_META_PATH, "w", encoding="utf-8") as f:
        json.dump({"model": config.EMBED_MODEL, "dim": int(emb.shape[1]), "n": len(ids)}, f)

    t2 = time.time()
    semantic.build_bm25(narratives)
    print(f"[precompute] BM25 index in {time.time() - t2:.1f}s", flush=True)

    print(f"[precompute] DONE total {time.time() - t0:.1f}s -> {config.ARTIFACTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
