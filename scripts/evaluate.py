#!/usr/bin/env python3
"""Offline evaluation — the build-time confidence gate (no ground truth).

Produces the deck/interview evidence the plan calls for:
  * honeypot-rate-in-top-100  (DQ guard — target 0)
  * %India in the top-10      (geo trap avoided)
  * ABLATION: skills-only vs narrative-only vs full pipeline — reproduces the V1
    finding that skills-only drags keyword-stuffers/honeypots into the top-100
    while the full pipeline keeps them out.
  * ANCHOR set: where the hand-verified strong positives and the geo/availability
    negatives land (strong positives should be high; negatives should be lower).
  * Cross-encoder lift: how the rerank moves the strong-positive anchors.

    python scripts/evaluate.py --candidates ./candidates.jsonl
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from redrob_ranker import pipeline, scoring  # noqa: E402
from redrob_ranker.textmatch import contains_any  # noqa: E402

# Hand-verified anchors from plan verification V2 (see docs/PROJECT_MEMORY.md).
STRONG_POSITIVES = [
    "CAND_0018499", "CAND_0039754", "CAND_0081846",
    "CAND_0075249", "CAND_0057563", "CAND_0086151",
]
GEO_AVAIL_NEGATIVES = [
    "CAND_0041611", "CAND_0055905", "CAND_0046526", "CAND_0040887", "CAND_0092278",
]
AI_SKILL_KEYWORDS = [
    "machine learning", "deep learning", "nlp", "computer vision", "tensorflow",
    "pytorch", "llm", "fine-tun", "rag", "transformer", "embedding", "retrieval",
    "ranking", "recommendation", "neural", "data science", "mlops",
]


def _ai_skill_count(c):
    return sum(1 for s in c.get("skills", []) if contains_any(s.get("name", ""), AI_SKILL_KEYWORDS))


def _top_report(name, order, ids, honeypot, india, stuffer, k=100):
    top = order[:k]
    top10 = order[:10]
    hp = int(honeypot[top].sum())
    st = int(stuffer[top].sum())
    india10 = int(india[top10].sum())
    print(f"  {name:<16} top100: honeypots={hp:>2}  stuffers={st:>3}  | top10 India={india10}/10")
    return hp


def sensitivity_sweep(detail, id_to_i):
    """Modifier-envelope sensitivity (PLAN_REVIEW_V2 Round 12).

    Without ground truth we cannot *tune* the geo/behavioral magnitudes, but we
    CAN prove the ranking is robust to them: re-derive the heuristic `final`
    under several envelope scales (0 = modifiers off -> 1.5 = widened) and report
    where the hand-verified strong positives / negatives land. Stable ordering
    across settings is the validatable property (and a strong Stage-5 line).
    """
    base = detail["base"]
    p, g, b = detail["p_plaus"], detail["f_geo"], detail["f_beh"]
    anchors = [a for a in STRONG_POSITIVES if a in id_to_i]
    negs = [a for a in GEO_AVAIL_NEGATIVES if a in id_to_i]
    print("\n=== MODIFIER SENSITIVITY SWEEP ===  (heuristic-final rank under scaled envelopes)")
    print(f"  {'scale':<7}" + "".join(f"{a[-4:]:>6}" for a in anchors)
          + "  |" + "".join(f"{n[-4:]:>7}" for n in negs))
    for scale in (0.0, 0.5, 1.0, 1.5):
        # Compress/expand each multiplier around 1.0 by `scale` (1.0 == shipped).
        g_s = 1.0 + (g - 1.0) * scale
        b_s = 1.0 + (b - 1.0) * scale
        final_s = (base * p * g_s * b_s).astype(np.float64)
        order = np.argsort(-final_s, kind="stable")
        rank_of = {int(idx): r + 1 for r, idx in enumerate(order)}
        pos = "".join(f"{rank_of[id_to_i[a]]:>6}" for a in anchors)
        neg = "".join(f"{rank_of[id_to_i[n]]:>7}" for n in negs)
        tag = " (shipped)" if scale == 1.0 else ""
        print(f"  {scale:<7}{pos}  |{neg}{tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    args = ap.parse_args()

    emb, ids, bm25 = pipeline.load_artifacts()
    cands = pipeline._align_candidates(args.candidates, ids)
    final, detail = pipeline._score_all(cands, emb, bm25, log=print)

    ids = np.array([c["candidate_id"] for c in cands])
    s_sem = detail["s_semantic"]
    honeypot = np.array([d["is_honeypot"] for d in detail["plausibility"]])
    india = np.array([d["is_india"] for d in detail["geo"]])
    offtarget = np.array([d["offtarget_title"] for d in detail["role"]])
    ai_skills = np.array([_ai_skill_count(c) for c in cands])
    stuffer = (ai_skills >= 5) & offtarget
    id_to_i = {cid: i for i, cid in enumerate(ids)}

    full_order = np.argsort(-final, kind="stable")
    narr_order = np.argsort(-s_sem, kind="stable")
    skills_order = np.argsort(-ai_skills, kind="stable")

    sensitivity_sweep(detail, id_to_i)

    print("\n=== ABLATION (top-100 composition) ===")
    _top_report("skills-only", skills_order, ids, honeypot, india, stuffer)
    _top_report("narrative-only", narr_order, ids, honeypot, india, stuffer)
    _top_report("full (heuristic)", full_order, ids, honeypot, india, stuffer)

    # Reranked submission ordering (the real output).
    shortlist, sl_score = pipeline._rerank_shortlist(cands, final, use_rerank=True, log=print)
    chosen = scoring.select_top_k([cands[i]["candidate_id"] for i in shortlist], sl_score, k=100)
    top100_ids = [cid for cid, _ in chosen]
    top_idx = np.array([id_to_i[c] for c in top100_ids])
    print("\n=== FINAL SUBMISSION (reranked) ===")
    print(f"  honeypots in top-100 : {int(honeypot[top_idx].sum())}  (DQ threshold > 10)")
    print(f"  stuffers  in top-100 : {int(stuffer[top_idx].sum())}")
    top10_idx = top_idx[:10]
    print(f"  top-10 India          : {int(india[top10_idx].sum())}/10")
    print(f"  top-10 product/eng    : {int((~offtarget[top10_idx]).sum())}/10 non-off-target titles")

    rerank_rank = {cid: r for r, cid in enumerate(top100_ids, start=1)}
    print("\n=== ANCHORS ===  (final-rank -> reranked-rank)")
    for cid in STRONG_POSITIVES:
        if cid not in id_to_i:
            print(f"  {cid}  [absent from this candidates file]")
            continue
        fr = int(np.where(full_order == id_to_i[cid])[0][0]) + 1
        rr = rerank_rank.get(cid, ">100")
        print(f"  +pos {cid}  final #{fr:<6} reranked #{rr}")
    for cid in GEO_AVAIL_NEGATIVES:
        if cid not in id_to_i:
            print(f"  {cid}  [absent]")
            continue
        fr = int(np.where(full_order == id_to_i[cid])[0][0]) + 1
        print(f"  -neg {cid}  final #{fr}")


if __name__ == "__main__":
    main()
