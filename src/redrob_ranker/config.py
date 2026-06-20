"""Single source of truth for paths, model names, weights and constants.

Every tunable lives here so the ranking is fully auditable and the Stage-5
interview answer to "where do these numbers come from?" is one file.
The blend weights below are *starting points*; `scripts/evaluate.py` calibrates
them against the hand-verified anchor set and the chosen values are recorded in
docs/PROJECT_MEMORY.md.
"""
from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DATA_DIR = REPO_ROOT / "data"
EMB_PATH = ARTIFACTS_DIR / "narrative_emb.npy"          # float32, (N, d), L2-normalized
EMB_IDS_PATH = ARTIFACTS_DIR / "candidate_ids.npy"      # (N,) str candidate_id, aligned to EMB_PATH
EMB_META_PATH = ARTIFACTS_DIR / "emb_meta.json"         # {"model": ..., "dim": ...}
BM25_TF_PATH = ARTIFACTS_DIR / "bm25_tf.npz"            # scipy CSR term-count matrix (N, V)
BM25_META_PATH = ARTIFACTS_DIR / "bm25_meta.npz"        # numpy: idf, doc_len, avgdl, vocab_*

# ---------------------------------------------------------------------------
# Models (small, CPU-friendly; weights cached locally so ranking needs no network)
# ---------------------------------------------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"                   # 384-dim, strong CPU retrieval
EMBED_DIM = 384
# bge-* retrieval convention: prefix the *query* with an instruction; passages get none.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Cross-encoder reranker. NDCG@10 = 50% of the composite is decided here, so this is
# the highest-leverage model in the system (docs/PLAN_REVIEW_V2.md, Round 4).
#
# MEASURED on this 4-core CPU box (not an estimate): reranking the full 1000-shortlist
# with BAAI/bge-reranker-base took ~25 MIN — ~5x OVER the 5-minute budget (a Stage-3
# DQ). So the default ships the fast 2019-era MiniLM (in-budget, ~3.5 min @ 1000), and
# the much stronger bge-reranker-base is OPT-IN via REDROB_CROSS_ENCODER and is only
# budget-safe when paired with a small RERANK_SHORTLIST_K (see below). The MiniLM is
# also the graceful offline FALLBACK if the preferred model can't be loaded.
CROSS_ENCODER_MODEL = os.environ.get("REDROB_CROSS_ENCODER", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CROSS_ENCODER_FALLBACK = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_MAX_LEN = 512                              # bge-reranker handles longer narratives
EMBED_BATCH = 256

# ---------------------------------------------------------------------------
# Pipeline knobs
# ---------------------------------------------------------------------------
SEED = 13
TOP_K = 100                 # submission size
SHORTLIST_K = 1000          # heuristic recall depth (top-N by `final`)
# How many of the shortlist actually get the (expensive) cross-encoder. For the fast
# MiniLM this equals SHORTLIST_K (rerank all 1000, ~seconds). For a heavy reranker like
# bge-reranker-base, set this small (~128) via env so the stage stays in budget — the
# top-100 output is covered with margin. None -> use SHORTLIST_K.
RERANK_SHORTLIST_K = int(os.environ.get("REDROB_RERANK_K", SHORTLIST_K))
USE_RERANK = True
SCORE_DECIMALS = 6          # rounding for the CSV `score` column

# ---------------------------------------------------------------------------
# Blend weights (calibrated; see PROJECT_MEMORY.md)
# ---------------------------------------------------------------------------
W_SEMANTIC = 0.45
W_ROLE = 0.35
W_TRAJECTORY = 0.20

# semantic = dense (primary) + sparse lexical (secondary)
W_DENSE = 0.70
W_SPARSE = 0.30

# Normalization shaping (PLAN_REVIEW_V2 Round 14). Robust-min-max alone clips the top
# ~1% of dense scores to 1.0, collapsing the elite cluster's ordering BEFORE shortlist
# selection. We (a) push the clip ceiling to 99.5 pct and (b) mix in a small rank01
# term so intra-elite ordering survives into the shortlist the reranker then sharpens.
ROBUST_HI_PCT = 99.5
SEMANTIC_RANK_MIX = 0.25        # fraction of rank01 blended into each robust-min-max leg

# BM25 (sparse lexical) hyperparameters — Robertson/Sparck-Jones defaults.
BM25_K1 = 1.5
BM25_B = 0.75

# Cross-encoder blend on the shortlist. The reranker is the genuine top-100
# discriminator (the heuristic `final` saturates in the elite cluster), and it decides
# NDCG@10 = 50% of the score, so we make it AUTHORITATIVE for the top-K ordering while
# the heuristic `final` governs shortlist membership / recall and breaks cross ties
# (PLAN_REVIEW_V2 Round 5). Was 0.60/0.40 (heuristic-led) with the weak MiniLM.
W_FINAL_IN_RERANK = 0.35
W_CROSS_IN_RERANK = 0.65
# Conservative blend used ONLY when the pipeline has to fall back to the weak 2019-era
# MiniLM reranker (preferred model unavailable): we do NOT let a weak reranker dominate
# the order, so the heuristic stays in the lead exactly as in the validated v1 run.
W_FINAL_IN_RERANK_FALLBACK = 0.60
W_CROSS_IN_RERANK_FALLBACK = 0.40

# ---------------------------------------------------------------------------
# Modifier envelopes
# ---------------------------------------------------------------------------
# Geography (no visa sponsorship in the JD -> non-India strongly penalized).
GEO_INDIA = 1.00
GEO_NONINDIA_RELOCATE = 0.45
GEO_NONINDIA = 0.25
PREFERRED_CITIES = {
    "noida", "pune", "hyderabad", "mumbai", "delhi", "new delhi",
    "gurgaon", "gurugram", "bengaluru", "bangalore", "ncr",
}
GEO_CITY_BONUS = 1.05       # applied (capped) when an India candidate is in/near a preferred hub

# Behavioral availability multiplier envelope.
BEHAVIOR_MIN = 0.60
BEHAVIOR_MAX = 1.15

# Plausibility (honeypot) gate.
PLAUSIBILITY_OK = 1.00
PLAUSIBILITY_SOFT = 0.85    # 1-2 expert/advanced-with-0-duration skills (mildly suspicious)
PLAUSIBILITY_HONEYPOT = 0.03
EXPERT_ZERO_DURATION_HARD = 3   # >= this many expert/adv skills at 0 months -> honeypot
JOB_OVER_SPAN_MONTHS = 3        # job duration exceeding its calendar span by > this -> impossible
# A single role claiming more months than the candidate's ENTIRE career timeline
# (earliest start -> reference date) allows is a hard impossibility — the "8 years at a
# company founded 3 years ago" class the spec names (PLAN_REVIEW_V2 Round 11). Margin
# kept generous so only unambiguous impossibilities trip it (no genuine-candidate FPs).
ROLE_OVER_CAREER_MONTHS = 6

# Experience band from the JD ("5-9 years", ideal 6-8).
EXP_BAND_LOW, EXP_BAND_HIGH = 5.0, 9.0
EXP_IDEAL_LOW, EXP_IDEAL_HIGH = 6.0, 8.0

# Reference "today" for recency/availability and plausibility date math.
# (Max activity date observed in the dataset is ~2026-05; we anchor just after.)
REFERENCE_DATE = _dt.date(2026, 6, 16)

# Consulting/services firms — the JD rejects consulting-only careers.
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mindtree", "ltimindtree", "mphasis",
    "ibm", "deloitte", "dxc",
}
