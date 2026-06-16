"""Single source of truth for paths, model names, weights and constants.

Every tunable lives here so the ranking is fully auditable and the Stage-5
interview answer to "where do these numbers come from?" is one file.
The blend weights below are *starting points*; `scripts/evaluate.py` calibrates
them against the hand-verified anchor set and the chosen values are recorded in
docs/PROJECT_MEMORY.md.
"""
from __future__ import annotations

import datetime as _dt
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
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBED_BATCH = 256

# ---------------------------------------------------------------------------
# Pipeline knobs
# ---------------------------------------------------------------------------
SEED = 13
TOP_K = 100                 # submission size
SHORTLIST_K = 1000          # candidates re-ranked by the cross-encoder
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

# BM25 (sparse lexical) hyperparameters — Robertson/Sparck-Jones defaults.
BM25_K1 = 1.5
BM25_B = 0.75

# cross-encoder blend on the shortlist
W_FINAL_IN_RERANK = 0.60
W_CROSS_IN_RERANK = 0.40

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
