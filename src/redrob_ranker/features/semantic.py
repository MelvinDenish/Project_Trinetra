"""Semantic fit = dense (sentence-transformer) + sparse (BM25), narrative-only.

Why narrative-only: profiling proved the `skills` array and education field are
decoys (uniform-random, role-independent). The empirical separation probe (V1)
showed skills-only ranking surfaces 85 keyword-stuffers into the top-100 while
narrative-only surfaces 75 genuine candidates and 0 stuffers. So both the dense
vectors and this BM25 index are built over the candidate *narrative* only.

This module owns the BM25 index lifecycle (build at precompute, load+query at
ranking) and the dense+sparse blend that produces S_semantic in [0, 1].
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .. import config
from ..normalize import rank01, robust_minmax01


def _shape01(x: np.ndarray) -> np.ndarray:
    """Squash a raw aggregate into [0,1] while preserving intra-elite ordering.

    Robust-min-max alone clips the top ~1% to 1.0 (elite-cluster collapse, which
    erases ordering BEFORE the shortlist the cross-encoder reranks). We raise the
    clip ceiling and mix in a small rank01 term so the elite stays separated
    (PLAN_REVIEW_V2 Round 14). SEMANTIC_RANK_MIX=0 recovers the old behavior.
    """
    rm = robust_minmax01(x, hi_pct=config.ROBUST_HI_PCT)
    mix = config.SEMANTIC_RANK_MIX
    if mix <= 0.0:
        return rm
    return ((1.0 - mix) * rm + mix * rank01(x)).astype(np.float32)

# Tokens: alphabetic-led, keep tech tokens like c++, ml-ops, gpt-4 reasonably intact.
_TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.\-]+\b"
_MIN_DF = 3
_analyzer = None


def _make_vectorizer():
    from sklearn.feature_extraction.text import CountVectorizer

    return CountVectorizer(
        lowercase=True,
        stop_words="english",
        min_df=_MIN_DF,
        token_pattern=_TOKEN_PATTERN,
    )


def tokenize(text: str) -> list[str]:
    """Tokenize a query string exactly like the indexed corpus was tokenized."""
    global _analyzer
    if _analyzer is None:
        _analyzer = _make_vectorizer().build_analyzer()
    return _analyzer(text or "")


# ---------------------------------------------------------------------------
# Build (offline precompute)
# ---------------------------------------------------------------------------
def _build_arrays(narratives: list[str]):
    vec = _make_vectorizer()
    tf = vec.fit_transform(narratives).tocsr().astype(np.float32)
    doc_len = np.asarray(tf.sum(axis=1)).ravel().astype(np.float32)
    df = np.asarray((tf > 0).sum(axis=0)).ravel().astype(np.float32)
    n = tf.shape[0]
    idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
    avgdl = float(doc_len.mean()) if n else 1.0
    return tf, vec.vocabulary_, idf, doc_len, avgdl


def build_bm25(narratives: list[str]) -> None:
    """Fit a BM25 index over candidate narratives and persist it to artifacts/."""
    tf, vocab, idf, doc_len, avgdl = _build_arrays(narratives)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(config.BM25_TF_PATH, tf)
    # Stored as numpy .npz (loaded with allow_pickle=False) so no untrusted-pickle
    # path exists. The vocab dict is split into parallel term/index arrays.
    vocab_terms = np.array(list(vocab.keys()))
    vocab_idx = np.array(list(vocab.values()), dtype=np.int64)
    np.savez(
        config.BM25_META_PATH,
        idf=idf, doc_len=doc_len, avgdl=np.float32(avgdl),
        vocab_terms=vocab_terms, vocab_idx=vocab_idx,
    )


def build_bm25_index(narratives: list[str]) -> "BM25Index":
    """Build a BM25 index in memory (no disk I/O) — used by the demo path."""
    tf, vocab, idf, doc_len, avgdl = _build_arrays(narratives)
    return BM25Index(tf, vocab, idf, doc_len, avgdl)


# ---------------------------------------------------------------------------
# Query (ranking step)
# ---------------------------------------------------------------------------
class BM25Index:
    """Vectorized BM25 (Okapi) over a precomputed term-count matrix.

    A custom vectorized implementation is used rather than `rank_bm25` because
    the latter stores one Python dict per document; at 100K docs that is both
    memory-heavy and slow to (de)serialize. Here the index is a single scipy
    sparse matrix + three numpy arrays — compact, fast, and deterministic.
    """

    def __init__(self, tf, vocab, idf, doc_len, avgdl,
                 k1: float = config.BM25_K1, b: float = config.BM25_B):
        self.tf = tf.tocsc()  # CSC -> fast column (term) slicing
        self.vocab = vocab
        self.idf = idf
        self.doc_len = doc_len
        self.avgdl = avgdl or 1.0
        self.k1 = k1
        self.b = b
        self._len_damp = (1.0 - b + b * (doc_len / self.avgdl)).astype(np.float32)

    @property
    def n_docs(self) -> int:
        return self.tf.shape[0]

    def scores(self, terms: list[str]) -> np.ndarray:
        """Raw BM25 score of every document against the query `terms`."""
        cols = [self.vocab[t] for t in terms if t in self.vocab]
        if not cols:
            return np.zeros(self.n_docs, dtype=np.float32)
        sub = self.tf[:, cols].toarray().astype(np.float32)         # (N, q)
        denom = sub + self.k1 * self._len_damp[:, None]
        contrib = (sub * (self.k1 + 1.0)) / np.maximum(denom, 1e-9)
        return (contrib * self.idf[cols][None, :]).sum(axis=1).astype(np.float32)


def load_bm25() -> BM25Index:
    meta = np.load(config.BM25_META_PATH, allow_pickle=False)  # safe: no pickle
    vocab = {t: int(i) for t, i in zip(meta["vocab_terms"], meta["vocab_idx"])}
    tf = sparse.load_npz(config.BM25_TF_PATH)
    return BM25Index(tf, vocab, meta["idf"], meta["doc_len"], float(meta["avgdl"]))


# ---------------------------------------------------------------------------
# The S_semantic sub-score
# ---------------------------------------------------------------------------
def _aggregate(per_aspect: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Combine an (N, A) per-aspect score into (N,) rewarding peak + breadth."""
    weighted_mean = per_aspect @ weights
    peak = per_aspect.max(axis=1)
    return 0.5 * peak + 0.5 * weighted_mean


def semantic_scores(
    cand_emb: np.ndarray,
    query_emb: np.ndarray,
    bm25: BM25Index,
    aspect_term_lists: list[list[str]],
    aspect_weights: np.ndarray | None = None,
):
    """Return (S_semantic in [0,1], detail dict) for all candidates.

    cand_emb: (N, d) L2-normalized candidate vectors.
    query_emb: (A, d) L2-normalized aspect-query vectors.
    aspect_term_lists: tokenized terms per aspect (for BM25).
    """
    a = query_emb.shape[0]
    if aspect_weights is None:
        aspect_weights = np.ones(a, dtype=np.float32)
    w = np.asarray(aspect_weights, dtype=np.float32)
    w = w / w.sum()

    dense = cand_emb @ query_emb.T                                   # (N, A) cosine
    sparse_mat = np.stack(
        [bm25.scores(terms) for terms in aspect_term_lists], axis=1
    )                                                                # (N, A)

    dense_agg = _aggregate(dense, w)
    sparse_agg = _aggregate(sparse_mat, w)
    d = _shape01(dense_agg)
    s = _shape01(sparse_agg)
    s_semantic = (config.W_DENSE * d + config.W_SPARSE * s).astype(np.float32)
    detail = {"dense": dense, "sparse": sparse_mat, "dense01": d, "sparse01": s}
    return s_semantic, detail
