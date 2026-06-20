"""Score combination + deterministic top-K selection.

The blend weights live in config.py (calibrated against the anchor set, not
invented). The selection step is written to satisfy the bundled validator
*exactly*: scores are rounded to SCORE_DECIMALS BEFORE sorting (so float drift
cannot flip adjacent ranks), the sort key is (score descending, candidate_id
ascending), ranks are the 1..K positions in that order, and the emitted score
column is the rounded value — guaranteeing it is non-increasing and that equal
scores are ordered by candidate_id ascending.
"""
from __future__ import annotations

import numpy as np

from . import config
from .normalize import rank01


def base_fit(s_semantic: np.ndarray, s_role: np.ndarray, s_trajectory: np.ndarray) -> np.ndarray:
    return (
        config.W_SEMANTIC * s_semantic
        + config.W_ROLE * s_role
        + config.W_TRAJECTORY * s_trajectory
    ).astype(np.float32)


def final_score(
    base: np.ndarray,
    plausibility: np.ndarray,
    geo: np.ndarray,
    behavioral: np.ndarray,
) -> np.ndarray:
    return (base * plausibility * geo * behavioral).astype(np.float32)


def blend_rerank(
    final_shortlist: np.ndarray,
    cross_shortlist: np.ndarray,
    w_final: float = config.W_FINAL_IN_RERANK,
    w_cross: float = config.W_CROSS_IN_RERANK,
) -> np.ndarray:
    """Blend heuristic `final` with the cross-encoder on the shortlist.

    Both are mapped to normalized ranks first so the cross-encoder's logit scale
    (which is unbounded) can't swamp the bounded heuristic score. The weights are
    supplied by the caller so a *strong* reranker can be made authoritative for
    the top-K while a *fallback* reranker stays subordinate (PLAN_REVIEW_V2 R5).
    """
    f = rank01(final_shortlist)
    c = rank01(cross_shortlist)
    return (w_final * f + w_cross * c).astype(np.float32)


def select_top_k(
    ids: list[str],
    scores: np.ndarray,
    k: int = config.TOP_K,
    decimals: int = config.SCORE_DECIMALS,
) -> list[tuple[str, float]]:
    """Return [(candidate_id, rounded_score)] for the top-k, validator-ordered."""
    rounded = np.round(np.asarray(scores, dtype=np.float64), decimals)
    order = sorted(range(len(ids)), key=lambda i: (-rounded[i], ids[i]))
    return [(ids[i], float(rounded[i])) for i in order[:k]]
