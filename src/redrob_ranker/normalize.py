"""Small, shared score-normalization helpers.

Keeping these in one place means every sub-score is squashed into [0, 1] the
same way, which is what makes the blend weights in `config.py` interpretable
(and the Stage-5 answer to "why 0.45?" honest).
"""
from __future__ import annotations

import numpy as np


def minmax01(x: np.ndarray) -> np.ndarray:
    """Plain min-max to [0, 1]. Constant input -> all zeros."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def robust_minmax01(x: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Percentile-clipped min-max — robust to a few extreme outliers.

    Used for population-level signals (dense/sparse aggregates) where a single
    freak value would otherwise compress everyone else toward zero.
    """
    x = np.asarray(x, dtype=np.float64)
    lo = float(np.percentile(x, lo_pct))
    hi = float(np.percentile(x, hi_pct))
    if hi - lo < 1e-12:
        return minmax01(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def rank01(x: np.ndarray) -> np.ndarray:
    """Map values to their normalized rank in [0, 1] (ties share average rank).

    Distribution-free; handy when two raw scores live on very different scales
    (e.g. blending the heuristic `final` with the cross-encoder logit).
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).reshape(-1)  # numpy 2.0 briefly returned (n,1); normalize shape
    cum = np.cumsum(counts)
    start = cum - counts
    avg = (start + cum - 1) / 2.0
    ranks = avg[inv]
    return (ranks / (n - 1)).astype(np.float32)
