"""Scoring + deterministic selection tests (must match the bundled validator)."""
import numpy as np

from redrob_ranker import config, scoring


def test_base_fit_weights():
    s = scoring.base_fit(np.array([1.0]), np.array([1.0]), np.array([1.0]))
    assert abs(float(s[0]) - (config.W_SEMANTIC + config.W_ROLE + config.W_TRAJECTORY)) < 1e-6


def test_final_is_product_of_modifiers():
    base = np.array([0.5])
    out = scoring.final_score(base, np.array([1.0]), np.array([0.25]), np.array([1.0]))
    assert abs(float(out[0]) - 0.125) < 1e-6


def test_select_top_k_orders_and_breaks_ties_by_id():
    # Two candidates tie on score -> the smaller candidate_id must come first.
    ids = ["CAND_0000050", "CAND_0000010", "CAND_0000099"]
    scores = np.array([0.5, 0.5, 0.9])
    out = scoring.select_top_k(ids, scores, k=3)
    assert [cid for cid, _ in out] == ["CAND_0000099", "CAND_0000010", "CAND_0000050"]


def test_select_top_k_scores_non_increasing():
    rng = np.random.default_rng(0)
    ids = [f"CAND_{i:07d}" for i in range(500)]
    scores = rng.random(500)
    out = scoring.select_top_k(ids, scores, k=100)
    vals = [s for _, s in out]
    assert len(out) == 100
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))


def test_select_top_k_is_deterministic():
    ids = [f"CAND_{i:07d}" for i in range(300)]
    scores = np.linspace(0, 1, 300)[::-1]
    a = scoring.select_top_k(ids, scores, k=100)
    b = scoring.select_top_k(ids, scores, k=100)
    assert a == b
