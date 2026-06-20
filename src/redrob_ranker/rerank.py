"""Local cross-encoder rerank of the shortlist (sharpens NDCG@10 = 50% of score).

A pretrained cross-encoder reads the (JD_summary, candidate_narrative) pair
jointly and scores relevance — strictly better than bi-encoder cosine for fine
top-k ordering, which is where 50% of the composite (NDCG@10) lives. It is run
ONLY on the ~1K shortlist so it fits the CPU budget.

Model selection (PLAN_REVIEW_V2 Round 4): we PREFER a modern reranker
(`BAAI/bge-reranker-base`, far stronger than the 2019-era ms-marco-MiniLM) and
fall back through `CROSS_ENCODER_FALLBACK` (the always-cached MiniLM) and finally
to the heuristic `final` if no reranker can be loaded. `load_cross_encoder`
returns BOTH the model and the name that actually loaded so the caller can choose
an appropriate blend weight (trust a strong reranker more than a weak fallback).
"""
from __future__ import annotations

import numpy as np

from . import config, jd_spec

_CE_CACHE: dict[str, object] = {}


def _candidate_models() -> list[str]:
    """Preferred model first, then the cached fallback (de-duplicated)."""
    seq = [config.CROSS_ENCODER_MODEL, config.CROSS_ENCODER_FALLBACK]
    out: list[str] = []
    for m in seq:
        if m and m not in out:
            out.append(m)
    return out


def load_cross_encoder(name: str | None = None) -> tuple[object, str]:
    """Load the best available cross-encoder. Returns (model, loaded_name).

    Tries `name` (or the configured preferred model) first, then the fallback;
    raises only if NONE can be loaded (caller then degrades to the heuristic).
    """
    order = [name] if name else _candidate_models()
    last_err: Exception | None = None
    for cand in order:
        if cand in _CE_CACHE:
            return _CE_CACHE[cand], cand
        try:
            import os

            import torch
            from sentence_transformers import CrossEncoder

            torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 4)))
            model = CrossEncoder(cand, max_length=config.CROSS_ENCODER_MAX_LEN, device="cpu")
            _CE_CACHE[cand] = model
            return model, cand
        except Exception as e:  # not cached / offline / incompatible — try the next
            last_err = e
            continue
    raise RuntimeError(f"no cross-encoder could be loaded (last error: {last_err})")


def cross_encoder_scores(
    narratives: list[str],
    jd_text: str = jd_spec.JD_SUMMARY,
    model=None,
    batch_size: int = 64,
) -> np.ndarray:
    """Relevance score for each narrative against the JD summary (higher=better)."""
    if model is None:
        model, _ = load_cross_encoder()
    pairs = [(jd_text, n) for n in narratives]
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    return np.asarray(scores, dtype=np.float32)
