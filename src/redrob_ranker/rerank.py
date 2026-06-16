"""Optional local cross-encoder rerank of the shortlist (sharpens NDCG@10).

A pretrained cross-encoder (ms-marco-MiniLM-L-6-v2) reads the (JD_summary,
candidate_narrative) pair jointly and scores relevance — strictly better than
bi-encoder cosine for fine top-k ordering, which is where 50% of the composite
(NDCG@10) lives. It is run ONLY on the ~1K shortlist so it fits the CPU budget,
and the whole stage is optional: if the model is unavailable the pipeline falls
back to the heuristic `final` score (USE_RERANK / graceful degradation).
"""
from __future__ import annotations

import numpy as np

from . import config, jd_spec

_CE_CACHE: dict[str, object] = {}


def load_cross_encoder(name: str = config.CROSS_ENCODER_MODEL):
    if name not in _CE_CACHE:
        import os

        import torch
        from sentence_transformers import CrossEncoder

        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 4)))
        _CE_CACHE[name] = CrossEncoder(name, max_length=256, device="cpu")
    return _CE_CACHE[name]


def cross_encoder_scores(
    narratives: list[str],
    jd_text: str = jd_spec.JD_SUMMARY,
    model=None,
    batch_size: int = 64,
) -> np.ndarray:
    """Relevance score for each narrative against the JD summary (higher=better)."""
    model = model or load_cross_encoder()
    pairs = [(jd_text, n) for n in narratives]
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    return np.asarray(scores, dtype=np.float32)
