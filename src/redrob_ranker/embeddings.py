"""Dense embedding model load + encode (precompute and query time).

The model weights are cached locally by the precompute step so the ranking step
can run with `HF_HUB_OFFLINE=1` (no network) — a hard Stage-3 requirement.
We use BAAI/bge-small-en-v1.5 (384-dim, strong CPU retrieval). BGE follows the
"prefix the query, not the passage" convention; passages (candidate narratives)
are encoded raw, queries (JD aspect-queries) get the instruction prefix.
"""
from __future__ import annotations

import numpy as np

from . import config

_MODEL_CACHE: dict[str, object] = {}


def load_embed_model(model_name: str = config.EMBED_MODEL):
    """Load (and cache) a SentenceTransformer on CPU.

    Imported lazily so that lightweight code paths (tests, validation) don't pay
    the torch import cost.
    """
    if model_name not in _MODEL_CACHE:
        import os

        import torch
        from sentence_transformers import SentenceTransformer

        # Use all physical cores for intra-op parallelism (torch defaults low on
        # some Windows builds). Bounded by the machine's CPU count.
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 4)))
        model = SentenceTransformer(model_name, device="cpu")
        # Narratives are short; 256 tokens covers headline+summary+role descriptions
        # for the vast majority and roughly halves CPU encode time vs the 512 default.
        model.max_seq_length = 256
        model._redrob_name = model_name  # remembered for the query-prefix decision
        _MODEL_CACHE[model_name] = model
    return _MODEL_CACHE[model_name]


def _is_bge(model_name: str) -> bool:
    return "bge" in model_name.lower()


def encode_passages(
    texts: list[str],
    model=None,
    batch_size: int = config.EMBED_BATCH,
    show_progress: bool = False,
) -> np.ndarray:
    """Encode candidate narratives to L2-normalized float32 vectors (N, d)."""
    model = model or load_embed_model()
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return np.ascontiguousarray(emb.astype(np.float32))


def encode_queries(queries: list[str], model=None, batch_size: int = 32) -> np.ndarray:
    """Encode JD aspect-queries to L2-normalized float32 vectors (A, d).

    Applies the BGE query instruction prefix when the active model is a BGE model.
    """
    model = model or load_embed_model()
    name = getattr(model, "_redrob_name", config.EMBED_MODEL)
    texts = [config.BGE_QUERY_PREFIX + q for q in queries] if _is_bge(name) else list(queries)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.ascontiguousarray(emb.astype(np.float32))
