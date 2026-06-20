"""End-to-end ranking orchestration.

Two entry points share one core:
  * rank_from_artifacts() — the scored path: load precomputed embeddings + BM25
    from artifacts/, align candidates by id, score, shortlist, rerank, select.
  * rank_in_memory() — the demo path (<=100 candidates): compute embeddings and
    the BM25 index on the fly, then run the identical scoring core. This is what
    makes the Streamlit sandbox reproduce the full system on a small sample.

Data flow (matches docs/PROJECT_MEMORY.md):
  base_fit = 0.45*S_semantic + 0.35*S_role + 0.20*S_trajectory
  final    = base_fit * P_plausible * geo_factor * behavioral_factor
  shortlist top-K by final -> cross-encoder rerank -> blend -> top-100 + reasoning
"""
from __future__ import annotations

import time

import numpy as np

from . import config, embeddings, io_utils, jd_spec, reasoning, scoring
from .features import behavioral, geo, plausibility, role, semantic, trajectory
from .narrative import build_narrative
from .normalize import rank01
from .textmatch import combined_text


def _log(msg: str, log) -> None:
    if log:
        log(f"[rank] {msg}")


# ---------------------------------------------------------------------------
# Artifact loading / candidate alignment
# ---------------------------------------------------------------------------
def load_artifacts():
    emb = np.load(config.EMB_PATH)
    ids = np.load(config.EMB_IDS_PATH, allow_pickle=False).astype(str)
    bm25 = semantic.load_bm25()
    return emb, ids, bm25


def _align_candidates(candidates_path: str, ids: np.ndarray) -> list[dict]:
    """Load candidates and return them in the SAME order as the embedding ids."""
    by_id = {c["candidate_id"]: c for c in io_utils.iter_candidates(candidates_path)}
    missing = [cid for cid in ids if cid not in by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} embedded ids are absent from {candidates_path} "
            f"(first: {missing[:3]}). Re-run precompute on this candidates file."
        )
    return [by_id[cid] for cid in ids]


# ---------------------------------------------------------------------------
# Scoring core (shared by both entry points)
# ---------------------------------------------------------------------------
def _aspect_inputs():
    queries = [a["query"] for a in jd_spec.ASPECTS]
    weights = np.array([a["weight"] for a in jd_spec.ASPECTS], dtype=np.float32)
    term_lists = [semantic.tokenize(" ".join(a["terms"])) for a in jd_spec.ASPECTS]
    return queries, weights, term_lists


def _score_all(cands, emb, bm25, log=None):
    """Compute every sub-score + multiplier + the heuristic `final` for all cands."""
    texts = [combined_text(c) for c in cands]

    queries, weights, term_lists = _aspect_inputs()
    query_emb = embeddings.encode_queries(queries)
    s_sem, sem_detail = semantic.semantic_scores(emb, query_emb, bm25, term_lists, weights)
    _log("semantic done", log)

    s_role, role_detail = role.role_scores(cands, texts=texts)
    s_traj, traj_detail = trajectory.trajectory_scores(cands, texts=texts)
    _log("role+trajectory done", log)

    p_plaus, plaus_detail = plausibility.plausibility_factor(cands)
    f_geo, geo_detail = geo.geo_factor(cands)
    f_beh, beh_detail = behavioral.behavioral_factor(cands)
    _log("modifiers done", log)

    base = scoring.base_fit(s_sem, s_role, s_traj)
    final = scoring.final_score(base, p_plaus, f_geo, f_beh)
    detail = {
        "s_semantic": s_sem, "s_role": s_role, "s_trajectory": s_traj,
        "base": base, "final": final,
        # raw multiplier arrays kept so evaluate.py can run modifier sensitivity
        # sweeps without recomputing the (expensive) sub-scores.
        "p_plaus": p_plaus, "f_geo": f_geo, "f_beh": f_beh,
        "role": role_detail, "trajectory": traj_detail, "plausibility": plaus_detail,
        "geo": geo_detail, "behavioral": beh_detail, "semantic": sem_detail,
    }
    return final, detail


def _rerank_shortlist(cands, final, use_rerank, log=None):
    """Return (shortlist_indices, shortlist_score) — blended if rerank is on."""
    n = len(cands)
    k = min(config.SHORTLIST_K, n)
    shortlist = np.argsort(-final, kind="stable")[:k]

    if not use_rerank:
        return shortlist, rank01(final[shortlist])

    try:
        from . import rerank as rr
        model, ce_name = rr.load_cross_encoder()
        # Only the top RERANK_SHORTLIST_K get the (expensive) cross-encoder; the rest
        # of the recall shortlist keep heuristic order strictly below the reranked
        # block. This lets a heavy reranker stay in budget (it MUST: bge-reranker-base
        # over the full 1000 measured ~25 min here) while still covering the top-100.
        rk = min(config.RERANK_SHORTLIST_K, k)
        rerank_idx = shortlist[:rk]
        narratives = [build_narrative(cands[i]) for i in rerank_idx]
        cross = rr.cross_encoder_scores(narratives, model=model)
        # Trust a STRONG reranker enough to lead the top-K ordering; the weak MiniLM
        # fallback stays subordinate to the heuristic (PLAN_REVIEW_V2 Round 5).
        is_strong = ce_name != config.CROSS_ENCODER_FALLBACK
        if is_strong:
            w_final, w_cross = config.W_FINAL_IN_RERANK, config.W_CROSS_IN_RERANK
            kind = "authoritative"
        else:
            w_final, w_cross = config.W_FINAL_IN_RERANK_FALLBACK, config.W_CROSS_IN_RERANK_FALLBACK
            kind = "conservative"
        blended = scoring.blend_rerank(final[rerank_idx], cross, w_final, w_cross)  # [0,1]
        _log(f"reranked {rk}/{k} with {ce_name} ({kind} blend {w_final}/{w_cross})", log)

        score = np.empty(k, dtype=np.float32)
        score[:rk] = blended
        if rk < k:  # un-reranked tail sits below the reranked block, heuristic order
            score[rk:] = (-1.0 + rank01(final[shortlist[rk:]])).astype(np.float32)
        return shortlist, score
    except Exception as e:  # graceful degradation — no model available at all
        _log(f"rerank skipped ({type(e).__name__}: {e}); using heuristic final", log)
        return shortlist, rank01(final[shortlist])


def _finalize(cands, detail, shortlist, shortlist_score, top_k):
    """Select top-k (validator-ordered) and attach grounded reasoning."""
    sl_ids = [cands[i]["candidate_id"] for i in shortlist]
    chosen = scoring.select_top_k(sl_ids, shortlist_score, k=top_k)

    idx_of = {cands[i]["candidate_id"]: i for i in shortlist}
    rows = []
    for rank, (cid, score) in enumerate(chosen, start=1):
        i = idx_of[cid]
        cdetail = {
            "role": detail["role"][i], "trajectory": detail["trajectory"][i],
            "behavioral": detail["behavioral"][i], "geo": detail["geo"][i],
            "plausibility": detail["plausibility"][i],
            "derived_years": detail["role"][i]["derived_years"],
            "s_semantic": float(detail["s_semantic"][i]),
        }
        text = reasoning.make_reasoning(cands[i], cdetail, rank)
        rows.append({"candidate_id": cid, "rank": rank, "score": score, "reasoning": text})
    return rows


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def rank_from_artifacts(candidates_path, top_k=config.TOP_K, use_rerank=config.USE_RERANK, log=print):
    t0 = time.time()
    emb, ids, bm25 = load_artifacts()
    _log(f"artifacts loaded: {emb.shape} ({time.time() - t0:.1f}s)", log)
    cands = _align_candidates(candidates_path, ids)
    _log(f"{len(cands):,} candidates aligned ({time.time() - t0:.1f}s)", log)

    final, detail = _score_all(cands, emb, bm25, log=log)
    shortlist, sl_score = _rerank_shortlist(cands, final, use_rerank, log=log)
    rows = _finalize(cands, detail, shortlist, sl_score, top_k)
    _log(f"done: {len(rows)} rows in {time.time() - t0:.1f}s", log)
    return rows


def rank_in_memory(cands, top_k=config.TOP_K, use_rerank=config.USE_RERANK, log=None):
    """Rank an in-memory candidate list (demo path); computes its own embeddings."""
    narratives = [build_narrative(c) for c in cands]
    emb = embeddings.encode_passages(narratives)
    bm25 = semantic.build_bm25_index(narratives)  # in-memory; never touches artifacts/

    final, detail = _score_all(cands, emb, bm25, log=log)
    use = use_rerank and len(cands) <= 2000  # keep the demo within budget
    shortlist, sl_score = _rerank_shortlist(cands, final, use, log=log)
    return _finalize(cands, detail, shortlist, sl_score, min(top_k, len(cands)))
