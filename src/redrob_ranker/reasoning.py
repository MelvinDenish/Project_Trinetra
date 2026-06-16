"""Evidence-Ledger Grounded NLG with controlled variation.

Produces the CSV `reasoning` column for the top-100. Design goals (Stage-4's six
checks): specific facts, JD connection, honest concerns, ZERO hallucination,
cross-row variation, and rank-consistent tone. It is fully deterministic so the
ranking step reproduces the exact same CSV byte-for-byte in the Stage-3 sandbox
(a local-LLM-polished CSV would not — see docs).

Pipeline:
  1. Evidence ledger  — collect pointers to REAL fields only (current title,
     capability keywords that literally appear in the narrative, derived years,
     and concern flags from the feature detail dicts). `evidence_tokens` returns
     the grounded substrings so a CI test can assert they all exist in the record.
  2. Content selection — 1-2 strongest positives + the single worst concern.
  3. Surface realization — lead phrase + connector chosen by a stable hash of
     (candidate_id, facets) for real variation; tone scaled by rank tier.
"""
from __future__ import annotations

import hashlib

from .textmatch import combined_text, norm, term_in, tokenset

# Capability keyword -> reader-facing phrase. Order = priority for selection.
_CAPABILITY_PRIORITY = [
    ("retrieval", "retrieval"),
    ("information retrieval", "information retrieval"),
    ("ranking", "ranking"),
    ("recommendation", "recommendation systems"),
    ("recommender", "recommendation systems"),
    ("recsys", "recommendation systems"),
    ("semantic", "semantic search"),
    ("search", "search"),
    ("embedding", "embeddings"),
    ("embeddings", "embeddings"),
    ("relevance", "search relevance"),
    ("nlp", "NLP"),
    ("vector", "vector search"),
    ("llm", "LLMs"),
    ("rag", "RAG"),
    ("matching", "matching"),
    ("personalization", "personalization"),
]

_LEADS = {
    "confident": ["Strong fit —", "Top-tier match —", "Excellent alignment —"],
    "standard": ["Solid fit —", "Good match —", "Relevant background —"],
    "hedged": ["Borderline fit —", "Adjacent profile —", "Near the cutoff —"],
}
_CONNECTORS = ["Concern:", "Caveat:", "Note:", "However,"]


def _tier(rank: int) -> str:
    if rank <= 10:
        return "confident"
    if rank <= 50:
        return "standard"
    return "hedged"


def _pick(cid: str, salt: str, n: int) -> int:
    h = hashlib.md5(f"{cid}|{salt}".encode("utf-8")).hexdigest()
    return int(h, 16) % n


def _capabilities(text: str) -> list[tuple[str, str]]:
    """Up to 2 (phrase, grounded_token) capabilities that appear in `text`."""
    toks = tokenset(text)
    found: list[tuple[str, str]] = []
    seen_phrases: set[str] = set()
    for token, phrase in _CAPABILITY_PRIORITY:
        if term_in(token, text, toks) and phrase not in seen_phrases:
            found.append((phrase, token))
            seen_phrases.add(phrase)
        if len(found) >= 2:
            break
    return found


def _concern(c: dict, detail: dict) -> tuple[str, list[str]] | None:
    """Worst applicable concern as (text, grounded_tokens), or None."""
    role = detail.get("role", {})
    traj = detail.get("trajectory", {})
    geo = detail.get("geo", {})
    beh = detail.get("behavioral", {})
    plaus = detail.get("plausibility", {})
    country = norm(c.get("profile", {}).get("country", ""))
    notice = int(beh.get("notice_days") or 0)

    if plaus.get("is_honeypot"):
        return "the profile contains internally inconsistent claims", []
    if not geo.get("is_india", True):
        if geo.get("willing_relocate"):
            return (f"based outside India (in {country}) though open to relocating; the JD does not sponsor visas", [country] if country else [])
        return (f"based outside India (in {country}); the JD does not sponsor visas", [country] if country else [])
    if traj.get("consulting_only"):
        return "the career has been entirely at services/consulting firms with no product-company experience", []
    if traj.get("research_only"):
        return "the background reads as research-only with little production deployment", []
    if role.get("cv_speech_only"):
        return "the domain looks like vision/speech rather than the NLP/IR this role needs", []
    if traj.get("framework_only"):
        return "the AI experience looks recent and framework-centric without deeper ML history", []
    if traj.get("title_chaser"):
        return "short tenures with rapid title escalation (the JD wants a 3+ year commitment)", []
    if traj.get("no_recent_code"):
        return "a senior/lead profile with limited recent hands-on coding", []
    if beh.get("response_rate", 1.0) < 0.2 or (beh.get("days_inactive") or 0) > 150:
        return "limited recent platform activity and a low recruiter response rate", []
    if notice >= 90:
        return (f"a long notice period ({notice} days)", [str(notice)])
    if role.get("offtarget_title") and detail.get("s_semantic", 0) > 0.4:
        return "the current title is off-target, though the narrative shows relevant work", []
    return None


def reason_and_evidence(c: dict, detail: dict, rank: int) -> tuple[str, list[str]]:
    """Return (reasoning_text, grounded_tokens)."""
    p = c.get("profile", {})
    cid = c.get("candidate_id", "")
    title = (p.get("current_title") or "").strip()
    text = combined_text(c)
    years = round(float(detail.get("derived_years") or 0))
    caps = _capabilities(text)

    grounded: list[str] = []
    tier = _tier(rank)
    lead = _LEADS[tier][_pick(cid, "lead" + tier, len(_LEADS[tier]))]

    # ---- positive clause ----
    subject = title if title else "candidate"
    if title:
        grounded.append(norm(title))
    years_clause = f" with ~{years} years' experience" if years >= 3 else ""

    if caps:
        for _, tok in caps:
            grounded.append(tok)
        names = [phrase for phrase, _ in caps]
        cap_clause = (
            f", showing hands-on {names[0]} experience"
            if len(names) == 1
            else f", showing hands-on {names[0]} and {names[1]} experience"
        )
    else:
        cap_clause = ""

    prod_clause = ""
    if detail.get("trajectory", {}).get("production_hits", 0) >= 2 and caps and _pick(cid, "prod", 2):
        prod_clause = "; has shipped systems to production"

    positive = f"{lead} {subject}{years_clause}{cap_clause}{prod_clause}."

    # ---- concern clause ----
    concern = _concern(c, detail)
    if concern is not None:
        ctext, ctokens = concern
        grounded.extend(t for t in ctokens if t)
        connector = _CONNECTORS[_pick(cid, "conn", len(_CONNECTORS))]
        concern_sentence = f" {connector} {ctext}."
    elif tier == "hedged":
        concern_sentence = " Included as a lower-confidence pick near the cutoff."
    else:
        concern_sentence = ""

    return (positive + concern_sentence).strip(), grounded


def make_reasoning(c: dict, detail: dict, rank: int) -> str:
    return reason_and_evidence(c, detail, rank)[0]


def faithfulness_violations(c: dict, detail: dict, rank: int) -> list[str]:
    """Grounded tokens that do NOT appear in the candidate record (should be [])."""
    import json

    haystack = json.dumps(c, ensure_ascii=False).lower()
    _, tokens = reason_and_evidence(c, detail, rank)
    return [t for t in tokens if t and t not in haystack]
