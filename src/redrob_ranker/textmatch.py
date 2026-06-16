"""Tiny lower-cased word-aware matching helpers shared by the rule features.

Deliberately simple and transparent: every negative/positive rule in the
role/trajectory features is "does this lexicon term appear in this text", which
keeps the system auditable (a recruiter can see exactly why a candidate scored
the way they did) — the opposite of an opaque learned model.

Matching is WORD-AWARE, not raw substring: a single alphanumeric term must match
a whole token (so "rag" does NOT match inside "storage"/"average"), while
multi-word or symbol-bearing terms ("hybrid search", "c++", "a/b") fall back to
substring. This avoids the short-token false positives that would otherwise
inflate domain scores and produce hallucinated reasoning.
"""
from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9][a-z0-9+#.]*")


def norm(text: str | None) -> str:
    return (text or "").lower()


def tokenset(text: str) -> set[str]:
    return set(_WORD.findall(norm(text)))


def _is_word_term(term: str) -> bool:
    """True if `term` is a single plain alphanumeric token (-> token match)."""
    return term.isalnum() and " " not in term


def term_in(term: str, text: str, toks: set[str] | None = None) -> bool:
    if _is_word_term(term):
        toks = toks if toks is not None else tokenset(text)
        return term in toks
    return term in norm(text)


def contains_any(text: str, terms, toks: set[str] | None = None) -> bool:
    toks = toks if toks is not None else tokenset(text)
    t = norm(text)
    return any(term in toks if _is_word_term(term) else term in t for term in terms)


def count_hits(text: str, terms, toks: set[str] | None = None) -> int:
    """Number of distinct lexicon terms that appear in `text`."""
    toks = toks if toks is not None else tokenset(text)
    t = norm(text)
    return sum(1 for term in terms if (term in toks if _is_word_term(term) else term in t))


def matched_terms(text: str, terms, toks: set[str] | None = None) -> list[str]:
    toks = toks if toks is not None else tokenset(text)
    t = norm(text)
    return [term for term in terms if (term in toks if _is_word_term(term) else term in t)]


def combined_text(c: dict) -> str:
    """Lower-cased full free-text of a candidate for rule matching.

    Unlike the (truncated) embedding narrative, this uses the *full* summary and
    every role description so negative rules (research-only, framework-only,
    no-recent-code) see all the evidence.
    """
    p = c.get("profile", {})
    parts = [p.get("headline", ""), p.get("summary", ""), p.get("current_title", "")]
    for j in c.get("career_history", []):
        parts.append(j.get("title", ""))
        parts.append(j.get("company", ""))
        parts.append(j.get("description", ""))
    return norm(" \n ".join(p for p in parts if p))
