"""Role / seniority / domain fit -> S_role in [0, 1].

Title is a SOFT prior, never a hard filter (the plan's key mitigation): a modest
title ("Backend Engineer") can still score well if the narrative proves real
domain work, and a strong title alone is not enough without domain evidence.
This is what lets a Tier-5 "Backend Engineer who built a production recommender"
out-rank a title-only match, while keeping HR Managers / Accountants (the decoy
population the sample submission ranks #1) near the floor.

S_role = 0.5*title + 0.3*domain + 0.2*experience-band, then a CV/speech/robotics
penalty for candidates whose only domain is vision/speech/robotics (the JD:
"primary expertise is computer vision, speech, or robotics without significant
NLP/IR exposure ... you'd be re-learning fundamentals here").
"""
from __future__ import annotations

import re

import numpy as np

from .. import jd_spec
from ..io_utils import derived_experience_years
from ..textmatch import combined_text, contains_any, count_hits, norm, tokenset

# NLP/IR subset that "rescues" a CV/speech/robotics profile.
_NLP_IR = [
    "nlp", "natural language", "retrieval", "information retrieval", "search",
    "ranking", "recommendation", "recommender", "embedding", "semantic",
]
# Short off-target title words matched on token boundaries (avoid substring noise).
_OFFTARGET_WORDS = {
    "hr", "finance", "sales", "accounting", "accountant", "marketing",
    "operations", "recruiter", "teacher", "professor", "support", "qa",
}
_OFFTARGET_PHRASES = [
    "human resources", "talent acquisition", "business analyst",
    "content writer", "copywriter", "graphic designer", "civil engineer",
    "mechanical engineer", "electrical engineer", "product manager",
    "project manager", "customer success", "consultant",
]


def _is_offtarget(current_title: str) -> bool:
    cur = norm(current_title)
    toks = set(re.findall(r"[a-z]+", cur))
    return bool(toks & _OFFTARGET_WORDS) or contains_any(cur, _OFFTARGET_PHRASES)


def _all_titles(c: dict) -> str:
    p = c.get("profile", {})
    titles = [p.get("current_title", "")]
    titles += [j.get("title", "") for j in c.get("career_history", [])]
    return norm(" | ".join(t for t in titles if t))


def _title_score(c: dict) -> tuple[float, bool]:
    titles = _all_titles(c)
    offtarget = _is_offtarget(c.get("profile", {}).get("current_title", ""))
    if contains_any(titles, jd_spec.TITLE_STRONG):
        return (0.85 if offtarget else 1.0), offtarget
    if contains_any(titles, jd_spec.TITLE_ADJACENT):
        return (0.35 if offtarget else 0.60), offtarget
    if offtarget:
        return 0.05, True
    return 0.25, False


def _exp_fit(years: float) -> float:
    if 6.0 <= years <= 8.0:
        return 1.0
    if 5.0 <= years <= 9.0:
        return 0.85
    if 4.0 <= years < 10.5:
        return 0.65
    if 3.0 <= years < 12.0:
        return 0.45
    return 0.25  # JD: "a range, not a requirement" — never zero


def role_scores(cands: list[dict], texts: list[str] | None = None) -> tuple[np.ndarray, list[dict]]:
    out = np.empty(len(cands), dtype=np.float32)
    detail: list[dict] = []
    for i, c in enumerate(cands):
        text = texts[i] if texts is not None else combined_text(c)
        toks = tokenset(text)
        title, offtarget = _title_score(c)
        domain_hits = count_hits(text, jd_spec.DOMAIN_POSITIVE, toks)
        domain = min(1.0, domain_hits / 4.0)
        years = derived_experience_years(c)
        exp = _exp_fit(years)

        cv_only = contains_any(text, jd_spec.DOMAIN_CV_SPEECH_ROBOTICS, toks) and not contains_any(text, _NLP_IR, toks)
        score = 0.5 * title + 0.3 * domain + 0.2 * exp
        if cv_only:
            score *= 0.4
        score = float(np.clip(score, 0.0, 1.0))
        out[i] = score
        detail.append({
            "title_score": title,
            "offtarget_title": offtarget,
            "domain_hits": domain_hits,
            "derived_years": years,
            "cv_speech_only": cv_only,
            "score": score,
        })
    return out, detail
