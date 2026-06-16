"""Honeypot / consistency gate (multiplicative, in [PLAUSIBILITY_HONEYPOT, 1.0]).

Honeypots are FORCED to relevance tier 0 in the hidden ground truth, and >10%
honeypot rate in the top-100 is an automatic Stage-3 disqualification, so this
gate is a hard safety net layered on top of the fact that honeypots also score
low on genuine fit (their narratives are generic by construction).

The signals here were calibrated against the full 100K pool and are
*unambiguous impossibilities only* (the plan warns that a naive yoe-vs-duration
check false-positives on genuine candidates):

  1. >=3 skills at expert/advanced proficiency with 0 months of usage.
     Observed counts in the pool: {0: 99979, 3: 8, 4: 5, 5: 8} — i.e. it is
     either 0 or >=3, never 1-2; a perfectly clean planted signal (21 records).
  2. A role whose claimed duration_months exceeds the calendar span of its own
     start_date->end_date by >3 months — claiming more tenure than the dates
     allow (19 records, disjoint from set 1).

Deliberately NOT used: "skill duration > derived career" — it matches 9,191
genuine candidates (skills aren't tied to employment in this data), so it is a
false-positive factory, not a honeypot signal.
"""
from __future__ import annotations

import numpy as np

from .. import config
from ..io_utils import months_between, parse_date


def _expert_zero_duration_count(c: dict) -> int:
    return sum(
        1
        for s in c.get("skills", [])
        if s.get("proficiency") in ("expert", "advanced")
        and (s.get("duration_months") or 0) == 0
    )


def _duration_date_inconsistency(c: dict, ref=config.REFERENCE_DATE) -> bool:
    for j in c.get("career_history", []):
        sd = parse_date(j.get("start_date"))
        if not sd:
            continue
        ed = parse_date(j.get("end_date")) or ref
        span = months_between(sd, ed)
        if (j.get("duration_months") or 0) > span + config.JOB_OVER_SPAN_MONTHS:
            return True
    return False


def assess(c: dict, ref=config.REFERENCE_DATE) -> dict:
    """Per-candidate plausibility assessment with human-readable reasons."""
    e0 = _expert_zero_duration_count(c)
    inconsistent = _duration_date_inconsistency(c, ref)
    reasons: list[str] = []
    if e0 >= config.EXPERT_ZERO_DURATION_HARD:
        reasons.append(f"{e0} skills marked expert/advanced with 0 months of actual use")
    if inconsistent:
        reasons.append("a role claims more tenure than its own start/end dates allow")

    is_honeypot = bool(reasons)
    if is_honeypot:
        factor = config.PLAUSIBILITY_HONEYPOT
    elif e0 >= 1:  # defensive soft tier (does not occur in the released pool)
        factor = config.PLAUSIBILITY_SOFT
        reasons.append(f"{e0} expert/advanced skill(s) with 0 months of use")
    else:
        factor = config.PLAUSIBILITY_OK
    return {"factor": float(factor), "is_honeypot": is_honeypot, "reasons": reasons}


def plausibility_factor(cands: list[dict], ref=config.REFERENCE_DATE) -> tuple[np.ndarray, list[dict]]:
    out = np.empty(len(cands), dtype=np.float32)
    detail: list[dict] = []
    for i, c in enumerate(cands):
        d = assess(c, ref)
        out[i] = d["factor"]
        detail.append(d)
    return out, detail
