"""Behavioral availability modifier in [0.6, 1.15].

The JD: "a perfect-on-paper candidate who hasn't logged in for 6 months and has
a 5% recruiter response rate is, for hiring purposes, not actually available.
Down-weight them appropriately." This factor MODULATES fit — it never dominates
it (the decoy submission's mistake was letting response-rate drive the ranking).
The envelope is intentionally asymmetric: an unavailable candidate can lose up
to 40%, while an ideal one gains at most 15%.
"""
from __future__ import annotations

import numpy as np

from .. import config
from ..io_utils import parse_date


def _recency01(last_active, ref) -> float:
    d = parse_date(last_active)
    if d is None:
        return 0.4
    days = (ref - d).days
    if days <= 30:
        return 1.0
    if days >= 210:
        return 0.0
    return max(0.0, 1.0 - (days - 30) / 180.0)


def _notice01(days: float) -> float:
    if days <= 30:
        return 1.0
    if days <= 60:
        return 0.7
    if days <= 90:
        return 0.5
    return 0.2


def _avail_score(c: dict, ref) -> tuple[float, dict]:
    s = c.get("redrob_signals", {})
    resp = float(s.get("recruiter_response_rate") or 0.0)
    recency = _recency01(s.get("last_active_date"), ref)
    otw = 1.0 if s.get("open_to_work_flag") else 0.4
    notice = float(s.get("notice_period_days") if s.get("notice_period_days") is not None else 90)
    notice_sc = _notice01(notice)
    saved = float(s.get("saved_by_recruiters_30d") or 0)
    gh = float(s.get("github_activity_score") if s.get("github_activity_score") is not None else -1)
    engagement = 0.0
    engagement += 0.5 if saved > 0 else 0.0
    engagement += 0.5 if gh > 20 else (0.2 if gh >= 0 else 0.0)

    avail = (
        0.30 * resp
        + 0.30 * recency
        + 0.15 * otw
        + 0.15 * notice_sc
        + 0.10 * engagement
    )
    d = parse_date(s.get("last_active_date"))
    detail = {
        "response_rate": resp,
        "days_inactive": (ref - d).days if d else None,
        "open_to_work": bool(s.get("open_to_work_flag")),
        "notice_days": notice,
        "github_activity": gh,
        "avail": float(avail),
    }
    return avail, detail


def behavioral_factor(cands: list[dict], ref=config.REFERENCE_DATE) -> tuple[np.ndarray, list[dict]]:
    out = np.empty(len(cands), dtype=np.float32)
    detail: list[dict] = []
    for i, c in enumerate(cands):
        avail, d = _avail_score(c, ref)
        if avail >= 0.5:
            factor = 1.0 + (avail - 0.5) / 0.5 * (config.BEHAVIOR_MAX - 1.0)
        else:
            factor = 1.0 - (0.5 - avail) / 0.5 * (1.0 - config.BEHAVIOR_MIN)
        factor = float(np.clip(factor, config.BEHAVIOR_MIN, config.BEHAVIOR_MAX))
        d["factor"] = factor
        out[i] = factor
        detail.append(d)
    return out, detail
