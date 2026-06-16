"""Streaming IO and schema-safe accessors for candidate records.

Key data lesson baked in here (see PROJECT_MEMORY.md):
`profile.years_of_experience` is unreliable (a genuine ML engineer in the data
has yoe=2.7 while their summary says 6.3 and career history sums to 75 months).
So `derived_experience_years` is computed robustly from multiple signals and is
what the ranking trusts, while the *stated* yoe is kept separately for the
plausibility gate (which looks for self-contradiction).
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import re
from pathlib import Path
from typing import Iterator

from . import config

_YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*years", re.IGNORECASE)


def _open(path: str | Path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_candidates(path: str | Path) -> Iterator[dict]:
    """Yield one candidate dict per JSONL line (gzip-aware, blank-line-safe)."""
    with _open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_candidates(path: str | Path, limit: int | None = None) -> list[dict]:
    out: list[dict] = []
    for i, c in enumerate(iter_candidates(path)):
        if limit is not None and i >= limit:
            break
        out.append(c)
    return out


def parse_date(s: str | None) -> dt.date | None:
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def months_between(start: dt.date, end: dt.date) -> float:
    return (end.year - start.year) * 12 + (end.month - start.month) + (end.day - start.day) / 30.0


def career_sum_months(c: dict) -> float:
    return float(sum((j.get("duration_months") or 0) for j in c.get("career_history", [])))


def career_span_months(c: dict, ref: dt.date = config.REFERENCE_DATE) -> float:
    """Calendar span from earliest start to latest end (or ref for current roles)."""
    starts, ends = [], []
    for j in c.get("career_history", []):
        sd = parse_date(j.get("start_date"))
        if sd:
            starts.append(sd)
            ed = parse_date(j.get("end_date")) or ref
            ends.append(ed)
    if not starts:
        return 0.0
    return max(0.0, months_between(min(starts), max(ends)))


def years_from_summary(c: dict) -> float:
    m = _YEARS_RE.search(c.get("profile", {}).get("summary", "") or "")
    return float(m.group(1)) if m else 0.0


def stated_yoe(c: dict) -> float:
    return float(c.get("profile", {}).get("years_of_experience") or 0.0)


def derived_experience_years(c: dict) -> float:
    """Robust experience estimate; resilient to the planted yoe noise.

    Uses the strongest corroborated signal: calendar span of the career, the
    self-reported years in the summary, and the stated yoe field.
    """
    span_years = career_span_months(c) / 12.0
    return round(max(span_years, years_from_summary(c), stated_yoe(c)), 2)


def dataset_reference_date(path: str | Path) -> dt.date:
    """Latest activity/end date seen in the file (used to anchor 'today')."""
    latest = config.REFERENCE_DATE
    for c in iter_candidates(path):
        d = parse_date(c.get("redrob_signals", {}).get("last_active_date"))
        if d and d > latest:
            latest = d
    return latest
