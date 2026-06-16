"""Build the candidate 'narrative document' used for semantic matching.

This deliberately EXCLUDES the `skills` array and `education.field_of_study`:
profiling proved both are decoys (uniformly random, role-independent). The real
fit signal lives in the free-text headline/summary and per-role descriptions —
that is what we embed and lexically match.
"""
from __future__ import annotations


def build_narrative(c: dict) -> str:
    """Concatenate headline + summary + each role's title/company/description."""
    p = c.get("profile", {})
    parts: list[str] = []
    if p.get("headline"):
        parts.append(str(p["headline"]))
    if p.get("summary"):
        parts.append(str(p["summary"]))
    for j in c.get("career_history", []):
        title = (j.get("title") or "").strip()
        company = (j.get("company") or "").strip()
        desc = (j.get("description") or "").strip()
        head = f"{title} at {company}".strip(" at")
        parts.append(f"{head}. {desc}".strip())
    return "\n".join(pt for pt in parts if pt)


def career_titles(c: dict) -> list[str]:
    titles = [c.get("profile", {}).get("current_title", "")]
    titles += [j.get("title", "") for j in c.get("career_history", [])]
    return [t for t in titles if t]
