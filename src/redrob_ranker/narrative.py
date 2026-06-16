"""Build the candidate 'narrative document' used for semantic matching.

This deliberately EXCLUDES the `skills` array and `education.field_of_study`:
profiling proved both are decoys (uniformly random, role-independent). The real
fit signal lives in the free-text headline/summary and per-role descriptions —
that is what we embed and lexically match.
"""
from __future__ import annotations

# The encoder truncates at a fixed token budget, so the narrative is bounded and
# front-loaded: headline + full summary + the most recent roles (career_history
# is stored most-recent-first), each description capped. This keeps the highest-
# signal text (recent experience) inside the truncation window and trims encode
# cost without losing the evidence the ranking depends on.
MAX_ROLES = 4
MAX_DESC_WORDS = 60


def _cap_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


def build_narrative(c: dict) -> str:
    """Concatenate headline + summary + recent roles' title/company/description."""
    p = c.get("profile", {})
    parts: list[str] = []
    if p.get("headline"):
        parts.append(str(p["headline"]))
    if p.get("summary"):
        parts.append(str(p["summary"]))
    for j in c.get("career_history", [])[:MAX_ROLES]:
        title = (j.get("title") or "").strip()
        company = (j.get("company") or "").strip()
        desc = _cap_words((j.get("description") or "").strip(), MAX_DESC_WORDS)
        head = f"{title} at {company}".strip(" at")
        parts.append(f"{head}. {desc}".strip())
    return "\n".join(pt for pt in parts if pt)


def career_titles(c: dict) -> list[str]:
    titles = [c.get("profile", {}).get("current_title", "")]
    titles += [j.get("title", "") for j in c.get("career_history", [])]
    return [t for t in titles if t]
