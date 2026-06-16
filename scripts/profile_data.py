#!/usr/bin/env python3
"""Reproducible Phase-0 data profiler.

Recomputes the headline numbers the architecture decisions rest on:
geography skew, title long-tail, the yoe-unreliability finding, behavioral
medians, the trap populations, the skills/education decoy uniformity, and the
honeypot counts. Run:

    python scripts/profile_data.py --candidates ./candidates.jsonl
"""
from __future__ import annotations

import argparse
import collections
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from redrob_ranker import config, io_utils, jd_spec  # noqa: E402
from redrob_ranker.features.plausibility import assess  # noqa: E402
from redrob_ranker.features.role import _is_offtarget  # noqa: E402
from redrob_ranker.textmatch import contains_any, norm  # noqa: E402

AI_SKILL_KEYWORDS = [
    "machine learning", "deep learning", "nlp", "natural language",
    "computer vision", "tensorflow", "pytorch", "llm", "fine-tun", "rag",
    "transformer", "embedding", "retrieval", "ranking", "recommendation",
    "neural", "data science", "mlops",
]


def _ai_skill_count(c: dict) -> int:
    return sum(1 for s in c.get("skills", []) if contains_any(s.get("name", ""), AI_SKILL_KEYWORDS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(config.DATA_DIR / "candidates.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    n = 0
    countries = collections.Counter()
    titles = collections.Counter()
    skill_names = collections.Counter()
    stated_yoe, derived_yoe = [], []
    resp, inactive_days, notice = [], [], []
    no_github = open_to_work = 0
    consulting_only = non_india = stuffer = honeypot = 0

    for i, c in enumerate(io_utils.iter_candidates(args.candidates)):
        if args.limit and i >= args.limit:
            break
        n += 1
        p = c.get("profile", {})
        s = c.get("redrob_signals", {})
        countries[p.get("country", "?")] += 1
        titles[p.get("current_title", "?")] += 1
        for sk in c.get("skills", []):
            skill_names[sk.get("name", "?")] += 1
        stated_yoe.append(io_utils.stated_yoe(c))
        derived_yoe.append(io_utils.derived_experience_years(c))
        resp.append(float(s.get("recruiter_response_rate") or 0))
        d = io_utils.parse_date(s.get("last_active_date"))
        if d:
            inactive_days.append((config.REFERENCE_DATE - d).days)
        notice.append(float(s.get("notice_period_days") or 0))
        no_github += 1 if (s.get("github_activity_score", -1) == -1) else 0
        open_to_work += 1 if s.get("open_to_work_flag") else 0

        is_india = "india" in norm(p.get("country"))
        non_india += 0 if is_india else 1
        ch = c.get("career_history", [])
        if ch and all(
            contains_any(j.get("company", ""), config.CONSULTING_FIRMS)
            or contains_any(j.get("industry", ""), jd_spec.SERVICES_INDUSTRY_MARKERS)
            for j in ch
        ):
            consulting_only += 1
        if _ai_skill_count(c) >= 5 and _is_offtarget(p.get("current_title", "")):
            stuffer += 1
        if assess(c)["is_honeypot"]:
            honeypot += 1

    def med(x):
        return round(statistics.median(x), 2) if x else 0

    print(f"N = {n:,}")
    print(f"\nGeography: India {100 * (n - non_india) / n:.1f}% | non-India {non_india:,}")
    print("  top countries:", countries.most_common(6))
    print(f"\nTitles (top 12 of {len(titles):,} unique):")
    for t, ct in titles.most_common(12):
        print(f"  {ct:>6,}  {t}")
    print(f"\nExperience: stated yoe median {med(stated_yoe)} | derived median {med(derived_yoe)}")
    print("  (yoe is unreliable — derived from career history is what the ranker trusts)")
    print(f"\nBehavioral medians: response_rate {med(resp)} | inactive_days {med(inactive_days)} | notice {med(notice)}")
    print(f"  no GitHub (-1): {100 * no_github / n:.1f}% | open_to_work: {100 * open_to_work / n:.1f}%")
    print("\nTrap populations:")
    print(f"  consulting-only careers : {consulting_only:,}")
    print(f"  non-India (no visa)     : {non_india:,}")
    print(f"  keyword-stuffer shape   : {stuffer:,}  (>=5 AI skills + off-target title)")
    print(f"  honeypots (gate-caught) : {honeypot:,}")
    print(f"\nSkills decoy check (top 6 of {len(skill_names)} skill names — near-uniform => decoy):")
    for nm, ct in skill_names.most_common(6):
        print(f"  {ct:>6,}  {nm}")


if __name__ == "__main__":
    main()
