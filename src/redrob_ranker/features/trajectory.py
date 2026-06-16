"""Career-trajectory fit -> S_trajectory in [0, 1].

Encodes the JD's "things we explicitly do NOT want" that are about the *shape*
of a career rather than the role itself:
  * consulting-ONLY careers (TCS/Infosys/... with no product-company experience);
  * title-chasers (sub-18-month hops with escalating seniority);
  * pure research with no production deployment;
  * "AI = LangChain in the last <12 months" with no pre-LLM ML to back it;
  * senior engineers with no recent hands-on coding.
Each is a bounded multiplicative penalty on a base that rewards production /
shipping evidence, so the signal stays in [0, 1] and no single rule dominates.
"""
from __future__ import annotations

import numpy as np

from .. import config, jd_spec
from ..io_utils import derived_experience_years
from ..textmatch import combined_text, contains_any, count_hits, norm, tokenset

_CODE_SIGNALS = [
    "code", "coded", "built", "develop", "implement", "engineered", "python",
    "deployed", "production", "shipped", "model", "pipeline", "system", "api",
]
_SENIORITY = [
    ("chief", 7), ("vp", 7), ("vice president", 7), ("head ", 6), ("director", 6),
    ("principal", 5), ("staff", 4), ("lead", 3), ("senior", 2), ("sr ", 2),
    ("associate", 0), ("junior", 0), ("jr ", 0), ("intern", -1),
]


def _seniority(title: str) -> int:
    t = norm(title)
    best = 1
    for key, val in _SENIORITY:
        if key in t:
            best = max(best, val)
    return best


def _is_services_role(company: str, industry: str) -> bool:
    return contains_any(company, config.CONSULTING_FIRMS) or contains_any(
        industry, jd_spec.SERVICES_INDUSTRY_MARKERS
    )


def _consulting_only(c: dict) -> bool:
    ch = c.get("career_history", [])
    if not ch:
        return False
    flags = [_is_services_role(j.get("company", ""), j.get("industry", "")) for j in ch]
    return all(flags)


def _title_chaser(c: dict) -> bool:
    ch = c.get("career_history", [])
    if len(ch) < 3:
        return False
    short = [j for j in ch if (j.get("duration_months") or 0) < 18 and not j.get("is_current")]
    # career_history is most-recent-first, so [0] is latest, [-1] earliest.
    escalation = _seniority(ch[0].get("title", "")) - _seniority(ch[-1].get("title", ""))
    return len(short) >= 2 and escalation >= 2


def _senior_no_recent_code(c: dict, derived_years: float) -> bool:
    cur_title = norm(c.get("profile", {}).get("current_title", ""))
    is_senior = derived_years >= 7.0 or _seniority(cur_title) >= 3
    if not is_senior:
        return False
    ch = c.get("career_history", [])
    recent_desc = norm(ch[0].get("description", "")) if ch else ""
    return not contains_any(recent_desc, _CODE_SIGNALS)


def trajectory_scores(cands: list[dict], texts: list[str] | None = None) -> tuple[np.ndarray, list[dict]]:
    out = np.empty(len(cands), dtype=np.float32)
    detail: list[dict] = []
    for i, c in enumerate(cands):
        text = texts[i] if texts is not None else combined_text(c)
        toks = tokenset(text)
        years = derived_experience_years(c)
        prod_hits = count_hits(text, jd_spec.PRODUCTION_SIGNALS, toks)
        base = 0.4 + 0.6 * min(1.0, prod_hits / 3.0)

        consulting_only = _consulting_only(c)
        research_only = contains_any(text, jd_spec.RESEARCH_ONLY_SIGNALS, toks) and prod_hits == 0
        framework_only = (
            contains_any(text, jd_spec.FRAMEWORK_LLM_ONLY, toks)
            and years < 3.5
            and not contains_any(text, jd_spec.PRELLM_ML_SIGNALS, toks)
        )
        title_chaser = _title_chaser(c)
        no_recent_code = _senior_no_recent_code(c, years)

        factor = base
        factor *= 0.25 if consulting_only else 1.0
        factor *= 0.40 if research_only else 1.0
        factor *= 0.50 if framework_only else 1.0
        factor *= 0.55 if title_chaser else 1.0
        factor *= 0.70 if no_recent_code else 1.0
        score = float(np.clip(factor, 0.0, 1.0))

        out[i] = score
        detail.append({
            "production_hits": prod_hits,
            "consulting_only": consulting_only,
            "research_only": research_only,
            "framework_only": framework_only,
            "title_chaser": title_chaser,
            "no_recent_code": no_recent_code,
            "score": score,
        })
    return out, detail
