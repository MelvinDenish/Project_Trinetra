"""Reasoning tests mapped to the Stage-4 checks: zero hallucination
(faithfulness), cross-row variation, and rank-consistent tone."""
from pathlib import Path

from redrob_ranker import io_utils, reasoning
from redrob_ranker.features import behavioral, geo, plausibility, role, trajectory

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample_candidates.jsonl"


def _details(cands):
    _, rd = role.role_scores(cands)
    _, td = trajectory.trajectory_scores(cands)
    _, bd = behavioral.behavioral_factor(cands)
    _, gd = geo.geo_factor(cands)
    _, pd = plausibility.plausibility_factor(cands)
    out = []
    for i in range(len(cands)):
        out.append({
            "role": rd[i], "trajectory": td[i], "behavioral": bd[i],
            "geo": gd[i], "plausibility": pd[i],
            "derived_years": rd[i]["derived_years"], "s_semantic": 0.5,
        })
    return out


def test_no_hallucinated_facts():
    cands = io_utils.load_candidates(str(SAMPLE), limit=120)
    details = _details(cands)
    for rank, (c, d) in enumerate(zip(cands, details), start=1):
        violations = reasoning.faithfulness_violations(c, d, rank)
        assert violations == [], (c["candidate_id"], violations)


def test_reasoning_varies_across_rows():
    cands = io_utils.load_candidates(str(SAMPLE), limit=100)
    details = _details(cands)
    texts = [reasoning.make_reasoning(c, d, r) for r, (c, d) in enumerate(zip(cands, details), 1)]
    # Substantively different, not a single templated string.
    assert len(set(texts)) >= int(0.8 * len(texts))


def test_rank_consistent_tone():
    cands = io_utils.load_candidates(str(SAMPLE), limit=100)
    details = _details(cands)
    top = reasoning.make_reasoning(cands[0], details[0], 1)
    bottom = reasoning.make_reasoning(cands[1], details[1], 100)
    assert any(lead in top for lead in reasoning._LEADS["confident"])
    assert any(lead in bottom for lead in reasoning._LEADS["hedged"])
