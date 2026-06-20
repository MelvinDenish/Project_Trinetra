"""Honeypot gate tests — the Stage-3 DQ guard must catch impossibilities and,
critically, must NOT false-positive on genuine candidates."""
from redrob_ranker import config
from redrob_ranker.features.plausibility import assess


def _skill(name, prof, dur):
    return {"name": name, "proficiency": prof, "endorsements": 0, "duration_months": dur}


def _candidate(skills=None, career=None):
    return {
        "candidate_id": "CAND_0000001",
        "profile": {"summary": "5 years of experience.", "years_of_experience": 5,
                    "current_title": "Engineer"},
        "skills": skills or [],
        "career_history": career or [
            {"company": "Acme", "title": "Engineer", "start_date": "2019-01-01",
             "end_date": "2024-01-01", "duration_months": 60, "is_current": False,
             "industry": "Software", "description": "built systems"},
        ],
    }


def test_expert_zero_duration_is_honeypot():
    skills = [_skill(n, "expert", 0) for n in ("Go", "Docker", "Hadoop")]
    a = assess(_candidate(skills=skills))
    assert a["is_honeypot"] is True
    assert a["factor"] == config.PLAUSIBILITY_HONEYPOT
    assert a["reasons"]


def test_duration_exceeds_date_span_is_honeypot():
    # A role claiming 171 months but whose dates span ~60 months -> impossible.
    career = [{
        "company": "Studio", "title": "Graphic Designer", "start_date": "2019-01-01",
        "end_date": "2024-01-01", "duration_months": 171, "is_current": False,
        "industry": "Design", "description": "marketing background",
    }]
    a = assess(_candidate(career=career))
    assert a["is_honeypot"] is True


def test_genuine_candidate_not_flagged():
    # Long skill duration but a consistent career is the 9,191-case false-positive
    # trap that the gate deliberately ignores.
    skills = [_skill("Python", "expert", 120), _skill("SQL", "advanced", 90)]
    a = assess(_candidate(skills=skills))
    assert a["is_honeypot"] is False
    assert a["factor"] == config.PLAUSIBILITY_OK


def test_two_expert_zero_is_soft_not_honeypot():
    skills = [_skill("Go", "expert", 0), _skill("Rust", "advanced", 0)]
    a = assess(_candidate(skills=skills))
    assert a["is_honeypot"] is False
    assert a["factor"] == config.PLAUSIBILITY_SOFT


def test_role_longer_than_entire_timeline_is_honeypot():
    # A role with a FUTURE end_date whose duration fills that future span is
    # consistent with its own dates (so the per-role check misses it) but claims
    # more months than the candidate has been working (earliest start -> today).
    career = [{
        "company": "FutureCorp", "title": "Engineer", "start_date": "2020-01-01",
        "end_date": "2031-01-01", "duration_months": 132, "is_current": False,
        "industry": "Software", "description": "built systems",
    }]
    a = assess(_candidate(career=career))
    assert a["is_honeypot"] is True
    assert any("entire career timeline" in r for r in a["reasons"])
