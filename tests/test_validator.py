"""End-to-end check that our selection produces a CSV the bundled validator
(the same script the portal runs) accepts."""
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redrob_ranker import scoring  # noqa: E402

_spec = importlib.util.spec_from_file_location("validate_submission", ROOT / "validate_submission.py")
vs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vs)


def _write(rows, path, n=None):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cid, sc) in enumerate(rows[:n] if n else rows, start=1):
            w.writerow([cid, rank, f"{sc:.6f}", "grounded reasoning, with a comma"])


def _chosen(seed=1, npool=3000):
    ids = [f"CAND_{i:07d}" for i in range(npool)]
    scores = np.random.default_rng(seed).random(npool)
    return scoring.select_top_k(ids, scores, k=100)


def test_generated_submission_is_valid(tmp_path):
    p = tmp_path / "team_test.csv"
    _write(_chosen(), p)
    errs = vs.validate_submission(str(p))
    assert errs == [], errs


def test_validator_rejects_truncated_submission(tmp_path):
    p = tmp_path / "team_test.csv"
    _write(_chosen(), p, n=99)
    errs = vs.validate_submission(str(p))
    assert errs  # 99 rows must fail


def test_tie_break_survives_validator(tmp_path):
    # Force many ties (all equal score) -> validator demands candidate_id ascending.
    ids = [f"CAND_{i:07d}" for i in range(200)]
    scores = np.zeros(200)
    chosen = scoring.select_top_k(ids, scores, k=100)
    p = tmp_path / "team_test.csv"
    _write(chosen, p)
    assert vs.validate_submission(str(p)) == []
