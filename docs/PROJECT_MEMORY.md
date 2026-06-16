# PROJECT_MEMORY — Redrob Evidence-Grounded Hybrid Ranker

The running reasoning trail the brief asks for: what the data is, every
architecture considered and why it was eliminated, the final design, the
calibrated weights, and what we learned at build time.

---

## Phase 0 — what the data actually is

100,000 candidate JSONL records (~465 MB). Fields: `profile`, `career_history[]`
(free-text `description` per role), `education[]` (with `tier`), `skills[]`
(name/proficiency/endorsements/duration_months), optional certifications/
languages, and 23 `redrob_signals` behavioral fields.

**The decisive finding — skills & education field are DECOYS.**
- 133 unique skill names, the top ~35 each occurring ~12,000× → skills are
  assigned near-uniformly at random, decoupled from role. `education.field_of_study`
  is the same. Any ranker keyed on the skills array or degree field walks into the
  planted trap. The provided `sample_submission.csv` is a deliberate decoy that
  ranks by "AI core skill count" and puts HR Managers/Accountants above ML Engineers.
- **The real fit signal is in free text** (`profile.summary` + `career_history[].description`)
  + `current_title` + behavioral signals. We therefore build a *narrative* document
  (headline + summary + recent role descriptions, **no skills array**) and rank on that.

Other distributions (reproduce with `scripts/profile_data.py`):
- Geography: ~75% India; non-India ~24.9K. JD: Noida/Pune, **no visa sponsorship**.
- Titles: ~68% obviously-irrelevant roles (~5.7K each); genuine AI/ML/IR titles are
  a thin long tail (<~750 each). Great matches are rare — as the JD states.
- `years_of_experience` is **unreliable** (an ideal ML candidate has yoe=2.7 while
  career history sums to 75 months). We derive experience from `career_history`.
- Behavioral: response-rate median ~0.44; ~half inactive 4+ months; ~64% no GitHub.

**Honeypots — calibrated on the full pool (the Stage-3 DQ guard).**
Dates are internally consistent (0 date contradictions), so honeypots are
*semantic/numeric impossibilities*. We scanned all 100K and found two clean,
disjoint signals:
- `>=3` skills at expert/advanced proficiency with **0 months** used — count
  distribution is `{0: 99979, 3: 8, 4: 5, 5: 8}` (never 1–2) → **21 records**.
- a role whose `duration_months` exceeds its own start→end span by >3 months →
  **19 records** (disjoint from the first set).
We deliberately **reject** the tempting "skill duration > career" rule — it matches
**9,191 genuine candidates** (skills aren't tied to employment here), so it is a
false-positive factory, not a honeypot signal. The other ~40 of the ~80 honeypots
are caught by naturally-low narrative fit (their narratives are generic).

### The 5 hardest challenges
| # | Challenge | Consequence |
|---|---|---|
| C1 | Skills/education are decoys | Keyword/whole-profile rankers are actively misled |
| C2 | Fit signal is plain-language free text | Needs semantic understanding, not keywords |
| C3 | Honeypots must be excluded | >10% in top-100 = disqualification |
| C4 | Negatives matter (consulting/non-India/CV-only/research/LangChain-only/no-code/title-chasing) | Needs structured rules |
| C5 | Behavioral availability is real | Must modulate by availability without dominating |
| C6 | Hard compute budget + defend-it interview | No per-candidate LLM; transparent, reproducible system |

---

## Architecture decision — 5 → 3 → 1

| ID | Architecture | Verdict |
|----|---|---|
| A1 | Lexical/BM25 on skills+title | **Eliminated** — *is* the planted trap (C1/C3). Empirically ranks 85 stuffers into the top-100. |
| A2 | Whole-profile dense bi-encoder (incl. skills) | **Eliminated** — adding decoy skills drags 24 stuffers back into the top-100; no negatives/availability/gate. |
| A3 | **Evidence-Grounded Hybrid Ranker** | **CHOSEN** — narrative dense+BM25 + structured features + honeypot gate + bounded modifiers + cross-encoder rerank + grounded reasoning. Solves C1–C6. |
| A4 | Learning-to-Rank (LightGBM) on weak labels | **Eliminated** — no ground truth ⇒ must fabricate labels; either imitates a heuristic (no gain) or overfits noise; weak Stage-5 answer. Its discipline (engineered features, **calibrated blend**) folded into A3. |
| A5 | Local small-LLM reasoning re-ranker | **Eliminated as core** — 100K local-LLM passes can't meet the 5-min CPU budget; non-deterministic. Its strength (reasoning) folded in as (1) a pretrained cross-encoder on the shortlist and (2) an optional demo-only `--polish`. |

**Verification on real data:** a TF-IDF separation probe confirmed skills-only
ranks 85 stuffers into the top-100, whole-profile 24, narrative-only 0 — so A1/A2
are data-disproven, not just argued. Honeypots never surfaced in any
representation, so the gate is a safety net (DQ de-risked).

---

## Final architecture & data flow

```
OFFLINE (declared; may exceed 5 min; network allowed only to download the model):
  candidates.jsonl -> narrative (summary + recent role descriptions, NO skills)
                   -> bge-small-en-v1.5 dense embeddings  -> artifacts/narrative_emb.npy
                   -> BM25 index over narratives           -> artifacts/bm25_*.npz

RANKING STEP (CPU, no network, <=5 min, deterministic):
  JD -> jd_spec (positive aspect-queries [embedded+BM25] + negative rules + prefs)
  for all 100K (vectorized):
     S_semantic   = hybrid(dense cosine vs aspect-queries, BM25 vs aspect terms)  [narrative only]
     S_role       = title(soft prior)+domain+experience-band; CV/speech-only penalty
     S_trajectory = production reward × consulting/title-chasing/research/framework/no-code penalties
     P_plausible  = honeypot gate            in {0.03, 0.85, 1.0}
     geo_factor   = location                 in {0.25, 0.45, 1.0, 1.05}
     behavioral   = availability modifier    in [0.60, 1.15]
  base_fit = 0.45*S_semantic + 0.35*S_role + 0.20*S_trajectory
  final    = base_fit * P_plausible * geo_factor * behavioral
  shortlist top-1000 by final -> cross-encoder(JD_summary, narrative) -> blend(0.6 final-rank + 0.4 cross-rank)
  top-100 -> evidence-ledger grounded reasoning -> CSV (score non-increasing; tie-break candidate_id asc)
```

### Calibrated weights (single source of truth: `src/redrob_ranker/config.py`)
- `base_fit = 0.45·S_semantic + 0.35·S_role + 0.20·S_trajectory`
- `S_semantic = 0.70·dense + 0.30·BM25` (both robust-min-max normalized)
- rerank blend `= 0.60·rank(final) + 0.40·rank(cross_encoder)`
- `geo`: India 1.00 (hub bonus 1.05), non-India+relocate 0.45, non-India 0.25
- `behavioral` ∈ [0.60, 1.15], asymmetric (penalize unavailability more than it rewards availability)
- `plausibility`: honeypot 0.03, soft 0.85, ok 1.00
- Starting point is the plan's design; intended to be tuned against the hand-verified
  anchor set via `scripts/evaluate.py` (anchor NDCG / separation), then frozen here.

---

## How each challenge is handled
| Challenge | Handling |
|---|---|
| C1 decoy skills | Narrative-only embeddings/BM25; skills used only inside the honeypot gate, never as primary signal |
| C2 plain-language fit | Dense aspect-query matching over summary+role descriptions (`features/semantic.py`) |
| C3 honeypots | `features/plausibility.py` gate (calibrated) + naturally-low fit; `evaluate.py` asserts top-100 honeypot rate |
| C4 negatives | `features/role.py` + `features/trajectory.py` + `features/geo.py` rules from `jd_spec.py` |
| C5 availability | `features/behavioral.py` bounded multiplier |
| C6 compute/defend | Precompute offline; ranking = numpy + light features + small CPU cross-encoder; deterministic & documented |

---

## Build-time learnings (kept honest)
- **CPU encoding is the long pole** (~13 docs/s for bge-small on an 8-core CPU →
  ~130 min for 100K). Mitigated by capping the narrative (front-loaded: headline +
  summary + 4 most-recent roles, each description ≤60 words) and `max_seq_length=256`.
  Precompute is offline/one-time; the ranking step reuses the cached vectors.
- **Word-aware matching matters.** Naive substring matching made `"rag"` match
  inside `"storage"`/`"average"`, inflating domain scores and producing a
  hallucinated "RAG experience". `textmatch.py` now matches single alphanumeric
  terms on token boundaries (phrases/symbol-terms still substring).
- **BM25 implementation:** `rank_bm25` stores one Python dict per document — at
  100K that is memory-heavy and slow to (de)serialize, so we use a vectorized BM25
  over a scipy sparse count matrix (compact `.npz`, fast, deterministic).
- **No pickle:** the BM25 index is stored as numpy `.npz` (loaded with
  `allow_pickle=False`) — no untrusted-deserialization path.
- **Offline by construction:** `rank.py` sets `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`
  before importing torch so Stage-3 reproduction never reaches the network.

## Reasoning generation — Evidence-Ledger Grounded NLG (decided)
Deterministic so the ranking step reproduces the exact CSV byte-for-byte in the
no-network sandbox (a local-LLM-polished CSV would not — a Stage-4 red flag).
Per top-100 candidate: build an evidence ledger of pointers to real fields only →
select 1–2 strongest positives + the worst concern → realize with a lead/connector
chosen by a stable hash for cross-row variation, tone scaled by rank tier. A CI
test (`faithfulness_violations`) asserts every emitted token exists in the record.
The local LLM is kept strictly as a demo-only `--polish` flag, never in the CSV.

---

## Measured results (full 100,000-candidate run)

Embedding model: `BAAI/bge-small-en-v1.5`; cross-encoder: `ms-marco-MiniLM-L-6-v2`.
Frozen weights as in `config.py`. Reproduce with `scripts/rank.py` + `scripts/evaluate.py`.

- **Runtime (ranking step):** ~190–210 s wall-clock on an 8-core CPU for all 100K
  → comfortably inside the 5-minute budget. (Offline precompute was a one-time
  ~14,000 s / ~4 h on this box, dominated by slow CPU encoding.)
- **Validator:** `python validate_submission.py submission.csv` → "Submission is valid."
- **Determinism:** two independent runs produced **byte-identical** CSVs (same SHA-256).
- **Honeypot rate in top-100: 0** (DQ threshold is >10). **Stuffers in top-100: 0.**
- **Top-10:** 10/10 India, 10/10 non-off-target titles.
- **Ablation (top-100 composition):**
  | ranking | honeypots | keyword-stuffers |
  |---|---|---|
  | skills-only (the decoy) | 0 | **37** |
  | narrative-only | 0 | 0 |
  | full pipeline | 0 | **0** |
  Skills-only drags 37 keyword-stuffers into the top-100; the narrative-driven
  full pipeline admits none — the core thesis, confirmed on the real data.
- **Cross-encoder lift (anchor final-rank → reranked-rank):** CAND_0039754
  #106→#38, CAND_0081846 #38→#9, CAND_0075249 #58→#15 — the rerank measurably
  sharpens the top, exactly where NDCG@10 (50% of the score) is decided.
- **Anchors vs negatives:** strong positives land #5/#9/#15/#38/#49 (5 of 6 inside
  top-50; the 6th, a more generic recsys persona, sits just outside the top-100 in
  a densely-packed elite cluster). Every geo/availability negative is buried far
  below — non-India anchors at #29.8K / #56.2K / #65.9K / #59.5K, the unavailable
  anchor at #1014. The Go/No-Go gate (anchors ≫ negatives, honeypot rate 0) passes.
- **Saturation note (honest):** robust-min-max maps the top ~1% of dense scores to
  1.0, so among the elite cluster the heuristic `final` is largely decided by the
  bounded behavioral/geo modifiers; the cross-encoder is what provides genuine
  top-100 discrimination. We did NOT hand-tune past this point — without ground
  truth, further weight-fiddling risks overfitting intuition (a stated plan risk).
