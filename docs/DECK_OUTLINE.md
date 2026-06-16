# Deck Outline — Redrob Evidence-Grounded Hybrid Ranker

Slide-ready outline for the submission deck / Stage-5 walkthrough.

### 1. Title
Evidence-Grounded Hybrid Ranker — ranking 100 of 100,000 for the Senior AI
Engineer JD. CPU-only, no-network, fully reproducible, interview-defensible.

### 2. The problem, honestly stated
- Rank top-100; scored on hidden graded ground truth:
  `0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10` (top-10 is half the score).
- Adversarial by design: naïve "embeddings over whole profile" or "count AI
  skills" are engineered to fail. Honeypots (>10% in top-100 = DQ).

### 3. The insight that wins it (one slide, the differentiator)
- **Skills & education field are decoys** — 133 skill names, top ~35 each ~12K×,
  uniform-random, decoupled from role. The provided `sample_submission.csv` ranks
  by AI-skill-count → HR Managers above ML Engineers.
- Real signal = the free-text **career narrative**. Show a "Graphic Designer" with
  Kafka/Airflow/Fine-tuning-LLMs as skills vs an ML Engineer whose *summary* tells
  the story.

### 4. Data profile (from `scripts/profile_data.py`)
- 75% India / 25% non-India (no visa sponsorship). 68% off-target titles; genuine
  AI/ML a thin long tail. `years_of_experience` unreliable → derive from history.
- Behavioral: half inactive 4+ months, response-rate median ~0.44.

### 5. Architecture decision — 5 → 3 → 1
- A1 skills/BM25 (the trap), A2 whole-profile dense, A3 hybrid+rules+gate,
  A4 LTR (no labels), A5 per-candidate LLM (busts compute budget).
- Decision matrix → **A3**, with A4's calibrated-blend discipline and A5's
  reasoning folded in (cross-encoder + grounded NLG). Data-backed eliminations.

### 6. Final architecture diagram
- `base_fit = 0.45·semantic + 0.35·role + 0.20·trajectory`,
  `final = base_fit · plausibility · geo · behavioral`, shortlist → cross-encoder
  → top-100 + reasoning. (Use the diagram in PROJECT_MEMORY.md.)

### 7. How traps are defeated
- C1 decoys → narrative-only. C2 plain language → dense aspect-queries.
  C3 honeypots → calibrated gate. C4 negatives → structured rules.
  C5 availability → bounded modifier. C6 compute → precompute + small CPU models.

### 8. Eval & ablations (from `scripts/evaluate.py`) — the money slide
- Ablation table: **skills-only** drags stuffers / honeypots into top-100;
  **narrative-only** ~0; **full pipeline** ~0. (Fill with the run's numbers.)
- Honeypot-rate in top-100 = 0 (DQ guard). %India in top-10. Anchor positions
  (strong positives high, geo/availability negatives lower). Cross-encoder lift.

### 9. Explainability (3 real examples)
- Paste 3 real grounded `reasoning` rows: a strong fit, a "modest title but real
  work" rescue, and a hedged near-cutoff pick with an honest concern.

### 10. Reproducibility & defense
- Deterministic, CPU-only, no-network ranking; byte-identical CSV; Docker for
  Stage-3; every weight in one file; reasoning faithfulness enforced by a test.
- Why no LLM in the scored path: it wouldn't reproduce byte-for-byte and reads as
  "reasoning generated independently of ranking" — a Stage-4 red flag.
