"""Redrob Evidence-Grounded Hybrid Ranker.

A transparent, reproducible, CPU-only candidate ranking system for the
Redrob "Intelligent Candidate Discovery & Ranking" challenge.

The central design insight (see docs/PROJECT_MEMORY.md): in this dataset the
`skills` array and `education.field_of_study` are *decoys* (uniformly random,
decoupled from the real role), so ranking must be driven by the free-text
career narrative + structured role/trajectory/geo/behavioral evidence, with a
plausibility gate against planted honeypots.
"""

__version__ = "1.0.0"
