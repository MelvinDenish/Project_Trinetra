"""Streamlit sandbox demo (Section 10.5 of the submission spec).

Accepts a small candidate sample (<=100 — upload a .jsonl or use the bundled
sample), runs the FULL ranking system end-to-end on CPU within the compute
budget, and shows the ranked results with the grounded reasoning. It uses the
same scoring core as the full pipeline via `rank_in_memory` (computing the
embeddings + BM25 index on the fly for the small sample).

Run locally:   streamlit run app/streamlit_app.py
Deploy:        Streamlit Cloud / HuggingFace Spaces (free tier is fine).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from redrob_ranker import io_utils, pipeline  # noqa: E402

SAMPLE = ROOT / "data" / "sample_candidates.jsonl"

st.set_page_config(page_title="Redrob Evidence-Grounded Ranker", layout="wide")
st.title("Redrob — Evidence-Grounded Hybrid Ranker")
st.caption(
    "Ranks a small candidate sample for the Senior AI Engineer JD. In this dataset "
    "the skills array is a decoy — we rank the free-text career narrative instead."
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Show top K", 5, 100, 20)
    use_rerank = st.checkbox("Cross-encoder rerank", value=True)
    st.markdown(
        "Upload a `candidates.jsonl` (≤100 rows) or use the bundled 200-row sample "
        "(first N used). First run downloads the embedding model (~1 min)."
    )


@st.cache_data(show_spinner=False)
def _load_sample(n: int = 100):
    return io_utils.load_candidates(str(SAMPLE), limit=n)


uploaded = st.file_uploader("candidates .jsonl", type=["jsonl", "json"])
if uploaded is not None:
    cands = []
    for line in uploaded.getvalue().decode("utf-8").splitlines():
        line = line.strip()
        if line:
            cands.append(json.loads(line))
    cands = cands[:100]
    st.success(f"Loaded {len(cands)} uploaded candidates (capped at 100).")
else:
    cands = _load_sample(100)
    st.info(f"Using the bundled sample ({len(cands)} candidates).")

if st.button("Rank candidates", type="primary"):
    with st.spinner("Embedding + scoring on CPU…"):
        rows = pipeline.rank_in_memory(cands, top_k=top_k, use_rerank=use_rerank)
    df = pd.DataFrame(rows)[["rank", "candidate_id", "score", "reasoning"]]
    st.subheader(f"Top {len(rows)}")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download ranking CSV",
        df.to_csv(index=False).encode("utf-8"),
        "submission_sample.csv",
        "text/csv",
    )
    st.subheader("Why these candidates (grounded reasoning)")
    for r in rows[: min(8, len(rows))]:
        with st.expander(f"#{r['rank']}  {r['candidate_id']}  —  score {r['score']:.4f}"):
            st.write(r["reasoning"])
