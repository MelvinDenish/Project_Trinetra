# Stage-3 reproduction container. Matches the judges' sandbox: Python 3.11,
# Linux, CPU-only. The embedding + cross-encoder models are baked in at BUILD
# time (network is available during `docker build`), so the ranking step runs
# fully offline at `docker run` time.
FROM python:3.11-slim

WORKDIR /app

# CPU-only torch first (keeps the image small and avoids CUDA wheels), then the
# rest of the pinned deps.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download model weights into the image cache (the only networked step).
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY . .

# From here on, no network: the ranking step loads models from the local cache.
ENV OMP_NUM_THREADS=8 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

# Default: precompute (offline, may exceed 5 min) then the <=5-min ranking step.
# Mount your candidates.jsonl and collect submission.csv, e.g.:
#   docker run --rm -v "$PWD:/data" redrob-ranker \
#       bash run.sh /data/candidates.jsonl /data/submission.csv
CMD ["bash", "run.sh"]
