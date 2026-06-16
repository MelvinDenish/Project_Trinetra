"""Structured intent model for the single released JD (extraction, not grep).

The JD ("Senior AI Engineer — Founding Team, Redrob AI") is parsed once, by
hand, into:
  * POSITIVE ASPECT-QUERIES — short natural-language statements of what the role
    actually needs, embedded and matched against candidate narratives. These
    encode "what the JD means", catching Tier-5 candidates who built recsys/
    search without writing the buzzword "RAG".
  * A JD_SUMMARY — the compact target text the cross-encoder pairs with each
    shortlisted narrative.
  * NEGATIVE / POSITIVE LEXICONS — the JD's explicit "do NOT want" list plus the
    "things you absolutely need", consumed by the role/trajectory features.
  * BEHAVIORAL PREFERENCES — active, responsive, open-to-work, <=30-day notice.

Every item here is traceable to a sentence in job_description.docx; that
traceability is the Stage-5 ("defend your design") answer.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Positive aspect-queries (embedded + BM25). `weight` reflects how central the
# aspect is to the JD's "things you absolutely need" vs "nice to have".
# `terms` are curated BM25 keywords (cleaner than tokenizing the prose query).
# ---------------------------------------------------------------------------
ASPECTS: list[dict] = [
    {
        "name": "retrieval_embeddings",
        "weight": 1.0,
        "query": (
            "built and deployed production embeddings-based retrieval and "
            "semantic search systems for real users, handling embedding drift, "
            "index refresh and retrieval-quality regression"
        ),
        "terms": [
            "embeddings", "embedding", "retrieval", "semantic", "search",
            "sentence-transformers", "bge", "e5", "vector",
        ],
    },
    {
        "name": "ranking_recsys",
        "weight": 1.0,
        "query": (
            "shipped a ranking, recommendation or search relevance system to "
            "real users at meaningful scale and improved engagement metrics"
        ),
        "terms": [
            "ranking", "rank", "recommendation", "recommender", "recsys",
            "relevance", "matching", "personalization", "learning-to-rank",
        ],
    },
    {
        "name": "vector_db_hybrid",
        "weight": 0.9,
        "query": (
            "operated vector databases or hybrid search infrastructure in "
            "production such as Pinecone, Weaviate, Qdrant, FAISS, OpenSearch "
            "or Elasticsearch, combining dense and lexical retrieval"
        ),
        "terms": [
            "vector", "pinecone", "weaviate", "qdrant", "milvus", "faiss",
            "opensearch", "elasticsearch", "hybrid", "index", "ann",
        ],
    },
    {
        "name": "eval_frameworks",
        "weight": 0.9,
        "query": (
            "designed evaluation frameworks for ranking systems using NDCG, "
            "MRR, MAP, offline-to-online correlation and A/B test interpretation"
        ),
        "terms": [
            "ndcg", "mrr", "map", "evaluation", "offline", "online",
            "a/b", "ab", "metrics", "benchmark", "ground-truth",
        ],
    },
    {
        "name": "ml_production_scale",
        "weight": 0.85,
        "query": (
            "applied machine learning engineer at a product company who took "
            "models from prototype to production at scale, owning latency, "
            "serving and quality in real systems"
        ),
        "terms": [
            "production", "deployed", "scale", "latency", "serving",
            "pipeline", "model", "machine", "learning", "shipped",
        ],
    },
    {
        "name": "llm_finetuning",
        "weight": 0.55,  # JD: "we'd like you to have but won't reject you for"
        "query": (
            "fine-tuned large language models with LoRA, QLoRA or PEFT and "
            "integrated LLMs into production with judgment about when to "
            "fine-tune versus prompt"
        ),
        "terms": [
            "llm", "fine-tuning", "finetuning", "lora", "qlora", "peft",
            "transformer", "language", "rag",
        ],
    },
]

# Compact target text for the cross-encoder (JD "between the lines" ideal).
JD_SUMMARY = (
    "Senior AI Engineer for the founding team of a product company. The ideal "
    "candidate has 6-8 years of experience, of which several are in applied "
    "ML/AI at product companies (not pure services or research). They have "
    "shipped at least one end-to-end ranking, search or recommendation system "
    "to real users at scale, have production experience with embeddings-based "
    "retrieval and vector or hybrid search infrastructure, and have designed "
    "rigorous ranking evaluation (NDCG, MRR, MAP, offline and online A/B "
    "testing). Strong Python and a scrappy, ship-first product-engineering "
    "attitude. Based in or willing to relocate to Noida or Pune, India, and "
    "active and responsive on the hiring platform."
)

# ---------------------------------------------------------------------------
# Role / domain lexicons (lower-cased substring matching against titles/text)
# ---------------------------------------------------------------------------
# Titles that ARE the role (strong positive prior).
TITLE_STRONG = [
    "machine learning engineer", "ml engineer", "ai engineer",
    "applied scientist", "applied ml", "research engineer",
    "nlp engineer", "search engineer", "relevance engineer",
    "recommendation", "recsys", "information retrieval", "ranking",
    "data scientist", "ml scientist", "research scientist",
]
# Adjacent engineering titles — could be a strong fit if the narrative proves it.
TITLE_ADJACENT = [
    "software engineer", "backend engineer", "back-end engineer",
    "data engineer", "platform engineer", "full stack", "fullstack",
    "staff engineer", "senior engineer", "principal engineer", "sde",
]
# Titles that are clearly off-target (strong negative prior).
TITLE_OFFTARGET = [
    "hr", "human resources", "recruiter", "talent acquisition",
    "accountant", "accounting", "finance", "marketing", "sales",
    "business analyst", "content writer", "copywriter", "graphic designer",
    "civil engineer", "mechanical engineer", "electrical engineer",
    "product manager", "project manager", "operations", "consultant",
    "teacher", "professor", "customer success", "qa engineer", "support",
]

# Domain signals in the narrative (the JD's "absolutely need" cluster).
DOMAIN_POSITIVE = [
    "retrieval", "embedding", "embeddings", "ranking", "recommendation",
    "recommender", "recsys", "search", "relevance", "semantic", "nlp",
    "natural language", "information retrieval", "vector", "hybrid search",
    "learning to rank", "matching", "personalization", "llm", "rag",
]
# Computer-vision / speech / robotics — negative ONLY when NLP/IR is absent.
DOMAIN_CV_SPEECH_ROBOTICS = [
    "computer vision", "image classification", "object detection",
    "segmentation", "ocr", "speech recognition", "asr", "text-to-speech",
    "robotics", "autonomous", "lidar", "slam", "point cloud",
]
# Production / shipping verbs (used to separate builders from researchers).
PRODUCTION_SIGNALS = [
    "production", "deployed", "deploy", "shipped", "ship", "launched",
    "real users", "at scale", "serving", "latency", "throughput",
    "in production", "live", "rolled out", "owned",
]
# Pure-research markers (negative when production signals are absent).
RESEARCH_ONLY_SIGNALS = [
    "phd", "postdoc", "research lab", "academic", "publication", "published",
    "paper", "papers", "thesis", "university research", "research-only",
]
# Framework-enthusiast / "AI = LangChain" markers (negative w/o pre-LLM ML).
FRAMEWORK_LLM_ONLY = [
    "langchain", "llama index", "llamaindex", "prompt engineering",
    "chatgpt wrapper", "openai api", "autogpt", "gpt wrapper",
]
# Pre-LLM / classical-ML markers that rescue a framework-enthusiast profile.
PRELLM_ML_SIGNALS = [
    "scikit", "xgboost", "lightgbm", "random forest", "logistic regression",
    "gradient boosting", "feature engineering", "tf-idf", "word2vec",
    "collaborative filtering", "matrix factorization", "svm",
]

# Services / consulting industries (the JD rejects consulting-only careers).
SERVICES_INDUSTRY_MARKERS = ["it services", "consulting", "outsourcing"]

# ---------------------------------------------------------------------------
# Behavioral preferences (from "weigh behavioral signals" in the JD).
# ---------------------------------------------------------------------------
PREF_NOTICE_DAYS = 30        # "We'd love sub-30-day notice."
PREF_MAX_INACTIVE_DAYS = 60  # active enough that recruiters can reach them
