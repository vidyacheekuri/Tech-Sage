# TechSage — AI Ecosystem Intelligence Engine

> An AI Ecosystem Intelligence Engine that monitors official AI labs, frameworks, infrastructure providers, research papers, and trending GitHub repositories, using transformer-based NLP models for semantic deduplication, release detection, summarization, and hybrid ranking.

## What TechSage Does

TechSage is a production-grade ML system that continuously ingests, classifies, and ranks AI ecosystem signals from multiple source tiers:

| Tier | Sources | Priority |
|------|---------|----------|
| **Tier 1** | OpenAI, Anthropic, Google DeepMind, Meta AI, Mistral, Stability AI | Highest — official lab announcements |
| **Tier 2** | LangChain, LlamaIndex, Hugging Face, Weights & Biases, Pinecone | Framework/tool releases |
| **Tier 3** | TechCrunch AI, VentureBeat AI, The Verge AI, Ars Technica | Media coverage |
| **Tier 4** | GitHub Trending (AI/LLM/Agent/GenAI/ML topics) | Repository signals |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     TechSage Architecture                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │ RSS Fetcher  │  │ GitHub API   │  │ Optional: NewsAPI │   │
│  │ (Tiered)     │  │ (Trending)   │  │ / GNews           │   │
│  └──────┬───────┘  └──────┬───────┘  └────────┬──────────┘   │
│         │                 │                    │              │
│         └────────┬────────┴────────────────────┘              │
│                  ▼                                            │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              Ingestion Pipeline                        │    │
│  │  Clean → Embed → Deduplicate → Classify → Summarize   │    │
│  │                    (tier-aware)   (2-pass)             │    │
│  └──────────────────────────┬───────────────────────────┘    │
│                             ▼                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │           PostgreSQL + pgvector                        │    │
│  │  articles (384-dim embeddings, release metadata)       │    │
│  │  users / user_profiles / user_interactions             │    │
│  └──────────────────────────┬───────────────────────────┘    │
│                             ▼                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │           Ranking Engine                               │    │
│  │  Score = 0.40 × ReleasePriority                        │    │
│  │        + 0.25 × EmbeddingSimilarity                    │    │
│  │        + 0.20 × RecencyScore                           │    │
│  │        + 0.10 × SourceTierWeight                       │    │
│  │        + 0.05 × CategoryAffinity                       │    │
│  └──────────────────────────┬───────────────────────────┘    │
│                             ▼                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  FastAPI + Frontend Dashboard                          │    │
│  │  Tabs: Models | Frameworks | Agents | Research |       │    │
│  │        GitHub | AI News | Ask (RAG) | Saved | Profile  │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

## ML Models (All Local — No External APIs)

| Model | Purpose | Dimension |
|-------|---------|-----------|
| `sentence-transformers/all-MiniLM-L6-v2` | Sentence embeddings for deduplication + personalization + RAG retrieval | 384-dim |
| `facebook/bart-large-mnli` | Zero-shot classification (topic + release detection) | — |
| `facebook/bart-large-cnn` | Abstractive summarization (~120–150 words) | — |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | RAG chat generation (lazy-loaded on first query, fast on CPU) | — |

All models are loaded at startup via the `ModelRegistry` singleton, except the RAG LLM which loads on first chat request.

## RAG Chat (Retrieval Augmented Generation)

Ask questions about the AI ecosystem in natural language. Flow:
1. **Embed** the question with the same sentence embedding model
2. **Retrieve** top-K articles by cosine similarity in pgvector
3. **Generate** an answer with SmolLM2 grounded in the retrieved context
4. **Cite sources** — each answer links to the articles it drew from

Access via the **💬 Ask** tab in the dashboard or `POST /api/v1/chat`.

## Two-Pass Classification

1. **Topic Classification**: Assigns articles to 8 AI-focused categories (LLMs, Agents, Infrastructure, Frameworks, Research, Policy, Startups, Applications)
2. **Release Classification**: Detects release types (Model Release, Framework Release, Agent Platform, Infrastructure Update, Research Paper, General AI News) and sets `is_release=True` for actual releases

## Tier-Aware Deduplication

When the same release appears across multiple sources (e.g., OpenAI blog + TechCrunch), TechSage:
- Prefers the **Tier 1** (official) source
- Marks lower-tier duplicates with `duplicate_of` linking to the primary
- Uses cosine similarity threshold of 0.85 on 384-dim embeddings

## Ranking Formula

```
Score = 0.40 × ReleasePriority + 0.25 × EmbeddingSimilarity + 0.20 × RecencyScore + 0.10 × SourceTierWeight + 0.05 × CategoryAffinity
```

**ReleasePriority**: model=1.0, framework=0.9, agent_platform=0.85, infra=0.8, github_repo=0.75, research=0.7, general_news=0.5

**SourceTierWeight**: Tier1=1.0, Tier2=0.9, Tier3=0.7, Tier4=0.8

## Tech Stack

- **Python 3.10+** with async/await throughout
- **FastAPI** — API + static file serving
- **PostgreSQL + pgvector** — vector similarity search on 384-dim embeddings
- **SQLAlchemy 2.0** — ORM with JSONB + Vector column types
- **HuggingFace Transformers + Sentence-Transformers** — all NLP
- **APScheduler** — background ingestion (news every 5 min, GitHub every 6 hours)
- **scikit-learn** — cosine similarity computation

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create PostgreSQL database
createdb techsage
psql techsage -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 3. Configure environment
cp .env.example .env  # Edit DATABASE_URL and optional GITHUB_TOKEN

# 4. Run
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 for the dashboard.

### Automatic article fetching (without keeping the server running)

Ingestion normally runs inside the web server (every 5 min for news, every 6 hours for GitHub). To fetch articles **even when the server is off**, use one of these:

**Option A — Standalone scheduler** (long-running process):
```bash
python -m scheduler.run_standalone
```
Runs in the foreground. Use `nohup` or `screen` to background it:
```bash
nohup python -m scheduler.run_standalone >> /tmp/techsage-scheduler.log 2>&1 &
```

**Option B — Cron** (no long-running process):
```bash
# News every 5 minutes
*/5 * * * * cd /path/to/TechSage && python -m scheduler.run_once --news >> /tmp/techsage-news.log 2>&1

# GitHub every 6 hours
0 */6 * * * cd /path/to/TechSage && python -m scheduler.run_once --github >> /tmp/techsage-github.log 2>&1
```
Replace `/path/to/TechSage` with your project directory.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | System health check |
| GET | `/api/v1/articles` | List articles (filter by category, release_type, content_type) |
| GET | `/api/v1/articles/{id}` | Get single article |
| GET | `/api/v1/articles/categories/summary` | Category distribution |
| GET | `/api/v1/articles/releases/summary` | Release type distribution |
| POST | `/api/v1/users` | Create user with optional interest seeding |
| GET | `/api/v1/users/{id}/feed` | Personalized ranked feed |
| POST | `/api/v1/users/{id}/interactions` | Record interaction (view/click/bookmark/like/dislike) |
| POST | `/api/v1/ingest` | Manual ingestion trigger |

## Project Structure

```
TechSage/
├── api/
│   ├── main.py              # FastAPI app, lifespan, model loading
│   ├── routes.py             # All API endpoints
│   └── schemas.py            # Pydantic validation models
├── config/
│   ├── settings.py           # Environment-based configuration
│   └── model_registry.py     # Singleton ML model loader
├── database/
│   ├── models.py             # SQLAlchemy ORM (Article, User, Profile)
│   └── session.py            # DB engine, session factory, init
├── news_ingestion/
│   ├── fetcher.py            # Tiered AI RSS source fetcher
│   ├── github_trending.py    # GitHub trending repo ingestion
│   ├── cleaner.py            # HTML/content cleaning pipeline
│   ├── embedding.py          # Sentence embedding generation
│   ├── deduplication.py      # Tier-aware semantic deduplication
│   ├── classifier.py         # Topic classification (8 categories)
│   ├── release_classifier.py # Release type detection (secondary pass)
│   └── summarizer.py         # Abstractive summarization
├── recommendation/
│   ├── ranking.py            # 5-component ranking engine
│   ├── interaction_tracker.py # Interaction logging + profile updates
│   └── user_profile.py       # User preference management
├── scheduler/
│   ├── ingestion_job.py      # Pipeline orchestration + scheduling
│   ├── run_standalone.py     # Standalone scheduler (no web server)
│   └── run_once.py           # One-shot ingestion (for cron)
├── static/
│   └── index.html            # Frontend dashboard
├── .env                      # Environment configuration
├── requirements.txt          # Python dependencies
└── README.md
```

## Architecture Diagram

*[Placeholder for detailed architecture diagram]*

---

Built as a production-grade ML system demonstrating transformer-based NLP, vector similarity search, multi-signal ranking, and real-time AI ecosystem monitoring.
