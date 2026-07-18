# RAG Project — Claude Code Rules

## Project context
Portfolio project for AI engineering job search. Code quality and architecture 
decisions matter as much as functionality. This will be reviewed by hiring managers.

## Non-negotiables
- No API keys or secrets in code — environment variables only via python-dotenv
- No print statements — use Python's logging module throughout
- All external API calls (Anthropic, OpenAI, Cohere, vector DB) must have try/except
- All functions need docstrings with parameters and return types
- Prompts live in generation/prompts.py as constants — never hardcoded inline

## Stack (do not substitute without asking)
- Python 3.11+ · FastAPI · Pydantic v2 for all data models
- LangChain is used **only** for `langchain-text-splitters` (chunking); retrieval and
  generation are hand-built — full `langchain`/`-community` are pruned, no LCEL, no chains
- ParadeDB (Postgres 16 + pgvector + pg_search) — HNSW dense index + BM25 lexical
- OpenAI text-embedding-3-small for embeddings
- Claude claude-sonnet-4-6 for generation (claude-sonnet-4-20250514 retired from the API);
  claude-haiku-4-5 for query rewrite; official metric runs override to claude-sonnet-5
  generation + claude-opus-4-8 judge via env only
- Cohere Rerank API for reranking

## Architecture rules
- Build retrieval and generation steps explicitly — do not use LangChain built-in chains
- Chunking strategies live in ingest/chunker.py — three strategies, recursive is default
- Reranking is never optional — if Cohere fails, fall back and log, never skip silently
- Score threshold for "no relevant content": 0.3 — return graceful message, never hallucinate

## After each phase
Tell me: what you built, if you made an unspecified decision and why, what to test manually.

---

## Commands

### Setup
```bash
# 1. Copy and fill in secrets
cp .env.example .env

# 2. Bring up the full stack (ParadeDB + api + ui, all healthchecked)
docker compose up --build

# — or, for local dev without containers —
docker compose up -d postgres            # just the DB
pip install -r requirements.txt -r requirements-dev.txt
```

### Run the API
```bash
uvicorn api.main:app --reload
# API docs: http://localhost:8000/docs
```

### Run the UI
```bash
streamlit run ui/app.py
# UI: http://localhost:8501
```

### Run tests
```bash
pytest tests/ -v
```

### Ingest a filing (curl)
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_paths": ["/path/to/filing.pdf"], "chunk_strategy": "recursive"}'
```

### Ask a question (curl)
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What was the total revenue in FY2023?"}'

# Streaming (Server-Sent Events: meta → delta* → citations → done)
curl -N -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What was the total revenue in FY2023?"}'
```

---

## Architectural decisions made during implementation

- **`asyncpg` + `psycopg2` dual drivers**: `asyncpg` for FastAPI async routes (performance); `psycopg2` only in the docker init script. No ORM — direct SQL keeps the schema transparent.
- **HNSW index (replaced IVFFlat)**: The original IVFFlat (lists=100) with pgvector's default probes=1 scanned <1% of vectors per query at this corpus size, zeroing dense recall on 20/27 eval questions (hit_rate@10 0.259 indexed vs 0.704 exact). HNSW (m=16, ef_construction=64) has no cluster blind spot; `hnsw.ef_search=100` is set at the database level so results are never capped below the retriever's branch limit.
- **SSE streaming shipped**: `POST /ask/stream` streams token deltas as Server-Sent Events (frames: meta → delta* → citations → done). `generation.generator.generate_stream()` is the single async-generator source of generation logic; the non-streaming `generate()` (used by `/ask` and evals) collects it into a string, so both paths share one code path.
- **pgvector embedding format**: Embeddings are passed as Python lists to asyncpg with `$1::vector` cast. The `_init_connection` hook registers a text codec so asyncpg can serialize/deserialize vectors.
- **Sentence chunking via RecursiveCharacterTextSplitter**: The plan called for `SentenceTransformersTokenTextSplitter` but that requires a heavy sentence-transformers download. Using RecursiveCharacterTextSplitter with sentence-friendly separators achieves the same goal without extra dependencies.
- **Citation deduplication by source+page**: Citations are deduplicated per source+page pair. Multiple chunks from the same page produce one citation entry to avoid redundancy in the UI.
- **`paradedb/paradedb:0.24.1-pg16` Docker image**: ParadeDB ships pg_search (true BM25) *and* pgvector pre-installed, so both lexical and dense search run in one database with no custom DB image. Pinned to a PG16 tag (`:latest` resolves to a PG18 layout incompatible with this compose file's volume). The app itself has its own multi-stage `Dockerfile`; `docker compose up` brings up postgres + api + ui, all healthchecked.