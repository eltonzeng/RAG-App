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
- Python 3.11+ · FastAPI · LangChain v0.2+ LCEL syntax only
- OpenAI text-embedding-3-small for embeddings
- Claude claude-sonnet-4-20250514 for generation
- Cohere Rerank API for reranking
- Pydantic v2 for all data models

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
# 1. Start the vector database
docker compose up -d

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and fill in secrets
cp .env.example .env
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
```

---

## Architectural decisions made during implementation

- **`asyncpg` + `psycopg2` dual drivers**: `asyncpg` for FastAPI async routes (performance); `psycopg2` only in the docker init script. No ORM — direct SQL keeps the schema transparent.
- **IVFFlat index with lists=100**: Appropriate for <1M vectors. HNSW would be faster but requires pgvector 0.5+; IVFFlat is universally supported.
- **Streaming collected, not SSE**: `client.messages.stream()` is used but collected into a complete string before returning. True SSE streaming is a planned enhancement — avoids Pydantic model complexity at MVP stage.
- **pgvector embedding format**: Embeddings are passed as Python lists to asyncpg with `$1::vector` cast. The `_init_connection` hook registers a text codec so asyncpg can serialize/deserialize vectors.
- **Sentence chunking via RecursiveCharacterTextSplitter**: The plan called for `SentenceTransformersTokenTextSplitter` but that requires a heavy sentence-transformers download. Using RecursiveCharacterTextSplitter with sentence-friendly separators achieves the same goal without extra dependencies.
- **Citation deduplication by source+page**: Citations are deduplicated per source+page pair. Multiple chunks from the same page produce one citation entry to avoid redundancy in the UI.
- **`pgvector/pgvector:pg16` Docker image**: Uses the official pgvector image which ships with the extension pre-installed, avoiding the need for a custom Dockerfile and `apt-get install postgresql-16-pgvector`.