# SEC Filings RAG Application

A production-quality Retrieval-Augmented Generation (RAG) system for querying SEC EDGAR filings (10-K, 10-Q, proxy statements). Built as a portfolio project demonstrating end-to-end AI engineering: ingestion, vector search, reranking, and grounded generation with citations.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (ui/app.py)                     │
│    Chat interface · Citation cards · Ingest form · Sidebar stats    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTP
┌──────────────────────────────▼──────────────────────────────────────┐
│                     FastAPI (api/main.py)                           │
│   POST /ask · POST /ingest · GET /health · asyncpg pool lifespan   │
└───────┬──────────────────────────────────────┬──────────────────────┘
        │                                       │
┌───────▼──────────┐                ┌───────────▼──────────────────┐
│  Ingestion       │                │  Query Pipeline              │
│                  │                │                              │
│  loader.py       │                │  retriever.py                │
│  ├─ load_pdf()   │                │  └─ embed query (OpenAI)     │
│  ├─ load_txt()   │                │     └─ cosine search         │
│  └─ load_urls()  │                │        (pgvector IVFFlat)    │
│                  │                │                              │
│  chunker.py      │                │  reranker.py                 │
│  ├─ fixed        │                │  └─ Cohere Rerank API        │
│  ├─ recursive ◄──┼── default      │     ├─ score threshold 0.3   │
│  └─ sentence     │                │     └─ fallback on failure   │
│                  │                │                              │
│  embedder.py     │                │  generator.py                │
│  └─ OpenAI       │                │  └─ Claude (streaming)       │
│     batch embed  │                │     └─ citation extraction   │
│     + pgvector   │                │                              │
│     upsert       │                └──────────────────────────────┘
└──────────────────┘
        │                                       │
┌───────▼───────────────────────────────────────▼──────────────────┐
│              pgvector (PostgreSQL 16 via Docker)                  │
│   chunks table · VECTOR(1536) · IVFFlat cosine index             │
└───────────────────────────────────────────────────────────────────┘
```

---

## Setup

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- API keys: OpenAI, Anthropic, Cohere

### 1. Clone and configure
```bash
git clone <repo>
cd rag-project
cp .env.example .env
# Edit .env and fill in your API keys
```

### 2. Start the vector database
```bash
docker compose up -d
# Verify: docker compose ps — postgres should show "healthy"
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the API server
```bash
uvicorn api.main:app --reload
# API docs: http://localhost:8000/docs
```

### 5. Start the Streamlit UI
```bash
streamlit run ui/app.py
# UI: http://localhost:8501
```

---

## Usage

### Ingest an SEC filing
```bash
# Via curl
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": ["/path/to/apple_10k_2023.pdf"],
    "chunk_strategy": "recursive"
  }'

# Via the Streamlit sidebar — paste file paths and click Ingest
```

### Ask a question
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What was Apple revenue in FY2023 and how does it compare to FY2022?"}'
```

Response includes the answer, source citations with page numbers, and pipeline latency.

### Run tests
```bash
pytest tests/ -v
```

---

## Tech Stack & Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Vector DB** | pgvector (PostgreSQL 16) | SQL familiarity, no extra infra, IVFFlat index for cosine similarity |
| **Embeddings** | OpenAI text-embedding-3-small | Cost-efficient, 1536-dim, strong performance on financial text |
| **Generation** | Claude claude-sonnet-4-20250514 | Strong instruction-following; grounded answers with "I don't know" |
| **Reranking** | Cohere Rerank v3 | Cross-encoder dramatically improves retrieval precision |
| **Chunking** | Recursive (default), Fixed, Sentence | Recursive preserves paragraph boundaries best for dense filings |
| **Framework** | FastAPI + asyncpg | Async-native; production-ready; Pydantic v2 validation |
| **UI** | Streamlit | Fast prototyping; native chat + expander for citations |

### Why explicit retrieval steps, not LangChain chains?

LangChain's built-in chains (e.g., `RetrievalQA`) abstract away the reranking step and make it difficult to inject Cohere between retrieval and generation. Building explicit steps in `retriever.py` → `reranker.py` → `generator.py` keeps each stage testable, observable, and swappable independently.

### Why a relevance threshold of 0.3?

Cohere relevance scores range [0, 1]. Empirically, scores below 0.3 indicate the top retrieved chunks are not meaningfully related to the query. Returning "I don't have enough information" rather than generating an answer from weak context prevents hallucination — critical for financial document Q&A.

### Why IVFFlat over HNSW?

IVFFlat with `lists=100` is appropriate for datasets up to ~1M vectors and requires less memory than HNSW. For larger corpora, switching to `HNSW (m=16, ef_construction=64)` would improve recall at the cost of build time.

### Streaming: collected vs. SSE

The Anthropic SDK streams tokens via `client.messages.stream()`, but the `/ask` endpoint collects the full response before returning JSON. This keeps the API simple (single Pydantic response model) at MVP stage. True Server-Sent Events would reduce time-to-first-token but adds complexity to both the API and UI.

---

## Known Limitations

1. **PDF table extraction**: `pypdf` extracts text but does not reconstruct table structure. Financial tables in 10-K filings may lose column alignment, affecting numerical comparisons. Future: integrate `pdfplumber` for table-aware extraction.

2. **No document deduplication**: Re-ingesting the same file creates duplicate chunks. Future: hash content before insert, skip if already exists.

3. **Single-turn Q&A only**: The `/ask` endpoint takes a single query; there is no conversation memory. Future: pass prior turns to Claude for multi-turn dialogue.

4. **IVFFlat requires data to build index**: The `ivfflat` index is created at schema init with `lists=100` but performs poorly until ~10,000 vectors are inserted. For small datasets, a sequential scan is automatically used.

5. **SEC EDGAR rate limits**: `load_urls()` fetches pages sequentially. Bulk EDGAR crawling should use their official bulk data download API to avoid rate limiting.

6. **No authentication on API**: The FastAPI routes have no auth. For deployment, add OAuth2 or API key middleware.
