# SEC Filings RAG

[![CI](https://github.com/eltonzeng/RAG-App/actions/workflows/ci.yml/badge.svg)](https://github.com/eltonzeng/RAG-App/actions/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
![Coverage](https://img.shields.io/badge/coverage-~71%25-brightgreen.svg)

A retrieval-augmented generation system for querying SEC EDGAR filings (10-K, 10-Q).
It combines **hybrid retrieval** (dense pgvector + BM25 lexical search, fused with
Reciprocal Rank Fusion), **LLM query rewriting** with metadata filters, **Cohere
reranking**, and **grounded Claude generation** with page-level citations — and it ships
with an **evaluation harness** that quantifies what each pipeline stage actually earns.

Built as a portfolio project: code quality, testing, and honest measurement are treated
as first-class deliverables alongside functionality.

---

## Architecture

```
                      Streamlit UI (ui/app.py)
              chat · live token streaming (SSE) · citations
                               │  HTTP
                               ▼
                     FastAPI (api/main.py, routes.py)
        request-id middleware · optional X-API-Key · JSON logs · CORS
        GET /health   POST /ingest   POST /ask   POST /ask/stream (SSE)
                               │
          ┌────────────────────┴─────────────────────┐
          ▼ ingestion                                 ▼ query pipeline
  loader.py  (PDF / TXT / URL)             query_rewriter.py  (Claude Haiku)
   └─ pdfplumber table → Markdown          └─ 3–4 query variants + metadata filters
  metadata.py (ticker/year/form)                           │
  chunker.py (fixed/recursive/sentence)                    │
  embedder.py                             retriever.py  — hybrid, per variant:
   └─ OpenAI batch embed                    ├─ semantic: pgvector cosine (HNSW)
   └─ content-hash dedup                    ├─ lexical:  ParadeDB pg_search BM25
   └─ pgvector upsert                       └─ Reciprocal Rank Fusion (RRF)
          │                                    │  (zero-result → unfiltered retry)
          │                                 reranker.py — Cohere Rerank v3.5
          │                                    └─ relevance gate 0.3
          │                                    └─ degraded cosine fallback if down
          │                                 generator.py — Claude, streaming
          │                                    └─ citations narrowed to filter
          ▼                                       │
     ParadeDB (Postgres 16 + pgvector + pg_search)
        chunks · VECTOR(1536) · sources JSONB (provenance + filters)
```

Every arrow is explicit code (no LangChain chains), so each stage is independently
testable, observable, and swappable — and measurable by the eval harness.

---

## Quickstart

```bash
git clone https://github.com/eltonzeng/RAG-App.git && cd RAG-App
cp .env.example .env          # fill in OPENAI / ANTHROPIC / COHERE keys

docker compose up --build     # postgres + api + ui, all with healthchecks
```

- API + docs: http://localhost:8000/docs
- UI: http://localhost:8501

Ingest a filing and ask a question:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_paths": ["/path/to/COHR_10-K_2025.pdf"], "chunk_strategy": "recursive"}'

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What was Coherent total revenue in fiscal 2025?"}'

# Streaming (Server-Sent Events: meta → delta* → citations → done)
curl -N -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" -d '{"query": "..."}'
```

### Local (without Docker)

```bash
pip install -r requirements.txt -r requirements-dev.txt
docker compose up -d postgres
uvicorn api.main:app --reload      # API
streamlit run ui/app.py            # UI
pytest                             # 105 tests, fully mocked (no DB/API needed)
```

---

## Configuration

Everything tunable is centralized in `core/config.py` (pydantic-settings) and
overridable via environment variables — so switching models for a benchmark run is a
pure env change, never a code edit.

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql://…@localhost:5434/ragdb` | ParadeDB DSN |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embeddings |
| `GENERATION_MODEL` | `claude-sonnet-4-6` | Answer generation |
| `QUERY_REWRITE_MODEL` | `claude-haiku-4-5-20251001` | Multi-query + filter extraction |
| `RERANK_MODEL` | `rerank-english-v3.0` | Cohere reranker |
| `JUDGE_MODEL` | `claude-sonnet-5` | LLM-as-judge (generation eval) |
| `RELEVANCE_THRESHOLD` | `0.3` | Below this → graceful "no info" answer |
| `PDF_EXTRACT_TABLES` | `true` | Render PDF tables as Markdown (vs. plain text) |
| `{OPENAI,ANTHROPIC}_TIMEOUT_S` / `_MAX_RETRIES` | 30/3, 60/2 | Client reliability |
| `COHERE_TIMEOUT_S` | `15` | Client reliability |
| `RAG_API_KEY` | _(unset)_ | When set, `/ask` + `/ingest` require `X-API-Key` |
| `CORS_ORIGINS` | `http://localhost:8501` | Allowed origins |
| `LOG_FORMAT` | `text` | `json` for structured logs (Docker default) |

---

## Evaluation

The harness (`evals/`, docs in [evals/README.md](evals/README.md)) measures the two
things that fail independently.

**Retrieval** — an ablation grid run through the *real* retriever internals, scored
against a 27-question hand-labeled gold set (recall@k, MRR, nDCG@k). This is the table
that justifies the pipeline: it shows what BM25, RRF, metadata filters, multi-query
rewrite, and rerank each contribute over a semantic-only baseline.

**Generation** — the full `/ask` pipeline per question, graded by an **independent**
LLM judge (a different, stronger model than the generator, to avoid self-consistency
bias) on faithfulness, citation accuracy, and answer relevance, plus a groundedness
pass-rate. Per-question judge **rationales** are persisted for inspection.

Results below are from the **official run** — 2026-07-17, commit
[`93f302d`](https://github.com/eltonzeng/RAG-App/commit/93f302d), `claude-sonnet-5`
generation, `claude-opus-4-8` judge, production Cohere key. Full per-question reports
(rationales, scores, run config) are committed at
[`evals/reports/official/`](evals/reports/official/); both runs recorded **zero**
`rerank_fallback` caveats, confirming the production key never throttled.

**Retrieval ablation** (n=27 gold questions) — as first run at v1.0.0, on the
original IVFFlat index:

| variant | recall@5 | recall@10 | nDCG@10 | MRR | hit_rate@10 |
|---|---|---|---|---|---|
| semantic_only | 0.173 | 0.210 | 0.154 | 0.163 | 0.259 |
| bm25_only | 0.506 | 0.580 | 0.383 | 0.337 | 0.593 |
| hybrid | 0.296 | 0.531 | 0.276 | 0.216 | 0.593 |
| hybrid_filters | 0.593 | 0.778 | 0.466 | 0.374 | 0.815 |
| hybrid_multiquery | 0.444 | 0.691 | 0.427 | 0.357 | 0.741 |
| hybrid_multiquery_rerank | 0.617 | 0.704 | 0.439 | 0.366 | 0.741 |

### The harness caught a bug: a dead dense path

The v1.0.0 table has an anomaly a healthy system shouldn't show: `semantic_only`
recall@10 of 0.210 — **20 of 27 questions retrieved zero gold pages** — and plain
`hybrid` *below* `bm25_only`, because RRF was fusing a strong lexical ranking with
near-noise. Root cause, confirmed by re-running one dead query three ways (default
index / exhaustive probes / sequential scan): the IVFFlat index was configured with
`lists=100` on only ~3.7K vectors and pgvector's default `probes=1`, so each query
scanned **under 1% of the corpus** — gold chunks in any other cluster were never
examined. Measuring the ceiling with exact search put true dense capability at
**hit_rate@10 = 0.704 vs the indexed 0.259**: the embeddings were fine; the index
was throwing their signal away.

The fix: swap IVFFlat for **HNSW** (`m=16, ef_construction=64`, `ef_search=100` —
graph search has no cluster blind spot and needs no probes tuning). Re-run of the
same suite, same gold set, same models
([full report](evals/reports/official/retrieval_official_20260717_hnsw.json)):

| variant | recall@5 | recall@10 | nDCG@10 | MRR | hit_rate@10 |
|---|---|---|---|---|---|
| semantic_only | 0.531 ▲ | 0.667 ▲ | 0.429 ▲ | 0.375 ▲ | 0.704 ▲ |
| bm25_only | 0.506 | 0.580 | 0.383 | 0.337 | 0.593 |
| hybrid | 0.469 ▲ | 0.691 ▲ | 0.422 ▲ | 0.346 ▲ | 0.704 ▲ |
| hybrid_filters | 0.580 | 0.765 | 0.459 | 0.369 | 0.778 |
| hybrid_multiquery | 0.432 | 0.704 | 0.438 | 0.372 | 0.741 |
| hybrid_multiquery_rerank | **0.617** | **0.778** | **0.463** | **0.378** | **0.815** |

Post-fix, the ablation shows the shape hybrid search is supposed to have:
`semantic_only` hit_rate@10 lands exactly on the measured 0.704 exact-search
ceiling (HNSW is effectively lossless here); `hybrid` now beats `bm25_only` on
recall@10 (+0.11); and the full pipeline (`hybrid_multiquery_rerank`) is best or
tied-best on every metric. The honest residual: 8 questions stay dense-hard even
under exact search — exact-number and table lookups where embeddings blur — which
is precisely the query class BM25 carries, and the empirical case for hybrid.

**Generation (LLM-as-judge, 1–5)** (n=27)

| faithfulness | citation_accuracy | answer_relevance | groundedness_rate |
|---|---|---|---|
| 4.963 | 4.926 | 4.778 | 0.963 |

Generation was scored at v1.0.0 (pre-index-fix); its retrieval feed
(`hybrid_multiquery_rerank`) improved further with HNSW, so these judge scores are
if anything a floor. Not re-run — the generation suite is the paid step, and the
index fix is upstream of an already-near-ceiling result.

> **⚠️ n=27 — read deltas, not absolutes.** One question moves any recall/hit-rate
> point by ~3.7pts, and the 95% CI on a proportion at this sample size is roughly
> ±18pts. Small gaps (under ~7pts) are noise; the deltas discussed above — the
> +0.46 semantic recovery, hybrid's +0.11 over BM25 — are outside that band.
>
> **On the judge:** the single non-grounded generation question is a positive
> signal, not a defect — the judge flagged a cross-company synthesis answer for
> overgeneralizing a claim that only one of the cited filings actually supported
> (see the full rationale in the committed report). That's the independent-judge
> design catching a real, subtle overreach, not rubber-stamping.
>
> Judge scores are directional (they compare configs and catch regressions), not a
> certified accuracy number. Reports auto-flag score-distorting artifacts (e.g. Cohere
> trial-key rate-limiting) as `caveats`.

```bash
python -m evals.run --suite retrieval
python -m evals.run --suite generation --concurrency 2 --pace 2
```

---

## Design decisions

- **Explicit stages, not LangChain chains.** Built-in chains hide the rerank step and
  make injecting Cohere between retrieval and generation awkward. Explicit
  `retriever → reranker → generator` keeps each stage testable and swappable.
- **Hybrid + RRF over pure vector search.** Dense search misses exact-term queries
  (ticker symbols, GAAP line items) that BM25 nails; RRF fuses both rankings without
  tuning score scales.
- **Table-aware PDF extraction.** Plain text extractors flatten a financial table
  into number-soup that loses which figure belongs to which column/year. `pdfplumber`
  recovers the grid; each table is rendered as a Markdown pipe table and emitted as
  its **own** chunk (so the splitter never breaks it mid-grid) with its cells removed
  from the page's narrative text (so figures aren't double-counted). Falls back to
  plain text per-page, then to `pypdf`, so a filing never fails to ingest.
- **Reranking is never optional.** If Cohere fails, retrieval falls back to a
  cosine-calibrated similarity gate (a separate threshold, since cosine and Cohere
  scores are distributed differently) — it never silently skips the relevance check.
- **Relevance gate at 0.3.** Below it, the app returns "I don't have enough
  information" instead of generating from weak context — essential for financial Q&A.
- **Content-hash dedup.** Re-ingesting shared/boilerplate text never re-pays for
  embeddings; provenance for every filing/page is accumulated in a `sources` JSONB
  array that also drives metadata filtering.
- **Config as the model-swap interface.** Dev uses cost-conscious models; the official
  benchmark overrides `GENERATION_MODEL`/`JUDGE_MODEL` via env only.

---

## Production notes

Things a real deployment would add, and where they'd go — deliberately out of scope
for a single-tenant portfolio demo:

- **Rate limiting** belongs at the API gateway / ingress, not a per-process in-memory
  limiter (which protects nothing behind a load balancer). The upstream LLM/rerank
  providers are the real scarce resource and are already timeout- and retry-guarded.
- **Observability**: JSON logs with request-id correlation are in place; a deployment
  would ship them to a log aggregator and add OpenTelemetry traces + a metrics endpoint.
- **Schema migrations**: the schema is one `init.sql`; a migration story (Alembic)
  would come with the first breaking schema change.
- **Auth**: `X-API-Key` fits single-tenant; multi-tenant would warrant OAuth2/JWT.

---

## Repository layout

```
api/         FastAPI app, routes, middleware (request-id, auth)
core/        config (pydantic-settings), shared API clients, logging
ingest/      loader · metadata · chunker · embedder
retrieval/   retriever (hybrid + RRF) · reranker (Cohere + fallback)
generation/  prompts · query_rewriter · generator (+ streaming)
evals/       metrics · variants · judge · retrieval/generation suites · report · run
ui/          Streamlit chat app (SSE consumer)
tests/       105 tests, fully mocked (no live DB/API)
```

## Stack

Python 3.11 · FastAPI · asyncpg · ParadeDB (Postgres 16 + pgvector + pg_search) ·
OpenAI `text-embedding-3-small` · Claude (generation, rewrite, judge) · Cohere Rerank ·
Streamlit · pytest / ruff / mypy · Docker Compose.
