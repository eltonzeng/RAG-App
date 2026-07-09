"""FastAPI route handlers for the SEC RAG API.

Routes:
  GET  /health  — liveness + DB connectivity + chunk count
  POST /ingest  — load → chunk → embed+store pipeline
  POST /ask     — retrieve → rerank → generate full pipeline
"""

import logging
import time

import psycopg2
from fastapi import APIRouter, HTTPException, Request

from api.models import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
)
from generation.generator import GENERATION_MODEL, generate
from generation.query_rewriter import rewrite_query
from ingest.chunker import chunk_fixed, chunk_recursive, chunk_sentence
from ingest.embedder import EMBEDDING_MODEL, embed_and_store
from ingest.loader import load_pdf, load_txt, load_urls
from ingest.metadata import extract_filing_metadata
from retrieval.reranker import rerank
from retrieval.retriever import hybrid_retrieve

logger = logging.getLogger(__name__)
router = APIRouter()

CHUNKING_STRATEGIES = {
    "fixed": chunk_fixed,
    "recursive": chunk_recursive,
    "sentence": chunk_sentence,
}


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(request: Request) -> HealthResponse:
    """Check service health, database connectivity, and chunk count.

    Args:
        request: FastAPI request with app.state.pool.

    Returns:
        HealthResponse with status, db connection, chunk count, model info.
    """
    db_connected = False
    chunk_count = 0

    try:
        pool = request.app.state.pool
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) AS cnt FROM chunks")
            chunk_count = int(row["cnt"])
            db_connected = True
    except Exception as e:
        logger.error("Health check DB query failed: %s", e)

    status = "ok" if db_connected else "degraded"
    return HealthResponse(
        status=status,
        database_connected=db_connected,
        chunk_count=chunk_count,
        embedding_model=EMBEDDING_MODEL,
        generation_model=GENERATION_MODEL,
    )


@router.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
async def ingest(request: Request, body: IngestRequest) -> IngestResponse:
    """Load, chunk, embed, and store documents into the vector store.

    Args:
        request: FastAPI request with app.state.pool.
        body: IngestRequest specifying file paths, URLs, and chunk strategy.

    Returns:
        IngestResponse with counts of documents loaded, chunks created, stored.

    Raises:
        HTTPException 400: If no sources provided or unknown chunk strategy.
        HTTPException 500: If loading, chunking, or embedding fails.
    """
    if not body.file_paths and not body.urls:
        raise HTTPException(status_code=400, detail="Provide at least one file_path or URL")

    strategy_fn = CHUNKING_STRATEGIES.get(body.chunk_strategy)
    if strategy_fn is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown chunk_strategy '{body.chunk_strategy}'. Use: {list(CHUNKING_STRATEGIES)}",
        )

    documents = []

    # Load files
    for path in body.file_paths:
        try:
            if path.endswith(".pdf"):
                documents.extend(load_pdf(path))
            elif path.endswith(".txt"):
                documents.extend(load_txt(path))
            else:
                logger.warning("Unsupported file type, skipping: %s", path)
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("Failed to load file %s: %s", path, e)
            raise HTTPException(status_code=500, detail=f"Failed to load {path}")

    # Load URLs
    if body.urls:
        try:
            documents.extend(load_urls(body.urls))
        except Exception as e:
            logger.error("Failed to load URLs: %s", e)
            raise HTTPException(status_code=500, detail="Failed to load one or more URLs")

    if not documents:
        raise HTTPException(status_code=400, detail="No content could be loaded from provided sources")

    # Extract best-effort filing metadata (ticker/year/quarter/form_type) per
    # document so chunks inherit it for query-time filtering.
    for document in documents:
        document.metadata.update(extract_filing_metadata(document))

    # Chunk
    try:
        chunks = strategy_fn(documents)
    except Exception as e:
        logger.error("Chunking failed: %s", e)
        raise HTTPException(status_code=500, detail="Chunking failed")

    # Embed and store (deduping by content hash)
    try:
        pool = request.app.state.pool
        embedded, skipped = await embed_and_store(chunks, pool)
    except Exception as e:
        logger.error("Embedding/storage failed: %s", e)
        raise HTTPException(status_code=500, detail="Embedding or database storage failed")

    return IngestResponse(
        documents_loaded=len(documents),
        chunks_created=len(chunks),
        chunks_embedded=embedded,
        chunks_skipped=skipped,
    )


@router.post("/ask", response_model=AskResponse, tags=["query"])
async def ask(request: Request, body: AskRequest) -> AskResponse:
    """Run the full RAG pipeline: retrieve → rerank → generate.

    Args:
        request: FastAPI request with app.state.pool.
        body: AskRequest with query, top_k, and top_n parameters.

    Returns:
        AskResponse with answer, citations, latency, and chunk counts.

    Raises:
        HTTPException 503: If retrieval, reranking, or generation fails.
    """
    start_time = time.perf_counter()
    pool = request.app.state.pool

    # Rewrite: expand into query variants + extract metadata filters. Degrades
    # gracefully to the original query on failure (never raises).
    rewrite = await rewrite_query(body.query)
    filters = rewrite.filters.as_containment()

    # Retrieve (hybrid semantic + BM25, multi-query RRF, metadata-filtered)
    try:
        scored_chunks = await hybrid_retrieve(
            rewrite.queries, filters, pool, top_k=body.top_k
        )
    except Exception as e:
        logger.error("Retrieval failed for query '%s': %s", body.query, e)
        raise HTTPException(status_code=503, detail="Retrieval service unavailable")

    chunks_retrieved = len(scored_chunks)

    # Rerank on the original query (its true intent) rather than a variant.
    try:
        reranked, is_relevant = await rerank(body.query, scored_chunks, top_n=body.top_n)
    except Exception as e:
        logger.error("Reranking failed: %s", e)
        raise HTTPException(status_code=503, detail="Reranking service unavailable")

    chunks_used = len(reranked)

    # Generate
    try:
        answer, citations = await generate(body.query, reranked, is_relevant)
    except Exception as e:
        logger.error("Generation failed: %s", e)
        raise HTTPException(status_code=503, detail="Generation service unavailable")

    latency_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "Ask complete: query='%s' retrieved=%d used=%d latency=%.0fms",
        body.query[:80], chunks_retrieved, chunks_used, latency_ms,
    )

    return AskResponse(
        answer=answer,
        citations=citations,
        latency_ms=round(latency_ms, 1),
        chunks_retrieved=chunks_retrieved,
        chunks_used=chunks_used,
        rewritten_queries=rewrite.queries,
        applied_filters=rewrite.filters,
    )
