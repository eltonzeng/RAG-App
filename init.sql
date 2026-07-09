-- Schema for the SEC filings RAG app.
--
-- Hybrid search (semantic + BM25) with content-hash dedup:
--   * pgvector `vector` type powers cosine similarity search.
--   * ParadeDB `pg_search` provides a true BM25 index over `content`.
--   * One row per unique `content_hash`; provenance (which filing/page a chunk
--     came from, plus extracted ticker/year/quarter/form_type) lives in the
--     `sources` JSONB array and is appended on re-ingest.
--
-- Runs only on a fresh volume. For an existing DB, apply db/migrations/001_hybrid.sql.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;

CREATE TABLE IF NOT EXISTS chunks (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_hash TEXT NOT NULL UNIQUE,          -- sha256 of normalized content
    content      TEXT NOT NULL,
    embedding    VECTOR(1536) NOT NULL,
    sources      JSONB NOT NULL DEFAULT '[]',   -- [{source_filename, page_number, chunk_index,
                                                --   ticker, fiscal_year, quarter, form_type}, ...]
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Semantic search: cosine-distance IVFFlat index (appropriate for < 1M vectors).
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Lexical search: ParadeDB BM25 index over content, keyed by the primary key.
CREATE INDEX IF NOT EXISTS idx_chunks_bm25
    ON chunks USING bm25 (id, content)
    WITH (key_field = 'id');

-- Metadata filtering: GIN index for JSONB containment (`sources @> ...`).
CREATE INDEX IF NOT EXISTS idx_chunks_sources
    ON chunks USING gin (sources jsonb_path_ops);
