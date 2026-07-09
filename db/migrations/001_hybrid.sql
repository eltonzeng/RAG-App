-- Migration 001: hybrid search + content-hash dedup.
--
-- Converts the original semantic-only `chunks` table (per-row source_filename /
-- page_number / line_range / chunk_index columns) to the hybrid schema:
--   * content_hash (unique) becomes the dedup identity
--   * provenance moves into a `sources` JSONB array
--   * adds the ParadeDB BM25 index and a GIN index for JSONB filters
--
-- init.sql only runs on a fresh volume, so apply this against an existing DB:
--   psql "$DATABASE_URL" -f db/migrations/001_hybrid.sql
--
-- NOTE: requires the ParadeDB image (pg_search extension). If you are on the old
-- pgvector image, switch docker-compose to paradedb/paradedb first.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pg_search;

-- New columns (nullable/defaulted so the ALTER succeeds on populated tables).
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_hash TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS sources JSONB NOT NULL DEFAULT '[]';

-- Backfill content_hash for existing rows (sha256 of normalized content).
-- Whitespace normalization here must match ingest.embedder._content_hash.
UPDATE chunks
SET content_hash = encode(
        sha256(convert_to(regexp_replace(trim(content), '\s+', ' ', 'g'), 'UTF8')),
        'hex')
WHERE content_hash IS NULL;

-- Fold legacy provenance columns into the sources array (one element per row).
UPDATE chunks
SET sources = jsonb_build_array(
        jsonb_strip_nulls(jsonb_build_object(
            'source_filename', source_filename,
            'page_number', page_number,
            'chunk_index', chunk_index
        )))
WHERE sources = '[]'::jsonb;

-- De-duplicate rows that now share a content_hash: keep the earliest, merge the
-- rest of their sources into it, then delete the extras.
WITH ranked AS (
    SELECT id, content_hash,
           row_number() OVER (PARTITION BY content_hash ORDER BY ingested_at, id) AS rn
    FROM chunks
), keep AS (
    SELECT content_hash, id AS keep_id FROM ranked WHERE rn = 1
), merged AS (
    SELECT k.keep_id,
           jsonb_agg(DISTINCT elem) AS all_sources
    FROM keep k
    JOIN chunks c ON c.content_hash = k.content_hash
    CROSS JOIN LATERAL jsonb_array_elements(c.sources) AS elem
    GROUP BY k.keep_id
)
UPDATE chunks c
SET sources = m.all_sources
FROM merged m
WHERE c.id = m.keep_id;

DELETE FROM chunks c
USING ranked r
WHERE c.id = r.id AND r.rn > 1;

-- Enforce the dedup identity and drop the legacy columns/index.
ALTER TABLE chunks ALTER COLUMN content_hash SET NOT NULL;
ALTER TABLE chunks ADD CONSTRAINT chunks_content_hash_key UNIQUE (content_hash);

DROP INDEX IF EXISTS idx_chunks_source;
ALTER TABLE chunks DROP COLUMN IF EXISTS source_filename;
ALTER TABLE chunks DROP COLUMN IF EXISTS page_number;
ALTER TABLE chunks DROP COLUMN IF EXISTS line_range;
ALTER TABLE chunks DROP COLUMN IF EXISTS chunk_index;
ALTER TABLE chunks DROP COLUMN IF EXISTS char_count;

-- New indexes.
CREATE INDEX IF NOT EXISTS idx_chunks_bm25
    ON chunks USING bm25 (id, content) WITH (key_field = 'id');
CREATE INDEX IF NOT EXISTS idx_chunks_sources
    ON chunks USING gin (sources jsonb_path_ops);

COMMIT;
