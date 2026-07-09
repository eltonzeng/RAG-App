"""Pydantic v2 data models for the RAG pipeline and API.

All request/response schemas and internal data models are defined here.
Models use strict validation and clear field descriptions for API documentation.
"""

from pydantic import BaseModel, Field


# --- Internal pipeline models ---


class Document(BaseModel):
    """A loaded document with content and source metadata.

    Attributes:
        content: The text content of the document.
        metadata: Source metadata (filename, page number, file type, etc.).
    """

    content: str
    metadata: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    """A text chunk derived from a Document, ready for embedding.

    Attributes:
        id: Unique identifier for this chunk.
        content: The text content of the chunk.
        chunk_index: Position of this chunk within its source document.
        char_count: Number of characters in the chunk content.
        metadata: Inherited and augmented metadata from the source document.
            After retrieval, ``metadata["sources"]`` holds the list of
            provenance objects (source_filename, page_number, and any extracted
            ticker/fiscal_year/quarter/form_type) for this deduplicated chunk.
    """

    id: str
    content: str
    chunk_index: int
    char_count: int
    metadata: dict = Field(default_factory=dict)


class ScoredChunk(BaseModel):
    """A chunk with a relevance score from retrieval or reranking.

    Attributes:
        chunk: The underlying Chunk object.
        score: Relevance score (higher is more relevant).
    """

    chunk: Chunk
    score: float


class Citation(BaseModel):
    """A citation reference extracted from a generated response.

    Attributes:
        source: Source filename or URL.
        page: Page number (for PDFs), or None.
        chunk_id: ID of the chunk that was cited.
    """

    source: str
    page: int | None = None
    chunk_id: str


# --- Query understanding models ---


class MetadataFilters(BaseModel):
    """Structured filters extracted from a user query by the rewrite step.

    All fields are optional; only the ones the model could confidently extract
    are set. They are matched against the per-chunk ``sources`` JSONB array via
    containment (``@>``) at retrieval time.

    Attributes:
        ticker: Stock ticker symbol (e.g. "AAPL"), uppercased.
        fiscal_year: Four-digit fiscal year (e.g. 2023).
        quarter: Fiscal quarter 1-4 (None for annual/10-K filings).
        form_type: SEC form type (e.g. "10-K", "10-Q").
    """

    ticker: str | None = None
    fiscal_year: int | None = None
    quarter: int | None = None
    form_type: str | None = None

    def as_containment(self) -> dict:
        """Return the non-null filters as a plain dict for JSONB containment.

        Returns:
            Dict of set filter keys → values, suitable for wrapping in a
            single-element JSON array and passing to ``sources @> $1``. Empty
            when no filters were extracted.
        """
        return {k: v for k, v in self.model_dump().items() if v is not None}


class QueryRewriteResult(BaseModel):
    """Output of the single Haiku query-rewrite call.

    Attributes:
        queries: 1-4 reformulated search queries for multi-query retrieval.
        filters: Structured metadata filters parsed from the user's question.
    """

    queries: list[str]
    filters: MetadataFilters = Field(default_factory=MetadataFilters)


# --- API request/response models ---


class AskRequest(BaseModel):
    """Request body for the /ask endpoint.

    Attributes:
        query: The user's natural language question.
        top_k: Number of chunks to retrieve before reranking.
        top_n: Number of chunks to keep after reranking.
    """

    query: str = Field(..., min_length=1, description="The question to answer")
    top_k: int = Field(default=20, ge=1, le=100, description="Chunks to retrieve")
    top_n: int = Field(default=5, ge=1, le=20, description="Chunks after reranking")


class AskResponse(BaseModel):
    """Response body for the /ask endpoint.

    Attributes:
        answer: The generated answer text.
        citations: List of source citations referenced in the answer.
        latency_ms: Total pipeline latency in milliseconds.
        chunks_retrieved: Number of chunks retrieved before reranking.
        chunks_used: Number of chunks passed to the generator.
        rewritten_queries: Query variants the rewrite step produced (for
            transparency/debugging).
        applied_filters: Metadata filters extracted and applied during retrieval.
    """

    answer: str
    citations: list[Citation]
    latency_ms: float
    chunks_retrieved: int
    chunks_used: int
    rewritten_queries: list[str] = Field(default_factory=list)
    applied_filters: MetadataFilters = Field(default_factory=MetadataFilters)


class IngestRequest(BaseModel):
    """Request body for the /ingest endpoint.

    Attributes:
        file_paths: List of local file paths to ingest (PDF or TXT).
        urls: List of URLs to fetch and ingest.
        chunk_strategy: Chunking strategy to use.
    """

    file_paths: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    chunk_strategy: str = Field(
        default="recursive",
        description="Chunking strategy: 'fixed', 'recursive', or 'sentence'",
    )


class IngestResponse(BaseModel):
    """Response body for the /ingest endpoint.

    Attributes:
        documents_loaded: Number of source documents loaded.
        chunks_created: Number of chunks generated.
        chunks_embedded: Number of new chunks embedded and stored.
        chunks_skipped: Number of chunks skipped as duplicates (content already
            stored); their new provenance was appended without re-embedding.
    """

    documents_loaded: int
    chunks_created: int
    chunks_embedded: int
    chunks_skipped: int = 0


class HealthResponse(BaseModel):
    """Response body for the /health endpoint.

    Attributes:
        status: Service health status string.
        database_connected: Whether the database is reachable.
        chunk_count: Total number of chunks stored.
        embedding_model: Name of the embedding model in use.
        generation_model: Name of the generation model in use.
    """

    status: str
    database_connected: bool
    chunk_count: int
    embedding_model: str
    generation_model: str
