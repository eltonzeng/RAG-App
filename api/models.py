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
    """

    answer: str
    citations: list[Citation]
    latency_ms: float
    chunks_retrieved: int
    chunks_used: int


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
        chunks_embedded: Number of chunks embedded and stored.
    """

    documents_loaded: int
    chunks_created: int
    chunks_embedded: int


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
