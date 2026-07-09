"""Text chunking strategies for SEC filings.

Three strategies are provided:
- fixed: Token-based fixed-size chunks using tiktoken
- recursive: Character-based recursive splitting (default)
- sentence: Sentence-aware splitting with overlap

All strategies preserve document metadata through to each chunk.
"""

import logging
import uuid

import tiktoken
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

from api.models import Chunk, Document

logger = logging.getLogger(__name__)

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_RECURSIVE_CHUNK_SIZE = 1000
DEFAULT_RECURSIVE_OVERLAP = 200
DEFAULT_SENTENCE_CHUNK_OVERLAP = 2


def _build_chunks(texts: list[str], doc: Document, strategy: str) -> list[Chunk]:
    """Convert split text segments into Chunk objects with metadata.

    Args:
        texts: List of text segments from a splitter.
        doc: Original Document to inherit metadata from.
        strategy: Name of the chunking strategy used.

    Returns:
        List of Chunk objects with preserved metadata.
    """
    chunks: list[Chunk] = []
    for i, text in enumerate(texts):
        chunks.append(Chunk(
            id=str(uuid.uuid4()),
            content=text,
            chunk_index=i,
            char_count=len(text),
            metadata={
                **doc.metadata,
                "chunk_strategy": strategy,
            },
        ))
    return chunks


def chunk_fixed(
    docs: list[Document],
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split documents into fixed-size token chunks.

    Uses tiktoken (cl100k_base) to count tokens, ensuring each chunk
    stays within the specified token budget.

    Args:
        docs: List of Document objects to chunk.
        size: Maximum number of tokens per chunk.
        overlap: Number of overlapping tokens between consecutive chunks.

    Returns:
        List of Chunk objects.
    """
    encoder = tiktoken.get_encoding("cl100k_base")
    all_chunks: list[Chunk] = []

    for doc in docs:
        tokens = encoder.encode(doc.content)
        step = max(size - overlap, 1)
        texts: list[str] = []

        for start in range(0, len(tokens), step):
            token_slice = tokens[start : start + size]
            texts.append(encoder.decode(token_slice))

        all_chunks.extend(_build_chunks(texts, doc, "fixed"))

    logger.info(
        "Fixed chunking: %d docs → %d chunks (size=%d, overlap=%d)",
        len(docs), len(all_chunks), size, overlap,
    )
    return all_chunks


def chunk_recursive(
    docs: list[Document],
    chunk_size: int = DEFAULT_RECURSIVE_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_RECURSIVE_OVERLAP,
) -> list[Chunk]:
    """Split documents using recursive character text splitting (default strategy).

    Uses LangChain's RecursiveCharacterTextSplitter with separators
    optimized for SEC filing text (paragraphs, sentences, words).

    Args:
        docs: List of Document objects to chunk.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        List of Chunk objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    all_chunks: list[Chunk] = []

    for doc in docs:
        texts = splitter.split_text(doc.content)
        all_chunks.extend(_build_chunks(texts, doc, "recursive"))

    logger.info(
        "Recursive chunking: %d docs → %d chunks (size=%d, overlap=%d)",
        len(docs), len(all_chunks), chunk_size, chunk_overlap,
    )
    return all_chunks


def chunk_sentence(
    docs: list[Document],
    chunk_overlap: int = DEFAULT_SENTENCE_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split documents with sentence-aware boundaries.

    Uses RecursiveCharacterTextSplitter with sentence-level separators
    to keep sentences intact where possible.

    Args:
        docs: List of Document objects to chunk.
        chunk_overlap: Number of sentences to overlap between chunks.

    Returns:
        List of Chunk objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function=len,
    )
    all_chunks: list[Chunk] = []

    for doc in docs:
        texts = splitter.split_text(doc.content)
        all_chunks.extend(_build_chunks(texts, doc, "sentence"))

    logger.info(
        "Sentence chunking: %d docs → %d chunks (overlap=%d)",
        len(docs), len(all_chunks), chunk_overlap,
    )
    return all_chunks
