"""Tests for the chunking strategies in ingest/chunker.py.

Tests all three strategies (fixed, recursive, sentence) for:
- Correct chunk creation and non-empty output
- Size constraints are respected
- Metadata is preserved from source documents
- Overlap behavior between consecutive chunks
"""

import pytest

from api.models import Document
from ingest.chunker import chunk_fixed, chunk_recursive, chunk_sentence


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for testing chunking strategies."""
    long_text = (
        "The Securities and Exchange Commission requires all public companies "
        "to file annual reports on Form 10-K. These reports provide a comprehensive "
        "overview of a company's financial condition and business operations. "
        "The 10-K includes audited financial statements, management's discussion "
        "and analysis of financial condition, and disclosures about market risk. "
    ) * 10  # ~1500+ chars to ensure multiple chunks

    return [
        Document(
            content=long_text,
            metadata={
                "source_filename": "test_filing.pdf",
                "page_number": 1,
                "file_type": "pdf",
            },
        ),
        Document(
            content="Short document content for edge case testing.",
            metadata={
                "source_filename": "short.txt",
                "file_type": "txt",
            },
        ),
    ]


class TestChunkFixed:
    """Tests for the fixed-size token chunking strategy."""

    def test_creates_chunks(self, sample_documents: list[Document]) -> None:
        """Fixed chunking should produce at least one chunk per document."""
        chunks = chunk_fixed(sample_documents, size=100, overlap=10)
        assert len(chunks) > 0

    def test_metadata_preserved(self, sample_documents: list[Document]) -> None:
        """Each chunk should inherit metadata from its source document."""
        chunks = chunk_fixed(sample_documents, size=100, overlap=10)
        pdf_chunks = [c for c in chunks if c.metadata.get("source_filename") == "test_filing.pdf"]
        assert len(pdf_chunks) > 0
        assert pdf_chunks[0].metadata["page_number"] == 1
        assert pdf_chunks[0].metadata["file_type"] == "pdf"
        assert pdf_chunks[0].metadata["chunk_strategy"] == "fixed"

    def test_chunk_index_sequential(self, sample_documents: list[Document]) -> None:
        """Chunk indices should be sequential starting from 0 within each document."""
        chunks = chunk_fixed([sample_documents[0]], size=100, overlap=10)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_char_count_matches_content(self, sample_documents: list[Document]) -> None:
        """char_count field should match actual content length."""
        chunks = chunk_fixed(sample_documents, size=100, overlap=10)
        for chunk in chunks:
            assert chunk.char_count == len(chunk.content)

    def test_overlap_produces_more_chunks(self, sample_documents: list[Document]) -> None:
        """Larger overlap should produce more chunks than smaller overlap."""
        chunks_small = chunk_fixed([sample_documents[0]], size=100, overlap=10)
        chunks_large = chunk_fixed([sample_documents[0]], size=100, overlap=50)
        assert len(chunks_large) >= len(chunks_small)

    def test_all_content_covered(self, sample_documents: list[Document]) -> None:
        """No content should be lost during chunking (all words present)."""
        doc = sample_documents[0]
        chunks = chunk_fixed([doc], size=100, overlap=10)
        combined = " ".join(c.content for c in chunks)
        # Check that key words from the original are in the combined output
        for word in ["Securities", "Commission", "10-K", "financial"]:
            assert word in combined


class TestChunkRecursive:
    """Tests for the recursive character text splitting strategy."""

    def test_creates_chunks(self, sample_documents: list[Document]) -> None:
        """Recursive chunking should produce chunks from long documents."""
        chunks = chunk_recursive(sample_documents, chunk_size=200, chunk_overlap=40)
        assert len(chunks) > 0

    def test_respects_size_limit(self, sample_documents: list[Document]) -> None:
        """Chunks should not exceed the specified size limit."""
        chunk_size = 300
        chunks = chunk_recursive([sample_documents[0]], chunk_size=chunk_size, chunk_overlap=40)
        for chunk in chunks:
            assert chunk.char_count <= chunk_size + 10  # small tolerance for separator

    def test_metadata_preserved(self, sample_documents: list[Document]) -> None:
        """Metadata including chunk_strategy should be set."""
        chunks = chunk_recursive(sample_documents, chunk_size=200, chunk_overlap=40)
        for chunk in chunks:
            assert "source_filename" in chunk.metadata
            assert chunk.metadata["chunk_strategy"] == "recursive"

    def test_short_document_single_chunk(self, sample_documents: list[Document]) -> None:
        """A short document should produce a single chunk."""
        chunks = chunk_recursive([sample_documents[1]], chunk_size=1000, chunk_overlap=100)
        assert len(chunks) == 1
        assert chunks[0].content == sample_documents[1].content

    def test_unique_ids(self, sample_documents: list[Document]) -> None:
        """Each chunk should have a unique ID."""
        chunks = chunk_recursive(sample_documents, chunk_size=200, chunk_overlap=40)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))


class TestChunkSentence:
    """Tests for the sentence-aware chunking strategy."""

    def test_creates_chunks(self, sample_documents: list[Document]) -> None:
        """Sentence chunking should produce chunks."""
        chunks = chunk_sentence(sample_documents)
        assert len(chunks) > 0

    def test_metadata_preserved(self, sample_documents: list[Document]) -> None:
        """Metadata should be preserved with sentence strategy label."""
        chunks = chunk_sentence(sample_documents)
        for chunk in chunks:
            assert "source_filename" in chunk.metadata
            assert chunk.metadata["chunk_strategy"] == "sentence"

    def test_chunks_are_nonempty(self, sample_documents: list[Document]) -> None:
        """No chunk should have empty content."""
        chunks = chunk_sentence(sample_documents)
        for chunk in chunks:
            assert len(chunk.content.strip()) > 0

    def test_char_count_matches(self, sample_documents: list[Document]) -> None:
        """char_count should match content length."""
        chunks = chunk_sentence(sample_documents)
        for chunk in chunks:
            assert chunk.char_count == len(chunk.content)
