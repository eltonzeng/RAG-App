"""Unit tests for citation/context source filtering (no DB or LLM).

Locks in the behavior that a deduplicated chunk whose ``sources`` span several
filings is narrowed to the applied metadata filter, so an answer never cites or
quotes a filing the user filtered out.
"""

from api.models import Chunk, ScoredChunk
from generation.generator import _extract_citations
from generation.prompts import build_context_block, relevant_sources


def _shared_chunk() -> ScoredChunk:
    """A deduped chunk of boilerplate shared across an MU and an SNDK filing."""
    sources = [
        {
            "source_filename": "MU_10-K_2025.pdf",
            "page_number": 42,
            "ticker": "MU",
            "fiscal_year": 2025,
        },
        {
            "source_filename": "SNDK_10-K_2025.pdf",
            "page_number": 7,
            "ticker": "SNDK",
            "fiscal_year": 2025,
        },
    ]
    return ScoredChunk(
        chunk=Chunk(
            id="c1",
            content="shared risk boilerplate",
            chunk_index=0,
            char_count=23,
            metadata={"sources": sources},
        ),
        score=0.9,
    )


class TestRelevantSources:
    def test_no_filter_keeps_all(self) -> None:
        sc = _shared_chunk()
        assert len(relevant_sources(sc.chunk.metadata["sources"], None)) == 2
        assert len(relevant_sources(sc.chunk.metadata["sources"], {})) == 2

    def test_filter_narrows_to_match(self) -> None:
        sc = _shared_chunk()
        kept = relevant_sources(sc.chunk.metadata["sources"], {"ticker": "MU"})
        assert [s["source_filename"] for s in kept] == ["MU_10-K_2025.pdf"]

    def test_multi_key_filter(self) -> None:
        sc = _shared_chunk()
        kept = relevant_sources(
            sc.chunk.metadata["sources"], {"ticker": "SNDK", "fiscal_year": 2025}
        )
        assert [s["source_filename"] for s in kept] == ["SNDK_10-K_2025.pdf"]

    def test_no_match_falls_back_to_all(self) -> None:
        # Retrieval guarantees at least one match; the fallback only guards
        # against unexpected states — it must never drop every source.
        sc = _shared_chunk()
        kept = relevant_sources(sc.chunk.metadata["sources"], {"ticker": "ZZZZ"})
        assert len(kept) == 2


class TestExtractCitations:
    def test_no_filter_emits_all_sources(self) -> None:
        cites = _extract_citations([_shared_chunk()])
        assert {(c.source, c.page) for c in cites} == {
            ("MU_10-K_2025.pdf", 42),
            ("SNDK_10-K_2025.pdf", 7),
        }

    def test_filter_emits_only_matching_source(self) -> None:
        cites = _extract_citations([_shared_chunk()], {"ticker": "MU"})
        assert [(c.source, c.page) for c in cites] == [("MU_10-K_2025.pdf", 42)]

    def test_absent_sources_uses_top_level_metadata(self) -> None:
        sc = ScoredChunk(
            chunk=Chunk(
                id="c2",
                content="legacy",
                chunk_index=0,
                char_count=6,
                metadata={"source_filename": "legacy.pdf", "page_number": 3},
            ),
            score=0.5,
        )
        cites = _extract_citations([sc], {"ticker": "MU"})
        assert [(c.source, c.page) for c in cites] == [("legacy.pdf", 3)]


class TestBuildContextBlock:
    def test_filtered_context_labels_only_matching_filing(self) -> None:
        block = build_context_block([_shared_chunk()], {"ticker": "MU"})
        assert "MU_10-K_2025.pdf, page 42" in block
        assert "SNDK_10-K_2025.pdf" not in block
