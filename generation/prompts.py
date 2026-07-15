"""Prompt templates for SEC filings RAG generation.

All prompts are defined as module-level constants. Never hardcode
prompts inline in generator.py or elsewhere.
"""

SYSTEM_PROMPT = """You are a financial analyst assistant specialized in SEC filings \
(10-K, 10-Q, proxy statements, and other EDGAR documents).

Your task is to answer questions based ONLY on the provided context chunks from \
SEC filings. You must not use any outside knowledge or make assumptions beyond \
what is explicitly stated in the context.

Rules:
1. Answer based strictly on the provided context. Do not hallucinate facts.
2. If the context does not contain enough information to answer the question, \
respond with: "I don't have enough information in the provided filings to answer \
this question."
3. When citing information, reference the source document and page number if \
available (e.g., "According to [filename], page [N]...").
4. Be precise with financial figures — always include units (millions, billions, %).
5. If multiple filings contain relevant information, synthesize across them clearly.
6. Keep your response concise and structured. Use bullet points for lists of facts.
"""

USER_PROMPT_TEMPLATE = """Here are the relevant excerpts from SEC filings:

{context}

---

Question: {query}

Please answer the question based only on the excerpts above. Include citations \
with source filenames and page numbers where available."""


NO_RELEVANT_CONTENT_RESPONSE = (
    "I don't have enough information in the provided filings to answer this question. "
    "The documents currently ingested do not appear to contain relevant content for "
    "your query. Please try rephrasing your question or ingest additional SEC filings "
    "that may cover this topic."
)


def relevant_sources(sources: list[dict], filters: dict | None) -> list[dict]:
    """Keep only the provenance entries that satisfy the applied metadata filters.

    A deduplicated chunk can carry sources from several filings/years. When a
    metadata filter was applied at retrieval time (e.g. ``{"ticker": "MU"}``),
    only the sources matching that filter are germane to the answer — surfacing
    the others would cite/quote filings the user filtered out. With no filter,
    every source is relevant.

    Args:
        sources: The chunk's ``metadata["sources"]`` list.
        filters: The applied containment filters (empty/None means no filter).

    Returns:
        The subset of sources matching every filter key/value. Falls back to all
        sources when the filter matches none — retrieval guarantees at least one
        match, so this only guards against unexpected states.
    """
    if not filters:
        return list(sources)
    matching = [src for src in sources if all(src.get(k) == v for k, v in filters.items())]
    return matching or list(sources)


def build_context_block(chunks: list, filters: dict | None = None) -> str:
    """Format a list of ScoredChunk objects into a numbered context block.

    Args:
        chunks: List of ScoredChunk objects to format.
        filters: Metadata filters applied at retrieval time. When set, a
            deduplicated chunk's sources are narrowed to those matching the
            filter so the excerpt is not labeled with filtered-out filings.

    Returns:
        Formatted string with numbered excerpts and source citations. When a
        deduplicated chunk appears in multiple (filter-matching) filings, all of
        those sources are listed.
    """
    parts: list[str] = []
    for i, scored_chunk in enumerate(chunks, start=1):
        chunk = scored_chunk.chunk
        sources = chunk.metadata.get("sources")
        if sources:
            labels = []
            for src in relevant_sources(sources, filters):
                name = src.get("source_filename", "Unknown source")
                page = src.get("page_number")
                labels.append(f"{name}, page {page}" if page else name)
            source_label = "; ".join(labels)
        else:
            source = chunk.metadata.get("source_filename", "Unknown source")
            page = chunk.metadata.get("page_number")
            source_label = f"{source}, page {page}" if page else source
        parts.append(f"[Excerpt {i} — {source_label}]\n{chunk.content}")
    return "\n\n".join(parts)


# --- Query rewriting (Haiku) ---

QUERY_REWRITE_SYSTEM_PROMPT = """You are a query-understanding component for a \
retrieval system over SEC filings (10-K, 10-Q, and related EDGAR documents).

Given a user's question, produce a search plan by calling the `search_plan` tool. \
Your job is twofold:

1. Rewrite the question into 3-4 diverse search queries that improve retrieval \
recall. Vary the phrasing: include a keyword-dense variant (financial terms, GAAP \
line items), a natural-language variant, and a variant using synonyms or expanded \
acronyms. Keep each query focused on the same underlying intent — do not invent new \
questions.

2. Extract any structured metadata the user explicitly or implicitly specified: \
company stock ticker, fiscal year, fiscal quarter, and SEC form type. Only include \
a field when you are confident it is present in the question. Leave a field null \
otherwise. Never guess a ticker from a company name unless it is unambiguous \
(e.g. "Apple" → "AAPL").

Always call the tool exactly once."""

# Anthropic tool schema forcing structured JSON output from the rewrite call.
QUERY_REWRITE_TOOL = {
    "name": "search_plan",
    "description": (
        "Return the rewritten search queries and any extracted metadata filters "
        "for retrieving relevant SEC filing chunks."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 4,
                "description": "3-4 reformulated search queries for the same intent.",
            },
            "filters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": ["string", "null"],
                        "description": "Uppercase stock ticker, e.g. AAPL.",
                    },
                    "fiscal_year": {
                        "type": ["integer", "null"],
                        "description": "Four-digit fiscal year, e.g. 2023.",
                    },
                    "quarter": {
                        "type": ["integer", "null"],
                        "description": "Fiscal quarter 1-4, or null for annual filings.",
                    },
                    "form_type": {
                        "type": ["string", "null"],
                        "description": "SEC form type, e.g. 10-K or 10-Q.",
                    },
                },
                "required": ["ticker", "fiscal_year", "quarter", "form_type"],
            },
        },
        "required": ["queries", "filters"],
    },
}
