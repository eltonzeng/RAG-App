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
    "The documents currently ingested do not appear to contain relevant content for your query. "
    "Please try rephrasing your question or ingest additional SEC filings that may cover this topic."
)


def build_context_block(chunks: list) -> str:
    """Format a list of ScoredChunk objects into a numbered context block.

    Args:
        chunks: List of ScoredChunk objects to format.

    Returns:
        Formatted string with numbered excerpts and source citations.
    """
    parts: list[str] = []
    for i, scored_chunk in enumerate(chunks, start=1):
        chunk = scored_chunk.chunk
        source = chunk.metadata.get("source_filename", "Unknown source")
        page = chunk.metadata.get("page_number")
        page_str = f", page {page}" if page else ""
        parts.append(
            f"[Excerpt {i} — {source}{page_str}]\n{chunk.content}"
        )
    return "\n\n".join(parts)
