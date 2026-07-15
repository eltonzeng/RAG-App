"""Streamlit chat interface for the SEC Filings RAG application.

Provides a chat UI with collapsible citation cards, sidebar stats,
and document ingestion directly from the UI.
"""

import json
import logging
import os
from collections.abc import Iterator

import requests
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base URL of the RAG API. Overridable so the containerized UI can reach the
# API service by its compose hostname (e.g. http://api:8000).
API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://localhost:8000")


def _render_citation(citation: dict) -> None:
    """Render one citation as a markdown bullet with source, page, and chunk id."""
    page_str = f" — page {citation['page']}" if citation.get("page") else ""
    chunk_id = citation["chunk_id"][:8]
    st.markdown(f"- **{citation['source']}**{page_str} `id:{chunk_id}`")


def _stream_answer(prompt: str, sink: dict) -> Iterator[str]:
    """Yield answer text deltas from /ask/stream, recording metadata in ``sink``.

    Parses the Server-Sent Events frames: ``delta`` frames are yielded (for
    st.write_stream); ``citations`` and ``done`` payloads are stored in ``sink``
    for the caller to render after the stream completes. An ``error`` frame
    raises so the caller can fall back to the blocking endpoint.
    """
    with requests.post(
        f"{API_BASE_URL}/ask/stream",
        json={"query": prompt},
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        event = None
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                payload = json.loads(line[len("data:") :].strip())
                if event == "delta":
                    yield payload["text"]
                elif event == "citations":
                    sink["citations"] = payload
                elif event == "done":
                    sink["done"] = payload
                elif event == "error":
                    raise RuntimeError(payload.get("detail", "stream error"))


def _finalize_turn(
    answer: str,
    citations: list,
    latency_ms: float,
    chunks_retrieved: int,
    chunks_used: int,
) -> None:
    """Render citations + latency caption and append the turn to chat history."""
    if citations:
        with st.expander(f"Sources ({len(citations)} cited)"):
            for citation in citations:
                _render_citation(citation)

    st.caption(
        f"Latency: {latency_ms:.0f} ms | Retrieved: {chunks_retrieved} | Used: {chunks_used}"
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "latency_ms": latency_ms,
            "chunks_retrieved": chunks_retrieved,
            "chunks_used": chunks_used,
        }
    )
    st.session_state.last_latency_ms = latency_ms
    st.session_state.total_queries += 1


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SEC Filings RAG",
    page_icon="📄",
    layout="wide",
)


# ── Session state initialization ─────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_latency_ms" not in st.session_state:
    st.session_state.last_latency_ms = None

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("SEC Filings RAG")
    st.caption("Powered by Claude + OpenAI + Cohere + ParadeDB (BM25 + pgvector hybrid)")

    st.divider()

    # Health / stats
    st.subheader("System Status")
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=5).json()
        st.metric("Chunks Indexed", health.get("chunk_count", 0))
        st.metric("DB Connected", "Yes" if health.get("database_connected") else "No")
        st.caption(f"Embedding: `{health.get('embedding_model', 'N/A')}`")
        st.caption(f"Generation: `{health.get('generation_model', 'N/A')}`")
    except Exception:
        st.error("API not reachable — is `uvicorn api.main:app` running?")

    if st.session_state.last_latency_ms is not None:
        st.metric("Last Query", f"{st.session_state.last_latency_ms:.0f} ms")

    st.metric("Total Queries", st.session_state.total_queries)

    st.divider()

    # Ingest section
    st.subheader("Ingest Documents")

    with st.form("ingest_form"):
        file_paths_input = st.text_area(
            "File paths (one per line)",
            placeholder="/path/to/apple_10k.pdf\n/path/to/msft_10q.pdf",
            height=100,
        )
        urls_input = st.text_area(
            "URLs (one per line)",
            placeholder="https://www.sec.gov/...",
            height=60,
        )
        chunk_strategy = st.selectbox(
            "Chunk strategy",
            ["recursive", "fixed", "sentence"],
            index=0,
        )
        ingest_submitted = st.form_submit_button("Ingest", use_container_width=True)

    if ingest_submitted:
        file_paths = [p.strip() for p in file_paths_input.splitlines() if p.strip()]
        urls = [u.strip() for u in urls_input.splitlines() if u.strip()]

        if not file_paths and not urls:
            st.error("Provide at least one file path or URL.")
        else:
            with st.spinner("Ingesting documents..."):
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/ingest",
                        json={
                            "file_paths": file_paths,
                            "urls": urls,
                            "chunk_strategy": chunk_strategy,
                        },
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(
                            f"Ingested {data['documents_loaded']} docs → "
                            f"{data['chunks_embedded']} chunks stored."
                        )
                        st.rerun()
                    else:
                        st.error(f"Ingest failed ({resp.status_code}): {resp.json().get('detail')}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_latency_ms = None
        st.session_state.total_queries = 0
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────

st.title("Ask about SEC Filings")
st.caption("Ask questions about ingested 10-K, 10-Q, or proxy statement documents.")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander(f"Sources ({len(msg['citations'])} cited)"):
                for citation in msg["citations"]:
                    _render_citation(citation)

        if msg["role"] == "assistant" and msg.get("latency_ms"):
            st.caption(
                f"Latency: {msg['latency_ms']:.0f} ms | "
                f"Retrieved: {msg.get('chunks_retrieved', 0)} | "
                f"Used: {msg.get('chunks_used', 0)}"
            )

# Chat input
if prompt := st.chat_input("What was Apple's revenue in FY2023?"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API — stream tokens live, falling back to the blocking endpoint.
    with st.chat_message("assistant"):
        try:
            sink: dict = {}
            answer = st.write_stream(_stream_answer(prompt, sink))
            done = sink.get("done", {})
            _finalize_turn(
                answer,
                sink.get("citations", []),
                done.get("latency_ms", 0),
                done.get("chunks_retrieved", 0),
                done.get("chunks_used", 0),
            )
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Run: `uvicorn api.main:app --reload`")
        except Exception as stream_err:
            # Streaming failed — fall back to the non-streaming /ask endpoint.
            logger.warning("Stream failed (%s); falling back to /ask", stream_err)
            try:
                resp = requests.post(f"{API_BASE_URL}/ask", json={"query": prompt}, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    st.markdown(data["answer"])
                    _finalize_turn(
                        data["answer"],
                        data.get("citations", []),
                        data.get("latency_ms", 0),
                        data.get("chunks_retrieved", 0),
                        data.get("chunks_used", 0),
                    )
                elif resp.status_code == 503:
                    st.error("A backend service is unavailable. Check API logs.")
                else:
                    detail = resp.json().get("detail", "Unknown error")
                    st.error(f"Error {resp.status_code}: {detail}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                logger.error("UI error: %s", e)
