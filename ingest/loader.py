"""Document loaders for SEC filings.

Supports PDF, plain text, and URL-based loading. Each loader returns
a list of Document objects with content and metadata preserved for
downstream chunking and citation.
"""

import logging
from pathlib import Path

import pdfplumber
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from api.models import Document
from ingest.pdf_tables import table_to_markdown, within_bbox

logger = logging.getLogger(__name__)


def _page_documents(
    page: pdfplumber.page.Page,
    page_num: int,
    filename: str,
    extract_tables: bool,
) -> list[Document]:
    """Turn one pdfplumber page into narrative and per-table Documents.

    When ``extract_tables`` is set, detected tables are rendered as Markdown and
    emitted as their own Documents (tagged ``content_type="table"``) so they stay
    atomic through chunking, and their cells are removed from the narrative text
    so figures are not duplicated as garbled inline prose. Any failure in the
    table path degrades gracefully to plain full-page text extraction.

    Args:
        page: A pdfplumber page.
        page_num: 1-based page number (for metadata/citations).
        filename: Source filename for metadata.
        extract_tables: Whether to attempt table-aware extraction.

    Returns:
        List of Documents for this page (may be empty if the page has no text).
    """
    base_meta = {
        "source_filename": filename,
        "page_number": page_num,
        "file_type": "pdf",
    }
    table_docs: list[Document] = []
    narrative = ""

    if extract_tables:
        try:
            found = page.find_tables()
            bboxes = [t.bbox for t in found]
            for idx, table in enumerate(found, start=1):
                markdown = table_to_markdown(table.extract())
                if not markdown:
                    continue
                table_docs.append(
                    Document(
                        content=f"Table {idx} (page {page_num}):\n{markdown}",
                        metadata={**base_meta, "content_type": "table", "table_index": idx},
                    )
                )
            # Narrative = page text with table regions removed, so numbers are
            # not double-counted between inline text and the rendered tables.
            narrative_page = page.filter(
                lambda obj: not any(within_bbox(obj, bbox) for bbox in bboxes)
            )
            narrative = (narrative_page.extract_text() or "").strip()
        except Exception as e:
            # Never let a malformed table drop a page: fall back to plain text.
            logger.warning(
                "Table extraction failed on page %d of %s (%s); using plain text",
                page_num,
                filename,
                e,
            )
            table_docs = []
            narrative = ""

    if not narrative and not table_docs:
        narrative = (page.extract_text() or "").strip()

    documents: list[Document] = []
    if narrative:
        documents.append(
            Document(content=narrative, metadata={**base_meta, "content_type": "text"})
        )
    documents.extend(table_docs)

    if not documents:
        logger.warning("Page %d of %s has no extractable text", page_num, filename)
    return documents


def _load_pdf_pypdf(file_path: Path) -> list[Document]:
    """Last-resort loader: extract plain per-page text with pypdf.

    Used only when ``pdfplumber`` cannot open the file at all, so a filing that
    defeats the table-aware path still ingests as plain text.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of one plain-text Document per page with extractable text.
    """
    reader = PdfReader(str(file_path))
    documents: list[Document] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            logger.warning("Page %d of %s has no extractable text", page_num, file_path.name)
            continue
        documents.append(
            Document(
                content=text,
                metadata={
                    "source_filename": file_path.name,
                    "page_number": page_num,
                    "file_type": "pdf",
                    "content_type": "text",
                },
            )
        )
    return documents


def load_pdf(path: str, *, extract_tables: bool = True) -> list[Document]:
    """Load a PDF file into Documents, with table-aware extraction.

    Each page yields a narrative-text Document plus one Document per detected
    table (rendered as Markdown so column/row alignment survives). Tables are
    kept as separate Documents so the chunker never splits them mid-grid.
    Page numbers are preserved on every Document for citations.

    Args:
        path: Filesystem path to the PDF file.
        extract_tables: When True (default), detect tables and render them as
            Markdown. When False, extract plain page text only.

    Returns:
        List of Document objects. A page contributes a narrative Document (when
        it has prose) and a Document per table; ``content_type`` metadata is
        "text" or "table".

    Raises:
        FileNotFoundError: If the PDF path does not exist.
        ValueError: If the PDF contains no extractable text.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    documents: list[Document] = []
    try:
        with pdfplumber.open(str(file_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                documents.extend(_page_documents(page, page_num, file_path.name, extract_tables))
    except Exception as e:
        # pdfplumber itself failed to open/parse — fall back to pypdf plain text.
        logger.warning(
            "pdfplumber failed on %s (%s); falling back to pypdf plain-text extraction",
            file_path.name,
            e,
        )
        documents = _load_pdf_pypdf(file_path)

    if not documents:
        raise ValueError(f"No extractable text found in {path}")

    table_count = sum(1 for d in documents if d.metadata.get("content_type") == "table")
    logger.info(
        "Loaded %d documents (%d tables) from PDF: %s",
        len(documents),
        table_count,
        file_path.name,
    )
    return documents


def load_txt(path: str) -> list[Document]:
    """Load a plain text file and return a single Document.

    Args:
        path: Filesystem path to the text file.

    Returns:
        List containing one Document with the full file content.

    Raises:
        FileNotFoundError: If the text file path does not exist.
        ValueError: If the file is empty.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty text file: {path}")

    logger.info("Loaded text file: %s (%d chars)", file_path.name, len(text))
    return [
        Document(
            content=text,
            metadata={
                "source_filename": file_path.name,
                "file_type": "txt",
            },
        )
    ]


def load_urls(url_list: list[str]) -> list[Document]:
    """Load web pages and extract text content.

    Args:
        url_list: List of URLs to fetch and parse.

    Returns:
        List of Document objects, one per successfully loaded URL.
    """
    documents: list[Document] = []

    for url in url_list:
        try:
            response = requests.get(
                url, timeout=30, headers={"User-Agent": "SEC-RAG-Research-Bot/1.0"}
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            text = soup.get_text(separator="\n", strip=True)
            if not text:
                logger.warning("No text content extracted from %s", url)
                continue

            documents.append(
                Document(
                    content=text,
                    metadata={
                        "source_filename": url,
                        "file_type": "url",
                    },
                )
            )
            logger.info("Loaded URL: %s (%d chars)", url, len(text))

        except requests.RequestException as e:
            logger.error("Failed to load URL %s: %s", url, e)

    return documents
