"""Document loaders for SEC filings.

Supports PDF, plain text, and URL-based loading. Each loader returns
a list of Document objects with content and metadata preserved for
downstream chunking and citation.
"""

import logging
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from api.models import Document

logger = logging.getLogger(__name__)


def load_pdf(path: str) -> list[Document]:
    """Load a PDF file and return one Document per page.

    Args:
        path: Filesystem path to the PDF file.

    Returns:
        List of Document objects, one per page, with page_number metadata.

    Raises:
        FileNotFoundError: If the PDF path does not exist.
        ValueError: If the PDF contains no extractable text.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(str(file_path))
    documents: list[Document] = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
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
                },
            )
        )

    if not documents:
        raise ValueError(f"No extractable text found in {path}")

    logger.info("Loaded %d pages from PDF: %s", len(documents), file_path.name)
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
