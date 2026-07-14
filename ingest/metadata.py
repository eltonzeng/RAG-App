"""Best-effort SEC filing metadata extraction for ingest.

Parses ticker, fiscal year, quarter, and form type from a document's filename
and the head of its content. These values are stored inside each chunk's
``sources`` element so that query-time metadata filters (JSONB containment) can
constrain retrieval.

Extraction is heuristic and deliberately conservative: a field is only returned
when a confident match is found, so that missing keys stay absent from the
stored JSON (keeping ``sources @>`` containment filters clean).
"""

import logging
import re

from api.models import Document

logger = logging.getLogger(__name__)

# How many leading characters of content to scan for metadata cues.
_CONTENT_SCAN_CHARS = 4000

# 10-K / 10-Q (and amendments like 10-K/A). Normalized to upper case. Uses
# explicit non-alphanumeric boundaries because filenames often use underscores
# (e.g. "AAPL_10-K_2023.pdf"), where \b does not match between "_" and "1".
_FORM_TYPE_RE = re.compile(
    r"(?<![0-9A-Za-z])(10[\-\s]?[KQ])(?:/A)?(?![A-Za-z])", re.IGNORECASE
)
# Four-digit fiscal years in a plausible range. Digit-boundary lookarounds
# (rather than \b) so years adjacent to "_" or letters in filenames still match.
_YEAR_RE = re.compile(r"(?<!\d)(19[9]\d|20[0-4]\d)(?!\d)")
# Explicit fiscal-year phrasing, e.g. "fiscal year 2023", "FY2023", "FY 2023".
_FY_PHRASE_RE = re.compile(
    r"(?:fiscal\s+year|fiscal\s+year\s+ended|fy)\s*[:\-]?\s*(20[0-4]\d|19[9]\d)",
    re.IGNORECASE,
)
# Quarter cues: "Q3", "third quarter", "quarterly period ended ... (Q2)".
_QUARTER_RE = re.compile(r"\bQ([1-4])\b", re.IGNORECASE)
_QUARTER_WORD_RE = re.compile(
    r"\b(first|second|third|fourth)\s+quarter\b", re.IGNORECASE
)
# Ticker in a filename like "AAPL_10-K_2023.pdf" or "aapl-10q.pdf".
_FILENAME_TICKER_RE = re.compile(r"\b([A-Z]{1,5})[\-_]?(?:10[\-]?[KQ])", re.IGNORECASE)
# Ticker in content, e.g. "(NASDAQ: AAPL)" or "Symbol: AAPL".
_CONTENT_TICKER_RE = re.compile(
    r"(?:NYSE|NASDAQ|symbol|ticker)[:\s]+([A-Z]{1,5})\b"
)

_WORD_TO_QUARTER = {"first": 1, "second": 2, "third": 3, "fourth": 4}


def _normalize_form_type(raw: str) -> str:
    """Normalize a matched form-type token to canonical form (e.g. '10-K').

    Args:
        raw: The raw regex match, such as "10K", "10 q", or "10-Q".

    Returns:
        Canonical uppercased form type with a hyphen, e.g. "10-K".
    """
    letter = raw.strip()[-1].upper()
    return f"10-{letter}"


def extract_filing_metadata(doc: Document) -> dict:
    """Extract best-effort filing metadata from a document.

    Scans the document's ``source_filename`` and the head of its content for a
    ticker, fiscal year, quarter, and form type. Only confidently matched
    fields are included in the result.

    Args:
        doc: The source Document (its metadata carries ``source_filename``;
            content is scanned for cues).

    Returns:
        Dict with any of the keys ``ticker`` (str), ``fiscal_year`` (int),
        ``quarter`` (int), ``form_type`` (str). Keys are omitted when no
        confident match is found.
    """
    filename = str(doc.metadata.get("source_filename", ""))
    head = doc.content[:_CONTENT_SCAN_CHARS]
    haystack = f"{filename}\n{head}"
    result: dict = {}

    form_match = _FORM_TYPE_RE.search(haystack)
    if form_match:
        result["form_type"] = _normalize_form_type(form_match.group(1))

    # Prefer explicit fiscal-year phrasing; fall back to filename year, then any
    # plausible year in the content head.
    fy_match = _FY_PHRASE_RE.search(head)
    year_match = fy_match or _YEAR_RE.search(filename) or _YEAR_RE.search(head)
    if year_match:
        result["fiscal_year"] = int(year_match.group(1))

    # A 10-K is an annual filing with no quarter. Skip quarter extraction for it
    # so incidental phrasing like "first quarter of fiscal 2025" in the MD&A does
    # not mislabel the whole filing as Q1 (which would corrupt its provenance and
    # any later quarter filter). Only 10-Q / unknown-form filings carry a quarter.
    if result.get("form_type") != "10-K":
        quarter_match = _QUARTER_RE.search(haystack)
        if quarter_match:
            result["quarter"] = int(quarter_match.group(1))
        else:
            word_match = _QUARTER_WORD_RE.search(head)
            if word_match:
                result["quarter"] = _WORD_TO_QUARTER[word_match.group(1).lower()]

    ticker_match = _FILENAME_TICKER_RE.search(filename) or _CONTENT_TICKER_RE.search(head)
    if ticker_match:
        result["ticker"] = ticker_match.group(1).upper()

    logger.debug("Extracted filing metadata from %s: %s", filename, result)
    return result
