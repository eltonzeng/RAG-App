"""Render extracted PDF tables as GitHub-flavored Markdown.

``pdfplumber`` recovers a table's geometric grid as a list of rows (each a list
of cell strings). Financial tables in 10-K/10-Q filings only carry meaning when
that grid is preserved: ``2,481 2,193 38.1% 35.4%`` is number-soup, but the same
values in an aligned pipe table keep each figure attached to its column and
year. Markdown is the target format because Claude reads it natively and it
survives embedding and BM25 as ordinary text.

This module is deliberately pure (no pdfplumber dependency) so the rendering
logic is unit-testable from plain Python row lists.
"""

import logging

logger = logging.getLogger(__name__)

# pdfplumber cell/bbox types are loosely typed; a cell is a string or None and a
# bbox is a 4-tuple (x0, top, x1, bottom) in PDF coordinate space.
Row = list[str | None]
BBox = tuple[float, float, float, float]


def _clean_cell(value: str | None) -> str:
    """Normalize a single table cell for Markdown rendering.

    Collapses internal newlines to spaces (a Markdown table row is one line) and
    escapes pipe characters so cell content never breaks the column grid.

    Args:
        value: Raw cell value from pdfplumber (may be None for empty cells).

    Returns:
        A single-line, pipe-safe cell string ("" for None/blank cells).
    """
    if value is None:
        return ""
    text = " ".join(value.split())
    return text.replace("|", "\\|")


def table_to_markdown(rows: list[Row]) -> str:
    """Render a pdfplumber table (list of rows of cells) as a Markdown table.

    The first row is treated as the header. Ragged rows are padded to the widest
    row so the pipe grid stays rectangular (never truncated — no cell is lost).
    Columns that are blank in every row are dropped (PDF financial statements
    often emit blank spacer columns between a "$" and its figure). A table with
    no cells,
    or with only a header and no body, is considered degenerate and yields an
    empty string (the caller then skips emitting a table document for it).

    Args:
        rows: Table rows as returned by ``pdfplumber``'s table extraction; each
            row is a list of cell strings (or None for empty cells).

    Returns:
        A GitHub-flavored Markdown table string, or "" if the table is empty or
        has no data rows.
    """
    # Drop rows that are entirely empty (pdfplumber often yields blank spacer
    # rows between ruled sections).
    cleaned: list[list[str]] = []
    for row in rows:
        cells = [_clean_cell(c) for c in row]
        if any(cells):
            cleaned.append(cells)

    if len(cleaned) < 2:
        # Need a header plus at least one data row to form a meaningful table.
        return ""

    width = max(len(r) for r in cleaned)
    if width == 0:
        return ""

    def _pad(row: list[str]) -> list[str]:
        return (row + [""] * width)[:width]

    padded = [_pad(r) for r in cleaned]

    # Drop columns that are empty in every row. Financial-statement PDFs often
    # render each "$" and its number in separate cells with blank spacer columns
    # between them; those all-blank columns add noise (and width) without adding
    # information. Keep a column only if some row has content there.
    keep = [col for col in range(width) if any(row[col] for row in padded)]
    if len(keep) < 1:
        return ""
    padded = [[row[col] for col in keep] for row in padded]
    width = len(keep)

    header = padded[0]
    body = padded[1:]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(r) + " |" for r in body)
    return "\n".join(lines)


def within_bbox(obj: dict, bbox: BBox) -> bool:
    """Return True if a pdfplumber page object's center falls inside ``bbox``.

    Used to exclude words that belong to a detected table from the page's
    narrative text (so table numbers are not duplicated once as garbled inline
    text and once as a rendered Markdown table).

    Args:
        obj: A pdfplumber page object (dict with x0/x1/top/bottom keys).
        bbox: A table bounding box (x0, top, x1, bottom).

    Returns:
        True when the object's midpoint lies within the bounding box.
    """
    x0, top, x1, bottom = bbox
    cx = (obj["x0"] + obj["x1"]) / 2
    cy = (obj["top"] + obj["bottom"]) / 2
    return x0 <= cx <= x1 and top <= cy <= bottom
