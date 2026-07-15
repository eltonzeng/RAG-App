"""Tests for PDF loading and table-aware extraction.

pdfplumber is mocked so no real PDF parsing or filesystem PDFs are needed; the
pure Markdown renderer is exercised directly from plain row lists.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ingest.loader import load_pdf
from ingest.pdf_tables import table_to_markdown, within_bbox


class TestTableToMarkdown:
    def test_clean_grid_renders_pipe_table(self) -> None:
        rows = [
            ["Metric", "FY2024", "FY2023"],
            ["Revenue", "2,481", "2,193"],
            ["Gross margin", "38.1%", "35.4%"],
        ]
        md = table_to_markdown(rows)
        lines = md.splitlines()
        assert lines[0] == "| Metric | FY2024 | FY2023 |"
        assert lines[1] == "| --- | --- | --- |"
        assert lines[2] == "| Revenue | 2,481 | 2,193 |"
        assert lines[3] == "| Gross margin | 38.1% | 35.4% |"

    def test_none_cells_become_empty(self) -> None:
        rows = [["A", None, "C"], ["1", "2", None]]
        md = table_to_markdown(rows)
        assert "| A |  | C |" in md
        assert "| 1 | 2 |  |" in md

    def test_ragged_rows_padded_to_widest(self) -> None:
        rows = [["A", "B", "C"], ["1", "2"], ["x", "y", "z", "extra"]]
        md = table_to_markdown(rows)
        # Grid widens to the longest row (4 cols → 5 pipes) and stays rectangular;
        # no cell is dropped and the short row is padded.
        assert all(line.count("|") == 5 for line in md.splitlines())
        assert "extra" in md

    def test_pipe_in_cell_is_escaped(self) -> None:
        rows = [["Header"], ["a|b"]]
        md = table_to_markdown(rows)
        assert r"a\|b" in md

    def test_internal_newline_collapsed(self) -> None:
        rows = [["Head"], ["line1\nline2"]]
        md = table_to_markdown(rows)
        assert "line1 line2" in md
        assert "line1\nline2" not in md

    def test_all_blank_columns_are_dropped(self) -> None:
        # Middle column blank in every row (a PDF spacer column) → removed.
        rows = [["Label", "", "$", "Value"], ["Revenue", "", "$", "100"]]
        md = table_to_markdown(rows)
        header = md.splitlines()[0]
        assert header == "| Label | $ | Value |"
        assert "|  |" not in md

    def test_empty_table_returns_empty_string(self) -> None:
        assert table_to_markdown([]) == ""

    def test_header_only_returns_empty_string(self) -> None:
        # No data rows → not a meaningful table.
        assert table_to_markdown([["A", "B"]]) == ""

    def test_all_blank_rows_returns_empty_string(self) -> None:
        assert table_to_markdown([[None, None], ["", ""]]) == ""


class TestWithinBbox:
    def test_center_inside(self) -> None:
        obj = {"x0": 10, "x1": 20, "top": 10, "bottom": 20}
        assert within_bbox(obj, (0, 0, 100, 100)) is True

    def test_center_outside(self) -> None:
        obj = {"x0": 200, "x1": 210, "top": 5, "bottom": 15}
        assert within_bbox(obj, (0, 0, 100, 100)) is False


def _fake_table(bbox: tuple, rows: list) -> MagicMock:
    """A pdfplumber-like table object exposing .bbox and .extract()."""
    table = MagicMock()
    table.bbox = bbox
    table.extract.return_value = rows
    return table


def _fake_page(tables: list, narrative: str, full_text: str = "") -> MagicMock:
    """A pdfplumber-like page.

    ``find_tables`` returns ``tables``; ``filter`` returns an object whose
    ``extract_text`` yields ``narrative``; the page's own ``extract_text``
    yields ``full_text`` (used by the fallback paths).
    """
    page = MagicMock()
    page.find_tables.return_value = tables
    filtered = SimpleNamespace(extract_text=lambda: narrative)
    page.filter.return_value = filtered
    page.extract_text.return_value = full_text
    return page


def _patch_pdfplumber(pages: list):
    """Patch ingest.loader.pdfplumber.open to yield a pdf with ``pages``."""
    pdf = MagicMock()
    pdf.pages = pages
    cm = MagicMock()
    cm.__enter__.return_value = pdf
    cm.__exit__.return_value = False
    return patch("ingest.loader.pdfplumber.open", return_value=cm)


class TestLoadPdf:
    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_pdf("/no/such/file.pdf")

    def test_page_yields_narrative_and_table_documents(self, tmp_path) -> None:
        pdf_file = tmp_path / "COHR_10-K_2025.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")
        table = _fake_table(
            (0, 0, 100, 50),
            [["Metric", "FY2025"], ["Revenue", "5,810"]],
        )
        page = _fake_page([table], narrative="Management discussion prose.")

        with _patch_pdfplumber([page]):
            docs = load_pdf(str(pdf_file))

        assert len(docs) == 2
        text_doc = next(d for d in docs if d.metadata["content_type"] == "text")
        table_doc = next(d for d in docs if d.metadata["content_type"] == "table")
        assert text_doc.content == "Management discussion prose."
        assert text_doc.metadata["page_number"] == 1
        assert table_doc.content.startswith("Table 1 (page 1):")
        assert "| Metric | FY2025 |" in table_doc.content
        assert table_doc.metadata["table_index"] == 1
        assert table_doc.metadata["page_number"] == 1

    def test_extract_tables_false_skips_tables(self, tmp_path) -> None:
        pdf_file = tmp_path / "f.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")
        # find_tables would return a table, but extract_tables=False must not call it.
        table = _fake_table((0, 0, 1, 1), [["a"], ["b"]])
        page = _fake_page([table], narrative="x", full_text="Plain page text.")

        with _patch_pdfplumber([page]):
            docs = load_pdf(str(pdf_file), extract_tables=False)

        assert len(docs) == 1
        assert docs[0].metadata["content_type"] == "text"
        assert docs[0].content == "Plain page text."

    def test_table_extraction_failure_falls_back_to_plain_text(self, tmp_path) -> None:
        pdf_file = tmp_path / "f.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")
        page = MagicMock()
        page.find_tables.side_effect = RuntimeError("bad grid")
        page.extract_text.return_value = "Fallback plain text."

        with _patch_pdfplumber([page]):
            docs = load_pdf(str(pdf_file))

        assert len(docs) == 1
        assert docs[0].content == "Fallback plain text."
        assert docs[0].metadata["content_type"] == "text"

    def test_pdfplumber_open_failure_falls_back_to_pypdf(self, tmp_path) -> None:
        pdf_file = tmp_path / "f.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")
        reader = MagicMock()
        reader.pages = [SimpleNamespace(extract_text=lambda: "pypdf page text")]

        with (
            patch("ingest.loader.pdfplumber.open", side_effect=RuntimeError("cannot open")),
            patch("ingest.loader.PdfReader", return_value=reader),
        ):
            docs = load_pdf(str(pdf_file))

        assert len(docs) == 1
        assert docs[0].content == "pypdf page text"
        assert docs[0].metadata["content_type"] == "text"

    def test_no_extractable_text_raises_value_error(self, tmp_path) -> None:
        pdf_file = tmp_path / "f.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")
        page = _fake_page([], narrative="", full_text="")

        with _patch_pdfplumber([page]), pytest.raises(ValueError, match="No extractable text"):
            load_pdf(str(pdf_file))
