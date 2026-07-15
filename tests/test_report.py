"""Tests for evals/report.py — table rendering, JSON payload, thresholds."""

import json

from evals.report import (
    check_thresholds,
    render_markdown_table,
    write_json_report,
)


class TestMarkdownTable:
    def test_renders_rows_and_columns(self) -> None:
        table = render_markdown_table(
            {"hybrid": {"recall@10": 0.8, "mrr": 0.5}}, row_label="variant"
        )
        assert "| variant | recall@10 | mrr |" in table
        assert "| hybrid | 0.800 | 0.500 |" in table

    def test_empty_results(self) -> None:
        assert render_markdown_table({}, row_label="variant") == "_(no results)_"


class TestWriteJsonReport:
    def test_payload_shape(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("evals.report.REPORTS_DIR", tmp_path)
        path = write_json_report(
            "generation",
            aggregates={"generation": {"faithfulness": 4.5}},
            rows=[{"id": "q1", "rationale": "clear", "rerank_fallback": False}],
            models={"generation_model": "claude-sonnet-5"},
            run_config={"concurrency": 2, "pace": 0.0},
            caveats=["1/1 questions used the similarity rerank fallback"],
        )
        payload = json.loads(path.read_text())
        assert payload["suite"] == "generation"
        assert payload["models"]["generation_model"] == "claude-sonnet-5"
        assert payload["run_config"]["concurrency"] == 2
        assert payload["caveats"][0].startswith("1/1")
        assert payload["rows"][0]["rationale"] == "clear"
        assert "git_commit" in payload  # present (value may be a hash or None)


class TestCheckThresholds:
    def test_pass_and_fail(self) -> None:
        results = {"hybrid": {"recall@10": 0.7}}
        assert check_thresholds(results, {"hybrid.recall@10": 0.6}) == []
        failures = check_thresholds(results, {"hybrid.recall@10": 0.8})
        assert len(failures) == 1 and "hybrid.recall@10" in failures[0]

    def test_missing_metric_reported(self) -> None:
        failures = check_thresholds({"hybrid": {}}, {"hybrid.mrr": 0.5})
        assert "no such variant/metric" in failures[0]
