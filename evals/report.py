"""Reporting for eval runs: markdown tables to stdout + JSON to evals/reports/.

Pure formatting — no dependency on tabulate or pandas.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

REPORTS_DIR = Path(__file__).parent / "reports"


def _fmt(value: float) -> str:
    """Format a metric value to 3 decimals."""
    return f"{value:.3f}"


def render_markdown_table(results: dict[str, dict[str, float]], row_label: str) -> str:
    """Render a nested results dict as a markdown table.

    Args:
        results: Mapping of row name → {metric: value}.
        row_label: Header for the first column (e.g. "variant" or "suite").

    Returns:
        A markdown table string. Empty note if there are no results.
    """
    if not results:
        return "_(no results)_"

    # Union of metric columns, preserving first-seen order.
    columns: list[str] = []
    for metrics in results.values():
        for key in metrics:
            if key not in columns:
                columns.append(key)

    header = f"| {row_label} | " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * (len(columns) + 1)) + " |"
    lines = [header, divider]
    for name, metrics in results.items():
        cells = " | ".join(_fmt(metrics.get(col, 0.0)) for col in columns)
        lines.append(f"| {name} | {cells} |")
    return "\n".join(lines)


def write_json_report(suite: str, results: dict) -> Path:
    """Write a timestamped JSON report and return its path.

    Args:
        suite: Suite name ("retrieval" or "generation").
        results: The results payload to persist.

    Returns:
        Path to the written report.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = REPORTS_DIR / f"{suite}_{stamp}.json"
    payload = {
        "suite": suite,
        "generated_at": stamp,
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def check_thresholds(
    results: dict[str, dict[str, float]],
    thresholds: dict[str, float],
) -> list[str]:
    """Check `variant.metric` thresholds against results.

    Args:
        results: Variant → metrics mapping.
        thresholds: Mapping of "variant.metric" → minimum value.

    Returns:
        List of human-readable failure messages (empty if all pass).
    """
    failures: list[str] = []
    for key, floor in thresholds.items():
        variant, _, metric = key.partition(".")
        actual = results.get(variant, {}).get(metric)
        if actual is None:
            failures.append(f"{key}: no such variant/metric to check")
        elif actual < floor:
            failures.append(f"{key}: {actual:.3f} < {floor:.3f}")
    return failures
