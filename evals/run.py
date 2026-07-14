"""CLI entrypoint for the eval harness.

Examples:
    python -m evals.run --suite retrieval
    python -m evals.run --suite retrieval --variants semantic_only,hybrid
    python -m evals.run --suite retrieval --fail-under hybrid.recall@10=0.6
    python -m evals.run --suite generation --concurrency 3
    python -m evals.run --suite generation --limit 5   # smoke test, bounds Opus judge cost

Requires a running ParadeDB (with filings ingested) and the relevant API keys in
the environment. The retrieval suite needs OpenAI + DB; the generation suite also
needs Anthropic (generator + judge) and Cohere (rerank).
"""

import argparse
import asyncio
import logging
import sys

from evals.db import open_pool
from evals.generation_eval import run_generation_eval
from evals.report import (
    check_thresholds,
    render_markdown_table,
    write_json_report,
)
from evals.retrieval_eval import run_retrieval_eval
from evals.variants import VARIANTS

logger = logging.getLogger(__name__)


def _parse_thresholds(tokens: list[str]) -> dict[str, float]:
    """Parse --fail-under "key=value" tokens into a dict.

    Args:
        tokens: Raw "variant.metric=floor" strings.

    Returns:
        Mapping of key → float floor.

    Raises:
        SystemExit: On a malformed token.
    """
    thresholds: dict[str, float] = {}
    for token in tokens:
        key, sep, value = token.partition("=")
        if not sep:
            raise SystemExit(f"Invalid --fail-under '{token}' (expected key=value)")
        thresholds[key.strip()] = float(value)
    return thresholds


async def _run(args: argparse.Namespace) -> int:
    """Execute the requested suite. Returns a process exit code."""
    pool = await open_pool()
    try:
        if args.suite == "retrieval":
            variant_names = (
                [v.strip() for v in args.variants.split(",")] if args.variants else None
            )
            results = await run_retrieval_eval(
                pool, variant_names=variant_names, limit=args.limit
            )
            table = render_markdown_table(results, row_label="variant")
        else:
            gen = await run_generation_eval(
                pool, concurrency=args.concurrency, limit=args.limit
            )
            results = {"generation": gen}
            table = render_markdown_table(results, row_label="suite")
    finally:
        await pool.close()

    print(f"\n## {args.suite} eval\n")
    print(table)
    report_path = write_json_report(args.suite, results)
    print(f"\nReport written to {report_path}")

    if args.fail_under:
        failures = check_thresholds(results, _parse_thresholds(args.fail_under))
        if failures:
            print("\nThreshold failures:")
            for f in failures:
                print(f"  - {f}")
            return 1
    return 0


def main() -> None:
    """Parse arguments and run the selected eval suite."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

    parser = argparse.ArgumentParser(description="RAG evaluation harness")
    parser.add_argument(
        "--suite", choices=["retrieval", "generation"], required=True,
        help="Which eval suite to run.",
    )
    parser.add_argument(
        "--variants", default=None,
        help=f"Comma-separated retrieval variants (default all: {', '.join(VARIANTS)}).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=3,
        help="Max concurrent pipeline+judge runs for the generation suite.",
    )
    parser.add_argument(
        "--fail-under", action="append", default=[], metavar="KEY=FLOOR",
        help="Exit non-zero if a metric is below the floor, e.g. hybrid.recall@10=0.6.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Evaluate only the first N dataset rows. Useful for smoke-testing "
             "before a full run, especially to bound Opus judge cost.",
    )
    args = parser.parse_args()

    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
