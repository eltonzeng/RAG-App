# Evaluation dataset

`retrieval_qa.jsonl` is the hand-labeled gold set that drives both eval suites.
One JSON object per line:

```json
{
  "id": "q1",
  "question": "What was total revenue in FY2023?",
  "filters": {"ticker": "AAPL", "fiscal_year": 2023},
  "gold": [{"source_filename": "aapl_10k_2023.pdf", "page_number": 45}]
}
```

## Fields

| Field | Required | Meaning |
|-------|----------|---------|
| `id` | yes | Stable identifier for the question. |
| `question` | yes | The natural-language query, exactly as a user would ask it. |
| `filters` | no | Metadata filters to apply in the `*_filters` / `*_multiquery` retrieval variants. Keys: `ticker`, `fiscal_year`, `quarter`, `form_type`. Omit or `{}` for none. |
| `gold` | yes (for retrieval scoring) | The chunk(s) that actually answer the question, identified by `source_filename` + `page_number`. One or more entries. |

## How gold matching works

Chunk IDs are fresh UUIDs on every ingest, so gold is matched on the **stable**
`(source_filename, page_number)` provenance carried in each chunk's `sources`
array — not on IDs. A retrieved chunk counts as relevant for a question if any of
its `sources` matches any `gold` entry. This also means re-ingesting the same
filing doesn't invalidate your labels.

## Labeling guide

1. Ingest the filing(s) you want to evaluate against (`POST /ingest`).
2. For each question, find the page(s) whose text genuinely answers it. Open the
   source PDF and read — don't guess. A question may have multiple gold pages.
3. Record the **exact `source_filename`** as stored (the PDF's basename, e.g.
   `aapl_10k_2023.pdf`) and the 1-indexed `page_number`.
4. Set `filters` only when the question implies them (a named company/year/quarter/
   form). Leave empty otherwise.

## Seed rows

The committed rows are **placeholders**: every `source_filename` is
`REPLACE_WITH_YOUR_FILING.pdf` and every `page_number` is `0`. They cover common
10-K topics (revenue, risk factors, R&D, segments, liquidity, cyber, cash flow,
governance) so you have a starting scaffold — replace the `gold` values with real
labels against your ingested filings before the numbers mean anything. Add more
rows freely; 30–50 well-labeled questions is a solid benchmark.
