# Evaluation Harness

Two suites measure the two things that fail independently in a RAG system:

- **Retrieval** (`--suite retrieval`) — did we fetch the right chunks? Offline
  ranking metrics (recall@k, MRR, nDCG@k) against a hand-labeled gold set, across
  an ablation grid of pipeline configurations. No LLM cost beyond embeddings.
- **Generation** (`--suite generation`) — is the answer faithful and correctly
  cited? An independent **LLM-as-judge** scores faithfulness, citation accuracy,
  and answer relevance on a 1–5 scale, plus a groundedness pass/fail.

The gold dataset (`datasets/retrieval_qa.jsonl`) drives both suites; see
[datasets/README.md](datasets/README.md) for its schema and labeling notes.

## Running

```bash
# Retrieval ablation table (all variants)
python -m evals.run --suite retrieval

# A subset of variants, or a smoke test on the first N questions
python -m evals.run --suite retrieval --variants semantic_only,hybrid
python -m evals.run --suite retrieval --limit 3

# Fail CI-style if a metric is below a floor
python -m evals.run --suite retrieval --fail-under hybrid.recall@10=0.6

# Generation suite (judged); --pace keeps a Cohere trial key under 10 req/min
python -m evals.run --suite generation --concurrency 3
python -m evals.run --suite generation --concurrency 1 --pace 8
```

All flags: `--suite` (required), `--variants`, `--concurrency`, `--pace`,
`--limit`, `--fail-under KEY=FLOOR` (repeatable).

## Report format

Each run writes a timestamped JSON to `reports/` (gitignored) with:

- `git_commit`, `models` (the exact model ids used), `run_config` — reproducibility
- `aggregates` — the summary-table metrics
- `rows` — **per-question** detail. Generation rows include the judge `rationale`
  and a `rerank_fallback` flag; retrieval rows carry per-question metric values.
- `caveats` — auto-generated warnings about score-distorting artifacts (e.g. "N/27
  questions used the similarity rerank fallback").

## Cost note — Cohere trial key

A Cohere **trial** key is capped at 10 requests/minute. Under load the reranker
falls back to cosine similarity (flagged per-row as `rerank_fallback` and summarized
in `caveats`), which understates rerank quality and can zero-out synthesis questions.
For clean numbers, either use a **production** Cohere key or pace the run
(`--concurrency 1 --pace 8`).

## Official metric run (paid, one-time)

The dev defaults are cost-conscious models. For the portfolio benchmark, override
models via the environment only — no code change (see `core/config.py`):

```bash
export GENERATION_MODEL=claude-sonnet-5
export JUDGE_MODEL=claude-opus-4-8
export COHERE_API_KEY=<production key>

# 1) Smoke first to confirm the overrides + cost (~$2 total budget)
python -m evals.run --suite generation --limit 2 --concurrency 1

# 2) Full retrieval (embeddings only) + full generation
python -m evals.run --suite retrieval
python -m evals.run --suite generation --concurrency 2 --pace 2
```

Confirm the generation report's `caveats` shows **zero** rerank fallbacks — that is
the artifact-free certification. Copy the two reports into `reports/official/` and
record the numbers (with model ids + date) in the top-level README.
