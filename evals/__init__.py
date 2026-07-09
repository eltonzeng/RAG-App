"""Evaluation harness for the SEC filings RAG pipeline.

Two suites:
- Retrieval: offline ranking metrics (recall@k, MRR, nDCG@k) over a hand-labeled
  gold set, with an ablation grid across pipeline configurations.
- Generation: an LLM-as-judge (Claude Opus 4.8) scoring faithfulness, citation
  accuracy, and answer relevance of the full pipeline's answers.
"""
