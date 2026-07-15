"""Tests for evals/metrics.py — pure ranking metrics, no DB or LLM.

Covers the credit-once / no-inflation rules for content-deduplicated chunks:
a chunk matching several gold labels is one hit at its position, and duplicate
content within the top-k is collapsed before scoring.
"""

import math

from evals.metrics import (
    gold_keys,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
    source_keys,
)


def chunk(identity: str, *sources: tuple[str, int]) -> tuple[str, set]:
    """Build a (content_identity, source_keys) ranked-list entry."""
    keys = {(name, page) for name, page in sources}
    return identity, keys


AAPL_45 = ("aapl_10k_2023.pdf", 45)
AAPL_46 = ("aapl_10k_2023.pdf", 46)
MSFT_10 = ("msft_10k_2023.pdf", 10)


class TestKeyHelpers:
    def test_source_keys_from_metadata(self) -> None:
        srcs = [
            {"source_filename": "aapl_10k_2023.pdf", "page_number": 45, "ticker": "AAPL"},
            {"source_filename": "aapl_10q.pdf", "page_number": 2},
        ]
        assert source_keys(srcs) == {("aapl_10k_2023.pdf", 45), ("aapl_10q.pdf", 2)}

    def test_gold_keys(self) -> None:
        gold = [{"source_filename": "aapl_10k_2023.pdf", "page_number": 45}]
        assert gold_keys(gold) == {AAPL_45}


class TestRecall:
    def test_perfect_hit(self) -> None:
        retrieved = [chunk("c1", AAPL_45)]
        assert recall_at_k(retrieved, {AAPL_45}, k=5) == 1.0

    def test_partial_recall(self) -> None:
        retrieved = [chunk("c1", AAPL_45), chunk("c2", MSFT_10)]
        # Two gold labels, one covered → 0.5
        assert recall_at_k(retrieved, {AAPL_45, AAPL_46}, k=5) == 0.5

    def test_cutoff_excludes_later_hits(self) -> None:
        retrieved = [chunk("c1", MSFT_10), chunk("c2", AAPL_45)]
        assert recall_at_k(retrieved, {AAPL_45}, k=1) == 0.0
        assert recall_at_k(retrieved, {AAPL_45}, k=2) == 1.0

    def test_empty_gold_is_zero(self) -> None:
        assert recall_at_k([chunk("c1", AAPL_45)], set(), k=5) == 0.0


class TestHitRate:
    def test_hit_and_miss(self) -> None:
        assert hit_rate_at_k([chunk("c1", AAPL_45)], {AAPL_45}, k=5) == 1.0
        assert hit_rate_at_k([chunk("c1", MSFT_10)], {AAPL_45}, k=5) == 0.0


class TestMRR:
    def test_first_position(self) -> None:
        assert mrr([chunk("c1", AAPL_45)], {AAPL_45}) == 1.0

    def test_second_position(self) -> None:
        retrieved = [chunk("c1", MSFT_10), chunk("c2", AAPL_45)]
        assert mrr(retrieved, {AAPL_45}) == 0.5

    def test_no_hit(self) -> None:
        assert mrr([chunk("c1", MSFT_10)], {AAPL_45}) == 0.0


class TestNDCG:
    def test_ideal_ordering_is_one(self) -> None:
        retrieved = [chunk("c1", AAPL_45), chunk("c2", AAPL_46)]
        assert ndcg_at_k(retrieved, {AAPL_45, AAPL_46}, k=5) == 1.0

    def test_relevant_lower_is_discounted(self) -> None:
        # Single gold at rank 2: DCG = 1/log2(3), IDCG = 1/log2(2) = 1
        retrieved = [chunk("c1", MSFT_10), chunk("c2", AAPL_45)]
        expected = (1.0 / math.log2(3)) / 1.0
        assert abs(ndcg_at_k(retrieved, {AAPL_45}, k=5) - expected) < 1e-9

    def test_empty_gold_is_zero(self) -> None:
        assert ndcg_at_k([chunk("c1", AAPL_45)], set(), k=5) == 0.0


class TestCreditOnceInflation:
    """The rules that stop boilerplate and duplicate content inflating scores."""

    def test_one_chunk_matching_many_gold_is_one_hit(self) -> None:
        # A boilerplate chunk carries provenance for two gold pages at once.
        boilerplate = chunk("boiler", AAPL_45, AAPL_46)
        retrieved = [boilerplate]
        gold = {AAPL_45, AAPL_46}
        # Recall credits both distinct gold labels...
        assert recall_at_k(retrieved, gold, k=5) == 1.0
        # ...but nDCG sees ONE relevant position (gain 1), not two.
        # DCG = 1/log2(2) = 1; IDCG over min(2,5)=2 ideal positions = 1 + 1/log2(3).
        idcg = 1.0 + 1.0 / math.log2(3)
        expected = 1.0 / idcg
        assert abs(ndcg_at_k(retrieved, gold, k=5) - expected) < 1e-9
        # A single position can't fully satisfy a 2-label ideal ranking.
        assert ndcg_at_k(retrieved, gold, k=5) < 1.0

    def test_duplicate_content_collapsed(self) -> None:
        # The same text block appears twice in the ranked list (same identity).
        retrieved = [
            chunk("dup", AAPL_45),
            chunk("dup", AAPL_45),  # collapsed — must not earn a second gain
            chunk("c3", AAPL_46),
        ]
        gold = {AAPL_45, AAPL_46}
        # After dedup: [AAPL_45 @1, AAPL_46 @2] → perfect.
        assert ndcg_at_k(retrieved, gold, k=5) == 1.0
        assert recall_at_k(retrieved, gold, k=5) == 1.0

    def test_second_chunk_covering_credited_gold_earns_nothing(self) -> None:
        # Two distinct chunks both point at the same single gold label.
        retrieved = [chunk("c1", AAPL_45), chunk("c2", AAPL_45)]
        gold = {AAPL_45}
        # Only rank 1 is relevant; rank 2 covers an already-credited label.
        assert mrr(retrieved, gold) == 1.0
        assert ndcg_at_k(retrieved, gold, k=5) == 1.0  # IDCG over 1 gold = 1
        assert recall_at_k(retrieved, gold, k=5) == 1.0
