"""Ranking metrics for retrieval evaluation.

Metrics are computed against a gold set of (source_filename, page_number) labels
rather than chunk ids, because chunk ids are fresh UUIDs on every ingest while
provenance is stable.

Credit-once / no-inflation rules (matter because chunks are content-deduplicated,
so one retrieved chunk can carry a sources list matching several gold items, and
different ranks can point at the same underlying text):

- Each distinct gold label is credited at most once, to the highest-ranked chunk
  that covers it.
- Each rank position contributes a binary gain of at most 1, regardless of how
  many gold labels that single chunk matches.
- Duplicate content within the top-k is collapsed before scoring, so two
  positions resolving to the same text block cannot both earn gain.

All functions are pure and dependency-free (no numpy).
"""

import math

# A gold label / source key is a (source_filename, page_number) tuple.
GoldKey = tuple[str, object]


def source_keys(chunk_sources: list[dict]) -> set[GoldKey]:
    """Extract the set of (source_filename, page_number) keys from a chunk.

    Args:
        chunk_sources: The chunk's ``metadata["sources"]`` list.

    Returns:
        Set of gold-comparable keys the chunk provides.
    """
    keys: set[GoldKey] = set()
    for src in chunk_sources or []:
        keys.add((src.get("source_filename"), src.get("page_number")))
    return keys


def gold_keys(gold: list[dict]) -> set[GoldKey]:
    """Normalize a dataset gold list into a set of comparable keys.

    Args:
        gold: List of {"source_filename", "page_number"} dicts.

    Returns:
        Set of gold keys.
    """
    return {(g.get("source_filename"), g.get("page_number")) for g in gold}


def _dedup_retrieved(retrieved: list[tuple[str, set[GoldKey]]]) -> list[set[GoldKey]]:
    """Collapse duplicate content within a ranked list, preserving order.

    Args:
        retrieved: Ranked list of (content_identity, source_keys) best-first.

    Returns:
        Ranked list of source-key sets with duplicate content identities removed
        (first occurrence kept).
    """
    seen: set[str] = set()
    deduped: list[set[GoldKey]] = []
    for identity, keys in retrieved:
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(keys)
    return deduped


def _relevant_positions(
    retrieved: list[tuple[str, set[GoldKey]]],
    gold: set[GoldKey],
    k: int,
) -> tuple[list[int], int]:
    """Compute per-rank binary relevance under the credit-once rules.

    Walks the deduplicated top-k. A position earns gain 1 iff its chunk covers at
    least one gold label not already credited to a higher rank; each gold label is
    consumed the first time it is covered.

    Args:
        retrieved: Ranked list of (content_identity, source_keys) best-first.
        gold: Set of gold keys for the query.
        k: Cutoff.

    Returns:
        Tuple of (per-position binary gains for the top-k, count of distinct gold
        labels covered within the top-k).
    """
    remaining = set(gold)
    gains: list[int] = []
    for keys in _dedup_retrieved(retrieved)[:k]:
        newly = keys & remaining
        if newly:
            remaining -= newly
            gains.append(1)
        else:
            gains.append(0)
    covered = len(gold) - len(remaining)
    return gains, covered


def recall_at_k(
    retrieved: list[tuple[str, set[GoldKey]]],
    gold: set[GoldKey],
    k: int,
) -> float:
    """Fraction of distinct gold labels covered within the top-k.

    Args:
        retrieved: Ranked (content_identity, source_keys) list, best-first.
        gold: Gold key set for the query.
        k: Cutoff.

    Returns:
        Recall in [0, 1]; 0.0 when the query has no gold labels.
    """
    if not gold:
        return 0.0
    _, covered = _relevant_positions(retrieved, gold, k)
    return covered / len(gold)


def hit_rate_at_k(
    retrieved: list[tuple[str, set[GoldKey]]],
    gold: set[GoldKey],
    k: int,
) -> float:
    """1.0 if at least one gold label appears in the top-k, else 0.0.

    Args:
        retrieved: Ranked (content_identity, source_keys) list, best-first.
        gold: Gold key set for the query.
        k: Cutoff.

    Returns:
        1.0 or 0.0 (0.0 when the query has no gold labels).
    """
    if not gold:
        return 0.0
    _, covered = _relevant_positions(retrieved, gold, k)
    return 1.0 if covered > 0 else 0.0


def mrr(
    retrieved: list[tuple[str, set[GoldKey]]],
    gold: set[GoldKey],
    k: int | None = None,
) -> float:
    """Reciprocal rank of the first position that covers a new gold label.

    Args:
        retrieved: Ranked (content_identity, source_keys) list, best-first.
        gold: Gold key set for the query.
        k: Optional cutoff; consider only the top-k positions.

    Returns:
        1/rank of the first relevant position (1-indexed), or 0.0 if none.
    """
    if not gold:
        return 0.0
    limit = k if k is not None else len(retrieved)
    gains, _ = _relevant_positions(retrieved, gold, limit)
    for i, gain in enumerate(gains):
        if gain:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    retrieved: list[tuple[str, set[GoldKey]]],
    gold: set[GoldKey],
    k: int,
) -> float:
    """Binary-relevance nDCG@k under the credit-once rules.

    IDCG is computed over min(#distinct_gold, k) ideal relevant positions, so the
    ceiling reflects distinct gold labels rather than repeated matches.

    Args:
        retrieved: Ranked (content_identity, source_keys) list, best-first.
        gold: Gold key set for the query.
        k: Cutoff.

    Returns:
        nDCG in [0, 1]; 0.0 when the query has no gold labels.
    """
    if not gold:
        return 0.0
    gains, _ = _relevant_positions(retrieved, gold, k)
    dcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(gains))
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0
