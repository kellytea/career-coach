"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009)."""


def reciprocal_rank_fusion(rankings: dict[str, list[int]], k: int = 60) -> list[tuple[int, float, dict[str, int]]]:
    """Fuse ranked lists of item IDs: RRF(d) = sum over lists of 1 / (k + rank(d)), rank from 1.

    Returns ``[(item, fused_score, {list_name: rank})]`` best first. Ties are broken
    by the best single rank, then by item ID, so results are deterministic.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for name, items in rankings.items():
        for position, item in enumerate(items, start=1):
            if name in ranks.get(item, {}):
                continue  # a list should not credit the same item twice
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + position)
            ranks.setdefault(item, {})[name] = position
    return sorted(((i, scores[i], ranks[i]) for i in scores),
                  key=lambda row: (-row[1], min(row[2].values()), row[0]))
