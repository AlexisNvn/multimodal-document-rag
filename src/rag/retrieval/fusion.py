from rag.db.models import Hit


def reciprocal_rank_fusion(rankings: list[list[Hit]], k=10, rank_constant=60) -> list[Hit]:
    if rank_constant < 0:
        raise ValueError("rank_constant must be nonnegative")
    merged = {}
    for ranking in rankings:
        seen = set()
        for rank, hit in enumerate(ranking, start=1):
            if hit.page.id in seen:
                continue
            seen.add(hit.page.id)
            if hit.page.id not in merged:
                merged[hit.page.id] = Hit(hit.page, 0.0, hit.text, [])
            current = merged[hit.page.id]
            current.score += 1 / (rank_constant + rank)
            current.channels = sorted(set(current.channels + hit.channels))
    return sorted(merged.values(), key=lambda h: (-h.score, h.page.id))[: max(0, k)]
