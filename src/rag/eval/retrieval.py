import math


def retrieval_metrics(ranked_ids: list[str], qrels: dict[str, float], k=5) -> dict[str, float]:
    if k < 1 or any(score < 0 or not math.isfinite(score) for score in qrels.values()):
        raise ValueError("Require k >= 1 and finite nonnegative relevance grades")
    ranked = list(dict.fromkeys(ranked_ids))[:k]
    relevant = {key for key, score in qrels.items() if score > 0}
    found = [index for index, key in enumerate(ranked, start=1) if key in relevant]
    dcg = sum(
        (2 ** qrels.get(key, 0) - 1) / math.log2(index + 2) for index, key in enumerate(ranked)
    )
    ideal = sum(
        (2**score - 1) / math.log2(index + 2)
        for index, score in enumerate(sorted(qrels.values(), reverse=True)[:k])
    )
    return {
        f"recall@{k}": len(found) / len(relevant) if relevant else 0.0,
        f"precision@{k}": len(found) / k,
        f"mrr@{k}": 1 / found[0] if found else 0.0,
        f"ndcg@{k}": dcg / ideal if ideal else 0.0,
    }
