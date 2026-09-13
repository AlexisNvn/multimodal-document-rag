import numpy as np

from rag.db.models import Hit


def search(query, chunks, pages, encoder, k=20):
    if not chunks or k < 1:
        return []
    vector = encoder.encode([query])[0]
    matrix = np.stack([embedding for _, embedding in chunks])
    scores = matrix @ vector
    best = {}
    for (chunk, _), score in zip(chunks, scores, strict=True):
        if score > 0 and (chunk.page_id not in best or score > best[chunk.page_id].score):
            best[chunk.page_id] = Hit(
                pages[chunk.page_id], float(score), chunk.parent_text, ["dense"]
            )
    return sorted(best.values(), key=lambda h: (-h.score, h.page.id))[:k]
