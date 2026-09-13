import math
from collections import Counter

from rag.db.models import Hit
from rag.embeddings.text import tokens


def search(query, chunks, pages, k=20, k1=1.5, b=0.75):
    if not chunks or k < 1:
        return []
    counts = [Counter(tokens(chunk.text)) for chunk, _ in chunks]
    lengths = [sum(count.values()) for count in counts]
    average = sum(lengths) / len(lengths) or 1
    df = Counter(token for count in counts for token in count)
    best = {}
    for (chunk, _), count, length in zip(chunks, counts, lengths, strict=True):
        score = 0.0
        for token in set(tokens(query)):
            frequency = count[token]
            idf = math.log(1 + (len(chunks) - df[token] + 0.5) / (df[token] + 0.5))
            score += idf * frequency * (k1 + 1) / (frequency + k1 * (1 - b + b * length / average))
        if score > 0 and (chunk.page_id not in best or score > best[chunk.page_id].score):
            best[chunk.page_id] = Hit(pages[chunk.page_id], score, chunk.parent_text, ["sparse"])
    return sorted(best.values(), key=lambda h: (-h.score, h.page.id))[:k]
