from rag.db.models import Hit
from rag.embeddings.visual import maxsim


def search(query, vectors, pages, encoder, k=20):
    if encoder is None or not vectors or k < 1:
        return []
    query_vector = encoder.encode_query(query)
    hits = [
        Hit(pages[page_id], maxsim(query_vector, vector), pages[page_id].text, ["visual"])
        for page_id, vector in vectors
    ]
    return sorted(hits, key=lambda h: (-h.score, h.page.id))[:k]
