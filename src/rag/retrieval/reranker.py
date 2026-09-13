from dataclasses import replace


class CrossEncoderReranker:
    """Optional text reranking; image-only pages retain their fused order at the end."""

    def __init__(self, model: str, device="cpu"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model, device=device)

    def rerank(self, query, hits):
        text_hits = [hit for hit in hits if hit.text.strip()]
        image_hits = [hit for hit in hits if not hit.text.strip()]
        if not text_hits:
            return image_hits
        scores = self.model.predict([(query, hit.text) for hit in text_hits])
        ranked = [
            replace(hit, score=float(score)) for hit, score in zip(text_hits, scores, strict=True)
        ]
        return sorted(ranked, key=lambda h: (-h.score, h.page.id)) + image_hits
