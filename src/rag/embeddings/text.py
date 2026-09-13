import hashlib
import re
from typing import Protocol

import numpy as np


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


class TextEncoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashEncoder:
    """Deterministic lexical smoke-test backend, NOT a semantic embedding model."""

    def __init__(self, dimensions=256):
        self.dimensions = dimensions

    def encode(self, texts):
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokens(text):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                matrix[row, int.from_bytes(digest[:4], "little") % self.dimensions] += (
                    1 if digest[4] % 2 else -1
                )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)


class SentenceTransformerEncoder:
    def __init__(self, model: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, device=device)

    def encode(self, texts):
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
