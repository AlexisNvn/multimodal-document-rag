import numpy as np
import pytest

from rag.embeddings.text import HashEncoder, tokens
from rag.embeddings.visual import maxsim

pytestmark = pytest.mark.unit


def test_hash_normalization_and_determinism():
    matrix = HashEncoder().encode(["Revenue revenue", "REVENUE", ""])
    np.testing.assert_allclose(matrix[0], matrix[1])
    assert np.linalg.norm(matrix[0]) == pytest.approx(1)
    assert not matrix[2].any()
    assert HashEncoder().encode([]).shape == (0, 256)
    assert tokens("Énergie 2025!") == ["énergie", "2025"]


def test_maxsim_uses_each_query_tokens_best_patch():
    query = np.eye(2)
    document = np.array([[1, 0], [0, 2], [-1, -1]])
    assert maxsim(query, document) == 3
    assert maxsim(np.empty((0, 2)), document) == 0
    with pytest.raises(ValueError):
        maxsim(query, np.ones((2, 3)))
