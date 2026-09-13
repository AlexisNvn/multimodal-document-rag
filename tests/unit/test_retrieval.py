import numpy as np
import pytest

from rag.db.models import Hit
from rag.retrieval import visual
from rag.retrieval.fusion import reciprocal_rank_fusion

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_relevant_page_first(indexed, mode):
    hits = indexed.retrieve("revenue", mode=mode)
    assert hits[0].page.id == "doc:p1"
    assert len({hit.page.id for hit in hits}) == len(hits)


def test_rrf_ignores_duplicate_votes(pages):
    a, b = Hit(pages[0], 999, "a", ["dense"]), Hit(pages[1], 1, "b", ["sparse"])
    hits = reciprocal_rank_fusion([[a, a], [b, a]])
    assert hits[0].page.id == a.page.id
    assert hits[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_document_filter_and_empty(indexed):
    assert indexed.retrieve("revenue", document_id="missing") == []
    assert indexed.retrieve("zzzznonexistent", mode="sparse") == []
    with pytest.raises(ValueError, match="disabled"):
        indexed.retrieve("revenue", mode="visual")


def test_visual_channel_with_controlled_vectors(pages):
    class Encoder:
        def encode_query(self, query):
            return np.eye(2)

    vectors = [(pages[0].id, np.eye(2)), (pages[1].id, np.zeros((1, 2)))]
    hits = visual.search("chart", vectors, {p.id: p for p in pages}, Encoder())
    assert hits[0].page.id == pages[0].id
    assert hits[0].score == 2
