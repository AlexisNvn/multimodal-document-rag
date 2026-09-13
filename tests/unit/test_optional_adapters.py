import sys
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from PIL import Image

from rag.db.models import Hit, Page
from rag.embeddings.text import SentenceTransformerEncoder
from rag.generation.answer import chat_answer
from rag.generation.context import build_context
from rag.retrieval.reranker import CrossEncoderReranker

pytestmark = pytest.mark.unit


def test_sentence_encoder_requests_normalized_embeddings(monkeypatch):
    class Model:
        def __init__(self, name, device):
            assert name == "test" and device == "cpu"

        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, texts, **kwargs):
            assert kwargs == {"normalize_embeddings": True, "convert_to_numpy": True}
            return np.ones((len(texts), 2))

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model)
    )
    encoder = SentenceTransformerEncoder("test")
    assert encoder.encode([]).shape == (0, 2)
    assert encoder.encode(["hello"]).shape == (1, 2)


def test_reranker_orders_text_and_retains_image_pages(monkeypatch, pages):
    class Model:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, pairs):
            return [0.1, 0.9]

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=Model))
    image = Page("image", "doc", 3, "")
    hits = [Hit(pages[0], 1, "A"), Hit(pages[1], 1, "B"), Hit(image, 1, "")]
    result = CrossEncoderReranker("test").rerank("question", hits)
    assert [hit.page.id for hit in result] == [pages[1].id, pages[0].id, "image"]
    assert hits[0].score == 1


def test_chat_sends_page_image_with_source_label(settings, tmp_path):
    import json

    path = tmp_path / "page.png"
    Image.new("RGB", (8, 8)).save(path)
    context = build_context([Hit(Page("p", "d", 1, "", str(path)), 1, "")])
    settings.llm_model = "test"

    def handler(request):
        payload = json.loads(request.content)
        content = payload["messages"][1]["content"]
        assert content[-2]["text"] == "Page image [S1]"
        assert content[-1]["image_url"]["url"].startswith("data:image/png;base64,")
        return httpx.Response(200, json={"choices": [{"message": {"content": "A page [S1]"}}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert not chat_answer("what?", context, settings, client).abstained


def test_visual_pipeline_persists_image_vectors(service, tmp_path):
    path = tmp_path / "page.png"
    Image.new("RGB", (8, 8)).save(path)
    service.pipeline.visual_encoder = SimpleNamespace(encode_image=lambda _: np.eye(2))
    service.pipeline.ingest_pages("d", "image", [Page("p", "d", 1, "", str(path))])
    vectors = service.repository.visual_vectors()
    assert vectors[0][0] == "p"
    np.testing.assert_array_equal(vectors[0][1], np.eye(2))
    service.repository.delete("d")
    assert service.repository.visual_vectors() == []
