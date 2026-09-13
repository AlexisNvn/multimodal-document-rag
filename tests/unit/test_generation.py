import httpx
import pytest

from rag.db.models import Hit, Page
from rag.generation.answer import ABSTENTION, chat_answer, extractive_answer
from rag.generation.citations import cited_sources
from rag.generation.context import build_context

pytestmark = pytest.mark.unit


def test_context_budget_and_deduplication(pages):
    hit = Hit(pages[0], 1, "x" * 200, ["dense"])
    context = build_context([hit, hit], max_chars=100)
    assert len(context.text) <= 100
    assert len(context.citations) == 1
    assert context.citations[0].page_number == 1


def test_empty_and_image_only_extractive_abstain():
    assert extractive_answer(build_context([])).abstained
    hit = Hit(Page("p", "d", 1, "", "page.png"), 1, "")
    assert extractive_answer(build_context([hit])).abstained


def test_citation_validation(pages):
    context = build_context([Hit(pages[0], 1, "Revenue 42")])
    assert cited_sources("42 [S1]", context.citations) == context.citations
    with pytest.raises(ValueError):
        cited_sources("42 [S99]", context.citations)


@pytest.mark.parametrize(
    "text,abstained",
    [
        ("42 million [S1]", False),
        ("42 million [S99]", True),
        ("42 million", True),
        (ABSTENTION, True),
    ],
)
def test_chat_requires_known_citations(settings, pages, text, abstained):
    settings.llm_model = "test-model"
    context = build_context([Hit(pages[0], 1, "Revenue 42")])

    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = chat_answer("Revenue?", context, settings, client)
    assert result.abstained is abstained
