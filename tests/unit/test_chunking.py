import pytest

from rag.chunking.parent_child import chunk_page
from rag.chunking.structural import sections
from rag.db.models import Page

pytestmark = pytest.mark.unit


def test_headings_and_tables_are_preserved():
    text = "intro\n# Results\n| Year | Value |\n| --- | --- |\n| 2025 | 42 |\n# End\nDone"
    result = sections(text)
    assert [heading for heading, _ in result] == ["", "Results", "End"]
    assert "| 2025 | 42 |" in result[1][1]


def test_overlap_parent_and_stable_ids():
    page = Page("p", "d", 1, " ".join(str(i) for i in range(25)))
    chunks = chunk_page(page, size=10, overlap=2)
    assert [chunk.id for chunk in chunks] == ["p:c0", "p:c1", "p:c2"]
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]
    assert all(chunk.parent_text == page.text for chunk in chunks)
    assert chunks[-1].text.endswith("24")


@pytest.mark.parametrize("size,overlap", [(0, 0), (2, 2), (2, -1)])
def test_invalid_window(size, overlap):
    with pytest.raises(ValueError):
        chunk_page(Page("p", "d", 1, "text"), size, overlap)


def test_empty_page():
    assert chunk_page(Page("p", "d", 1, "")) == []
