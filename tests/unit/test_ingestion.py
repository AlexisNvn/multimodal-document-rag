from types import SimpleNamespace

import pytest
from PIL import Image

from rag.ingestion.enrich_figures import enrich_figures
from rag.ingestion.pdf_renderer import render_pdf
from rag.ingestion.tables import rows_to_markdown

pytestmark = pytest.mark.unit


def test_render_real_pdf(tmp_path, pdf_bytes):
    path = tmp_path / "report.pdf"
    path.write_bytes(pdf_bytes)
    pages = render_pdf(path, tmp_path / "images", "doc")
    assert len(pages) == 1
    assert "42 million" in pages[0].text
    with Image.open(pages[0].image_path) as image:
        assert image.width > 600
    with pytest.raises(ValueError):
        render_pdf(path, tmp_path / "other", "doc", max_pages=0)


def test_tables_escape_delimiters():
    assert "a\\|b" in rows_to_markdown(["Name"], [["a|b"]])
    with pytest.raises(ValueError):
        rows_to_markdown(["A"], [["1", "2"]])


def test_figure_enrichment_keeps_generated_text_separate(tmp_path):
    picture = SimpleNamespace(
        prov=[SimpleNamespace(page_no=1, bbox=SimpleNamespace(model_dump=lambda **_: {"l": 0}))],
        get_image=lambda doc: Image.new("RGB", (10, 10)),
        caption_text=lambda doc: "Source caption",
    )
    output = enrich_figures(SimpleNamespace(pictures=[picture]), tmp_path, lambda _: "A chart")
    assert output[1][0]["caption"] == "Source caption"
    assert output[1][0]["generated_description"] == "A chart"
