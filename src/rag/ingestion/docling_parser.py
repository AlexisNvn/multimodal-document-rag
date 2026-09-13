from pathlib import Path

from rag.ingestion.enrich_figures import enrich_figures
from rag.ingestion.tables import table_markdown


def enrich_pages(path: Path, pages: list, output: Path, captioner=None) -> list:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.generate_picture_images = True
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options),
        }
    )
    result = converter.convert(path)
    if str(result.status.value) != "success":
        raise ValueError(f"Docling conversion incomplete: {result.status}")
    document = result.document
    figures = enrich_figures(document, output / "figures", captioner)
    for page in pages:
        page.text = document.export_to_markdown(page_no=page.number)
        page.metadata["figures"] = figures.get(page.number, [])
        page.metadata["tables"] = [
            table_markdown(table, document)
            for table in document.tables
            if any(prov.page_no == page.number for prov in table.prov)
        ]
        for figure in page.metadata["figures"]:
            if figure.get("generated_description"):
                page.text += "\n\n[Generated figure description] " + figure["generated_description"]
    return pages
