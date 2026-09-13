from pathlib import Path

import pypdfium2 as pdfium

from rag.db.models import Page


def render_pdf(path: Path, output: Path, document_id: str, max_pages: int = 100) -> list[Page]:
    output.mkdir(parents=True, exist_ok=True)
    pages = []
    with pdfium.PdfDocument(path) as pdf:
        if not 0 < len(pdf) <= max_pages:
            raise ValueError(f"PDF must contain 1–{max_pages} pages")
        for index in range(len(pdf)):
            page = pdf[index]
            try:
                text_page = page.get_textpage()
                try:
                    text = text_page.get_text_range()
                finally:
                    text_page.close()
                bitmap = page.render(scale=1.5)
                try:
                    image = bitmap.to_pil()
                    image_path = output / f"{index + 1}.png"
                    image.save(image_path)
                finally:
                    bitmap.close()
                pages.append(
                    Page(
                        f"{document_id}:p{index + 1}",
                        document_id,
                        index + 1,
                        text,
                        str(image_path.resolve()),
                    )
                )
            finally:
                page.close()
    return pages
