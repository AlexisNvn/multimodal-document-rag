import hashlib
from pathlib import Path

from rag.chunking.parent_child import chunk_page
from rag.ingestion.pdf_renderer import render_pdf


class IngestionPipeline:
    def __init__(self, repository, text_encoder, settings, visual_encoder=None):
        self.repository = repository
        self.text_encoder = text_encoder
        self.settings = settings
        self.visual_encoder = visual_encoder

    def ingest_pages(self, document_id, name, pages):
        if not pages:
            raise ValueError("No pages to ingest")
        chunks = [
            chunk
            for page in pages
            for chunk in chunk_page(page, self.settings.chunk_words, self.settings.chunk_overlap)
        ]
        embeddings = self.text_encoder.encode([chunk.text for chunk in chunks])
        visual = {}
        if self.visual_encoder:
            for page in pages:
                if page.image_path:
                    visual[page.id] = self.visual_encoder.encode_image(page.image_path)
        self.repository.save_document(document_id, name, pages, chunks, embeddings, visual)
        return {
            "document_id": document_id,
            "pages": len(pages),
            "chunks": len(chunks),
            "visual_pages": len(visual),
        }

    def ingest_pdf(self, path: Path, name: str | None = None):
        path = Path(path)
        document_id = hashlib.sha256(path.read_bytes()).hexdigest()[:24]
        output = self.settings.data_dir / "pages" / document_id
        pages = render_pdf(path, output, document_id, self.settings.max_pages)
        if self.settings.parser == "docling":
            from rag.ingestion.docling_parser import enrich_pages

            pages = enrich_pages(path, pages, output)
        return self.ingest_pages(document_id, name or path.name, pages)
