import pytest

from rag.config import Settings
from rag.db.models import Page
from rag.service import RAGService


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path / "data", chunk_words=20, chunk_overlap=4)


@pytest.fixture
def service(settings):
    return RAGService(settings)


@pytest.fixture
def pages():
    return [
        Page("doc:p1", "doc", 1, "# Revenue\nAnnual revenue grew to 42 million euros."),
        Page("doc:p2", "doc", 2, "# Staffing\nThe engineering team has twelve employees."),
    ]


@pytest.fixture
def indexed(service, pages):
    service.pipeline.ingest_pages("doc", "Annual report", pages)
    return service


@pytest.fixture
def pdf_bytes():
    """A real one-page text PDF, generated without additional runtime dependencies."""
    stream = b"BT /F1 18 Tf 50 700 Td (Annual revenue grew to 42 million euros.) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
    ]
    data = b"%PDF-1.4\n"
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(data)
    data += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode()
    for offset in offsets[1:]:
        data += f"{offset:010d} 00000 n \n".encode()
    data += (
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    return data
