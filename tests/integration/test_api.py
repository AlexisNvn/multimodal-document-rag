import pytest
from fastapi.testclient import TestClient

from rag.api.app import create_app
from rag.db.models import Page

pytestmark = pytest.mark.integration


def test_delete_namespaced_dataset_document(settings):
    with TestClient(create_app(settings)) as client:
        document_id = "vidore/test:book"
        client.app.state.service.pipeline.ingest_pages(
            document_id, "book", [Page("p", document_id, 1, "Revenue")]
        )
        assert client.delete(f"/documents/{document_id}").status_code == 204


def test_upload_query_restart_delete(settings, pdf_bytes):
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json()["status"] == "ok"
        response = client.post("/documents", files={"file": ("report.pdf", pdf_bytes)})
        assert response.status_code == 201, response.text
        document_id = response.json()["document_id"]
        duplicate = client.post("/documents", files={"file": ("report.pdf", pdf_bytes)})
        assert duplicate.json()["document_id"] == document_id
        result = client.post("/query", json={"query": "revenue", "mode": "sparse"}).json()
        assert "42 million" in result["answer"]
        assert result["citations"][0]["document_id"] == document_id
        assert "image_path" not in result["hits"][0]
    with TestClient(create_app(settings)) as client:
        assert len(client.get("/documents").json()) == 1
        assert client.delete(f"/documents/{document_id}").status_code == 204
        assert client.post("/query", json={"query": "revenue"}).json()["abstained"]
        assert client.delete(f"/documents/{document_id}").status_code == 404


@pytest.mark.parametrize(
    "body", [{"query": " "}, {"query": "x", "k": 0}, {"query": "x", "mode": "unknown"}]
)
def test_query_validation(settings, body):
    with TestClient(create_app(settings)) as client:
        assert client.post("/query", json=body).status_code == 422


def test_upload_validation_and_disabled_visual(settings):
    settings.max_upload_mb = 1
    with TestClient(create_app(settings)) as client:
        assert client.post("/documents", files={"file": ("fake.pdf", b"no")}).status_code == 415
        assert client.post("/documents", files={"file": ("empty.pdf", b"")}).status_code == 400
        assert (
            client.post(
                "/documents", files={"file": ("big.pdf", b"%PDF-" + b"0" * (1024 * 1024))}
            ).status_code
            == 413
        )
        assert client.post("/search", json={"query": "chart", "mode": "visual"}).status_code == 422
