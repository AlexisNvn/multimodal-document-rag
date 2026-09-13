import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)


@router.post("", status_code=201)
def upload_document(request: Request, file: UploadFile):
    service = request.app.state.service
    limit = service.settings.max_upload_mb * 1024 * 1024
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
            temp_path = Path(temp.name)
            total = 0
            while block := file.file.read(1024 * 1024):
                if total == 0 and not block.startswith(b"%PDF-"):
                    raise HTTPException(415, "Upload a PDF file")
                total += len(block)
                if total > limit:
                    raise HTTPException(413, "PDF exceeds upload limit")
                temp.write(block)
        if total == 0:
            raise HTTPException(400, "Empty upload")
        with service.lock:
            return service.pipeline.ingest_pdf(
                temp_path, Path(file.filename or "document.pdf").name
            )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("Document ingestion failed")
        raise HTTPException(422, "PDF could not be ingested; inspect server logs") from exc
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)
        file.file.close()


@router.get("")
def list_documents(request: Request):
    return request.app.state.service.repository.documents()


@router.delete("/{document_id:path}", status_code=204)
def delete_document(document_id: str, request: Request):
    service = request.app.state.service
    with service.lock:
        if not service.repository.delete(document_id):
            raise HTTPException(404, "Document not found")
