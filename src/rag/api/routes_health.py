from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health(request: Request):
    service = request.app.state.service
    with service.repository.connect() as conn:
        conn.execute("SELECT 1")
    return {
        "status": "ok",
        "text_backend": service.settings.text_backend,
        "visual_backend": service.settings.visual_backend,
        "answer_backend": service.settings.answer_backend,
    }
