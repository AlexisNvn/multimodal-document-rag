from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag.api import routes_documents, routes_health, routes_query
from rag.config import Settings
from rag.service import RAGService


def create_app(settings: Settings | None = None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.service = RAGService(settings or Settings())
        yield

    app = FastAPI(title="Multimodal Document RAG", version="0.1.0", lifespan=lifespan)
    for router in (routes_health.router, routes_documents.router, routes_query.router):
        app.include_router(router)
    return app


app = create_app()
