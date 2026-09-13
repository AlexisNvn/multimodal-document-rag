from dataclasses import asdict
from typing import Literal

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    k: int = Field(default=5, ge=1, le=100)
    mode: Literal["hybrid", "dense", "sparse", "visual"] = "hybrid"
    document_id: str | None = None

    @field_validator("query")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("Query cannot be blank")
        return value.strip()


def public_hit(hit):
    return {
        "page_id": hit.page.id,
        "document_id": hit.page.document_id,
        "page_number": hit.page.number,
        "score": hit.score,
        "channels": hit.channels,
        "text": hit.text,
    }


@router.post("/search")
def search(body: QueryRequest, request: Request):
    try:
        hits = request.app.state.service.retrieve(**body.model_dump())
        return {"hits": [public_hit(hit) for hit in hits]}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/query")
def query(body: QueryRequest, request: Request):
    try:
        answer, hits = request.app.state.service.answer(**body.model_dump())
        return {**asdict(answer), "hits": [public_hit(hit) for hit in hits]}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (httpx.HTTPError, KeyError, TypeError) as exc:
        raise HTTPException(502, "Answer provider returned an invalid response") from exc
