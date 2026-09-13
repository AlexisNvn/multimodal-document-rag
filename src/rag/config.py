from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")
    data_dir: Path = Path("data")
    text_backend: Literal["hash", "sentence-transformers"] = "hash"
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    visual_backend: Literal["none", "colqwen2"] = "none"
    visual_model: str = "vidore/colqwen2-v1.0"
    device: str = "cpu"
    parser: Literal["pdfium", "docling"] = "pdfium"
    reranker_model: str = ""
    answer_backend: Literal["extractive", "chat"] = "extractive"
    llm_base_url: str = "http://localhost:8001/v1"
    llm_model: str = ""
    llm_api_key: str = ""
    max_upload_mb: int = Field(default=25, ge=1)
    max_pages: int = Field(default=100, ge=1)
    chunk_words: int = Field(default=180, ge=10)
    chunk_overlap: int = Field(default=30, ge=0)
    context_chars: int = Field(default=16000, ge=100)

    @model_validator(mode="after")
    def valid_overlap(self):
        if self.chunk_overlap >= self.chunk_words:
            raise ValueError("chunk_overlap must be smaller than chunk_words")
        return self
