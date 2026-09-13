FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH" RAG_DATA_DIR=/app/data
RUN useradd --create-home rag && mkdir /app/data && chown rag:rag /app/data
USER rag
EXPOSE 8000
CMD ["uvicorn", "rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
