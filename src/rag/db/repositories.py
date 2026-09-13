"""Transactional local persistence. Exact vector scans are intended for portfolio-sized corpora."""

import io
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from rag.db.models import Chunk, Page

# Packaged with the wheel as well as supplied as a reviewable migration.
SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_versions (version INTEGER PRIMARY KEY);
CREATE TABLE IF NOT EXISTS documents (
 id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS pages (
 id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
 number INTEGER NOT NULL CHECK(number > 0), text TEXT NOT NULL, image_path TEXT,
 metadata TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks (
 id TEXT PRIMARY KEY, page_id TEXT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
 text TEXT NOT NULL, parent_text TEXT NOT NULL, heading TEXT NOT NULL, embedding BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS visual_vectors (
 page_id TEXT PRIMARY KEY REFERENCES pages(id) ON DELETE CASCADE, embedding BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS index_config (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS pages_document ON pages(document_id);
CREATE INDEX IF NOT EXISTS chunks_page ON chunks(page_id);
INSERT OR IGNORE INTO schema_versions(version) VALUES (1);
"""


def pack(vector: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(vector, dtype=np.float32), allow_pickle=False)
    return buffer.getvalue()


def unpack(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob), allow_pickle=False)


class Repository:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def configure(self, signature: str):
        with self.connect() as conn:
            conn.execute("INSERT OR IGNORE INTO index_config VALUES ('signature', ?)", (signature,))
            saved = conn.execute("SELECT value FROM index_config WHERE key='signature'").fetchone()
            if saved[0] != signature:
                raise ValueError("Index configuration changed; use a new RAG_DATA_DIR or rebuild.")

    def save_document(self, document_id, name, pages, chunks, embeddings, visual=None):
        if len(chunks) != len(embeddings):
            raise ValueError("One embedding is required per chunk")
        with self.connect() as conn:
            conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
            conn.execute("INSERT INTO documents(id,name) VALUES (?,?)", (document_id, name))
            for page in pages:
                if page.document_id != document_id:
                    raise ValueError("Page belongs to another document")
                conn.execute(
                    "INSERT INTO pages VALUES (?,?,?,?,?,?)",
                    (
                        page.id,
                        document_id,
                        page.number,
                        page.text,
                        page.image_path,
                        json.dumps(page.metadata),
                    ),
                )
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                conn.execute(
                    "INSERT INTO chunks VALUES (?,?,?,?,?,?)",
                    (
                        chunk.id,
                        chunk.page_id,
                        chunk.text,
                        chunk.parent_text,
                        chunk.heading,
                        pack(embedding),
                    ),
                )
            for page_id, vector in (visual or {}).items():
                conn.execute("INSERT INTO visual_vectors VALUES (?,?)", (page_id, pack(vector)))

    def pages(self) -> dict[str, Page]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM pages ORDER BY id").fetchall()
        return {
            r["id"]: Page(
                r["id"],
                r["document_id"],
                r["number"],
                r["text"],
                r["image_path"],
                json.loads(r["metadata"]),
            )
            for r in rows
        }

    def chunks(self):
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM chunks ORDER BY id").fetchall()
        return [
            (
                Chunk(r["id"], r["page_id"], r["text"], r["parent_text"], r["heading"]),
                unpack(r["embedding"]),
            )
            for r in rows
        ]

    def visual_vectors(self):
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM visual_vectors ORDER BY page_id").fetchall()
        return [(r["page_id"], unpack(r["embedding"])) for r in rows]

    def documents(self):
        with self.connect() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT d.*, count(p.id) AS pages FROM documents d LEFT JOIN pages p "
                    "ON d.id=p.document_id GROUP BY d.id ORDER BY d.id"
                )
            ]

    def delete(self, document_id: str) -> bool:
        with self.connect() as conn:
            return conn.execute("DELETE FROM documents WHERE id=?", (document_id,)).rowcount > 0
