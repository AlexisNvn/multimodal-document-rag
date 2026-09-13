from pathlib import Path

import pytest

from rag.db.models import Page
from rag.db.repositories import Repository
from rag.service import RAGService

pytestmark = pytest.mark.integration


def test_index_persists_and_reingestion_is_idempotent(indexed, settings, pages):
    indexed.pipeline.ingest_pages("doc", "Annual report", pages)
    restarted = RAGService(settings)
    assert len(restarted.repository.pages()) == 2
    assert len(restarted.repository.chunks()) == 2
    assert restarted.retrieve("revenue")[0].page.id == "doc:p1"
    assert restarted.repository.delete("doc")
    assert not restarted.repository.pages()
    assert not restarted.repository.chunks()


def test_failed_replace_rolls_back(indexed):
    with pytest.raises(ValueError, match="another document"):
        indexed.repository.save_document("doc", "broken", [Page("p", "other", 1, "")], [], [])
    assert len(indexed.repository.pages()) == 2


def test_embedding_configuration_mismatch_fails(indexed, settings):
    settings.chunk_words = 30
    with pytest.raises(ValueError, match="configuration changed"):
        RAGService(settings)


def test_migration_can_run_repeatedly(tmp_path):
    path = tmp_path / "test.sqlite3"
    Repository(path)
    repository = Repository(path)
    with repository.connect() as conn:
        assert conn.execute("SELECT version FROM schema_versions").fetchone()[0] == 1


def test_packaged_schema_matches_reviewable_migration(tmp_path):
    import sqlite3

    repository = Repository(tmp_path / "packaged.sqlite3")
    migration = Path(__file__).resolve().parents[2] / "migrations/001_initial.sql"
    with repository.connect() as packaged, sqlite3.connect(":memory:") as migrated:
        migrated.executescript(migration.read_text())
        query = "SELECT type, name, tbl_name FROM sqlite_master ORDER BY type, name"
        assert [tuple(row) for row in packaged.execute(query)] == migrated.execute(query).fetchall()
        for table in ("documents", "pages", "chunks", "visual_vectors", "index_config"):
            assert [tuple(row) for row in packaged.execute(f"PRAGMA table_info({table})")] == (
                migrated.execute(f"PRAGMA table_info({table})").fetchall()
            )
