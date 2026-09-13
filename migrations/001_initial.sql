CREATE TABLE IF NOT EXISTS schema_versions (version INTEGER PRIMARY KEY);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS pages (
    id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    number INTEGER NOT NULL CHECK(number > 0), text TEXT NOT NULL,
    image_path TEXT, metadata TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY, page_id TEXT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    text TEXT NOT NULL, parent_text TEXT NOT NULL, heading TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS visual_vectors (
    page_id TEXT PRIMARY KEY REFERENCES pages(id) ON DELETE CASCADE, embedding BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS index_config (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS pages_document ON pages(document_id);
CREATE INDEX IF NOT EXISTS chunks_page ON chunks(page_id);
INSERT OR IGNORE INTO schema_versions(version) VALUES (1);
