
-- Deduplicate content within each pipeline, not globally.
ALTER TABLE documents
    DROP CONSTRAINT IF EXISTS documents_content_hash_key;

CREATE UNIQUE INDEX IF NOT EXISTS documents_pipeline_content_hash_uidx
    ON documents (pipeline_id, content_hash);
