-- Add GIN indexes on chunks and documents metadata for fast JSONB containment queries.
CREATE INDEX IF NOT EXISTS chunks_metadata_gin_idx
    ON chunks USING gin (metadata jsonb_path_ops);

CREATE INDEX IF NOT EXISTS documents_metadata_gin_idx
    ON documents USING gin (metadata jsonb_path_ops);
