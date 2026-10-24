-- Update chunks.embedding dimension to 384 for sentence-transformers/all-MiniLM-L6-v2
DROP INDEX IF EXISTS chunks_embedding_hnsw_idx;

ALTER TABLE chunks
    ALTER COLUMN embedding TYPE vector(384);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
