
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename TEXT NOT NULL,
    source_type VARCHAR(20) NOT NULL
        CHECK (source_type IN ('pdf', 'docx', 'image', 'csv', 'url', 'text')),
    content_hash TEXT UNIQUE NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    pipeline_id UUID,
    status VARCHAR(20) NOT NULL DEFAULT 'processing'
        CHECK (status IN ('processing', 'completed', 'failed')),
    chunk_count INT CHECK (chunk_count IS NULL OR chunk_count >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1536),
    chunk_index INT NOT NULL CHECK (chunk_index >= 0),
    token_count INT NOT NULL CHECK (token_count >= 0),
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx
    ON chunks (document_id);

CREATE INDEX IF NOT EXISTS chunks_content_fts_idx
    ON chunks USING gin (to_tsvector('english', content));

CREATE TABLE IF NOT EXISTS pipelines (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT UNIQUE NOT NULL,
    config JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pipeline_id UUID NOT NULL REFERENCES pipelines(id),
    query TEXT NOT NULL,
    retrieved_chunk_ids UUID[],
    generation TEXT,
    latency_ms INT CHECK (latency_ms IS NULL OR latency_ms >= 0),
    input_tokens INT CHECK (input_tokens IS NULL OR input_tokens >= 0),
    output_tokens INT CHECK (output_tokens IS NULL OR output_tokens >= 0),
    model_used TEXT,
    status VARCHAR(20) NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'failed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS evaluations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL REFERENCES pipeline_runs(id),
    faithfulness FLOAT CHECK (faithfulness BETWEEN 0 AND 1),
    answer_relevance FLOAT CHECK (answer_relevance BETWEEN 0 AND 1),
    context_precision FLOAT CHECK (context_precision BETWEEN 0 AND 1),
    context_recall FLOAT CHECK (context_recall BETWEEN 0 AND 1),
    overall_score FLOAT CHECK (overall_score BETWEEN 0 AND 1),
    judge_model TEXT,
    user_rating INT CHECK (user_rating BETWEEN 1 AND 5),
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS finetune_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    provider_job_id TEXT,
    base_model TEXT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    training_pair_count INT CHECK (
        training_pair_count IS NULL OR training_pair_count >= 0
    ),
    mlflow_run_id TEXT,
    metrics JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS training_pairs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL REFERENCES pipeline_runs(id),
    system_prompt TEXT,
    user_message TEXT NOT NULL,
    assistant_message TEXT NOT NULL,
    quality_score FLOAT CHECK (quality_score BETWEEN 0 AND 1),
    included_in_job UUID REFERENCES finetune_jobs(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS documents_pipeline_id_idx ON documents(pipeline_id);
CREATE INDEX IF NOT EXISTS pipeline_runs_pipeline_id_idx ON pipeline_runs(pipeline_id);
CREATE INDEX IF NOT EXISTS evaluations_run_id_idx ON evaluations(run_id);
CREATE INDEX IF NOT EXISTS training_pairs_run_id_idx ON training_pairs(run_id);
