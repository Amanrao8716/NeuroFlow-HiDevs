-- Migration 009: Pipeline configuration versioning, analytics, and run metadata

-- 1. Extend pipelines table with version, status, description, updated_at
ALTER TABLE pipelines ADD COLUMN IF NOT EXISTS description TEXT;
ALTER TABLE pipelines ADD COLUMN IF NOT EXISTS version INT NOT NULL DEFAULT 1;
ALTER TABLE pipelines ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'active'
    CHECK (status IN ('active', 'archived'));
ALTER TABLE pipelines ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

-- 2. Create immutable pipeline_versions table
CREATE TABLE IF NOT EXISTS pipeline_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pipeline_id UUID NOT NULL REFERENCES pipelines(id) ON DELETE CASCADE,
    version INT NOT NULL CHECK (version >= 1),
    config JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (pipeline_id, version)
);

CREATE INDEX IF NOT EXISTS pipeline_versions_pipeline_id_idx ON pipeline_versions(pipeline_id);

-- 3. Extend pipeline_runs table with version, retrieval_latency_ms, cost_usd, execution_metadata
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS pipeline_version INT;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS retrieval_latency_ms INT
    CHECK (retrieval_latency_ms IS NULL OR retrieval_latency_ms >= 0);
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS cost_usd FLOAT
    CHECK (cost_usd IS NULL OR cost_usd >= 0);
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS execution_metadata JSONB DEFAULT '{}';

-- 4. Enable Row-Level Security on pipeline_versions
ALTER TABLE pipeline_versions ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE policyname = 'pipeline_versions_pipeline_isolation'
    ) THEN
        CREATE POLICY pipeline_versions_pipeline_isolation ON pipeline_versions
            USING (
                pipeline_id = NULLIF(current_setting('app.pipeline_id', true), '')::UUID
            )
            WITH CHECK (
                pipeline_id = NULLIF(current_setting('app.pipeline_id', true), '')::UUID
            );
    END IF;
END $$;

GRANT ALL ON TABLE pipeline_versions TO neuroflow_app;
