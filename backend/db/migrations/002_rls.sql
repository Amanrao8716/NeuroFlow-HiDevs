
-- Enable and force Row Level Security on tenant-scoped tables.
-- The application must set app.pipeline_id for each transaction.
-- Missing or invalid context returns no tenant rows.

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents FORCE ROW LEVEL SECURITY;

ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks FORCE ROW LEVEL SECURITY;

ALTER TABLE pipelines ENABLE ROW LEVEL SECURITY;
ALTER TABLE pipelines FORCE ROW LEVEL SECURITY;

ALTER TABLE pipeline_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE pipeline_runs FORCE ROW LEVEL SECURITY;

ALTER TABLE evaluations ENABLE ROW LEVEL SECURITY;
ALTER TABLE evaluations FORCE ROW LEVEL SECURITY;

ALTER TABLE training_pairs ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_pairs FORCE ROW LEVEL SECURITY;

ALTER TABLE finetune_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE finetune_jobs FORCE ROW LEVEL SECURITY;

-- Documents: pipeline_id must match the transaction's tenant context.
CREATE POLICY documents_pipeline_isolation ON documents
    USING (
        pipeline_id = NULLIF(current_setting('app.pipeline_id', true), '')::UUID
    )
    WITH CHECK (
        pipeline_id = NULLIF(current_setting('app.pipeline_id', true), '')::UUID
    );

-- Chunks inherit tenant ownership through their parent document.
CREATE POLICY chunks_pipeline_isolation ON chunks
    USING (
        EXISTS (
            SELECT 1
            FROM documents d
            WHERE d.id = chunks.document_id
              AND d.pipeline_id =
                  NULLIF(current_setting('app.pipeline_id', true), '')::UUID
        )
    )
    WITH CHECK (
        EXISTS (
            SELECT 1
            FROM documents d
            WHERE d.id = chunks.document_id
              AND d.pipeline_id =
                  NULLIF(current_setting('app.pipeline_id', true), '')::UUID
        )
    );

CREATE POLICY pipelines_pipeline_isolation ON pipelines
    USING (
        id = NULLIF(current_setting('app.pipeline_id', true), '')::UUID
    )
    WITH CHECK (
        id = NULLIF(current_setting('app.pipeline_id', true), '')::UUID
    );

CREATE POLICY pipeline_runs_pipeline_isolation ON pipeline_runs
    USING (
        pipeline_id =
            NULLIF(current_setting('app.pipeline_id', true), '')::UUID
    )
    WITH CHECK (
        pipeline_id =
            NULLIF(current_setting('app.pipeline_id', true), '')::UUID
    );

CREATE POLICY evaluations_pipeline_isolation ON evaluations
    USING (
        EXISTS (
            SELECT 1
            FROM pipeline_runs r
            WHERE r.id = evaluations.run_id
              AND r.pipeline_id =
                  NULLIF(current_setting('app.pipeline_id', true), '')::UUID
        )
    )
    WITH CHECK (
        EXISTS (
            SELECT 1
            FROM pipeline_runs r
            WHERE r.id = evaluations.run_id
              AND r.pipeline_id =
                  NULLIF(current_setting('app.pipeline_id', true), '')::UUID
        )
    );

CREATE POLICY training_pairs_pipeline_isolation ON training_pairs
    USING (
        EXISTS (
            SELECT 1
            FROM pipeline_runs r
            WHERE r.id = training_pairs.run_id
              AND r.pipeline_id =
                  NULLIF(current_setting('app.pipeline_id', true), '')::UUID
        )
    )
    WITH CHECK (
        EXISTS (
            SELECT 1
            FROM pipeline_runs r
            WHERE r.id = training_pairs.run_id
              AND r.pipeline_id =
                  NULLIF(current_setting('app.pipeline_id', true), '')::UUID
        )
    );

-- A fine-tuning job is visible when its training data belongs to this tenant.
CREATE POLICY finetune_jobs_pipeline_isolation ON finetune_jobs
    USING (
        EXISTS (
            SELECT 1
            FROM training_pairs tp
            JOIN pipeline_runs r ON r.id = tp.run_id
            WHERE tp.included_in_job = finetune_jobs.id
              AND r.pipeline_id =
                  NULLIF(current_setting('app.pipeline_id', true), '')::UUID
        )
    )
    WITH CHECK (
        EXISTS (
            SELECT 1
            FROM training_pairs tp
            JOIN pipeline_runs r ON r.id = tp.run_id
            WHERE tp.included_in_job = finetune_jobs.id
              AND r.pipeline_id =
                  NULLIF(current_setting('app.pipeline_id', true), '')::UUID
        )
    );
