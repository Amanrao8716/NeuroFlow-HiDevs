CREATE TABLE IF NOT EXISTS pipeline_quality_anomalies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pipeline_id UUID NOT NULL REFERENCES pipelines(id) ON DELETE CASCADE,
    run_id UUID NOT NULL REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    rule_id TEXT NOT NULL,
    latest_score DOUBLE PRECISION NOT NULL CHECK (latest_score BETWEEN 0 AND 1),
    baseline_mean DOUBLE PRECISION NOT NULL CHECK (baseline_mean BETWEEN 0 AND 1),
    baseline_stddev DOUBLE PRECISION NOT NULL CHECK (baseline_stddev >= 0),
    suggestions JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (pipeline_id, run_id, rule_id)
);

CREATE INDEX IF NOT EXISTS pipeline_quality_anomalies_recent_idx
    ON pipeline_quality_anomalies (pipeline_id, created_at DESC);

ALTER TABLE pipeline_quality_anomalies ENABLE ROW LEVEL SECURITY;
ALTER TABLE pipeline_quality_anomalies FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS pipeline_quality_anomalies_pipeline_isolation
    ON pipeline_quality_anomalies;
CREATE POLICY pipeline_quality_anomalies_pipeline_isolation
    ON pipeline_quality_anomalies
    USING (
        pipeline_id =
            NULLIF(current_setting('app.pipeline_id', true), '')::UUID
    )
    WITH CHECK (
        pipeline_id =
            NULLIF(current_setting('app.pipeline_id', true), '')::UUID
    );

GRANT SELECT, INSERT ON TABLE pipeline_quality_anomalies TO neuroflow_app;
