-- Add prompt, citations, and error_message to pipeline_runs table for Task 6 RAG generation tracking.
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS prompt TEXT;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS citations JSONB;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS error_message TEXT;
