-- Migration 010: Fine-tuning jobs schema enhancements and indexes

-- 1. Ensure pipeline_runs has user_rating column if not already present
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'pipeline_runs' AND column_name = 'user_rating'
    ) THEN
        ALTER TABLE pipeline_runs ADD COLUMN user_rating INT CHECK (user_rating BETWEEN 1 AND 5);
    END IF;
END $$;

-- 2. Update finetune_jobs status check constraint to include 'succeeded'
ALTER TABLE finetune_jobs DROP CONSTRAINT IF EXISTS finetune_jobs_status_check;
ALTER TABLE finetune_jobs ADD CONSTRAINT finetune_jobs_status_check
    CHECK (status IN ('pending', 'running', 'completed', 'succeeded', 'failed', 'cancelled'));

-- 3. Add fine-tuning details columns to finetune_jobs
ALTER TABLE finetune_jobs ADD COLUMN IF NOT EXISTS fine_tuned_model TEXT;
ALTER TABLE finetune_jobs ADD COLUMN IF NOT EXISTS trained_tokens INT;
ALTER TABLE finetune_jobs ADD COLUMN IF NOT EXISTS error_message TEXT;
ALTER TABLE finetune_jobs ADD COLUMN IF NOT EXISTS extraction_report JSONB;
ALTER TABLE finetune_jobs ADD COLUMN IF NOT EXISTS pipeline_id UUID REFERENCES pipelines(id) ON DELETE SET NULL;

-- 4. Create performance indexes
CREATE INDEX IF NOT EXISTS training_pairs_included_in_job_idx ON training_pairs(included_in_job);
CREATE INDEX IF NOT EXISTS training_pairs_quality_score_idx ON training_pairs(quality_score);
CREATE INDEX IF NOT EXISTS finetune_jobs_provider_job_id_idx ON finetune_jobs(provider_job_id);
CREATE INDEX IF NOT EXISTS finetune_jobs_status_idx ON finetune_jobs(status);
