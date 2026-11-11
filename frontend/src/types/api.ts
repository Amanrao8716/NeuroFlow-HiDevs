/**
 * API Type Definitions for NeuroFlow RAG Platform
 * Aligned with backend Pydantic models in backend/models/pipeline.py and API endpoints.
 */

export type ChunkingStrategy = "auto" | "fixed" | "semantic" | "hierarchical";
export type RerankerType = "cross-encoder" | "none" | "llm";
export type SystemPromptVariant = "default" | "concise" | "detailed" | "technical";
export type ExtractorType = "pdf" | "docx" | "csv" | "image" | "url" | "text";

export interface IngestionConfig {
  chunking_strategy?: ChunkingStrategy;
  chunk_size_tokens?: number;
  chunk_overlap_tokens?: number;
  extractors_enabled?: ExtractorType[];
}

export interface RetrievalConfig {
  dense_k?: number;
  sparse_k?: number;
  reranker?: RerankerType;
  top_k_after_rerank?: number;
  query_expansion?: boolean;
  metadata_filters_enabled?: boolean;
}

export interface GenerationConfig {
  model_routing?: string;
  max_context_tokens?: number;
  temperature?: number;
  system_prompt_variant?: SystemPromptVariant;
}

export interface EvaluationConfig {
  auto_evaluate?: boolean;
  training_threshold?: number;
}

export interface PipelineConfig {
  name: string;
  description?: string | null;
  ingestion?: IngestionConfig;
  retrieval?: RetrievalConfig;
  generation?: GenerationConfig;
  evaluation?: EvaluationConfig;
}

export interface PipelineLatestMetrics {
  run_count?: number;
  last_run_at?: string | null;
  run_count_7d?: number;
  avg_overall_score?: number | null;
  daily_scores_7d?: { date: string; score: number | null }[];
}

export interface PipelineResponse {
  id: string;
  name: string;
  description: string | null;
  version: number;
  status: string;
  config: PipelineConfig;
  created_at: string;
  updated_at: string;
  aggregate_scores?: {
    faithfulness?: number | null;
    answer_relevance?: number | null;
    context_precision?: number | null;
    context_recall?: number | null;
    overall_score?: number | null;
  } | null;
  latest_metrics?: PipelineLatestMetrics | null;
}

export interface PipelineCreateRequest {
  name?: string;
  description?: string | null;
  config?: PipelineConfig;
  ingestion?: IngestionConfig;
  retrieval?: RetrievalConfig;
  generation?: GenerationConfig;
  evaluation?: EvaluationConfig;
}

export interface PipelineUpdateRequest {
  name?: string;
  description?: string | null;
  config_updates?: Partial<PipelineConfig>;
  ingestion?: Partial<IngestionConfig>;
  retrieval?: Partial<RetrievalConfig>;
  generation?: Partial<GenerationConfig>;
  evaluation?: Partial<EvaluationConfig>;
}

export interface DailyQueryCount {
  date: string;
  count: number;
}

export interface AnalyticsResponse {
  pipeline_id: string;
  version: number;
  retrieval_latency_p50: number | null;
  retrieval_latency_p95: number | null;
  retrieval_latency_p99: number | null;
  avg_generation_latency_ms: number | null;
  avg_evaluation_scores: {
    faithfulness: number | null;
    answer_relevance: number | null;
    context_precision: number | null;
    context_recall: number | null;
    overall_score: number | null;
  };
  estimated_cost_per_query: number | null;
  daily_query_counts: DailyQueryCount[];
  total_queries_30d: number;
  percentile_method: string;
  cost_trends_30d?: { date: string; cost_per_query: number | null; total_cost: number | null }[];
  recent_failed_runs?: {
    run_id: string;
    query: string;
    error_message: string;
    latency_ms: number | null;
    created_at: string | null;
  }[];
  daily_scores_7d?: { date: string; score: number | null }[];
}

export interface CitationItem {
  reference: string;
  source_number: number;
  chunk_id: string | null;
  document_name: string | null;
  page_number: number | null;
  content_preview: string | null;
  is_valid: boolean;
  error?: string | null;
}

export interface RetrievedSource {
  source_index: number;
  chunk_id: string;
  document_id: string;
  filename: string;
  page_number?: number | null;
  score: number;
  content?: string;
}

export interface RetrievalMetadata {
  query: string;
  query_type?: string;
  dense_count: number;
  sparse_count: number;
  meta_count: number;
  fused_count: number;
  reranked_count: number;
  top_candidates_count: number;
  context_chunks_used: number;
  total_tokens: number;
}

export interface QueryRequest {
  query: string;
  pipeline_id: string;
  stream?: boolean;
  top_k?: number;
  max_context_tokens?: number;
  rerank?: boolean;
  pipeline_version?: number | null;
}

export interface QueryResponse {
  run_id: string;
  query: string;
  query_type: string;
  answer: string;
  citations: CitationItem[];
  sources: RetrievedSource[];
  chunks_used: number;
  input_tokens: number;
  output_tokens: number;
  latency_ms: number;
  model_used: string;
  status: string;
}

export interface StreamInitResponse {
  run_id: string;
  stream_url: string;
  status: string;
}

export interface EvaluationItem {
  eval_id: string;
  run_id: string;
  pipeline_id: string;
  pipeline_name?: string;
  query?: string;
  answer?: string;
  retrieved_chunk_ids?: string[];
  faithfulness: number | null;
  answer_relevance: number | null;
  context_precision: number | null;
  context_recall: number | null;
  overall_score: number | null;
  judge_model?: string | null;
  user_rating?: number | null;
  evaluated_at?: string | null;
}

export interface RatingRequest {
  rating: number;
  pipeline_id?: string | null;
}

export interface RatingResponse {
  run_id: string;
  user_rating: number;
  normalized_rating: number;
  overall_score: number | null;
  calibration_needed: boolean;
  discrepancy: number | null;
  faithfulness?: number | null;
  answer_relevance?: number | null;
  context_precision?: number | null;
  context_recall?: number | null;
  message: string;
}

export interface DocumentItem {
  id: string;
  document_id: string;
  filename: string;
  source_type: string;
  status: "queued" | "processing" | "completed" | "failed" | string;
  chunk_count: number;
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface ChunkItem {
  chunk_id: string;
  document_id: string;
  content: string;
  chunk_index: number;
  token_count: number;
  metadata: Record<string, unknown>;
  filename?: string;
  source_type?: string;
  created_at?: string | null;
  score?: number;
}
