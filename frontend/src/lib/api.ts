import axios, { AxiosError } from "axios";
import type {
  AnalyticsResponse,
  ChunkItem,
  DocumentItem,
  EvaluationItem,
  PipelineCreateRequest,
  PipelineResponse,
  PipelineUpdateRequest,
  QueryRequest,
  QueryResponse,
  RatingResponse,
  StreamInitResponse,
} from "@/types/api";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 30000,
});

apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError<{ detail?: string; error?: string }>) => {
    const message =
      error.response?.data?.detail ||
      error.response?.data?.error ||
      error.message ||
      "An unexpected error occurred.";
    return Promise.reject(new Error(message));
  }
);

// ── Pipeline Endpoints ────────────────────────────────────────────────────────

export async function fetchPipelines(): Promise<PipelineResponse[]> {
  const res = await apiClient.get<PipelineResponse[]>("/pipelines");
  return res.data;
}

export async function fetchPipeline(id: string): Promise<PipelineResponse> {
  const res = await apiClient.get<PipelineResponse>(`/pipelines/${id}`);
  return res.data;
}

export async function createPipeline(payload: PipelineCreateRequest): Promise<PipelineResponse> {
  const res = await apiClient.post<PipelineResponse>("/pipelines", payload);
  return res.data;
}

export async function updatePipeline(
  id: string,
  payload: PipelineUpdateRequest
): Promise<PipelineResponse> {
  const res = await apiClient.patch<PipelineResponse>(`/pipelines/${id}`, payload);
  return res.data;
}

export async function deletePipeline(id: string): Promise<{ id: string; status: string }> {
  const res = await apiClient.delete<{ id: string; status: string }>(`/pipelines/${id}`);
  return res.data;
}

export async function fetchPipelineAnalytics(id: string): Promise<AnalyticsResponse> {
  const res = await apiClient.get<AnalyticsResponse>(`/pipelines/${id}/analytics`);
  return res.data;
}

// ── Query & Stream Endpoints ──────────────────────────────────────────────────

export async function postQuery(payload: QueryRequest): Promise<QueryResponse> {
  const res = await apiClient.post<QueryResponse>("/query", {
    ...payload,
    stream: false,
  });
  return res.data;
}

export async function initQueryStream(payload: QueryRequest): Promise<StreamInitResponse> {
  const res = await apiClient.post<StreamInitResponse>("/query", {
    ...payload,
    stream: true,
  });
  return res.data;
}

export function getStreamUrl(runId: string): string {
  return `${API_BASE_URL}/query/${runId}/stream`;
}

export function getEvaluationsStreamUrl(): string {
  return `${API_BASE_URL}/evaluations/stream`;
}

// ── Evaluations & Ratings ─────────────────────────────────────────────────────

export async function submitRating(
  runId: string,
  rating: number,
  pipelineId?: string
): Promise<RatingResponse> {
  const res = await apiClient.patch<RatingResponse>(`/runs/${runId}/rating`, {
    rating,
    pipeline_id: pipelineId,
  });
  return res.data;
}

export async function fetchRunEvaluation(
  runId: string,
  pipelineId?: string
): Promise<EvaluationItem> {
  const params = pipelineId ? { pipeline_id: pipelineId } : undefined;
  const res = await apiClient.get<EvaluationItem>(`/runs/${runId}/evaluation`, { params });
  return res.data;
}

export async function fetchEvaluations(
  pipelineId?: string,
  limit: number = 50
): Promise<EvaluationItem[]> {
  const params: Record<string, unknown> = { limit };
  if (pipelineId) params.pipeline_id = pipelineId;
  const res = await apiClient.get<EvaluationItem[]>("/evaluations", { params });
  return res.data;
}

// ── Documents & Ingestion ─────────────────────────────────────────────────────

export async function fetchDocuments(
  pipelineId: string,
  limit: number = 100,
  offset: number = 0
): Promise<DocumentItem[]> {
  const res = await apiClient.get<DocumentItem[]>("/documents", {
    params: { pipeline_id: pipelineId, limit, offset },
  });
  return res.data;
}

export async function fetchDocument(documentId: string, pipelineId: string): Promise<DocumentItem> {
  const res = await apiClient.get<DocumentItem>(`/documents/${documentId}`, {
    params: { pipeline_id: pipelineId },
  });
  return res.data;
}

export async function uploadDocument(
  pipelineId: string,
  file: File,
  onProgress?: (percent: number) => void
): Promise<{ document_id: string; status: string; duplicate?: boolean }> {
  const formData = new FormData();
  formData.append("pipeline_id", pipelineId);
  formData.append("file", file);

  const res = await apiClient.post("/ingest", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
    onUploadProgress: (progressEvent) => {
      if (progressEvent.total && onProgress) {
        const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onProgress(percent);
      }
    },
  });
  return res.data;
}

export async function ingestUrl(
  pipelineId: string,
  url: string
): Promise<{ document_id: string; status: string; duplicate?: boolean }> {
  const res = await apiClient.post("/ingest/url", {
    pipeline_id: pipelineId,
    url,
  });
  return res.data;
}

export async function fetchDocumentChunks(
  documentId: string,
  pipelineId?: string
): Promise<ChunkItem[]> {
  const res = await apiClient.get<ChunkItem[]>(`/documents/${documentId}/chunks`, {
    params: pipelineId ? { pipeline_id: pipelineId } : undefined,
  });
  return res.data;
}

export async function fetchChunk(chunkId: string): Promise<ChunkItem> {
  const res = await apiClient.get<ChunkItem>(`/chunks/${chunkId}`);
  return res.data;
}

export async function retrieveSimilarChunks(
  query: string,
  pipelineId: string,
  topK: number = 5
): Promise<{ chunks: ChunkItem[] }> {
  const res = await apiClient.post("/retrieve", {
    query,
    pipeline_id: pipelineId,
    top_k: topK,
    rerank: true,
  });
  return res.data;
}
