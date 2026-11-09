"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import type { CitationItem, RetrievalMetadata, RetrievedSource } from "@/types/api";

export type StreamStatus =
  | "idle"
  | "connecting"
  | "retrieving"
  | "streaming"
  | "completed"
  | "error";

export interface SSEStreamState {
  status: StreamStatus;
  tokens: string;
  sources: RetrievedSource[];
  citations: CitationItem[];
  retrievalMetadata: RetrievalMetadata | null;
  error: string | null;
  startStream: (runId: string) => void;
  stopStream: () => void;
  resetStream: () => void;
}

export function useSSEStream(initialRunId?: string | null): SSEStreamState {
  const [currentRunId, setCurrentRunId] = useState<string | null>(initialRunId || null);
  const [status, setStatus] = useState<StreamStatus>("idle");
  const [tokens, setTokens] = useState<string>("");
  const [sources, setSources] = useState<RetrievedSource[]>([]);
  const [citations, setCitations] = useState<CitationItem[]>([]);
  const [retrievalMetadata, setRetrievalMetadata] = useState<RetrievalMetadata | null>(null);
  const [error, setError] = useState<string | null>(null);

  const eventSourceRef = useRef<EventSource | null>(null);

  const stopStream = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  }, []);

  const resetStream = useCallback(() => {
    stopStream();
    setCurrentRunId(null);
    setStatus("idle");
    setTokens("");
    setSources([]);
    setCitations([]);
    setRetrievalMetadata(null);
    setError(null);
  }, [stopStream]);

  const startStream = useCallback(
    (runId: string) => {
      resetStream();
      setCurrentRunId(runId);
    },
    [resetStream]
  );

  useEffect(() => {
    if (!currentRunId) return;

    const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const streamUrl = `${apiBaseUrl}/query/${currentRunId}/stream`;

    setStatus("connecting");
    setError(null);

    const es = new EventSource(streamUrl);
    eventSourceRef.current = es;

    es.onopen = () => {
      setStatus("retrieving");
    };

    es.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        const eventType = payload.type;

        if (eventType === "retrieval_start") {
          setStatus("retrieving");
        } else if (eventType === "retrieval_complete") {
          if (Array.isArray(payload.sources)) {
            const validSources = payload.sources.filter(
              (source: unknown): source is RetrievedSource =>
                typeof source === "object" &&
                source !== null &&
                "source_index" in source &&
                typeof source.source_index === "number" &&
                "chunk_id" in source &&
                typeof source.chunk_id === "string" &&
                "document_id" in source &&
                typeof source.document_id === "string" &&
                "filename" in source &&
                typeof source.filename === "string" &&
                "score" in source &&
                typeof source.score === "number"
            );
            setSources(validSources);
          }
          if (payload.retrieval_metadata) {
            setRetrievalMetadata(payload.retrieval_metadata);
          }
          setStatus("streaming");
        } else if (eventType === "token") {
          setStatus("streaming");
          if (payload.delta) {
            setTokens((prev) => prev + payload.delta);
          }
        } else if (eventType === "done") {
          setStatus("completed");
          if (payload.citations) {
            setCitations(payload.citations);
          }
          es.close();
          eventSourceRef.current = null;
        } else if (eventType === "error") {
          setStatus("error");
          setError(payload.error || "An error occurred during query generation.");
          es.close();
          eventSourceRef.current = null;
        }
      } catch (err: any) {
        console.error("Failed to parse SSE event data:", err);
      }
    };

    es.onerror = () => {
      // EventSource readyState: 2 = CLOSED
      if (es.readyState === EventSource.CLOSED) {
        setStatus((prev) => (prev === "streaming" ? "completed" : prev));
      } else {
        setStatus("error");
        setError("Connection to query stream was lost.");
      }
      es.close();
      eventSourceRef.current = null;
    };

    return () => {
      es.close();
      eventSourceRef.current = null;
    };
  }, [currentRunId]);

  return {
    status,
    tokens,
    sources,
    citations,
    retrievalMetadata,
    error,
    startStream,
    stopStream,
    resetStream,
  };
}
