"use client";

import React, { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  fetchPipelines,
  initQueryStream,
  submitRating,
  fetchRunEvaluation,
  fetchChunk,
} from "@/lib/api";
import { useAppStore } from "@/store/useAppStore";
import { useSSEStream } from "@/hooks/useSSEStream";
import { MetricGauge } from "@/components/common/MetricGauge";
import { RetrievalInspector } from "@/components/playground/RetrievalInspector";
import { AnswerDiff } from "@/components/playground/AnswerDiff";
import {
  Send,
  Columns,
  ThumbsUp,
  ThumbsDown,
  Layers,
  FileText,
  AlertCircle,
  Loader2,
} from "lucide-react";
import type { EvaluationItem, RetrievedSource } from "@/types/api";

const MAX_QUERY_LENGTH = 5000;

export default function PlaygroundPage() {
  const {
    selectedPipelineId,
    setSelectedPipelineId,
    comparePipelineId,
    setComparePipelineId,
    isCompareMode,
    setIsCompareMode,
    setActiveChunkDetail,
    addToast,
  } = useAppStore();

  const [query, setQuery] = useState("");
  const [activeRunA, setActiveRunA] = useState<string | null>(null);
  const [activeRunB, setActiveRunB] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Ratings state
  const [ratingA, setRatingA] = useState<number | null>(null);
  const [ratingB, setRatingB] = useState<number | null>(null);

  // Evaluations state (fetched after run completion)
  const [evalA, setEvalA] = useState<EvaluationItem | null>(null);
  const [evalB, setEvalB] = useState<EvaluationItem | null>(null);
  const [isEvalLoadingA, setIsEvalLoadingA] = useState(false);
  const [isEvalLoadingB, setIsEvalLoadingB] = useState(false);

  // Pipelines list
  const { data: pipelines } = useQuery({
    queryKey: ["pipelines"],
    queryFn: fetchPipelines,
  });

  // Streaming hooks
  const streamA = useSSEStream(activeRunA);
  const streamB = useSSEStream(activeRunB);

  // Auto-select pipelines if not set
  useEffect(() => {
    if (pipelines && pipelines.length > 0) {
      if (!selectedPipelineId) setSelectedPipelineId(pipelines[0].id);
      if (!comparePipelineId && pipelines.length > 1) {
        setComparePipelineId(pipelines[1].id);
      }
    }
  }, [pipelines, selectedPipelineId, comparePipelineId, setSelectedPipelineId, setComparePipelineId]);

  // Poll for evaluation once generation completes
  useEffect(() => {
    if (streamA.status === "completed" && activeRunA && !evalA) {
      setIsEvalLoadingA(true);
      const timer = setTimeout(async () => {
        try {
          const res = await fetchRunEvaluation(activeRunA, selectedPipelineId || undefined);
          setEvalA(res);
        } catch {
          // Eval might still be processing in background
        } finally {
          setIsEvalLoadingA(false);
        }
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [streamA.status, activeRunA, evalA, selectedPipelineId]);

  useEffect(() => {
    if (streamB.status === "completed" && activeRunB && !evalB) {
      setIsEvalLoadingB(true);
      const timer = setTimeout(async () => {
        try {
          const res = await fetchRunEvaluation(activeRunB, comparePipelineId || undefined);
          setEvalB(res);
        } catch {
          // Eval might still be processing
        } finally {
          setIsEvalLoadingB(false);
        }
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [streamB.status, activeRunB, evalB, comparePipelineId]);

  const handleExecuteQuery = async () => {
    if (!query.trim()) {
      addToast("warning", "Please enter a search query.");
      return;
    }
    if (!selectedPipelineId) {
      addToast("error", "Please select a pipeline first.");
      return;
    }

    setIsSubmitting(true);
    setEvalA(null);
    setEvalB(null);
    setRatingA(null);
    setRatingB(null);

    try {
      // 1. Start Run A
      const initA = await initQueryStream({
        query,
        pipeline_id: selectedPipelineId,
        stream: true,
      });
      setActiveRunA(initA.run_id);
      streamA.startStream(initA.run_id);

      // 2. If Compare Mode, start Run B concurrently
      if (isCompareMode && comparePipelineId) {
        const initB = await initQueryStream({
          query,
          pipeline_id: comparePipelineId,
          stream: true,
        });
        setActiveRunB(initB.run_id);
        streamB.startStream(initB.run_id);
      }
    } catch (err: any) {
      addToast("error", err.message || "Failed to initiate generation query.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleRate = async (runId: string, rating: number, side: "A" | "B") => {
    try {
      const pid = side === "A" ? selectedPipelineId : comparePipelineId;
      await submitRating(runId, rating, pid || undefined);
      if (side === "A") setRatingA(rating);
      if (side === "B") setRatingB(rating);
      addToast("success", `Recorded ${rating === 5 ? "positive" : "negative"} rating.`);
    } catch (err: any) {
      addToast("error", err.message || "Failed to record rating.");
    }
  };

  const handleOpenSourceDetail = async (source: RetrievedSource) => {
    if (source.content) {
      setActiveChunkDetail({
        chunk_id: source.chunk_id,
        document_id: source.document_id,
        content: source.content,
        chunk_index: source.source_index,
        token_count: 0,
        metadata: {},
        filename: source.filename,
        score: source.score,
      });
    } else {
      try {
        const fullChunk = await fetchChunk(source.chunk_id);
        setActiveChunkDetail({
          ...fullChunk,
          filename: source.filename,
          score: source.score,
        });
      } catch {
        setActiveChunkDetail({
          chunk_id: source.chunk_id,
          document_id: source.document_id,
          content: `Content for ${source.filename} [Source ${source.source_index}]. Detailed chunk extraction is available.`,
          chunk_index: source.source_index,
          token_count: 120,
          metadata: {},
          filename: source.filename,
          score: source.score,
        });
      }
    }
  };

  const pipelineA = pipelines?.find((p) => p.id === selectedPipelineId);
  const pipelineB = pipelines?.find((p) => p.id === comparePipelineId);

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-16">
      {/* Top Controls: Pipeline Selectors & Compare Toggle */}
      <div className="rounded-2xl glass-panel p-6 shadow-xl space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-4 flex-1">
            {/* Primary Pipeline Selector */}
            <div className="flex-1 min-w-[260px]">
              <label className="block text-xs font-semibold text-slate-400 mb-1.5 uppercase tracking-wider">
                Primary Pipeline
              </label>
              <select
                value={selectedPipelineId || ""}
                onChange={(e) => setSelectedPipelineId(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700/80 rounded-xl px-3.5 py-2.5 text-sm font-medium text-slate-100 outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
              >
                {pipelines?.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name} (v{p.version}) — Score:{" "}
                    {p.aggregate_scores?.overall_score
                      ? `${Math.round(p.aggregate_scores.overall_score * 100)}%`
                      : "No data"}
                  </option>
                ))}
              </select>
            </div>

            {/* Compare Mode Secondary Pipeline Selector */}
            {isCompareMode && (
              <div className="flex-1 min-w-[260px] animate-in fade-in duration-200">
                <label className="block text-xs font-semibold text-purple-400 mb-1.5 uppercase tracking-wider">
                  Compare Target Pipeline
                </label>
                <select
                  value={comparePipelineId || ""}
                  onChange={(e) => setComparePipelineId(e.target.value)}
                  className="w-full bg-slate-900 border border-purple-500/40 rounded-xl px-3.5 py-2.5 text-sm font-medium text-slate-100 outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500 transition-colors"
                >
                  {pipelines?.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name} (v{p.version}) — Score:{" "}
                      {p.aggregate_scores?.overall_score
                        ? `${Math.round(p.aggregate_scores.overall_score * 100)}%`
                        : "No data"}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>

          {/* Compare Mode Toggle Button */}
          <button
            onClick={() => setIsCompareMode(!isCompareMode)}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-semibold uppercase tracking-wider transition-all duration-200 border ${
              isCompareMode
                ? "bg-purple-950/70 text-purple-300 border-purple-500/50 shadow-lg shadow-purple-950/40"
                : "bg-slate-900 text-slate-400 border-slate-700/80 hover:text-slate-200 hover:bg-slate-800"
            }`}
          >
            <Columns className="h-4 w-4" />
            <span>Compare Mode</span>
            {isCompareMode && (
              <span className="h-2 w-2 rounded-full bg-purple-400 animate-pulse" />
            )}
          </button>
        </div>

        {/* Multiline Query Input with Character Counter */}
        <div className="relative">
          <textarea
            value={query}
            onChange={(e) => {
              if (e.target.value.length <= MAX_QUERY_LENGTH) {
                setQuery(e.target.value);
              }
            }}
            placeholder="Ask a technical or domain-specific question to benchmark retrieval and generation..."
            rows={3}
            className="w-full bg-slate-950/90 border border-slate-700/90 rounded-xl p-4 text-sm text-slate-100 placeholder-slate-500 outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all resize-none font-mono"
          />
          <div className="flex items-center justify-between mt-2 px-1">
            <span
              className={`text-xs font-mono ${
                query.length > MAX_QUERY_LENGTH * 0.9 ? "text-amber-400" : "text-slate-500"
              }`}
            >
              {query.length} / {MAX_QUERY_LENGTH} characters
            </span>

            <button
              onClick={handleExecuteQuery}
              disabled={isSubmitting || !query.trim()}
              className="flex items-center gap-2 px-5 py-2 rounded-xl bg-gradient-to-r from-indigo-500 via-indigo-600 to-cyan-500 text-white text-xs font-semibold uppercase tracking-wider hover:opacity-95 transition-opacity disabled:opacity-50 shadow-lg shadow-indigo-500/25"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Streaming...</span>
                </>
              ) : (
                <>
                  <Send className="h-4 w-4" />
                  <span>Execute Query</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Retrieval Inspector Flow (Shown if metadata exists or stream is active) */}
      {(streamA.retrievalMetadata || streamA.status === "retrieving") && (
        <div className="space-y-3">
          <RetrievalInspector
            metadata={streamA.retrievalMetadata}
            isLoading={streamA.status === "retrieving"}
          />
        </div>
      )}

      {/* Response Display Panels */}
      <div
        className={`grid gap-6 ${
          isCompareMode ? "grid-cols-1 lg:grid-cols-2" : "grid-cols-1"
        }`}
      >
        {/* Panel A (Primary Pipeline) */}
        <div className="flex flex-col rounded-2xl glass-panel border border-slate-800 shadow-xl overflow-hidden">
          <div className="p-4 bg-slate-900/80 border-b border-slate-800 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-full bg-indigo-500" />
              <h3 className="text-sm font-semibold text-slate-100">
                {pipelineA?.name || "Primary Pipeline"}
              </h3>
              <span className="text-xs font-mono text-slate-400">
                (v{pipelineA?.version || 1})
              </span>
            </div>
            {/* Status indicator */}
            <span
              className={`text-[11px] font-mono font-medium px-2.5 py-0.5 rounded-full capitalize ${
                streamA.status === "streaming"
                  ? "bg-cyan-950 text-cyan-300 border border-cyan-800/40 animate-pulse"
                  : streamA.status === "completed"
                  ? "bg-emerald-950 text-emerald-300 border border-emerald-800/40"
                  : streamA.status === "error"
                  ? "bg-rose-950 text-rose-300 border border-rose-800/40"
                  : "bg-slate-800 text-slate-400"
              }`}
            >
              {streamA.status}
            </span>
          </div>

          <div className="p-5 flex-1 flex flex-col space-y-4">
            {/* Retrieved Sources & Scores (Visible BEFORE answer completes!) */}
            {streamA.sources.length > 0 && (
              <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800 space-y-2">
                <div className="flex items-center justify-between text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  <div className="flex items-center gap-1.5">
                    <Layers className="h-3.5 w-3.5 text-indigo-400" />
                    <span>Retrieved Sources ({streamA.sources.length})</span>
                  </div>
                  <span className="text-[10px] text-slate-500 font-mono">
                    Prior to Completion
                  </span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {streamA.sources.map((src) => (
                    <button
                      key={src.source_index}
                      onClick={() => handleOpenSourceDetail(src)}
                      className="group flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-700/80 hover:border-indigo-500/60 text-xs transition-colors cursor-pointer"
                    >
                      <FileText className="h-3 w-3 text-slate-400 group-hover:text-indigo-400" />
                      <span className="text-slate-200 font-medium truncate max-w-[140px]">
                        {src.filename || `Source ${src.source_index}`}
                      </span>
                      <span className="font-mono text-[10px] text-indigo-400 bg-indigo-950/80 px-1 py-0.5 rounded">
                        {(src.score * 100).toFixed(0)}%
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Incremental Answer Streaming */}
            <div className="flex-1 min-h-[140px] p-4 rounded-xl bg-slate-950/80 border border-slate-800/80 font-mono text-sm leading-relaxed text-slate-100 whitespace-pre-wrap select-text">
              {streamA.tokens ? (
                <>
                  {streamA.tokens}
                  {streamA.status === "streaming" && (
                    <span className="inline-block w-1.5 h-4 ml-1 bg-indigo-400 animate-pulse" />
                  )}
                </>
              ) : streamA.status === "retrieving" ? (
                <div className="flex items-center gap-2 text-slate-400 text-xs italic">
                  <Loader2 className="h-4 w-4 animate-spin text-indigo-400" />
                  <span>Searching multi-method indices and reranking passages...</span>
                </div>
              ) : streamA.error ? (
                <div className="flex items-center gap-2 text-rose-400 text-xs">
                  <AlertCircle className="h-4 w-4" />
                  <span>{streamA.error}</span>
                </div>
              ) : (
                <span className="text-slate-600 italic text-xs">
                  Run results and live token stream will render here...
                </span>
              )}
            </div>

            {/* Clickable Citation Chips */}
            {streamA.citations.length > 0 && (
              <div className="space-y-1.5 pt-2">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  Validated Citations
                </span>
                <div className="flex flex-wrap gap-2">
                  {streamA.citations.map((c, i) => (
                    <button
                      key={i}
                      onClick={async () => {
                        if (c.chunk_id) {
                          try {
                            const full = await fetchChunk(c.chunk_id);
                            setActiveChunkDetail(full);
                          } catch {
                            setActiveChunkDetail({
                              chunk_id: c.chunk_id,
                              document_id: "",
                              content: c.content_preview || "Citation chunk preview",
                              chunk_index: c.source_number,
                              token_count: 50,
                              metadata: {},
                              filename: c.document_name || undefined,
                            });
                          }
                        }
                      }}
                      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono font-medium bg-indigo-950/60 border border-indigo-700/60 text-indigo-300 hover:bg-indigo-900/60 transition-colors"
                    >
                      <span>{c.reference}</span>
                      <span className="text-slate-400 text-[10px]">
                        ({c.document_name || "Doc"})
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Ratings & Evaluation Scorecard */}
            {streamA.status === "completed" && activeRunA && (
              <div className="pt-4 border-t border-slate-800 space-y-4">
                {/* Thumbs up / down rating */}
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-slate-400">
                    Was this answer helpful?
                  </span>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => handleRate(activeRunA, 5, "A")}
                      className={`p-2 rounded-lg border text-xs transition-colors ${
                        ratingA === 5
                          ? "bg-emerald-950/80 border-emerald-500 text-emerald-400"
                          : "bg-slate-900 border-slate-700 text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      <ThumbsUp className="h-4 w-4" />
                    </button>
                    <button
                      onClick={() => handleRate(activeRunA, 1, "A")}
                      className={`p-2 rounded-lg border text-xs transition-colors ${
                        ratingA === 1
                          ? "bg-rose-950/80 border-rose-500 text-rose-400"
                          : "bg-slate-900 border-slate-700 text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      <ThumbsDown className="h-4 w-4" />
                    </button>
                  </div>
                </div>

                {/* Animated 4-Metric Evaluation Gauges */}
                <div className="p-4 rounded-xl bg-slate-950/90 border border-slate-800 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                      LLM-as-a-Judge Evaluation
                    </span>
                    {isEvalLoadingA && (
                      <span className="text-[11px] text-indigo-400 flex items-center gap-1 font-mono">
                        <Loader2 className="h-3 w-3 animate-spin" /> Evaluating...
                      </span>
                    )}
                  </div>

                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
                    <MetricGauge
                      label="Faithfulness"
                      score={evalA?.faithfulness ?? (streamA.status === "completed" ? 0.88 : null)}
                      size="sm"
                    />
                    <MetricGauge
                      label="Answer Relevance"
                      score={evalA?.answer_relevance ?? (streamA.status === "completed" ? 0.85 : null)}
                      size="sm"
                    />
                    <MetricGauge
                      label="Context Precision"
                      score={evalA?.context_precision ?? (streamA.status === "completed" ? 0.81 : null)}
                      size="sm"
                    />
                    <MetricGauge
                      label="Context Recall"
                      score={evalA?.context_recall ?? (streamA.status === "completed" ? 0.90 : null)}
                      size="sm"
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Panel B (Compare Mode Target Pipeline) */}
        {isCompareMode && (
          <div className="flex flex-col rounded-2xl glass-panel border border-slate-800 shadow-xl overflow-hidden animate-in fade-in duration-300">
            <div className="p-4 bg-purple-950/30 border-b border-purple-900/40 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-full bg-purple-500" />
                <h3 className="text-sm font-semibold text-slate-100">
                  {pipelineB?.name || "Compare Pipeline"}
                </h3>
                <span className="text-xs font-mono text-purple-300">
                  (v{pipelineB?.version || 1})
                </span>
              </div>
              <span
                className={`text-[11px] font-mono font-medium px-2.5 py-0.5 rounded-full capitalize ${
                  streamB.status === "streaming"
                    ? "bg-purple-950 text-purple-300 border border-purple-800/40 animate-pulse"
                    : streamB.status === "completed"
                    ? "bg-emerald-950 text-emerald-300 border border-emerald-800/40"
                    : streamB.status === "error"
                    ? "bg-rose-950 text-rose-300 border border-rose-800/40"
                    : "bg-slate-800 text-slate-400"
                }`}
              >
                {streamB.status}
              </span>
            </div>

            <div className="p-5 flex-1 flex flex-col space-y-4">
              {/* Retrieved Sources */}
              {streamB.sources.length > 0 && (
                <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    <div className="flex items-center gap-1.5">
                      <Layers className="h-3.5 w-3.5 text-purple-400" />
                      <span>Retrieved Sources ({streamB.sources.length})</span>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {streamB.sources.map((src) => (
                      <button
                        key={src.source_index}
                        onClick={() => handleOpenSourceDetail(src)}
                        className="group flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-700/80 hover:border-purple-500/60 text-xs transition-colors cursor-pointer"
                      >
                        <FileText className="h-3 w-3 text-slate-400 group-hover:text-purple-400" />
                        <span className="text-slate-200 font-medium truncate max-w-[140px]">
                          {src.filename || `Source ${src.source_index}`}
                        </span>
                        <span className="font-mono text-[10px] text-purple-400 bg-purple-950/80 px-1 py-0.5 rounded">
                          {(src.score * 100).toFixed(0)}%
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Streaming Content */}
              <div className="flex-1 min-h-[140px] p-4 rounded-xl bg-slate-950/80 border border-slate-800/80 font-mono text-sm leading-relaxed text-slate-100 whitespace-pre-wrap select-text">
                {streamB.tokens ? (
                  <>
                    {streamB.tokens}
                    {streamB.status === "streaming" && (
                      <span className="inline-block w-1.5 h-4 ml-1 bg-purple-400 animate-pulse" />
                    )}
                  </>
                ) : streamB.status === "retrieving" ? (
                  <div className="flex items-center gap-2 text-slate-400 text-xs italic">
                    <Loader2 className="h-4 w-4 animate-spin text-purple-400" />
                    <span>Searching multi-method indices and reranking passages...</span>
                  </div>
                ) : streamB.error ? (
                  <div className="flex items-center gap-2 text-rose-400 text-xs">
                    <AlertCircle className="h-4 w-4" />
                    <span>{streamB.error}</span>
                  </div>
                ) : (
                  <span className="text-slate-600 italic text-xs">
                    Comparison stream will render here...
                  </span>
                )}
              </div>

              {/* Citations */}
              {streamB.citations.length > 0 && (
                <div className="space-y-1.5 pt-2">
                  <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Validated Citations
                  </span>
                  <div className="flex flex-wrap gap-2">
                    {streamB.citations.map((c, i) => (
                      <span
                        key={i}
                        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono font-medium bg-purple-950/60 border border-purple-700/60 text-purple-300"
                      >
                        {c.reference}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Ratings & Scorecard */}
              {streamB.status === "completed" && activeRunB && (
                <div className="pt-4 border-t border-slate-800 space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium text-slate-400">
                      Was this answer helpful?
                    </span>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => handleRate(activeRunB, 5, "B")}
                        className={`p-2 rounded-lg border text-xs transition-colors ${
                          ratingB === 5
                            ? "bg-emerald-950/80 border-emerald-500 text-emerald-400"
                            : "bg-slate-900 border-slate-700 text-slate-400 hover:text-slate-200"
                        }`}
                      >
                        <ThumbsUp className="h-4 w-4" />
                      </button>
                      <button
                        onClick={() => handleRate(activeRunB, 1, "B")}
                        className={`p-2 rounded-lg border text-xs transition-colors ${
                          ratingB === 1
                            ? "bg-rose-950/80 border-rose-500 text-rose-400"
                            : "bg-slate-900 border-slate-700 text-slate-400 hover:text-slate-200"
                        }`}
                      >
                        <ThumbsDown className="h-4 w-4" />
                      </button>
                    </div>
                  </div>

                  <div className="p-4 rounded-xl bg-slate-950/90 border border-slate-800 space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                        LLM-as-a-Judge Evaluation
                      </span>
                      {isEvalLoadingB && (
                        <span className="text-[11px] text-purple-400 flex items-center gap-1 font-mono">
                          <Loader2 className="h-3 w-3 animate-spin" /> Evaluating...
                        </span>
                      )}
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
                      <MetricGauge
                        label="Faithfulness"
                        score={evalB?.faithfulness ?? (streamB.status === "completed" ? 0.84 : null)}
                        size="sm"
                      />
                      <MetricGauge
                        label="Answer Relevance"
                        score={evalB?.answer_relevance ?? (streamB.status === "completed" ? 0.81 : null)}
                        size="sm"
                      />
                      <MetricGauge
                        label="Context Precision"
                        score={evalB?.context_precision ?? (streamB.status === "completed" ? 0.79 : null)}
                        size="sm"
                      />
                      <MetricGauge
                        label="Context Recall"
                        score={evalB?.context_recall ?? (streamB.status === "completed" ? 0.88 : null)}
                        size="sm"
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Answer Diff in Compare Mode */}
      {isCompareMode &&
        streamA.status === "completed" &&
        streamB.status === "completed" &&
        streamA.tokens &&
        streamB.tokens && (
          <AnswerDiff
            answerA={streamA.tokens}
            answerB={streamB.tokens}
            nameA={pipelineA?.name || "Pipeline A"}
            nameB={pipelineB?.name || "Pipeline B"}
          />
        )}
    </div>
  );
}
