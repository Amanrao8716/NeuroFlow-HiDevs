"use client";

import React, { useState, useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchEvaluations, fetchPipelines, getEvaluationsStreamUrl } from "@/lib/api";
import { EvaluationCard } from "@/components/evaluations/EvaluationCard";
import {
  Activity,
  RefreshCw,
  AlertCircle,
  SlidersHorizontal,
} from "lucide-react";
import type { EvaluationItem } from "@/types/api";

export default function EvaluationsPage() {
  const [evaluations, setEvaluations] = useState<EvaluationItem[]>([]);
  const [newEvalIds, setNewEvalIds] = useState<Set<string>>(new Set());
  const [connectionStatus, setConnectionStatus] = useState<
    "connecting" | "connected" | "offline" | "error"
  >("connecting");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Filter States
  const [selectedPipelineId, setSelectedPipelineId] = useState<string>("all");
  const [metricFilter, setMetricFilter] = useState<string>("all");
  const [dateRange, setDateRange] = useState<string>("all");

  const eventSourceRef = useRef<EventSource | null>(null);

  // Pre-load pipelines for filtering
  const { data: pipelines } = useQuery({
    queryKey: ["pipelines"],
    queryFn: fetchPipelines,
  });

  // Pre-load initial historical evaluations
  const { data: initialEvals, isLoading: isInitialLoading, refetch } = useQuery({
    queryKey: ["evaluations", selectedPipelineId],
    queryFn: () =>
      fetchEvaluations(selectedPipelineId === "all" ? undefined : selectedPipelineId, 50),
  });

  useEffect(() => {
    if (initialEvals) {
      setEvaluations(initialEvals);
    }
  }, [initialEvals]);

  // Connect to live SSE evaluations stream
  useEffect(() => {
    const streamUrl = getEvaluationsStreamUrl();
    setConnectionStatus("connecting");
    setErrorMessage(null);

    const es = new EventSource(streamUrl);
    eventSourceRef.current = es;

    es.onopen = () => {
      setConnectionStatus("connected");
      setErrorMessage(null);
    };

    es.addEventListener("evaluation", (event: MessageEvent) => {
      try {
        const item: EvaluationItem = JSON.parse(event.data);
        setEvaluations((prev) => {
          // Avoid duplicate items
          if (prev.some((e) => e.eval_id === item.eval_id)) return prev;
          return [item, ...prev];
        });

        // Highlight new incoming card for 3s
        setNewEvalIds((prev) => new Set(prev).add(item.eval_id));
        setTimeout(() => {
          setNewEvalIds((prev) => {
            const next = new Set(prev);
            next.delete(item.eval_id);
            return next;
          });
        }, 3000);
      } catch (err: any) {
        console.error("Failed to parse evaluation stream event:", err);
      }
    });

    es.onerror = () => {
      setConnectionStatus("error");
      setErrorMessage("Live stream connection lost. Auto-reconnecting in background...");
    };

    return () => {
      es.close();
      eventSourceRef.current = null;
    };
  }, []);

  // Filter evaluations list
  const filteredEvaluations = evaluations.filter((item) => {
    // 1. Pipeline filter
    if (selectedPipelineId !== "all" && item.pipeline_id !== selectedPipelineId) {
      return false;
    }

    // 2. Metric threshold filter
    if (metricFilter === "faithfulness_low") {
      if ((item.faithfulness ?? 1) >= 0.7) return false;
    } else if (metricFilter === "overall_low") {
      if ((item.overall_score ?? 1) >= 0.8) return false;
    } else if (metricFilter === "high_quality") {
      if ((item.overall_score ?? 0) < 0.8) return false;
    }

    // 3. Date range filter
    if (dateRange !== "all" && item.evaluated_at) {
      const itemTime = new Date(item.evaluated_at).getTime();
      const now = Date.now();
      if (dateRange === "24h" && now - itemTime > 24 * 60 * 60 * 1000) return false;
      if (dateRange === "7d" && now - itemTime > 7 * 24 * 60 * 60 * 1000) return false;
    }

    return true;
  });

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-16">
      {/* Header and Live Connection Status */}
      <div className="p-6 rounded-2xl glass-panel shadow-xl flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3.5">
          <div className="p-3 rounded-xl bg-gradient-to-tr from-cyan-500/20 to-indigo-500/20 border border-cyan-500/30 text-cyan-400">
            <Activity className="h-6 w-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-slate-100">Live Evaluation Feed</h1>
            <p className="text-xs text-slate-400">
              Real-time LLM-as-a-judge scorecards streamed directly via Server-Sent Events
            </p>
          </div>
        </div>

        {/* Live SSE Status Badge */}
        <div className="flex items-center gap-3">
          <div
            className={`flex items-center gap-2 px-3 py-1.5 rounded-xl border text-xs font-mono font-medium ${
              connectionStatus === "connected"
                ? "bg-emerald-950/70 border-emerald-500/40 text-emerald-300"
                : connectionStatus === "connecting"
                ? "bg-amber-950/70 border-amber-500/40 text-amber-300"
                : "bg-rose-950/70 border-rose-500/40 text-rose-300"
            }`}
          >
            <span
              className={`h-2 w-2 rounded-full ${
                connectionStatus === "connected"
                  ? "bg-emerald-400 animate-pulse"
                  : connectionStatus === "connecting"
                  ? "bg-amber-400 animate-ping"
                  : "bg-rose-400"
              }`}
            />
            <span>
              {connectionStatus === "connected"
                ? "Live SSE Connected"
                : connectionStatus === "connecting"
                ? "Connecting Stream..."
                : "Stream Offline (Reconnecting)"}
            </span>
          </div>

          <button
            onClick={() => refetch()}
            className="p-2.5 rounded-xl bg-slate-900 border border-slate-700/80 hover:bg-slate-800 text-slate-300 transition-colors"
            title="Refresh Historical Evaluations"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Recoverable Error Banner */}
      {errorMessage && (
        <div className="p-4 rounded-xl bg-amber-950/40 border border-amber-800/50 flex items-center gap-3 text-xs text-amber-300">
          <AlertCircle className="h-4 w-4 text-amber-400 flex-shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Filters Bar */}
      <div className="p-4 rounded-2xl glass-panel border border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 text-xs font-semibold text-slate-400 uppercase tracking-wider">
            <SlidersHorizontal className="h-3.5 w-3.5" />
            <span>Filters:</span>
          </div>

          {/* Pipeline filter */}
          <select
            value={selectedPipelineId}
            onChange={(e) => setSelectedPipelineId(e.target.value)}
            className="bg-slate-900 border border-slate-700/80 rounded-xl px-3 py-1.5 text-xs font-medium text-slate-200 outline-none cursor-pointer"
          >
            <option value="all">All Pipelines</option>
            {pipelines?.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>

          {/* Metric threshold filter */}
          <select
            value={metricFilter}
            onChange={(e) => setMetricFilter(e.target.value)}
            className="bg-slate-900 border border-slate-700/80 rounded-xl px-3 py-1.5 text-xs font-medium text-slate-200 outline-none cursor-pointer"
          >
            <option value="all">All Metric Thresholds</option>
            <option value="faithfulness_low">Faithfulness &lt; 0.7 (Hallucination Alert)</option>
            <option value="overall_low">Overall Score &lt; 0.8</option>
            <option value="high_quality">High Quality (&ge; 0.8 Training Candidates)</option>
          </select>

          {/* Date range filter */}
          <select
            value={dateRange}
            onChange={(e) => setDateRange(e.target.value)}
            className="bg-slate-900 border border-slate-700/80 rounded-xl px-3 py-1.5 text-xs font-medium text-slate-200 outline-none cursor-pointer"
          >
            <option value="all">All Time</option>
            <option value="24h">Past 24 Hours</option>
            <option value="7d">Past 7 Days</option>
          </select>
        </div>

        <span className="text-xs font-mono text-slate-400">
          Showing {filteredEvaluations.length} evaluation{filteredEvaluations.length === 1 ? "" : "s"}
        </span>
      </div>

      {/* Evaluations Feed List */}
      {isInitialLoading ? (
        <div className="p-16 text-center space-y-3">
          <RefreshCw className="h-6 w-6 animate-spin text-indigo-400 mx-auto" />
          <p className="text-xs text-slate-400 font-mono">Loading evaluation history...</p>
        </div>
      ) : filteredEvaluations.length > 0 ? (
        <div className="space-y-4">
          {filteredEvaluations.map((item) => (
            <EvaluationCard
              key={item.eval_id}
              evaluation={item}
              isNew={newEvalIds.has(item.eval_id)}
            />
          ))}
        </div>
      ) : (
        <div className="p-12 rounded-2xl glass-panel text-center space-y-3 max-w-lg mx-auto">
          <div className="p-3 rounded-2xl bg-slate-900 border border-slate-800 w-fit mx-auto text-cyan-400">
            <Activity className="h-6 w-6" />
          </div>
          <h3 className="text-sm font-semibold text-slate-200">No Evaluations Match Filters</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Run queries in the Query Playground or adjust filter criteria. Completed runs are evaluated asynchronously and streamed here in real-time.
          </p>
        </div>
      )}
    </div>
  );
}
