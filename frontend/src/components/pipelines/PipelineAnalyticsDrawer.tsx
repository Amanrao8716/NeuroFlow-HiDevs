"use client";

import React from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchPipelineAnalytics } from "@/lib/api";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";
import { X, BarChart3, AlertOctagon, TrendingUp, DollarSign, Activity } from "lucide-react";
import type { PipelineResponse } from "@/types/api";

interface PipelineAnalyticsDrawerProps {
  pipeline: PipelineResponse | null;
  isOpen: boolean;
  onClose: () => void;
}

export function PipelineAnalyticsDrawer({
  pipeline,
  isOpen,
  onClose,
}: PipelineAnalyticsDrawerProps) {
  const {
    data: analytics,
    isLoading,
    isError,
    error,
    refetch,
  } = useQuery({
    queryKey: ["analytics", pipeline?.id],
    queryFn: () => (pipeline ? fetchPipelineAnalytics(pipeline.id) : null),
    enabled: Boolean(pipeline && isOpen),
  });

  if (!isOpen || !pipeline) return null;

  // Latency data
  const latencyData = [
    {
      metric: "P50 (Median)",
      latency: analytics?.retrieval_latency_p50 ?? null,
    },
    {
      metric: "P95",
      latency: analytics?.retrieval_latency_p95 ?? null,
    },
    {
      metric: "P99",
      latency: analytics?.retrieval_latency_p99 ?? null,
    },
  ];
  const hasLatencyData = latencyData.some((item) => item.latency !== null);

  // 30-Day Cost Trend
  const costTrendData = (analytics?.cost_trends_30d ?? [])
    .filter((cost) => typeof cost.cost_per_query === "number")
    .map((c) => ({
      date: c.date.slice(5),
      cost: c.cost_per_query,
    }));

  // Radar Data for the 4 Evaluation Metrics
  const evalScores = analytics?.avg_evaluation_scores;
  const radarData = [
    {
      metric: "Faithfulness",
      score: evalScores?.faithfulness,
    },
    {
      metric: "Relevance",
      score: evalScores?.answer_relevance,
    },
    {
      metric: "Precision",
      score: evalScores?.context_precision,
    },
    {
      metric: "Recall",
      score: evalScores?.context_recall,
    },
  ]
    .filter((item): item is { metric: string; score: number } => typeof item.score === "number")
    .map((item) => ({ ...item, score: Math.round(item.score * 100) }));

  const failedRuns = analytics?.recent_failed_runs || [];
  const availableMetricCount = radarData.length;
  const averageCost = analytics?.estimated_cost_per_query;

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/60 backdrop-blur-sm transition-opacity">
      <div className="relative flex h-full w-full max-w-2xl flex-col bg-slate-900 border-l border-slate-800 shadow-2xl p-6 overflow-y-auto animate-in slide-in-from-right duration-300 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              <BarChart3 className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-100">
                Pipeline Analytics — {pipeline.name}
              </h2>
              <p className="text-xs text-slate-400">
                30-day telemetry, latency percentiles, and quality metrics
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {isLoading ? (
          <div className="p-8 rounded-2xl bg-slate-950/70 border border-slate-800 text-center text-sm text-slate-400">
            Loading pipeline analytics...
          </div>
        ) : isError ? (
          <div className="p-8 rounded-2xl bg-rose-950/30 border border-rose-800/50 text-center space-y-3">
            <p className="text-sm text-rose-300">
              Unable to load pipeline analytics
              {error instanceof Error ? `: ${error.message}` : "."}
            </p>
            <button
              onClick={() => refetch()}
              className="px-3 py-1.5 rounded-lg bg-slate-900 border border-rose-800/60 text-xs text-rose-200 hover:bg-slate-800"
            >
              Retry
            </button>
          </div>
        ) : (
          <>
        {/* Latency Percentiles (P50, P95, P99) */}
        <div className="p-5 rounded-2xl bg-slate-950/70 border border-slate-800 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-cyan-400" />
              <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
                Retrieval Latency Percentiles (ms)
              </h3>
            </div>
            <span className="text-[11px] font-mono text-slate-400">
              Method: Linear Interpolation
            </span>
          </div>
          <div className="h-44 w-full">
            {hasLatencyData ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={latencyData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <XAxis dataKey="metric" stroke="#64748b" fontSize={11} />
                  <YAxis stroke="#64748b" fontSize={11} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0f172a",
                      border: "1px solid #334155",
                      borderRadius: "8px",
                      fontSize: "12px",
                    }}
                  />
                  <Bar dataKey="latency" fill="#06b6d4" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-xs text-slate-400">
                No retrieval latency data available yet.
              </div>
            )}
          </div>
        </div>

        {/* 30-Day Cost Trend */}
        <div className="p-5 rounded-2xl bg-slate-950/70 border border-slate-800 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <DollarSign className="h-4 w-4 text-emerald-400" />
              <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
                Cost-per-Query Trend (Last 30 Days)
              </h3>
            </div>
            <span className="text-[11px] font-mono text-emerald-400">
              Avg: {averageCost === null || averageCost === undefined ? "Unavailable" : `$${averageCost}`}
            </span>
          </div>
          <div className="h-44 w-full">
            {costTrendData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={costTrendData} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
                <defs>
                  <linearGradient id="costGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="date" stroke="#64748b" fontSize={10} />
                <YAxis stroke="#64748b" fontSize={10} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0f172a",
                    border: "1px solid #334155",
                    borderRadius: "8px",
                    fontSize: "12px",
                  }}
                  formatter={(value: any) => [`$${value}`, "Cost"]}
                />
                <Area
                  type="monotone"
                  dataKey="cost"
                  stroke="#10b981"
                  strokeWidth={2}
                  fillOpacity={1}
                  fill="url(#costGrad)"
                />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-xs text-slate-400">
                No cost trend data available yet.
              </div>
            )}
          </div>
        </div>

        {/* Evaluation Metrics Radar Chart */}
        <div className="p-5 rounded-2xl bg-slate-950/70 border border-slate-800 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-purple-400" />
              <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
                Evaluation Quality Radar (4 Metrics)
              </h3>
            </div>
          </div>
          <div className="h-56 w-full flex items-center justify-center">
            {availableMetricCount === 4 ? (
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData} outerRadius="75%">
                  <PolarGrid stroke="#334155" />
                  <PolarAngleAxis dataKey="metric" stroke="#94a3b8" fontSize={11} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#475569" fontSize={10} />
                  <Radar
                    name="Pipeline"
                    dataKey="score"
                    stroke="#8b5cf6"
                    fill="#8b5cf6"
                    fillOpacity={0.4}
                  />
                </RadarChart>
              </ResponsiveContainer>
            ) : availableMetricCount > 0 ? (
              <div className="w-full space-y-2 px-6">
                <p className="text-center text-xs text-slate-400">
                  {availableMetricCount} of 4 evaluation metrics available.
                </p>
                {radarData.map((item) => (
                  <div key={item.metric} className="flex justify-between text-xs text-slate-300">
                    <span>{item.metric}</span>
                    <span className="font-mono">{item.score}%</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-xs text-slate-400">
                No evaluation metrics available yet.
              </div>
            )}
          </div>
        </div>

        {/* Recent Failed Runs */}
        <div className="p-5 rounded-2xl bg-slate-950/70 border border-slate-800 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertOctagon className="h-4 w-4 text-rose-400" />
              <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
                Recent Failed Runs & Errors
              </h3>
            </div>
            <span className="text-xs font-mono text-slate-400">
              {failedRuns.length} recorded
            </span>
          </div>

          {failedRuns.length > 0 ? (
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {failedRuns.map((r) => (
                <div
                  key={r.run_id}
                  className="p-3 rounded-xl bg-slate-900 border border-rose-950/60 space-y-1 text-xs"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-slate-300 truncate max-w-sm font-medium">
                      {r.query}
                    </span>
                    <span className="text-[10px] text-slate-500 font-mono">
                      {r.created_at ? new Date(r.created_at).toLocaleDateString() : ""}
                    </span>
                  </div>
                  <p className="text-rose-400 font-mono text-[11px]">{r.error_message}</p>
                </div>
              ))}
            </div>
          ) : (
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 text-center text-xs text-slate-400 italic">
              No failed runs recorded for this pipeline.
            </div>
          )}
        </div>
          </>
        )}
      </div>
    </div>
  );
}
