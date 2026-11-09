"use client";

import React, { useState } from "react";
import { ChevronDown, ChevronUp, Clock } from "lucide-react";
import { getScoreColor } from "@/components/common/MetricGauge";
import type { EvaluationItem } from "@/types/api";

interface EvaluationCardProps {
  evaluation: EvaluationItem;
  isNew?: boolean;
}

function MetricBar({ label, value }: { label: string; value: number | null }) {
  const colors = getScoreColor(value);
  const percentage = value !== null && value !== undefined ? Math.round(value * 100) : 0;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-[11px] font-medium">
        <span className="text-slate-400 capitalize">{label.replace("_", " ")}</span>
        <span className={`font-mono font-semibold ${colors.text}`}>
          {value !== null && value !== undefined ? `${percentage}%` : "--"}
        </span>
      </div>
      <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500`}
          style={{
            width: `${percentage}%`,
            backgroundColor: colors.stroke,
          }}
        />
      </div>
    </div>
  );
}

export function EvaluationCard({ evaluation, isNew }: EvaluationCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const scoreColors = getScoreColor(evaluation.overall_score);

  const formattedDate = evaluation.evaluated_at
    ? new Date(evaluation.evaluated_at).toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      })
    : "Just now";

  return (
    <div
      className={`rounded-2xl glass-panel p-5 border transition-all duration-300 shadow-lg ${
        isNew
          ? "border-indigo-500/80 bg-indigo-950/20 shadow-indigo-500/10"
          : "border-slate-800 hover:border-slate-700/80"
      }`}
    >
      {/* Card Header: Pipeline, Timestamp, Overall Badge */}
      <div className="flex items-start justify-between gap-4 mb-3">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-indigo-400" />
            <span className="text-xs font-bold text-slate-200">
              {evaluation.pipeline_name || "RAG Pipeline"}
            </span>
            <span className="text-[10px] font-mono text-slate-500">
              Run: {evaluation.run_id.slice(0, 8)}...
            </span>
          </div>
          <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
            <Clock className="h-3 w-3 text-slate-500" />
            <span>{formattedDate}</span>
            {evaluation.judge_model && (
              <>
                <span className="text-slate-600">•</span>
                <span className="font-mono text-slate-400">Judge: {evaluation.judge_model}</span>
              </>
            )}
          </div>
        </div>

        {/* Overall Score Badge */}
        <div
          className={`flex items-center gap-1.5 px-3 py-1 rounded-xl border ${scoreColors.bg} ${scoreColors.border}`}
        >
          <span className="text-[10px] uppercase font-semibold text-slate-400">Overall</span>
          <span className={`text-sm font-bold font-mono ${scoreColors.text}`}>
            {evaluation.overall_score !== null && evaluation.overall_score !== undefined
              ? `${Math.round(evaluation.overall_score * 100)}%`
              : "--"}
          </span>
        </div>
      </div>

      {/* Query Preview */}
      <div className="mb-4">
        <p className="text-sm font-medium text-slate-200 line-clamp-2">
          {evaluation.query || "No query prompt recorded"}
        </p>
      </div>

      {/* 4 Metric Bars Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 py-3 border-y border-slate-800/80">
        <MetricBar label="Faithfulness" value={evaluation.faithfulness} />
        <MetricBar label="Answer Relevance" value={evaluation.answer_relevance} />
        <MetricBar label="Context Precision" value={evaluation.context_precision} />
        <MetricBar label="Context Recall" value={evaluation.context_recall} />
      </div>

      {/* Expand / Collapse Section */}
      <div className="pt-3">
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center justify-between w-full text-xs font-semibold text-slate-400 hover:text-slate-200 transition-colors"
        >
          <span>{isExpanded ? "Hide Full Details" : "View Query, Chunks & Answer"}</span>
          {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        </button>

        {isExpanded && (
          <div className="mt-4 pt-4 border-t border-slate-800/80 space-y-4 animate-in fade-in duration-200">
            {/* Full Query */}
            <div className="space-y-1">
              <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
                Full Query Prompt
              </span>
              <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 text-xs font-mono text-slate-200 select-text whitespace-pre-wrap">
                {evaluation.query}
              </div>
            </div>

            {/* Retrieved Chunks IDs */}
            {evaluation.retrieved_chunk_ids && evaluation.retrieved_chunk_ids.length > 0 && (
              <div className="space-y-1">
                <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
                  Retrieved Chunks ({evaluation.retrieved_chunk_ids.length})
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {evaluation.retrieved_chunk_ids.map((cid, i) => (
                    <span
                      key={i}
                      className="px-2 py-0.5 rounded bg-slate-800/80 border border-slate-700/60 text-[11px] font-mono text-indigo-300"
                    >
                      {cid.slice(0, 8)}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Generated Answer */}
            <div className="space-y-1">
              <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
                Generated Answer
              </span>
              <div className="p-4 rounded-xl bg-slate-950/90 border border-slate-800 text-xs leading-relaxed font-mono text-slate-200 select-text whitespace-pre-wrap">
                {evaluation.answer || "No answer recorded for this run."}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
