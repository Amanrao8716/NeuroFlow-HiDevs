"use client";

import React, { useMemo } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  Edge,
  Node,
  Position,
  Handle,
} from "@xyflow/react";
import { Database, Search, Filter, Shuffle, Award, CheckCircle2 } from "lucide-react";
import type { RetrievalMetadata } from "@/types/api";

interface RetrievalInspectorProps {
  metadata: RetrievalMetadata | null;
  isLoading?: boolean;
}

function CustomNode({ data }: { data: any }) {
  const Icon = data.icon || Database;

  return (
    <div className="px-4 py-3 rounded-xl bg-slate-900 border border-slate-700/80 shadow-xl min-w-[190px] text-xs">
      <Handle type="target" position={Position.Left} className="!bg-indigo-500 !w-2.5 !h-2.5" />
      <div className="flex items-center justify-between gap-2 mb-1.5">
        <div className="flex items-center gap-1.5">
          <div className={`p-1.5 rounded-lg ${data.iconBg || "bg-indigo-500/10 text-indigo-400"}`}>
            <Icon className="h-3.5 w-3.5" />
          </div>
          <span className="font-semibold text-slate-200">{data.label}</span>
        </div>
      </div>
      <div className="flex items-center justify-between text-[11px] text-slate-400 border-t border-slate-800/80 pt-1.5">
        <span>{data.metricLabel || "Candidates"}</span>
        <span className="font-mono font-bold text-indigo-300 bg-indigo-950/60 px-1.5 py-0.5 rounded border border-indigo-800/30">
          {data.count ?? "--"}
        </span>
      </div>
      <Handle type="source" position={Position.Right} className="!bg-indigo-500 !w-2.5 !h-2.5" />
    </div>
  );
}

const nodeTypes = {
  custom: CustomNode,
};

export function RetrievalInspector({ metadata, isLoading }: RetrievalInspectorProps) {
  const nodes: Node[] = useMemo(() => {
    const denseCount = metadata ? metadata.dense_count : 0;
    const sparseCount = metadata ? metadata.sparse_count : 0;
    const metaCount = metadata ? metadata.meta_count : 0;
    const fusedCount = metadata ? metadata.fused_count : 0;
    const rerankedCount = metadata ? metadata.reranked_count : 0;
    const finalCount = metadata ? metadata.context_chunks_used : 0;

    return [
      {
        id: "query",
        type: "custom",
        position: { x: 20, y: 150 },
        data: {
          label: "User Query",
          metricLabel: "Type",
          count: metadata?.query_type || "Semantic",
          icon: Search,
          iconBg: "bg-blue-500/10 text-blue-400",
        },
      },
      {
        id: "dense",
        type: "custom",
        position: { x: 250, y: 40 },
        data: {
          label: "Dense Retrieval",
          metricLabel: "Chunks",
          count: denseCount,
          icon: Database,
          iconBg: "bg-indigo-500/10 text-indigo-400",
        },
      },
      {
        id: "sparse",
        type: "custom",
        position: { x: 250, y: 150 },
        data: {
          label: "Sparse Lexical",
          metricLabel: "Chunks",
          count: sparseCount,
          icon: Search,
          iconBg: "bg-cyan-500/10 text-cyan-400",
        },
      },
      {
        id: "metadata_filter",
        type: "custom",
        position: { x: 250, y: 260 },
        data: {
          label: "Metadata Filter",
          metricLabel: "Chunks",
          count: metaCount,
          icon: Filter,
          iconBg: "bg-teal-500/10 text-teal-400",
        },
      },
      {
        id: "rrf",
        type: "custom",
        position: { x: 500, y: 150 },
        data: {
          label: "RRF Fusion",
          metricLabel: "Fused Chunks",
          count: fusedCount,
          icon: Shuffle,
          iconBg: "bg-purple-500/10 text-purple-400",
        },
      },
      {
        id: "rerank",
        type: "custom",
        position: { x: 740, y: 150 },
        data: {
          label: "Cross-Encoder",
          metricLabel: "Reranked",
          count: rerankedCount,
          icon: Award,
          iconBg: "bg-amber-500/10 text-amber-400",
        },
      },
      {
        id: "context",
        type: "custom",
        position: { x: 980, y: 150 },
        data: {
          label: "Final Context",
          metricLabel: "Assembled",
          count: `${finalCount} (${metadata?.total_tokens || 0} tok)`,
          icon: CheckCircle2,
          iconBg: "bg-emerald-500/10 text-emerald-400",
        },
      },
    ];
  }, [metadata]);

  const edges: Edge[] = useMemo(() => {
    return [
      { id: "e-q-dense", source: "query", target: "dense", animated: true },
      { id: "e-q-sparse", source: "query", target: "sparse", animated: true },
      { id: "e-q-meta", source: "query", target: "metadata_filter", animated: true },
      { id: "e-dense-rrf", source: "dense", target: "rrf", animated: true },
      { id: "e-sparse-rrf", source: "sparse", target: "rrf", animated: true },
      { id: "e-meta-rrf", source: "metadata_filter", target: "rrf", animated: true },
      { id: "e-rrf-rerank", source: "rrf", target: "rerank", animated: true },
      { id: "e-rerank-ctx", source: "rerank", target: "context", animated: true },
    ];
  }, []);

  return (
    <div className="w-full h-80 rounded-2xl bg-slate-950/80 border border-slate-800/80 overflow-hidden relative shadow-inner">
      <div className="absolute top-3 left-4 z-10 flex items-center gap-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Retrieval Flow Inspector
        </span>
        {isLoading && (
          <span className="flex items-center gap-1 text-[11px] text-cyan-400 font-mono">
            <span className="h-1.5 w-1.5 rounded-full bg-cyan-400 animate-ping" />
            Analyzing stages...
          </span>
        )}
      </div>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#1e293b" gap={16} size={1} />
        <Controls />
      </ReactFlow>
    </div>
  );
}
