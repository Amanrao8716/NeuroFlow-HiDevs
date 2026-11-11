"use client";

import React, { useState } from "react";
import { X, Copy, Check, FileText, Sparkles, Hash, Layers } from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import { retrieveSimilarChunks } from "@/lib/api";
import type { ChunkItem } from "@/types/api";

export function ChunkDrawer() {
  const { activeChunkDetail, setActiveChunkDetail, selectedPipelineId, addToast } = useAppStore();
  const [copied, setCopied] = useState(false);
  const [findingSimilar, setFindingSimilar] = useState(false);
  const [similarChunks, setSimilarChunks] = useState<ChunkItem[]>([]);

  if (!activeChunkDetail) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(activeChunkDetail.content);
    setCopied(true);
    addToast("success", "Chunk content copied to clipboard!");
    setTimeout(() => setCopied(false), 2000);
  };

  const handleFindSimilar = async () => {
    if (!selectedPipelineId) {
      addToast("warning", "Select an active pipeline first to search across chunks.");
      return;
    }
    setFindingSimilar(true);
    try {
      const res = await retrieveSimilarChunks(
        activeChunkDetail.content.slice(0, 300),
        selectedPipelineId,
        4
      );
      setSimilarChunks(res.chunks || []);
      addToast("success", `Found ${(res.chunks || []).length} similar chunks.`);
    } catch (err: any) {
      addToast("error", err.message || "Failed to find similar chunks.");
    } finally {
      setFindingSimilar(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/60 backdrop-blur-sm transition-opacity">
      <div className="relative flex h-full w-full max-w-xl flex-col bg-slate-900 border-l border-slate-800 shadow-2xl p-6 overflow-y-auto animate-in slide-in-from-right duration-300">
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              <FileText className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-slate-100">
                {activeChunkDetail.filename || "Chunk Details"}
              </h2>
              <p className="text-xs text-slate-400">
                ID: <span className="font-mono text-slate-300">{activeChunkDetail.chunk_id.slice(0, 8)}...</span>
              </p>
            </div>
          </div>
          <button
            onClick={() => setActiveChunkDetail(null)}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Metadata badges */}
        <div className="flex flex-wrap items-center gap-2 py-4 border-b border-slate-800">
          {activeChunkDetail.chunk_index !== undefined && (
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-medium bg-slate-800 text-slate-300 border border-slate-700">
              <Hash className="h-3.5 w-3.5 text-slate-400" />
              Index #{activeChunkDetail.chunk_index}
            </span>
          )}
          {activeChunkDetail.token_count && (
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-medium bg-slate-800 text-slate-300 border border-slate-700">
              <Layers className="h-3.5 w-3.5 text-slate-400" />
              {activeChunkDetail.token_count} tokens
            </span>
          )}
          {activeChunkDetail.score !== undefined && (
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-semibold bg-indigo-950/60 text-indigo-300 border border-indigo-800/40">
              Relevance: {(activeChunkDetail.score * 100).toFixed(1)}%
            </span>
          )}
        </div>

        {/* Chunk Content */}
        <div className="flex-1 py-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">
              Chunk Passage
            </span>
            <button
              onClick={handleCopy}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-md bg-slate-800 text-slate-300 hover:bg-slate-700 transition-colors border border-slate-700"
            >
              {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
              {copied ? "Copied" : "Copy"}
            </button>
          </div>
          <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 text-sm leading-relaxed text-slate-200 font-mono whitespace-pre-wrap select-text">
            {activeChunkDetail.content}
          </div>
        </div>

        {/* Find Similar Action */}
        <div className="pt-4 border-t border-slate-800 space-y-3">
          <button
            onClick={handleFindSimilar}
            disabled={findingSimilar}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-gradient-to-r from-indigo-600 to-cyan-600 text-white text-sm font-medium hover:opacity-95 transition-opacity disabled:opacity-50 shadow-lg shadow-indigo-500/20"
          >
            <Sparkles className="h-4 w-4" />
            {findingSimilar ? "Searching Across Knowledge Base..." : "Find Similar Chunks"}
          </button>

          {similarChunks.length > 0 && (
            <div className="space-y-2 mt-3">
              <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">
                Top Similar Chunks
              </span>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {similarChunks.map((chunk, idx) => (
                  <div
                    key={idx}
                    className="p-3 rounded-lg bg-slate-800/60 border border-slate-700 text-xs space-y-1 hover:border-indigo-500/40 transition-colors cursor-pointer"
                    onClick={() => setActiveChunkDetail(chunk)}
                  >
                    <div className="flex items-center justify-between font-semibold text-slate-300">
                      <span>{chunk.filename || `Chunk #${chunk.chunk_index}`}</span>
                      {chunk.score !== undefined && (
                        <span className="text-indigo-400 font-mono">
                          Score: {chunk.score.toFixed(3)}
                        </span>
                      )}
                    </div>
                    <p className="text-slate-400 line-clamp-2">{chunk.content}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
