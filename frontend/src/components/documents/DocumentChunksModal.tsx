"use client";

import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchDocumentChunks, retrieveSimilarChunks } from "@/lib/api";
import { X, FileText, Sparkles, Hash, Loader2 } from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import type { ChunkItem, DocumentItem } from "@/types/api";

interface DocumentChunksModalProps {
  document: DocumentItem | null;
  pipelineId: string;
  isOpen: boolean;
  onClose: () => void;
}

export function DocumentChunksModal({
  document,
  pipelineId,
  isOpen,
  onClose,
}: DocumentChunksModalProps) {
  const { setActiveChunkDetail, addToast } = useAppStore();
  const [selectedChunk, setSelectedChunk] = useState<ChunkItem | null>(null);
  const [similarChunks, setSimilarChunks] = useState<ChunkItem[]>([]);
  const [isSearchingSimilar, setIsSearchingSimilar] = useState(false);

  const { data: chunks, isLoading } = useQuery({
    queryKey: ["chunks", document?.id],
    queryFn: () => (document ? fetchDocumentChunks(document.id, pipelineId) : []),
    enabled: Boolean(document && isOpen),
  });

  if (!isOpen || !document) return null;

  const handleFindSimilar = async (chunk: ChunkItem) => {
    setSelectedChunk(chunk);
    setIsSearchingSimilar(true);
    setSimilarChunks([]);

    try {
      const res = await retrieveSimilarChunks(chunk.content.slice(0, 300), pipelineId, 4);
      setSimilarChunks(res.chunks || []);
      addToast("success", `Found ${(res.chunks || []).length} similar chunks.`);
    } catch (err: any) {
      addToast("error", err.message || "Failed to search similar chunks.");
    } finally {
      setIsSearchingSimilar(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-md p-4 animate-in fade-in duration-200">
      <div className="relative w-full max-w-4xl h-[85vh] flex flex-col rounded-2xl bg-slate-900 border border-slate-700 shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="p-5 border-b border-slate-800 flex items-center justify-between bg-slate-950/70">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              <FileText className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-100">{document.filename}</h2>
              <p className="text-xs text-slate-400">
                Document Chunks ({chunks?.length ?? document.chunk_count}) • Status:{" "}
                <span className="capitalize font-mono text-cyan-400">{document.status}</span>
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content Body: Split view or list */}
        <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
          {/* Chunks List */}
          <div className="flex-1 p-5 overflow-y-auto space-y-3 border-r border-slate-800">
            {isLoading ? (
              <div className="p-12 text-center space-y-2">
                <Loader2 className="h-6 w-6 animate-spin text-indigo-400 mx-auto" />
                <p className="text-xs text-slate-400 font-mono">Loading document chunks...</p>
              </div>
            ) : chunks && chunks.length > 0 ? (
              chunks.map((chunk) => (
                <div
                  key={chunk.chunk_id}
                  className={`p-4 rounded-xl border transition-all space-y-2 ${
                    selectedChunk?.chunk_id === chunk.chunk_id
                      ? "bg-slate-800/90 border-indigo-500/60 shadow-lg"
                      : "bg-slate-950/60 border-slate-800 hover:border-slate-700"
                  }`}
                >
                  <div className="flex items-center justify-between text-xs font-semibold">
                    <span className="flex items-center gap-1.5 text-slate-300 font-mono">
                      <Hash className="h-3.5 w-3.5 text-slate-500" />
                      Chunk #{chunk.chunk_index}
                    </span>
                    <div className="flex items-center gap-2">
                      <span className="text-slate-500 text-[11px] font-mono">
                        {chunk.token_count} tokens
                      </span>
                      <button
                        onClick={() => handleFindSimilar(chunk)}
                        disabled={isSearchingSimilar}
                        className="flex items-center gap-1 px-2.5 py-1 rounded-md bg-indigo-600/20 hover:bg-indigo-600/40 border border-indigo-500/30 text-indigo-300 text-[11px] font-medium transition-colors"
                      >
                        <Sparkles className="h-3 w-3" />
                        <span>Find Similar</span>
                      </button>
                    </div>
                  </div>
                  <p className="text-xs font-mono text-slate-300 whitespace-pre-wrap line-clamp-4 select-text">
                    {chunk.content}
                  </p>
                </div>
              ))
            ) : (
              <div className="p-8 text-center text-xs text-slate-400 italic">
                No chunks produced yet. Document is either processing or queued.
              </div>
            )}
          </div>

          {/* Similar Chunks Analysis Panel */}
          <div className="w-full md:w-80 p-5 bg-slate-950/40 overflow-y-auto space-y-4">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400 border-b border-slate-800 pb-2">
              <Sparkles className="h-4 w-4 text-cyan-400" />
              <span>Cross-Pipeline Similarity</span>
            </div>

            {isSearchingSimilar ? (
              <div className="p-8 text-center space-y-2">
                <Loader2 className="h-5 w-5 animate-spin text-cyan-400 mx-auto" />
                <p className="text-xs text-slate-400 font-mono">Comparing embeddings...</p>
              </div>
            ) : similarChunks.length > 0 ? (
              <div className="space-y-3">
                <span className="text-[11px] font-medium text-slate-400 block">
                  Top matches for Chunk #{selectedChunk?.chunk_index}:
                </span>
                {similarChunks.map((sim, i) => (
                  <div
                    key={i}
                    onClick={() => setActiveChunkDetail(sim)}
                    className="p-3 rounded-xl bg-slate-900 border border-slate-800 hover:border-cyan-500/50 transition-colors cursor-pointer space-y-1.5 text-xs"
                  >
                    <div className="flex items-center justify-between font-medium text-slate-300">
                      <span className="truncate max-w-[140px]">
                        {sim.filename || `Chunk #${sim.chunk_index}`}
                      </span>
                      {sim.score !== undefined && (
                        <span className="font-mono text-cyan-400 text-[11px] bg-cyan-950/80 px-1.5 py-0.5 rounded border border-cyan-800/40">
                          {(sim.score * 100).toFixed(1)}%
                        </span>
                      )}
                    </div>
                    <p className="text-slate-400 text-[11px] font-mono line-clamp-3">
                      {sim.content}
                    </p>
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-6 text-center text-xs text-slate-500 italic">
                Click &quot;Find Similar&quot; on any chunk to perform cross-document vector search and highlight related passages.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
