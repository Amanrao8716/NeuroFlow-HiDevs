"use client";

import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchDocuments } from "@/lib/api";
import { useAppStore } from "@/store/useAppStore";
import { DocumentUploadZone } from "@/components/documents/DocumentUploadZone";
import { DocumentChunksModal } from "@/components/documents/DocumentChunksModal";
import {
  FolderOpen,
  FileText,
  FileSpreadsheet,
  Image as ImageIcon,
  FileCode,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Search,
} from "lucide-react";
import type { DocumentItem } from "@/types/api";

export default function DocumentsPage() {
  const { selectedPipelineId } = useAppStore();

  const [selectedDoc, setSelectedDoc] = useState<DocumentItem | null>(null);
  const [isChunksModalOpen, setIsChunksModalOpen] = useState(false);
  const [searchFilter, setSearchFilter] = useState("");

  const { data: documents, isRefetching, refetch } = useQuery({
    queryKey: ["documents", selectedPipelineId],
    queryFn: () => (selectedPipelineId ? fetchDocuments(selectedPipelineId) : []),
    enabled: Boolean(selectedPipelineId),
    refetchInterval: (query) => {
      // Poll every 3s if any document is processing or queued
      const docs = query.state.data as DocumentItem[] | undefined;
      const isProcessing = docs?.some(
        (d) => d.status === "processing" || d.status === "queued"
      );
      return isProcessing ? 3000 : false;
    },
  });

  const getSourceIcon = (sourceType: string) => {
    switch (sourceType.toLowerCase()) {
      case "pdf":
        return <FileText className="h-4 w-4 text-rose-400" />;
      case "csv":
        return <FileSpreadsheet className="h-4 w-4 text-emerald-400" />;
      case "image":
        return <ImageIcon className="h-4 w-4 text-amber-400" />;
      default:
        return <FileCode className="h-4 w-4 text-cyan-400" />;
    }
  };

  const handleOpenChunks = (doc: DocumentItem) => {
    setSelectedDoc(doc);
    setIsChunksModalOpen(true);
  };

  const filteredDocs =
    documents?.filter((d) =>
      d.filename.toLowerCase().includes(searchFilter.toLowerCase())
    ) || [];

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-16">
      {/* Top Banner */}
      <div className="p-6 rounded-2xl glass-panel shadow-xl flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3.5">
          <div className="p-3 rounded-xl bg-gradient-to-tr from-cyan-500/20 to-indigo-500/20 border border-cyan-500/30 text-cyan-400">
            <FolderOpen className="h-6 w-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-slate-100">Document Management</h1>
            <p className="text-xs text-slate-400">
              Ingest, extract, chunk, and embed tenant knowledge files
            </p>
          </div>
        </div>

        <button
          onClick={() => refetch()}
          disabled={isRefetching}
          className="p-2.5 rounded-xl bg-slate-900 border border-slate-700/80 hover:bg-slate-800 text-slate-300 transition-colors w-fit"
          title="Refresh Documents"
        >
          <RefreshCw className={`h-4 w-4 ${isRefetching ? "animate-spin" : ""}`} />
        </button>
      </div>

      {/* Upload Zone */}
      <div className="rounded-2xl glass-panel p-6 border border-slate-800 shadow-xl space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
            Ingest Knowledge Documents
          </h2>
          {selectedPipelineId && (
            <span className="text-xs font-mono text-indigo-400 bg-indigo-950/60 px-2 py-0.5 rounded border border-indigo-800/40">
              Target Pipeline Active
            </span>
          )}
        </div>
        <DocumentUploadZone
          pipelineId={selectedPipelineId || ""}
          onUploadComplete={() => refetch()}
        />
      </div>

      {/* Document Table Section */}
      <div className="rounded-2xl glass-panel border border-slate-800 shadow-xl overflow-hidden space-y-4 p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
              Document Knowledge Base
            </h2>
            <span className="text-xs font-mono text-slate-400">
              ({filteredDocs.length} total)
            </span>
          </div>

          <div className="relative max-w-xs w-full">
            <Search className="h-4 w-4 text-slate-500 absolute left-3 top-2.5" />
            <input
              type="text"
              value={searchFilter}
              onChange={(e) => setSearchFilter(e.target.value)}
              placeholder="Search documents by name..."
              className="w-full bg-slate-900 border border-slate-700/80 rounded-xl pl-9 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 outline-none focus:border-indigo-500 font-mono"
            />
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 font-semibold uppercase tracking-wider text-[11px]">
                <th className="py-3 px-4">Filename</th>
                <th className="py-3 px-4">Type</th>
                <th className="py-3 px-4">Processing Status</th>
                <th className="py-3 px-4">Chunks</th>
                <th className="py-3 px-4">Ingestion Date</th>
                <th className="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {filteredDocs.length > 0 ? (
                filteredDocs.map((doc) => {
                  const isProcessing =
                    doc.status === "processing" || doc.status === "queued";
                  const isCompleted = doc.status === "completed";
                  const isFailed = doc.status === "failed";

                  return (
                    <tr
                      key={doc.id}
                      className="hover:bg-slate-900/60 transition-colors group"
                    >
                      {/* Filename */}
                      <td className="py-3.5 px-4 font-sans font-medium text-slate-200">
                        <div className="flex items-center gap-2.5">
                          {getSourceIcon(doc.source_type)}
                          <span className="truncate max-w-xs">{doc.filename}</span>
                        </div>
                      </td>

                      {/* Source Type */}
                      <td className="py-3.5 px-4">
                        <span className="px-2 py-0.5 rounded-md bg-slate-800 border border-slate-700 text-slate-300 uppercase text-[10px]">
                          {doc.source_type}
                        </span>
                      </td>

                      {/* Processing Status (With animated pulsing blue indicator when processing!) */}
                      <td className="py-3.5 px-4">
                        {isProcessing ? (
                          <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-blue-950/80 border border-blue-500/50 text-blue-400 text-xs font-sans">
                            <span className="relative flex h-2 w-2">
                              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                              <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500"></span>
                            </span>
                            <span className="capitalize font-medium">Processing...</span>
                          </div>
                        ) : isCompleted ? (
                          <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-950/60 border border-emerald-800/50 text-emerald-400 text-xs font-sans">
                            <CheckCircle2 className="h-3.5 w-3.5" />
                            <span>Completed</span>
                          </div>
                        ) : isFailed ? (
                          <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-rose-950/60 border border-rose-800/50 text-rose-400 text-xs font-sans">
                            <AlertCircle className="h-3.5 w-3.5" />
                            <span>Failed</span>
                          </div>
                        ) : (
                          <span className="text-slate-400 capitalize">{doc.status}</span>
                        )}
                      </td>

                      {/* Chunk Count */}
                      <td className="py-3.5 px-4">
                        <span className="text-slate-300 font-semibold">
                          {doc.chunk_count}
                        </span>
                      </td>

                      {/* Ingestion Date */}
                      <td className="py-3.5 px-4 text-slate-400 text-[11px]">
                        {new Date(doc.created_at).toLocaleDateString(undefined, {
                          month: "short",
                          day: "numeric",
                          year: "numeric",
                        })}
                      </td>

                      {/* Actions */}
                      <td className="py-3.5 px-4 text-right">
                        <button
                          onClick={() => handleOpenChunks(doc)}
                          className="px-3 py-1 rounded-lg bg-indigo-600/10 hover:bg-indigo-600/20 border border-indigo-500/30 text-indigo-300 hover:text-indigo-200 font-sans text-xs font-medium transition-colors"
                        >
                          View Chunks
                        </button>
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={6} className="py-12 text-center text-slate-500 italic font-sans">
                    {searchFilter
                      ? "No documents match the search query."
                      : "No documents uploaded for this pipeline yet. Use the upload zone above to ingest files."}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Document Chunks Modal */}
      <DocumentChunksModal
        document={selectedDoc}
        pipelineId={selectedPipelineId || ""}
        isOpen={isChunksModalOpen}
        onClose={() => setIsChunksModalOpen(false)}
      />
    </div>
  );
}
