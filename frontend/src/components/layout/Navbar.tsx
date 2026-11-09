"use client";

import React from "react";
import { usePathname } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { fetchPipelines } from "@/lib/api";
import { useAppStore } from "@/store/useAppStore";
import { Database, CheckCircle2, AlertCircle, Info } from "lucide-react";

export function Navbar() {
  const pathname = usePathname();
  const { selectedPipelineId, setSelectedPipelineId, toasts, removeToast } = useAppStore();

  const { data: pipelines } = useQuery({
    queryKey: ["pipelines"],
    queryFn: fetchPipelines,
  });

  // Auto-select first pipeline if none selected
  React.useEffect(() => {
    if (!selectedPipelineId && pipelines && pipelines.length > 0) {
      setSelectedPipelineId(pipelines[0].id);
    }
  }, [pipelines, selectedPipelineId, setSelectedPipelineId]);

  const getPageTitle = () => {
    if (pathname.startsWith("/playground")) return "Query Playground";
    if (pathname.startsWith("/pipelines")) return "Pipeline Manager";
    if (pathname.startsWith("/evaluations")) return "Live Evaluation Feed";
    if (pathname.startsWith("/documents")) return "Document Ingestion";
    return "NeuroFlow";
  };

  return (
    <>
      <header className="h-16 border-b border-slate-800/80 bg-slate-950/60 backdrop-blur-xl px-8 flex items-center justify-between sticky top-0 z-30">
        <div className="flex items-center gap-3">
          <h2 className="text-base font-semibold text-slate-100">{getPageTitle()}</h2>
          <span className="text-slate-600">/</span>
          <span className="text-xs text-slate-400 font-mono">Workspace</span>
        </div>

        <div className="flex items-center gap-4">
          {/* Active Pipeline Selector */}
          <div className="flex items-center gap-2 bg-slate-900/80 border border-slate-800 rounded-xl px-3 py-1.5 shadow-sm">
            <Database className="h-3.5 w-3.5 text-indigo-400" />
            <span className="text-xs text-slate-400">Pipeline:</span>
            <select
              value={selectedPipelineId || ""}
              onChange={(e) => setSelectedPipelineId(e.target.value)}
              className="bg-transparent text-xs font-medium text-slate-200 outline-none cursor-pointer pr-1"
            >
              {pipelines && pipelines.length > 0 ? (
                pipelines.map((p) => (
                  <option key={p.id} value={p.id} className="bg-slate-900 text-slate-200">
                    {p.name} (v{p.version})
                  </option>
                ))
              ) : (
                <option value="" className="bg-slate-900 text-slate-400">
                  No pipelines available
                </option>
              )}
            </select>
          </div>

          {/* Health Badge */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-950/50 border border-emerald-800/40 text-emerald-400 text-xs font-mono font-medium">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
            <span>API Online</span>
          </div>
        </div>
      </header>

      {/* Floating Toast Notification Container */}
      <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-2 max-w-sm w-full pointer-events-none">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className={`pointer-events-auto flex items-start gap-3 p-3.5 rounded-xl border shadow-xl backdrop-blur-lg animate-in fade-in slide-in-from-bottom-3 duration-200 ${
              toast.type === "success"
                ? "bg-slate-900/90 border-emerald-500/40 text-emerald-300"
                : toast.type === "error"
                ? "bg-slate-900/90 border-rose-500/40 text-rose-300"
                : toast.type === "warning"
                ? "bg-slate-900/90 border-amber-500/40 text-amber-300"
                : "bg-slate-900/90 border-indigo-500/40 text-indigo-300"
            }`}
          >
            {toast.type === "success" && <CheckCircle2 className="h-4 w-4 mt-0.5 text-emerald-400" />}
            {toast.type === "error" && <AlertCircle className="h-4 w-4 mt-0.5 text-rose-400" />}
            {toast.type === "warning" && <AlertCircle className="h-4 w-4 mt-0.5 text-amber-400" />}
            {toast.type === "info" && <Info className="h-4 w-4 mt-0.5 text-indigo-400" />}
            <p className="text-xs font-medium text-slate-200 flex-1 leading-snug">{toast.text}</p>
            <button
              onClick={() => removeToast(toast.id)}
              className="text-slate-500 hover:text-slate-300 text-xs"
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </>
  );
}
