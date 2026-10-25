"use client";

import React, { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import {
  UploadCloud,
  FileText,
  FileSpreadsheet,
  Image as ImageIcon,
  FileCode,
  CheckCircle2,
  AlertCircle,
  X,
} from "lucide-react";
import { uploadDocument } from "@/lib/api";
import { useAppStore } from "@/store/useAppStore";

interface UploadFileItem {
  id: string;
  file: File;
  progress: number;
  status: "pending" | "uploading" | "completed" | "error";
  error?: string;
}

export function DocumentUploadZone({
  pipelineId,
  onUploadComplete,
}: {
  pipelineId: string;
  onUploadComplete: () => void;
}) {
  const { addToast } = useAppStore();
  const [fileItems, setFileItems] = useState<UploadFileItem[]>([]);
  const [isUploadingAll, setIsUploadingAll] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newItems: UploadFileItem[] = acceptedFiles.map((f) => ({
      id: Math.random().toString(36).substring(2, 9),
      file: f,
      progress: 0,
      status: "pending",
    }));
    setFileItems((prev) => [...prev, ...newItems]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
      "text/csv": [".csv"],
      "image/jpeg": [".jpg", ".jpeg"],
      "image/png": [".png"],
      "image/webp": [".webp"],
      "text/plain": [".txt"],
    },
    maxSize: 100 * 1024 * 1024, // 100 MB
  });

  const getFileIcon = (filename: string) => {
    const ext = filename.split(".").pop()?.toLowerCase();
    if (ext === "pdf") return <FileText className="h-5 w-5 text-rose-400" />;
    if (ext === "csv") return <FileSpreadsheet className="h-5 w-5 text-emerald-400" />;
    if (["jpg", "jpeg", "png", "webp"].includes(ext || ""))
      return <ImageIcon className="h-5 w-5 text-amber-400" />;
    return <FileCode className="h-5 w-5 text-cyan-400" />;
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const handleUploadAll = async () => {
    if (!pipelineId) {
      addToast("warning", "Select a pipeline first before uploading documents.");
      return;
    }
    const pending = fileItems.filter((f) => f.status === "pending" || f.status === "error");
    if (pending.length === 0) return;

    setIsUploadingAll(true);

    for (const item of pending) {
      setFileItems((prev) =>
        prev.map((f) => (f.id === item.id ? { ...f, status: "uploading", progress: 10 } : f))
      );

      try {
        await uploadDocument(pipelineId, item.file, (percent) => {
          setFileItems((prev) =>
            prev.map((f) => (f.id === item.id ? { ...f, progress: percent } : f))
          );
        });

        setFileItems((prev) =>
          prev.map((f) => (f.id === item.id ? { ...f, status: "completed", progress: 100 } : f))
        );
      } catch (err: any) {
        setFileItems((prev) =>
          prev.map((f) =>
            f.id === item.id
              ? { ...f, status: "error", error: err.message || "Upload failed." }
              : f
          )
        );
      }
    }

    setIsUploadingAll(false);
    onUploadComplete();
    addToast("success", "Document ingestion batch queued for background processing.");
  };

  const removeFile = (id: string) => {
    setFileItems((prev) => prev.filter((f) => f.id !== id));
  };

  return (
    <div className="space-y-4">
      {/* Drag & Drop Area */}
      <div
        {...getRootProps()}
        className={`p-8 rounded-2xl border-2 border-dashed transition-all duration-300 text-center cursor-pointer flex flex-col items-center justify-center ${
          isDragActive
            ? "border-cyan-400 bg-cyan-950/20 shadow-xl shadow-cyan-500/10 scale-[1.01]"
            : "border-slate-700/80 bg-slate-950/60 hover:border-indigo-500/50 hover:bg-slate-900/40"
        }`}
      >
        <input {...getInputProps()} />
        <div
          className={`p-4 rounded-2xl mb-3 transition-colors ${
            isDragActive ? "bg-cyan-500/20 text-cyan-300" : "bg-slate-900 text-indigo-400"
          }`}
        >
          <UploadCloud className="h-8 w-8 animate-bounce" />
        </div>
        <h3 className="text-sm font-semibold text-slate-100">
          {isDragActive ? "Drop documents here to ingest" : "Drag & drop documents here, or browse"}
        </h3>
        <p className="text-xs text-slate-400 mt-1">
          Supports PDF, DOCX, CSV, PNG, JPG, and WebP (Up to 100 MB per file)
        </p>
      </div>

      {/* Selected Files List & Individual Progress */}
      {fileItems.length > 0 && (
        <div className="p-4 rounded-2xl glass-panel border border-slate-800 space-y-3">
          <div className="flex items-center justify-between text-xs font-semibold text-slate-300">
            <span>Staged Documents ({fileItems.length})</span>
            <button
              onClick={handleUploadAll}
              disabled={isUploadingAll || fileItems.every((f) => f.status === "completed")}
              className="px-4 py-1.5 rounded-lg bg-gradient-to-r from-indigo-500 to-cyan-500 text-white text-xs font-semibold uppercase tracking-wider hover:opacity-95 transition-opacity disabled:opacity-40"
            >
              {isUploadingAll ? "Uploading..." : "Start Batch Ingestion"}
            </button>
          </div>

          <div className="space-y-2 max-h-56 overflow-y-auto">
            {fileItems.map((item) => (
              <div
                key={item.id}
                className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between gap-3 text-xs"
              >
                <div className="flex items-center gap-3 min-w-0 flex-1">
                  {getFileIcon(item.file.name)}
                  <div className="min-w-0 flex-1">
                    <p className="font-medium text-slate-200 truncate">{item.file.name}</p>
                    <p className="text-[11px] font-mono text-slate-500">
                      {formatFileSize(item.file.size)}
                    </p>
                  </div>
                </div>

                {/* Status / Progress */}
                <div className="flex items-center gap-3">
                  {item.status === "uploading" && (
                    <div className="flex items-center gap-2">
                      <div className="w-24 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-cyan-400 transition-all duration-300"
                          style={{ width: `${item.progress}%` }}
                        />
                      </div>
                      <span className="font-mono text-cyan-400 text-[11px]">
                        {item.progress}%
                      </span>
                    </div>
                  )}
                  {item.status === "completed" && (
                    <span className="flex items-center gap-1 text-emerald-400 text-xs font-medium">
                      <CheckCircle2 className="h-4 w-4" /> Queued
                    </span>
                  )}
                  {item.status === "error" && (
                    <span
                      className="flex items-center gap-1 text-rose-400 text-xs font-medium"
                      title={item.error}
                    >
                      <AlertCircle className="h-4 w-4" /> Failed
                    </span>
                  )}
                  {item.status === "pending" && (
                    <span className="text-slate-400 text-[11px] font-mono">Ready</span>
                  )}

                  <button
                    onClick={() => removeFile(item.id)}
                    className="p-1 rounded text-slate-500 hover:text-slate-300 transition-colors"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
