"use client";

import React, { useState, useEffect } from "react";
import Editor from "@monaco-editor/react";
import { X, CheckCircle2, AlertTriangle, Save, FileCode } from "lucide-react";
import type { PipelineConfig, PipelineResponse } from "@/types/api";

interface PipelineEditorModalProps {
  pipeline?: PipelineResponse | null;
  isOpen: boolean;
  onClose: () => void;
  onSave: (config: PipelineConfig, isEdit: boolean, id?: string) => Promise<void>;
}

const DEFAULT_PIPELINE_CONFIG: PipelineConfig = {
  name: "New-RAG-Pipeline",
  description: "Optimized hybrid RAG pipeline with cross-encoder reranking.",
  ingestion: {
    chunking_strategy: "auto",
    chunk_size_tokens: 512,
    chunk_overlap_tokens: 50,
    extractors_enabled: ["pdf", "docx", "csv", "image", "url", "text"],
  },
  retrieval: {
    dense_k: 10,
    sparse_k: 10,
    reranker: "cross-encoder",
    top_k_after_rerank: 5,
    query_expansion: true,
    metadata_filters_enabled: true,
  },
  generation: {
    model_routing: "gpt-4o-mini",
    max_context_tokens: 4000,
    temperature: 0.7,
    system_prompt_variant: "default",
  },
  evaluation: {
    auto_evaluate: true,
    training_threshold: 0.8,
  },
};

export function PipelineEditorModal({
  pipeline,
  isOpen,
  onClose,
  onSave,
}: PipelineEditorModalProps) {
  const [jsonText, setJsonText] = useState("");
  const [errors, setErrors] = useState<string[]>([]);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    if (pipeline) {
      setJsonText(JSON.stringify(pipeline.config, null, 2));
    } else {
      setJsonText(JSON.stringify(DEFAULT_PIPELINE_CONFIG, null, 2));
    }
    setErrors([]);
  }, [pipeline, isOpen]);

  if (!isOpen) return null;

  // Validate JSON against actual backend PipelineConfig Pydantic schema
  const validateConfig = (content: string): { isValid: boolean; parsed?: PipelineConfig } => {
    const errs: string[] = [];

    let parsed: any;
    try {
      parsed = JSON.parse(content);
    } catch (e: any) {
      setErrors([`JSON Syntax Error: ${e.message}`]);
      return { isValid: false };
    }

    if (!parsed || typeof parsed !== "object") {
      setErrors(["Configuration must be a valid JSON object."]);
      return { isValid: false };
    }

    // Name validation
    if (!parsed.name || typeof parsed.name !== "string" || parsed.name.trim().length === 0) {
      errs.push("name: Pipeline name is required (string length 1..100).");
    } else if (parsed.name.length > 100) {
      errs.push("name: Pipeline name cannot exceed 100 characters.");
    }

    // Ingestion validation
    if (parsed.ingestion) {
      const { chunk_size_tokens, chunk_overlap_tokens } = parsed.ingestion;
      if (chunk_size_tokens !== undefined) {
        if (typeof chunk_size_tokens !== "number" || chunk_size_tokens <= 0 || chunk_size_tokens > 8192) {
          errs.push("ingestion.chunk_size_tokens must be an integer between 1 and 8192.");
        }
      }
      if (chunk_overlap_tokens !== undefined) {
        if (typeof chunk_overlap_tokens !== "number" || chunk_overlap_tokens < 0) {
          errs.push("ingestion.chunk_overlap_tokens must be >= 0.");
        }
        if (
          chunk_size_tokens &&
          chunk_overlap_tokens >= chunk_size_tokens
        ) {
          errs.push(
            `ingestion.chunk_overlap_tokens (${chunk_overlap_tokens}) must be strictly less than chunk_size_tokens (${chunk_size_tokens}).`
          );
        }
      }
    }

    // Retrieval validation
    if (parsed.retrieval) {
      const { dense_k = 10, sparse_k = 10, top_k_after_rerank = 5 } = parsed.retrieval;
      if (dense_k <= 0 || dense_k > 100) {
        errs.push("retrieval.dense_k must be between 1 and 100.");
      }
      if (sparse_k <= 0 || sparse_k > 100) {
        errs.push("retrieval.sparse_k must be between 1 and 100.");
      }
      if (top_k_after_rerank <= 0 || top_k_after_rerank > 100) {
        errs.push("retrieval.top_k_after_rerank must be between 1 and 100.");
      }
      if (top_k_after_rerank > dense_k + sparse_k) {
        errs.push(
          `retrieval.top_k_after_rerank (${top_k_after_rerank}) cannot exceed dense_k + sparse_k (${dense_k + sparse_k}).`
        );
      }
    }

    // Generation validation
    if (parsed.generation) {
      const { temperature, max_context_tokens } = parsed.generation;
      if (temperature !== undefined && (temperature < 0.0 || temperature > 2.0)) {
        errs.push("generation.temperature must be between 0.0 and 2.0.");
      }
      if (max_context_tokens !== undefined && (max_context_tokens <= 0 || max_context_tokens > 32000)) {
        errs.push("generation.max_context_tokens must be between 1 and 32000.");
      }
    }

    // Evaluation validation
    if (parsed.evaluation) {
      const { training_threshold } = parsed.evaluation;
      if (training_threshold !== undefined && (training_threshold < 0.0 || training_threshold > 1.0)) {
        errs.push("evaluation.training_threshold must be between 0.0 and 1.0.");
      }
    }

    setErrors(errs);
    return { isValid: errs.length === 0, parsed };
  };

  const handleEditorChange = (value: string | undefined) => {
    const text = value || "";
    setJsonText(text);
    validateConfig(text);
  };

  const handleSave = async () => {
    const { isValid, parsed } = validateConfig(jsonText);
    if (!isValid || !parsed) return;

    setIsSaving(true);
    try {
      await onSave(parsed, Boolean(pipeline), pipeline?.id);
      onClose();
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-md p-4 animate-in fade-in duration-200">
      <div className="relative w-full max-w-4xl h-[85vh] flex flex-col rounded-2xl bg-slate-900 border border-slate-700 shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="p-5 border-b border-slate-800 flex items-center justify-between bg-slate-950/70">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              <FileCode className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-100">
                {pipeline ? `Edit Configuration — ${pipeline.name}` : "Create New Pipeline"}
              </h2>
              <p className="text-xs text-slate-400">
                Schema validated against backend PipelineConfig Pydantic model
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

        {/* Monaco Editor Container */}
        <div className="flex-1 relative bg-slate-950">
          <Editor
            height="100%"
            defaultLanguage="json"
            theme="vs-dark"
            value={jsonText}
            onChange={handleEditorChange}
            options={{
              minimap: { enabled: false },
              fontSize: 13,
              formatOnPaste: true,
              formatOnType: true,
              scrollBeyondLastLine: false,
              automaticLayout: true,
            }}
          />
        </div>

        {/* Validation Errors & Footer */}
        <div className="p-4 border-t border-slate-800 bg-slate-950/90 space-y-3">
          {errors.length > 0 ? (
            <div className="p-3 rounded-xl bg-rose-950/50 border border-rose-800/60 max-h-28 overflow-y-auto space-y-1">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-rose-300">
                <AlertTriangle className="h-4 w-4 text-rose-400" />
                <span>Validation Errors ({errors.length})</span>
              </div>
              <ul className="list-disc list-inside text-xs text-rose-300/90 font-mono space-y-0.5 pl-2">
                {errors.map((err, i) => (
                  <li key={i}>{err}</li>
                ))}
              </ul>
            </div>
          ) : (
            <div className="flex items-center gap-2 text-xs text-emerald-400 font-mono">
              <CheckCircle2 className="h-4 w-4 text-emerald-400" />
              <span>Pipeline configuration passes all schema constraints.</span>
            </div>
          )}

          <div className="flex items-center justify-end gap-3 pt-2">
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={isSaving || errors.length > 0}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-indigo-500 to-cyan-500 text-white text-xs font-semibold uppercase tracking-wider hover:opacity-95 transition-opacity disabled:opacity-50 shadow-lg shadow-indigo-500/25"
            >
              <Save className="h-4 w-4" />
              <span>{isSaving ? "Validating & Saving..." : pipeline ? "Deploy Update" : "Create Pipeline"}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
