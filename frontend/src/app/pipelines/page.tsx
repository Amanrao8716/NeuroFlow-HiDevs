"use client";

import React, { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchPipelines,
  createPipeline,
  updatePipeline,
  deletePipeline,
} from "@/lib/api";
import { PipelineCard } from "@/components/pipelines/PipelineCard";
import { PipelineEditorModal } from "@/components/pipelines/PipelineEditorModal";
import { PipelineAnalyticsDrawer } from "@/components/pipelines/PipelineAnalyticsDrawer";
import { useAppStore } from "@/store/useAppStore";
import { Plus, GitFork, Loader2, RefreshCw } from "lucide-react";
import type { PipelineConfig, PipelineResponse } from "@/types/api";

export default function PipelinesPage() {
  const queryClient = useQueryClient();
  const { addToast } = useAppStore();

  const [isEditorOpen, setIsEditorOpen] = useState(false);
  const [editingPipeline, setEditingPipeline] = useState<PipelineResponse | null>(null);

  const [isAnalyticsOpen, setIsAnalyticsOpen] = useState(false);
  const [analyticsPipeline, setAnalyticsPipeline] = useState<PipelineResponse | null>(null);

  const { data: pipelines, isLoading, isRefetching, refetch } = useQuery({
    queryKey: ["pipelines"],
    queryFn: fetchPipelines,
  });

  const saveMutation = useMutation({
    mutationFn: async ({
      config,
      isEdit,
      id,
    }: {
      config: PipelineConfig;
      isEdit: boolean;
      id?: string;
    }) => {
      if (isEdit && id) {
        return updatePipeline(id, {
          name: config.name,
          description: config.description,
          config_updates: config,
        });
      }
      return createPipeline({
        name: config.name,
        description: config.description,
        config: config,
      });
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ["pipelines"] });
      addToast(
        "success",
        variables.isEdit
          ? "Pipeline configuration updated and new version deployed!"
          : "New pipeline created successfully!"
      );
    },
    onError: (err: any) => {
      addToast("error", err.message || "Failed to save pipeline configuration.");
    },
  });

  const deleteMutation = useMutation({
    mutationFn: deletePipeline,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["pipelines"] });
      addToast("success", "Pipeline archived successfully.");
    },
    onError: (err: any) => {
      addToast("error", err.message || "Failed to archive pipeline.");
    },
  });

  const handleOpenCreate = () => {
    setEditingPipeline(null);
    setIsEditorOpen(true);
  };

  const handleOpenEdit = (p: PipelineResponse) => {
    setEditingPipeline(p);
    setIsEditorOpen(true);
  };

  const handleOpenAnalytics = (p: PipelineResponse) => {
    setAnalyticsPipeline(p);
    setIsAnalyticsOpen(true);
  };

  const handleDelete = (id: string) => {
    if (confirm("Are you sure you want to archive this pipeline? Its historical runs will be preserved.")) {
      deleteMutation.mutate(id);
    }
  };

  const handleSaveConfig = async (config: PipelineConfig, isEdit: boolean, id?: string) => {
    await saveMutation.mutateAsync({ config, isEdit, id });
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-16">
      {/* Top Banner & Actions */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-6 rounded-2xl glass-panel shadow-xl">
        <div className="flex items-center gap-3.5">
          <div className="p-3 rounded-xl bg-gradient-to-tr from-indigo-500/20 to-cyan-500/20 border border-indigo-500/30 text-indigo-400">
            <GitFork className="h-6 w-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-slate-100">Pipeline Manager</h1>
            <p className="text-xs text-slate-400">
              Configure, version, benchmark, and optimize tenant RAG pipelines
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => refetch()}
            disabled={isRefetching}
            className="p-2.5 rounded-xl bg-slate-900 border border-slate-700/80 hover:bg-slate-800 text-slate-300 transition-colors"
            title="Refresh Pipelines"
          >
            <RefreshCw className={`h-4 w-4 ${isRefetching ? "animate-spin" : ""}`} />
          </button>
          <button
            onClick={handleOpenCreate}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-indigo-500 via-indigo-600 to-cyan-500 text-white text-xs font-semibold uppercase tracking-wider hover:opacity-95 transition-opacity shadow-lg shadow-indigo-500/20"
          >
            <Plus className="h-4 w-4" />
            <span>Create Pipeline</span>
          </button>
        </div>
      </div>

      {/* Pipelines Grid */}
      {isLoading ? (
        <div className="flex flex-col items-center justify-center p-16 space-y-3">
          <Loader2 className="h-8 w-8 animate-spin text-indigo-400" />
          <p className="text-xs text-slate-400 font-mono">Loading configured pipelines...</p>
        </div>
      ) : pipelines && pipelines.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {pipelines.map((pipeline) => (
            <PipelineCard
              key={pipeline.id}
              pipeline={pipeline}
              onEdit={handleOpenEdit}
              onOpenAnalytics={handleOpenAnalytics}
              onDelete={handleDelete}
            />
          ))}
        </div>
      ) : (
        <div className="p-12 rounded-2xl glass-panel text-center space-y-4 max-w-lg mx-auto">
          <div className="p-4 rounded-2xl bg-slate-900 border border-slate-800 w-fit mx-auto text-indigo-400">
            <GitFork className="h-8 w-8" />
          </div>
          <h3 className="text-base font-semibold text-slate-200">No Pipelines Configured</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Create your first RAG pipeline to configure document extraction, vector/lexical retrieval, and LLM provider parameters.
          </p>
          <button
            onClick={handleOpenCreate}
            className="px-5 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold"
          >
            Create Pipeline
          </button>
        </div>
      )}

      {/* Editor Modal (Monaco) */}
      <PipelineEditorModal
        isOpen={isEditorOpen}
        pipeline={editingPipeline}
        onClose={() => setIsEditorOpen(false)}
        onSave={handleSaveConfig}
      />

      {/* Analytics Drawer */}
      <PipelineAnalyticsDrawer
        isOpen={isAnalyticsOpen}
        pipeline={analyticsPipeline}
        onClose={() => setIsAnalyticsOpen(false)}
      />
    </div>
  );
}
