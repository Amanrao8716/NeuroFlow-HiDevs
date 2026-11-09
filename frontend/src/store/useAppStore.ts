import { create } from "zustand";
import type { ChunkItem } from "@/types/api";

export interface ToastMessage {
  id: string;
  type: "success" | "error" | "info" | "warning";
  text: string;
}

interface AppState {
  selectedPipelineId: string | null;
  comparePipelineId: string | null;
  isCompareMode: boolean;
  activeChunkDetail: ChunkItem | null;
  toasts: ToastMessage[];

  setSelectedPipelineId: (id: string | null) => void;
  setComparePipelineId: (id: string | null) => void;
  setIsCompareMode: (enabled: boolean) => void;
  setActiveChunkDetail: (chunk: ChunkItem | null) => void;
  addToast: (type: ToastMessage["type"], text: string) => void;
  removeToast: (id: string) => void;
}

export const useAppStore = create<AppState>((set) => ({
  selectedPipelineId: null,
  comparePipelineId: null,
  isCompareMode: false,
  activeChunkDetail: null,
  toasts: [],

  setSelectedPipelineId: (id) => set({ selectedPipelineId: id }),
  setComparePipelineId: (id) => set({ comparePipelineId: id }),
  setIsCompareMode: (enabled) => set({ isCompareMode: enabled }),
  setActiveChunkDetail: (chunk) => set({ activeChunkDetail: chunk }),
  addToast: (type, text) => {
    const id = Math.random().toString(36).substring(2, 9);
    set((state) => ({ toasts: [...state.toasts, { id, type, text }] }));
    setTimeout(() => {
      set((state) => ({
        toasts: state.toasts.filter((t) => t.id !== id),
      }));
    }, 4000);
  },
  removeToast: (id) =>
    set((state) => ({
      toasts: state.toasts.filter((t) => t.id !== id),
    })),
}));
