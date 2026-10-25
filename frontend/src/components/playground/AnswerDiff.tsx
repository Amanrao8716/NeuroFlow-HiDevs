"use client";

import React from "react";
import { GitCompare } from "lucide-react";

interface AnswerDiffProps {
  answerA: string;
  answerB: string;
  nameA: string;
  nameB: string;
}

export function AnswerDiff({ answerA, answerB, nameA, nameB }: AnswerDiffProps) {
  const wordsA = answerA.trim().split(/\s+/);
  const wordsB = answerB.trim().split(/\s+/);

  const setB = new Set(wordsB.map((w) => w.toLowerCase()));
  const setA = new Set(wordsA.map((w) => w.toLowerCase()));

  return (
    <div className="rounded-2xl bg-slate-900/60 border border-slate-800 p-6 space-y-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <GitCompare className="h-4 w-4 text-indigo-400" />
          <h3 className="text-sm font-semibold text-slate-200">Answer Divergence Analysis</h3>
        </div>
        <div className="flex items-center gap-4 text-xs font-medium">
          <span className="flex items-center gap-1.5 text-cyan-400">
            <span className="h-2 w-2 rounded-full bg-cyan-400" /> Unique to {nameA}
          </span>
          <span className="flex items-center gap-1.5 text-purple-400">
            <span className="h-2 w-2 rounded-full bg-purple-400" /> Unique to {nameB}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm leading-relaxed font-mono">
        {/* Pipeline A with highlights */}
        <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800/80">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
            {nameA} Generation
          </div>
          <p className="text-slate-300">
            {wordsA.map((word, idx) => {
              const isUnique = !setB.has(word.toLowerCase());
              return (
                <span
                  key={idx}
                  className={
                    isUnique
                      ? "bg-cyan-500/20 text-cyan-300 px-0.5 rounded font-semibold"
                      : "text-slate-300"
                  }
                >
                  {word}{" "}
                </span>
              );
            })}
          </p>
        </div>

        {/* Pipeline B with highlights */}
        <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800/80">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
            {nameB} Generation
          </div>
          <p className="text-slate-300">
            {wordsB.map((word, idx) => {
              const isUnique = !setA.has(word.toLowerCase());
              return (
                <span
                  key={idx}
                  className={
                    isUnique
                      ? "bg-purple-500/20 text-purple-300 px-0.5 rounded font-semibold"
                      : "text-slate-300"
                  }
                >
                  {word}{" "}
                </span>
              );
            })}
          </p>
        </div>
      </div>
    </div>
  );
}
