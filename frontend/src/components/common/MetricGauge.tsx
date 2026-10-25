"use client";

import React from "react";

interface MetricGaugeProps {
  label: string;
  score: number | null | undefined;
  size?: "sm" | "md" | "lg";
  showLabel?: boolean;
}

export function getScoreColor(score: number | null | undefined): {
  text: string;
  bg: string;
  border: string;
  stroke: string;
} {
  if (score === null || score === undefined) {
    return {
      text: "text-slate-400",
      bg: "bg-slate-800/50",
      border: "border-slate-700",
      stroke: "#64748b",
    };
  }
  if (score >= 0.8) {
    return {
      text: "text-emerald-400",
      bg: "bg-emerald-950/40",
      border: "border-emerald-800/50",
      stroke: "#10b981",
    };
  }
  if (score >= 0.6) {
    return {
      text: "text-amber-400",
      bg: "bg-amber-950/40",
      border: "border-amber-800/50",
      stroke: "#f59e0b",
    };
  }
  return {
    text: "text-rose-400",
    bg: "bg-rose-950/40",
    border: "border-rose-800/50",
    stroke: "#ef4444",
  };
}

export function MetricGauge({ label, score, size = "md", showLabel = true }: MetricGaugeProps) {
  const colors = getScoreColor(score);
  const normalizedScore = score !== null && score !== undefined ? Math.max(0, Math.min(1, score)) : 0;
  const percentage = Math.round(normalizedScore * 100);

  // SVG dimensions
  const dimensions = {
    sm: { width: 56, strokeWidth: 5, radius: 22, text: "text-xs" },
    md: { width: 76, strokeWidth: 6, radius: 30, text: "text-sm font-semibold" },
    lg: { width: 100, strokeWidth: 8, radius: 40, text: "text-lg font-bold" },
  }[size];

  const circumference = 2 * Math.PI * dimensions.radius;
  const strokeDashoffset = circumference - (circumference * percentage) / 100;

  return (
    <div className="flex flex-col items-center justify-center gap-1.5">
      <div className="relative inline-flex items-center justify-center">
        <svg
          width={dimensions.width}
          height={dimensions.width}
          className="rotate-[-90deg] transform transition-all duration-500"
        >
          {/* Background circle */}
          <circle
            cx={dimensions.width / 2}
            cy={dimensions.width / 2}
            r={dimensions.radius}
            fill="transparent"
            stroke="rgba(255, 255, 255, 0.08)"
            strokeWidth={dimensions.strokeWidth}
          />
          {/* Animated progress circle */}
          <circle
            cx={dimensions.width / 2}
            cy={dimensions.width / 2}
            r={dimensions.radius}
            fill="transparent"
            stroke={colors.stroke}
            strokeWidth={dimensions.strokeWidth}
            strokeDasharray={circumference}
            strokeDashoffset={score !== null && score !== undefined ? strokeDashoffset : circumference}
            strokeLinecap="round"
            className="transition-all duration-700 ease-out"
          />
        </svg>
        <span className={`absolute ${dimensions.text} ${colors.text}`}>
          {score !== null && score !== undefined ? `${percentage}%` : "--"}
        </span>
      </div>
      {showLabel && (
        <span className="text-center text-xs font-medium text-slate-400 capitalize">
          {label.replace("_", " ")}
        </span>
      )}
    </div>
  );
}
