"use client";

import { DetectionResults } from "./ImageCanvas";

interface StatsBarProps {
  results: DetectionResults | null;
}

export default function StatsBar({ results }: StatsBarProps) {
  if (!results || !results.detections || results.detections.length === 0) {
    return null;
  }

  // group by label
  const counts: Record<string, number> = {};
  for (const det of results.detections) {
    counts[det.label] = (counts[det.label] || 0) + 1;
  }

  const dotColors: Record<string, string> = {
    bbox: "var(--accent-blue)",
    obb: "var(--accent-amber)",
    segmentation: "var(--accent-green)",
  };
  const dotColor = dotColors[results.type] || "var(--accent-cyan)";

  return (
    <div className="stats-bar">
      <div className="stat-chip">
        <span className="dot" style={{ background: dotColor }} />
        <span style={{ color: "var(--text-secondary)" }}>
          {results.type === "bbox"
            ? "Detection"
            : results.type === "obb"
            ? "Oriented Detection"
            : "Segmentation"}
        </span>
        <span className="count">{results.count} objects</span>
      </div>
      {Object.entries(counts).map(([label, count]) => (
        <div className="stat-chip" key={label}>
          <span>{label}</span>
          <span className="count">{count}</span>
        </div>
      ))}
    </div>
  );
}
