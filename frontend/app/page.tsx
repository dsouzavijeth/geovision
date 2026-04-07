"use client";

import { useState, useCallback } from "react";
import { CopilotSidebar } from "@copilotkit/react-ui";
import { useCopilotAction, useCopilotReadable } from "@copilotkit/react-core";
import ImageUpload from "@/components/ImageUpload";
import ImageCanvas, { DetectionResults } from "@/components/ImageCanvas";
import StatsBar from "@/components/StatsBar";

export default function HomePage() {
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [results, setResults] = useState<DetectionResults | null>(null);

  // ---- make image available to the agent as readable context ----
  useCopilotReadable({
    description: "Whether the user has uploaded an image",
    value: imageBase64
      ? { hasImage: true, fileName: fileName }
      : { hasImage: false },
  });

  // ---- make previous results available so agent can answer follow-ups ----
  useCopilotReadable({
    description: "Previous detection or segmentation results from the last analysis",
    value: results
      ? {
          hasResults: true,
          type: results.type,
          count: results.count,
          detections: results.detections,
        }
      : { hasResults: false },
  });

  // ---- action: agent pushes detection results to the frontend ----
  useCopilotAction({
    name: "displayResults",
    description:
      "Display detection or segmentation results on the image canvas. Call this after running inference to show overlays.",
    parameters: [
      {
        name: "resultsJson",
        type: "string",
        description:
          'JSON string of detection results with structure: { type: "bbox"|"obb"|"segmentation", count: number, detections: [...] }',
        required: true,
      },
    ],
    handler: async ({ resultsJson }) => {
      try {
        const parsed: DetectionResults = JSON.parse(resultsJson);
        setResults(parsed);
        return `Displayed ${parsed.count} ${parsed.type} results on the canvas.`;
      } catch {
        return "Failed to parse results JSON.";
      }
    },
  });

  const handleUpload = useCallback(
    async (base64: string, preview: string, name: string) => {
      setImageBase64(base64);
      setPreviewUrl(preview);
      setFileName(name);
      setResults(null);

      // send the image to the backend session store
      try {
        await fetch("http://127.0.0.1:8000/upload-image", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_b64: base64 }),
        });
      } catch (err) {
        console.error("Failed to upload image to backend:", err);
      }
    },
    []
  );

  return (
    <div className="app-container">
      <main className="main-panel">
        {/* Header */}
        <div className="header">
          <div className="header-icon">🌍</div>
          <div>
            <h1>
              Geo<span>Vision</span>
            </h1>
            <p className="header-subtitle">
              Conversational geospatial image analysis
            </p>
          </div>
        </div>

        {/* Upload or Canvas */}
        {!previewUrl ? (
          <ImageUpload onUpload={handleUpload} />
        ) : (
          <>
            <div className="canvas-container">
              <ImageCanvas imageSrc={previewUrl} results={results} />
            </div>
            <StatsBar results={results} />
            <button
              onClick={() => {
                setImageBase64(null);
                setPreviewUrl(null);
                setFileName(null);
                setResults(null);
              }}
              style={{
                marginTop: 12,
                padding: "8px 20px",
                borderRadius: 10,
                border: "1px solid var(--border-subtle)",
                background: "var(--bg-card)",
                color: "var(--text-secondary)",
                cursor: "pointer",
                fontSize: 13,
                fontFamily: "'Outfit', sans-serif",
                transition: "all 0.2s",
              }}
            >
              Upload new image
            </button>
          </>
        )}
      </main>

      {/* CopilotKit chat sidebar */}
      <CopilotSidebar
        defaultOpen={true}
        clickOutsideToClose={false}
        labels={{
          title: "GeoVision Assistant",
          initial:
            "Upload a satellite or aerial image, then ask me to detect objects, segment boundaries, or analyze what's in the scene.",
        }}
        instructions={`You are GeoVision, a geospatial image analysis assistant.
The user's uploaded image is available in your context as base64.
When running detection or segmentation, pass the base64 image to your tools.
After receiving results, call the displayResults action with the JSON to render overlays on the canvas.
For follow-up questions about previous results, use the cached results from context — do not re-run inference.`}
      />
    </div>
  );
}
