"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { CopilotSidebar } from "@copilotkit/react-ui";
import { useCopilotChat } from "@copilotkit/react-core";
import ImageUpload from "@/components/ImageUpload";
import ImageCanvas, { DetectionResults } from "@/components/ImageCanvas";
import StatsBar from "@/components/StatsBar";

export default function HomePage() {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [results, setResults] = useState<DetectionResults | null>(null);

  const { isLoading, reset: resetChat } = useCopilotChat();
  const prevLoading = useRef(false);
  const lastUpdatedRef = useRef<number>(0);

  useEffect(() => {
    if (prevLoading.current && !isLoading) {
      fetch("http://127.0.0.1:8000/latest-results")
        .then((r) => r.json())
        .then((data) => {
          const updatedAt = data.updated_at ?? 0;
          if (data.results && updatedAt > lastUpdatedRef.current) {
            lastUpdatedRef.current = updatedAt;
            setResults(data.results as DetectionResults);
          }
        })
        .catch((err) => console.error("Failed to fetch results:", err));
    }
    prevLoading.current = isLoading;
  }, [isLoading]);

  const clearAll = useCallback(async () => {
    setPreviewUrl(null);
    setFileName(null);
    setResults(null);
    lastUpdatedRef.current = 0;

    // Clear chat history in the CopilotKit UI
    try {
      resetChat();
    } catch (err) {
      console.error("Failed to reset chat:", err);
    }

    // Clear the backend session + agent memory
    try {
      await fetch("http://127.0.0.1:8000/reset", { method: "POST" });
    } catch (err) {
      console.error("Failed to reset backend:", err);
    }
  }, [resetChat]);

  const handleUpload = useCallback(
    async (base64: string, preview: string, name: string) => {
      // Full reset before loading the new image
      setResults(null);
      lastUpdatedRef.current = 0;

      try {
        resetChat();
      } catch (err) {
        console.error("Failed to reset chat:", err);
      }

      try {
        await fetch("http://127.0.0.1:8000/reset", { method: "POST" });
      } catch (err) {
        console.error("Failed to reset backend:", err);
      }

      setPreviewUrl(preview);
      setFileName(name);

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
    [resetChat]
  );

  const buttonStyle = {
    padding: "8px 20px",
    borderRadius: 10,
    border: "1px solid var(--border-subtle)",
    background: "var(--bg-card)",
    color: "var(--text-secondary)",
    fontSize: 13,
    fontFamily: "'Outfit', sans-serif",
    transition: "all 0.2s",
  } as const;

  return (
    <div className="app-container">
      <main className="main-panel">
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

        {!previewUrl ? (
          <ImageUpload onUpload={handleUpload} />
        ) : (
          <>
            <div className="canvas-container">
              <ImageCanvas imageSrc={previewUrl} results={results} />
            </div>
            <StatsBar results={results} />
            <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
              <button
                onClick={() => setResults(null)}
                disabled={!results}
                style={{
                  ...buttonStyle,
                  cursor: results ? "pointer" : "not-allowed",
                  opacity: results ? 1 : 0.5,
                }}
              >
                Clear results
              </button>
              <button
                onClick={clearAll}
                style={{
                  ...buttonStyle,
                  cursor: "pointer",
                }}
              >
                Upload new image
              </button>
            </div>
          </>
        )}
      </main>

      <CopilotSidebar
        defaultOpen={true}
        clickOutsideToClose={false}
        labels={{
          title: "GeoVision Assistant",
          initial:
            "Upload a satellite or aerial image, then ask me to detect objects or segment boundaries.",
        }}
      />
    </div>
  );
}
