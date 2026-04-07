"use client";

import { useRef, useState, DragEvent, ChangeEvent } from "react";

const MAX_SIZE_MB = 20;
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024;

interface ImageUploadProps {
  onUpload: (base64: string, previewUrl: string, fileName: string) => void;
}

export default function ImageUpload({ onUpload }: ImageUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function processFile(file: File) {
    setError(null);

    if (!file.type.startsWith("image/")) {
      setError("Please upload an image file (JPEG, PNG, or TIFF).");
      return;
    }
    if (file.size > MAX_SIZE_BYTES) {
      setError(`File exceeds ${MAX_SIZE_MB}MB limit.`);
      return;
    }

    const previewUrl = URL.createObjectURL(file);

    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      // strip the data:image/...;base64, prefix to get raw base64
      const base64 = dataUrl.split(",")[1];
      onUpload(base64, previewUrl, file.name);
    };
    reader.readAsDataURL(file);
  }

  function handleDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) processFile(file);
  }

  function handleChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) processFile(file);
  }

  return (
    <div
      className={`upload-zone ${dragging ? "dragging" : ""}`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*,.tif,.tiff"
        onChange={handleChange}
        style={{ display: "none" }}
      />
      <div className="upload-icon">🛰️</div>
      <h3>Upload Satellite or Aerial Image</h3>
      <p>
        Drag & drop or click to browse — JPEG, PNG, GeoTIFF up to {MAX_SIZE_MB}
        MB
      </p>
      {error && (
        <p style={{ color: "var(--accent-red)", marginTop: 12 }}>{error}</p>
      )}
    </div>
  );
}
