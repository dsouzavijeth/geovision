"use client";

import { useRef, useEffect, useCallback } from "react";

// ---------- colour palette for different object classes ----------
const COLORS = [
  "#63b3ed", // blue
  "#4ade80", // green
  "#fbbf24", // amber
  "#f87171", // red
  "#a78bfa", // purple
  "#22d3ee", // cyan
  "#fb923c", // orange
  "#e879f9", // pink
  "#2dd4bf", // teal
  "#facc15", // yellow
];

function colorFor(label: string): string {
  let hash = 0;
  for (let i = 0; i < label.length; i++) {
    hash = label.charCodeAt(i) + ((hash << 5) - hash);
  }
  return COLORS[Math.abs(hash) % COLORS.length];
}

// ---------- types ----------

export interface BBoxDetection {
  label: string;
  confidence: number;
  bbox: { x1: number; y1: number; x2: number; y2: number };
}

export interface OBBDetection {
  label: string;
  confidence: number;
  obb_points: { x: number; y: number }[];
}

export interface SegDetection {
  label: string;
  confidence: number;
  polygon: { x: number; y: number }[];
}

export interface DetectionResults {
  type: "bbox" | "obb" | "segmentation";
  count: number;
  detections: (BBoxDetection | OBBDetection | SegDetection)[];
}

interface ImageCanvasProps {
  imageSrc: string | null; // data URL or object URL
  results: DetectionResults | null;
}

export default function ImageCanvas({ imageSrc, results }: ImageCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  // ---------- draw everything ----------
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // size canvas to image
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    // draw base image
    ctx.drawImage(img, 0, 0);

    if (!results || !results.detections) return;

    const scale = 1; // 1:1 with natural pixels

    if (results.type === "bbox") {
      for (const det of results.detections as BBoxDetection[]) {
        const c = colorFor(det.label);
        const { x1, y1, x2, y2 } = det.bbox;
        const w = x2 - x1;
        const h = y2 - y1;

        ctx.strokeStyle = c;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1 * scale, y1 * scale, w * scale, h * scale);

        // label background
        const text = `${det.label} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = "bold 14px 'JetBrains Mono', monospace";
        const tm = ctx.measureText(text);
        const pad = 4;
        ctx.fillStyle = c;
        ctx.fillRect(
          x1 * scale,
          y1 * scale - 22,
          tm.width + pad * 2,
          22
        );
        ctx.fillStyle = "#000";
        ctx.fillText(text, x1 * scale + pad, y1 * scale - 6);
      }
    }

    if (results.type === "obb") {
      for (const det of results.detections as OBBDetection[]) {
        const c = colorFor(det.label);
        const pts = det.obb_points;
        if (pts.length < 4) continue;

        ctx.strokeStyle = c;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(pts[0].x * scale, pts[0].y * scale);
        for (let i = 1; i < pts.length; i++) {
          ctx.lineTo(pts[i].x * scale, pts[i].y * scale);
        }
        ctx.closePath();
        ctx.stroke();

        // label
        const text = `${det.label} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = "bold 14px 'JetBrains Mono', monospace";
        const tm = ctx.measureText(text);
        const pad = 4;
        ctx.fillStyle = c;
        ctx.fillRect(pts[0].x * scale, pts[0].y * scale - 22, tm.width + pad * 2, 22);
        ctx.fillStyle = "#000";
        ctx.fillText(text, pts[0].x * scale + pad, pts[0].y * scale - 6);
      }
    }

    if (results.type === "segmentation") {
      for (const det of results.detections as SegDetection[]) {
        const c = colorFor(det.label);
        const poly = det.polygon;
        if (poly.length < 3) continue;

        // filled semi-transparent mask
        ctx.fillStyle = c + "33"; // ~20% alpha
        ctx.strokeStyle = c;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(poly[0].x * scale, poly[0].y * scale);
        for (let i = 1; i < poly.length; i++) {
          ctx.lineTo(poly[i].x * scale, poly[i].y * scale);
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        // label at centroid
        const cx = poly.reduce((s, p) => s + p.x, 0) / poly.length;
        const cy = poly.reduce((s, p) => s + p.y, 0) / poly.length;
        const text = `${det.label} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = "bold 13px 'JetBrains Mono', monospace";
        const tm = ctx.measureText(text);
        const pad = 4;
        ctx.fillStyle = c;
        ctx.fillRect(cx * scale - tm.width / 2 - pad, cy * scale - 11, tm.width + pad * 2, 22);
        ctx.fillStyle = "#000";
        ctx.fillText(text, cx * scale - tm.width / 2, cy * scale + 4);
      }
    }
  }, [results]);

  // load image when src changes
  useEffect(() => {
    if (!imageSrc) return;
    const img = new Image();
    img.onload = () => {
      imgRef.current = img;
      draw();
    };
    img.src = imageSrc;
  }, [imageSrc, draw]);

  // redraw when results change
  useEffect(() => {
    if (imgRef.current) draw();
  }, [results, draw]);

  return <canvas ref={canvasRef} style={{ maxWidth: "100%", maxHeight: "100%" }} />;
}
