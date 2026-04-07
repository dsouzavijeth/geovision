"""
Modal GPU functions for YOLOv8 inference.

Deploy with:
    modal deploy modal_inference/inference.py

This creates three serverless GPU endpoints:
    - detect_bbox: axis-aligned bounding boxes (YOLOv8n)
    - detect_obb: oriented bounding boxes (YOLOv8n-obb)
    - segment: instance segmentation (YOLOv8n-seg)
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "ultralytics>=8.3.0",
        "opencv-python-headless>=4.10.0",
        "Pillow>=10.0.0",
        "numpy>=1.26.0",
    )
)

app = modal.App("geovision-yolo", image=image)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _decode_image(base64_str: str):
    """Decode a base64 image string into a PIL Image."""
    import base64
    import io
    from PIL import Image

    img_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_bytes))


# ---------------------------------------------------------------------------
# Axis-aligned bounding box detection
# ---------------------------------------------------------------------------

@app.function(gpu="A10G", max_containers=4, timeout=120)
def detect_bbox(image_b64: str, conf: float = 0.25) -> dict:
    """Run YOLOv8n detection, return axis-aligned bounding boxes."""
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    img = _decode_image(image_b64)
    results = model.predict(source=img, conf=conf, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "label": r.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 4),
                "bbox": {
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1),
                },
            })

    return {
        "type": "bbox",
        "count": len(detections),
        "detections": detections,
    }


# ---------------------------------------------------------------------------
# Oriented bounding box detection
# ---------------------------------------------------------------------------

@app.function(gpu="A10G", max_containers=4, timeout=120)
def detect_obb(image_b64: str, conf: float = 0.25) -> dict:
    """Run YOLOv8n-obb detection, return oriented bounding boxes."""
    from ultralytics import YOLO
    import numpy as np

    model = YOLO("yolov8n-obb.pt")
    img = _decode_image(image_b64)
    results = model.predict(source=img, conf=conf, verbose=False)

    detections = []
    for r in results:
        if r.obb is not None:
            for i, obb in enumerate(r.obb):
                # obb.xyxyxyxy gives four corner points
                points = obb.xyxyxyxy[0].tolist()
                detections.append({
                    "label": r.names[int(obb.cls[0])],
                    "confidence": round(float(obb.conf[0]), 4),
                    "obb_points": [
                        {"x": round(p[0], 1), "y": round(p[1], 1)}
                        for p in points
                    ],
                })

    return {
        "type": "obb",
        "count": len(detections),
        "detections": detections,
    }


# ---------------------------------------------------------------------------
# Instance segmentation
# ---------------------------------------------------------------------------

@app.function(gpu="A10G", max_containers=2, timeout=180)
def segment(image_b64: str, conf: float = 0.25) -> dict:
    """Run YOLOv8n-seg, return polygon masks."""
    from ultralytics import YOLO

    model = YOLO("yolov8n-seg.pt")
    img = _decode_image(image_b64)
    results = model.predict(source=img, conf=conf, verbose=False)

    detections = []
    for r in results:
        if r.masks is not None:
            for i, mask in enumerate(r.masks):
                # mask.xy gives polygon points as list of [x, y]
                polygon = mask.xy[0].tolist()
                # Simplify polygon to reduce payload size
                step = max(1, len(polygon) // 100)
                simplified = polygon[::step]

                detections.append({
                    "label": r.names[int(r.boxes[i].cls[0])],
                    "confidence": round(float(r.boxes[i].conf[0]), 4),
                    "polygon": [
                        {"x": round(p[0], 1), "y": round(p[1], 1)}
                        for p in simplified
                    ],
                })

    return {
        "type": "segmentation",
        "count": len(detections),
        "detections": detections,
    }
