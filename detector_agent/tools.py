"""Tools for the Detector Agent — bbox and OBB detection via Modal."""

import json
from langchain_core.tools import tool


@tool
async def detect_objects(image_b64: str, conf: float = 0.25) -> str:
    """Detect objects with axis-aligned bounding boxes using YOLOv8.

    Use this for standard object detection: counting vehicles, finding
    buildings, identifying objects in parking lots, etc.

    Args:
        image_b64: Base64-encoded image string.
        conf: Confidence threshold (0.0 to 1.0). Default 0.25.

    Returns:
        JSON string with detection results including labels, confidence, and bbox coordinates.
    """
    from modal_inference.client import run_detection

    result = await run_detection(image_b64, conf)
    return json.dumps(result, indent=2)


@tool
async def detect_oriented_objects(image_b64: str, conf: float = 0.25) -> str:
    """Detect objects with oriented (rotated) bounding boxes using YOLOv8-OBB.

    Use this when objects are at angles: vehicles on diagonal roads,
    aircraft on runways, ships at various headings, tilted buildings, etc.

    Args:
        image_b64: Base64-encoded image string.
        conf: Confidence threshold (0.0 to 1.0). Default 0.25.

    Returns:
        JSON string with OBB results including labels, confidence, and four corner points.
    """
    from modal_inference.client import run_obb

    result = await run_obb(image_b64, conf)
    return json.dumps(result, indent=2)


def get_detector_tools():
    return [detect_objects, detect_oriented_objects]
