"""Tools for the Segmentation Agent — instance segmentation via Modal."""

import json
from langchain_core.tools import tool


@tool
async def segment_objects(image_b64: str, conf: float = 0.25) -> str:
    """Segment objects in the image, returning polygon masks for each instance.

    Use this for boundary detection, area analysis, building footprints,
    land cover mapping, or any request involving outlines/masks/boundaries.

    Args:
        image_b64: Base64-encoded image string.
        conf: Confidence threshold (0.0 to 1.0). Default 0.25.

    Returns:
        JSON string with segmentation results including labels, confidence, and polygon points.
    """
    from modal_inference.client import run_segmentation

    result = await run_segmentation(image_b64, conf)
    return json.dumps(result, indent=2)


def get_segmentation_tools():
    return [segment_objects]
