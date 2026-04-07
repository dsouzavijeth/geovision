"""
Client wrapper to call Modal inference functions from local agents.

Usage:
    from modal_inference.client import run_detection, run_obb, run_segmentation
    result = await run_detection(base64_image_string)
"""

import modal


async def run_detection(image_b64: str, conf: float = 0.25) -> dict:
    """Call the remote detect_bbox Modal function."""
    detect_fn = modal.Function.from_name("geovision-yolo", "detect_bbox")
    result = await detect_fn.remote.aio(image_b64=image_b64, conf=conf)
    return result


async def run_obb(image_b64: str, conf: float = 0.25) -> dict:
    """Call the remote detect_obb Modal function."""
    detect_fn = modal.Function.from_name("geovision-yolo", "detect_obb")
    result = await detect_fn.remote.aio(image_b64=image_b64, conf=conf)
    return result


async def run_segmentation(image_b64: str, conf: float = 0.25) -> dict:
    """Call the remote segment Modal function."""
    seg_fn = modal.Function.from_name("geovision-yolo", "segment")
    result = await seg_fn.remote.aio(image_b64=image_b64, conf=conf)
    return result
