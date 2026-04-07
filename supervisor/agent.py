"""
Supervisor Agent — orchestrates Detector and Segmentation agents.
"""

import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

# In-memory image store (single-session for v1)
SESSION_IMAGE: dict = {"b64": None}


def set_session_image(b64: str):
    SESSION_IMAGE["b64"] = b64


def make_model():
    return ChatOpenAI(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=os.environ.get("NEBIUS_API_KEY"),
        base_url="https://api.tokenfactory.nebius.com/v1/",
        streaming=True,
    )


@tool
async def run_detection(mode: str = "bbox", conf: float = 0.25) -> str:
    """Run object detection on the user's uploaded image.

    Args:
        mode: 'bbox' for axis-aligned boxes, 'obb' for oriented/rotated boxes.
        conf: Confidence threshold (0.0 to 1.0).

    Returns:
        JSON with detection results (type, count, detections with labels/confidence/coordinates).
    """
    from modal_inference.client import run_detection as modal_detect, run_obb as modal_obb

    image_b64 = SESSION_IMAGE.get("b64")
    if not image_b64:
        return json.dumps({"error": "No image uploaded yet."})

    if mode == "obb":
        result = await modal_obb(image_b64, conf)
    else:
        result = await modal_detect(image_b64, conf)

    return json.dumps(result, indent=2)


@tool
async def run_segmentation(conf: float = 0.25) -> str:
    """Run instance segmentation on the user's uploaded image.

    Args:
        conf: Confidence threshold (0.0 to 1.0).

    Returns:
        JSON with segmentation results (type, count, detections with labels/confidence/polygons).
    """
    from modal_inference.client import run_segmentation as modal_seg

    image_b64 = SESSION_IMAGE.get("b64")
    if not image_b64:
        return json.dumps({"error": "No image uploaded yet."})

    result = await modal_seg(image_b64, conf)
    return json.dumps(result, indent=2)


SUPERVISOR_PROMPT = """You are GeoVision, an AI supervisor for geospatial image analysis.

You help users analyze satellite and aerial images by delegating to specialized detection
and segmentation models running on GPU. The user's uploaded image is already available
to your tools — you do NOT need to pass it as an argument.

## Your Tools
1. **run_detection** — detects objects with bounding boxes
   - mode="bbox" for standard detection (vehicles, buildings, people, etc.)
   - mode="obb" for oriented/rotated detection (objects at angles on roads, runways)
2. **run_segmentation** — segments objects with polygon masks (boundaries, footprints, outlines)

## Decision Rules
- "detect", "find", "count", "how many", "identify", "locate" → run_detection (mode="bbox")
- "at an angle", "oriented", "rotated", "tilted", "on the road/runway" → run_detection (mode="obb")
- "segment", "boundaries", "outlines", "masks", "footprints" → run_segmentation
- "detect and segment", "full analysis" → run BOTH
- Follow-up questions about previous results → answer from conversation history, do NOT re-run inference

## Response Format
After receiving results:
1. Summarize what was found (counts by category, notable detections)
2. Include the raw JSON in your response wrapped in a ```json code block so the frontend can parse and overlay it
3. Answer the user's specific question

Do NOT ask for confirmation before running inference — execute immediately.
"""

checkpointer = MemorySaver()

graph = create_react_agent(
    make_model(),
    tools=[run_detection, run_segmentation],
    prompt=SUPERVISOR_PROMPT,
    checkpointer=checkpointer,
)
