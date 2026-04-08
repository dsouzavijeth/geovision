"""Supervisor Agent — orchestrates detection and segmentation via Modal."""

import os
import json
import time
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

# Module-level session stores
SESSION_IMAGE: dict = {"b64": None}
LATEST_RESULTS: dict = {"value": None, "updated_at": 0.0}


def set_session_image(b64: str):
    SESSION_IMAGE["b64"] = b64
    LATEST_RESULTS["value"] = None
    LATEST_RESULTS["updated_at"] = 0.0


def get_latest_results():
    return {
        "results": LATEST_RESULTS["value"],
        "updated_at": LATEST_RESULTS["updated_at"],
    }


def _store_result(result: dict):
    LATEST_RESULTS["value"] = result
    LATEST_RESULTS["updated_at"] = time.time()


def make_model():
    return ChatOpenAI(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=os.environ.get("NEBIUS_API_KEY"),
        base_url="https://api.tokenfactory.nebius.com/v1/",
        streaming=True,
    )


@tool
async def run_detection(mode: str = "bbox", conf: float = 0.4) -> str:
    """Run object detection on the user's uploaded image.

    Args:
        mode: 'bbox' for axis-aligned boxes, 'obb' for oriented/rotated boxes.
        conf: Confidence threshold (0.0 to 1.0).
    """
    from modal_inference.client import run_detection as modal_detect, run_obb as modal_obb

    image_b64 = SESSION_IMAGE.get("b64")
    if not image_b64:
        return json.dumps({"error": "No image uploaded yet."})

    if mode == "obb":
        result = await modal_obb(image_b64, conf)
    else:
        result = await modal_detect(image_b64, conf)

    _store_result(result)

    counts: dict = {}
    for det in result.get("detections", []):
        counts[det["label"]] = counts.get(det["label"], 0) + 1
    return json.dumps({
        "type": result.get("type"),
        "total": result.get("count"),
        "by_category": counts,
    })


@tool
async def run_segmentation(conf: float = 0.4) -> str:
    """Run instance segmentation on the user's uploaded image."""
    from modal_inference.client import run_segmentation as modal_seg

    image_b64 = SESSION_IMAGE.get("b64")
    if not image_b64:
        return json.dumps({"error": "No image uploaded yet."})

    result = await modal_seg(image_b64, conf)
    _store_result(result)

    counts: dict = {}
    for det in result.get("detections", []):
        counts[det["label"]] = counts.get(det["label"], 0) + 1
    return json.dumps({
        "type": result.get("type"),
        "total": result.get("count"),
        "by_category": counts,
    })


@tool
async def filter_results(labels: list[str]) -> str:
    """Filter the most recent detection/segmentation results to only include the given labels.

    Use this when the user asks to show only certain object types after a detection
    has already been run (e.g. "show just the aircraft", "only vehicles").

    Args:
        labels: List of label names to keep (e.g. ["plane"], ["plane", "ship"]).
    """
    current = LATEST_RESULTS["value"]
    if not current:
        return json.dumps({"error": "No previous results to filter."})

    labels_lower = {l.lower() for l in labels}
    filtered_detections = [
        d for d in current.get("detections", [])
        if d.get("label", "").lower() in labels_lower
    ]
    filtered = {
        "type": current.get("type"),
        "count": len(filtered_detections),
        "detections": filtered_detections,
    }
    _store_result(filtered)

    counts: dict = {}
    for det in filtered_detections:
        counts[det["label"]] = counts.get(det["label"], 0) + 1
    return json.dumps({
        "type": filtered["type"],
        "total": filtered["count"],
        "by_category": counts,
    })


SUPERVISOR_PROMPT = """You are GeoVision, an AI supervisor for geospatial image analysis.

The user's uploaded image is already available to your tools — never pass it as an argument.

## Your Tools
1. **run_detection** — bounding boxes
   - mode="bbox" for standard COCO detection
   - mode="obb" for oriented detection on aerial imagery (best for aircraft, vehicles, ships)
2. **run_segmentation** — polygon masks for boundaries
3. **filter_results** — filter the MOST RECENT results by label(s). Use this when the user
   asks to show/keep only specific object types after a detection has already run.
   Example: if you previously detected planes and small_vehicles, and the user says
   "show only the aircraft", call filter_results(labels=["plane"]).

## Decision Rules
- For aerial/satellite images of aircraft, vehicles, ships → ALWAYS prefer mode="obb"
- "segment", "boundaries", "outlines" → run_segmentation
- "show only X", "just the X", "filter to X", "hide the Y" (after a prior detection)
  → filter_results, do NOT re-run detection
- Pure follow-up questions (counts, confidence, which had the most) → answer from history

## Response Format
After a tool runs, write a SHORT one-sentence summary with total counts and categories.
The visual overlays appear automatically on the canvas — do NOT include coordinates or JSON.
Example: "I found 10 aircraft and 4 vehicles in the image."

Execute tools immediately. Never ask for confirmation.
"""

checkpointer = MemorySaver()

graph = create_agent(
    make_model(),
    tools=[run_detection, run_segmentation, filter_results],
    system_prompt=SUPERVISOR_PROMPT,
    checkpointer=checkpointer,
)
