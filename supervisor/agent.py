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
ORIGINAL_RESULTS: dict = {"value": None}
LATEST_RESULTS: dict = {"value": None, "updated_at": 0.0}


def set_session_image(b64: str):
    SESSION_IMAGE["b64"] = b64
    ORIGINAL_RESULTS["value"] = None
    LATEST_RESULTS["value"] = None
    LATEST_RESULTS["updated_at"] = 0.0


def reset_session():
    """Wipe everything — image, results, and (caller must also rotate thread IDs)."""
    SESSION_IMAGE["b64"] = None
    ORIGINAL_RESULTS["value"] = None
    LATEST_RESULTS["value"] = None
    LATEST_RESULTS["updated_at"] = 0.0


def get_latest_results():
    return {
        "results": LATEST_RESULTS["value"],
        "updated_at": LATEST_RESULTS["updated_at"],
    }


def _store_original_and_display(result: dict):
    ORIGINAL_RESULTS["value"] = result
    LATEST_RESULTS["value"] = result
    LATEST_RESULTS["updated_at"] = time.time()


def _store_display(result: dict):
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

    _store_original_and_display(result)

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
    _store_original_and_display(result)

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
    """Filter the original detection/segmentation results to only include the given labels.

    Always filters against the full original result set from the last run_detection
    or run_segmentation call, so switching filters works correctly.

    Args:
        labels: List of label names to keep (e.g. ["plane"], ["plane", "ship"]).
    """
    original = ORIGINAL_RESULTS["value"]
    if not original:
        return json.dumps({"error": "No previous results to filter. Run detection first."})

    labels_lower = {l.lower() for l in labels}
    filtered_detections = [
        d for d in original.get("detections", [])
        if d.get("label", "").lower() in labels_lower
    ]
    filtered = {
        "type": original.get("type"),
        "count": len(filtered_detections),
        "detections": filtered_detections,
    }
    _store_display(filtered)

    counts: dict = {}
    for det in filtered_detections:
        counts[det["label"]] = counts.get(det["label"], 0) + 1
    return json.dumps({
        "type": filtered["type"],
        "total": filtered["count"],
        "by_category": counts,
        "available_labels_in_original": sorted({
            d.get("label", "") for d in original.get("detections", [])
        }),
    })


@tool
async def reset_filter() -> str:
    """Restore the full original detection/segmentation results.

    Use this when the user asks to show everything again, e.g. "show all",
    "reset", "show everything", "unfilter".
    """
    original = ORIGINAL_RESULTS["value"]
    if not original:
        return json.dumps({"error": "No previous results to restore."})

    _store_display(original)

    counts: dict = {}
    for det in original.get("detections", []):
        counts[det["label"]] = counts.get(det["label"], 0) + 1
    return json.dumps({
        "type": original.get("type"),
        "total": original.get("count"),
        "by_category": counts,
    })


SUPERVISOR_PROMPT = """You are GeoVision, an AI supervisor for geospatial image analysis.

The user's uploaded image is already available to your tools — never pass it as an argument.

## Your Tools
1. **run_detection** — bounding boxes
   - mode="bbox" for standard COCO detection
   - mode="obb" for oriented detection on aerial imagery (best for aircraft, vehicles, ships)
2. **run_segmentation** — polygon masks for boundaries
3. **filter_results** — filter the ORIGINAL results by label. Always filters against the
   full original detection, so switching filters works correctly.
4. **reset_filter** — restore the full original results (show everything again).

## Decision Rules
- For aerial/satellite images of aircraft, vehicles, ships → ALWAYS prefer mode="obb"
- "segment", "boundaries", "outlines" → run_segmentation
- "show only X", "just the X", "filter to X" (after a prior detection) → filter_results
- "show all", "show everything", "reset" → reset_filter
- Pure follow-up questions (counts, confidence, which had the most) → answer from history

## Important
- NEVER call filter_results twice in a row with the same arguments.
- If a tool returns empty results or an error, report it to the user and STOP. Do not retry.
- If the user asks for a label that isn't in the original detection, tell them what labels
  ARE available (the filter tool returns this in 'available_labels_in_original').

## Response Format
Write a SHORT one-sentence summary with total counts and categories.
The visual overlays appear automatically on the canvas — do NOT include coordinates or JSON.
Example: "I found 10 aircraft and 4 vehicles in the image."

Execute tools immediately. Never ask for confirmation.
"""

checkpointer = MemorySaver()

graph = create_agent(
    make_model(),
    tools=[run_detection, run_segmentation, filter_results, reset_filter],
    system_prompt=SUPERVISOR_PROMPT,
    checkpointer=checkpointer,
)
