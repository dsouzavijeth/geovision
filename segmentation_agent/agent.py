"""Segmentation Agent — handles instance segmentation requests."""

import os
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


def make_model():
    return ChatOpenAI(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=os.environ.get("NEBIUS_API_KEY"),
        base_url="https://api.tokenfactory.nebius.com/v1/",
    )


SEGMENTATION_PROMPT = """You are a Segmentation Agent specialized in instance segmentation of geospatial imagery.

You have one tool:
- segment_objects — runs YOLOv8-seg to produce polygon masks for every detected object

When you receive a request:
- Run segmentation immediately using the provided base64 image
- Report the total count of segmented objects
- List object labels and confidence scores
- Provide the full result JSON so it can be used for visualization

Use this for: boundaries, outlines, footprints, masks, area coverage,
building boundaries, road surfaces, water bodies, vegetation coverage, etc.

Do NOT ask for clarification. Run segmentation immediately with the provided image.
"""


def create_segmentation_agent():
    from segmentation_agent.tools import get_segmentation_tools

    tools = get_segmentation_tools()
    agent = create_react_agent(
        make_model(),
        tools=tools,
        prompt=SEGMENTATION_PROMPT,
    )
    return agent
