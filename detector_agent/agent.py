"""Detector Agent — handles bbox and OBB detection requests."""

import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent


def make_model():
    return ChatOpenAI(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=os.environ.get("NEBIUS_API_KEY"),
        base_url="https://api.tokenfactory.nebius.com/v1/",
    )


DETECTOR_PROMPT = """You are a Detection Agent specialized in geospatial object detection.

You have two detection tools:
1. detect_objects — standard axis-aligned bounding boxes (YOLO26)
2. detect_oriented_objects — oriented/rotated bounding boxes (YOLO26-OBB)

When you receive a request:
- If the user asks about objects at angles, rotated, tilted, on roads/runways → use detect_oriented_objects
- For all other detection requests → use detect_objects
- The image is provided as a base64 string in the request

After running detection:
- Report the total count of objects found
- List the object labels and their confidence scores
- Provide the full detection result JSON so it can be used for visualization

Do NOT ask for clarification. Run detection immediately with the provided image.
"""


def create_detector_agent():
    from detector_agent.tools import get_detector_tools

    tools = get_detector_tools()
    agent = create_agent(
        make_model(),
        tools=tools,
        system_prompt=DETECTOR_PROMPT,
    )
    return agent
