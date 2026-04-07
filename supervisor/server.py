"""
Supervisor server — FastAPI + AG-UI endpoint for CopilotKit frontend.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from ag_ui_langgraph import LangGraphAgent, add_langgraph_fastapi_endpoint
from supervisor.agent import graph, set_session_image

load_dotenv()

app = FastAPI(title="GeoVision Supervisor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agui_agent = LangGraphAgent(
    name="geoVisionAgent",
    graph=graph,
    config={"recursion_limit": 50},
)

add_langgraph_fastapi_endpoint(app, agui_agent, "/copilotkit")


class ImagePayload(BaseModel):
    image_b64: str


@app.post("/upload-image")
async def upload_image(payload: ImagePayload):
    set_session_image(payload.image_b64)
    return {"status": "ok", "size": len(payload.image_b64)}


@app.get("/health")
def health():
    return {"status": "geovision supervisor is running"}


def main():
    uvicorn.run(
        "supervisor.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
