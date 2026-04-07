# GeoVision

A conversational geospatial image analysis tool. Upload a satellite or aerial image, ask natural language questions, and get AI-powered object detection and segmentation results overlaid directly on your image.

---

## Architecture

```
User (CopilotKit UI + Image Canvas)
        │
        ▼
  Supervisor Agent            ← LangGraph orchestrator
    ├── run_detection()       ← Modal GPU (YOLOv8n / YOLOv8n-obb)
    └── run_segmentation()    ← Modal GPU (YOLOv8n-seg)
```

The Supervisor Agent receives user queries plus the uploaded image (base64), decides which model to invoke, calls Modal GPU inference, and returns annotated results that the frontend renders as overlays on the HTML Canvas.

---

## Stack

| Layer             | Technology                                      |
|-------------------|------------------------------------------------|
| Agent framework   | LangGraph                                       |
| Frontend UI       | CopilotKit + Next.js 15                         |
| Agent protocol    | CopilotKit SDK (FastAPI integration)            |
| LLM provider      | Nebius (Llama-3.1-70B via OpenAI-compatible API)|
| GPU inference     | Modal.com (A10G)                                |
| Detection models  | YOLOv8n, YOLOv8n-obb, YOLOv8n-seg             |
| Image overlays    | HTML Canvas API                                 |

---

## Getting Started

### Prerequisites

- Python ≥ 3.11 with `uv` installed
- Node.js + npm
- Modal account (free tier works)
- Nebius API key

### 1. Clone and configure

```bash
cd geovision
cp .env.example .env
# Edit .env with your API keys
uv sync
```

### 2. Authenticate and deploy Modal inference functions

```bash
uv run modal setup
uv run modal deploy modal_inference/inference.py
```

This creates three serverless GPU endpoints on Modal:
- `detect_bbox` — axis-aligned bounding boxes (A10G, 4 concurrent)
- `detect_obb` — oriented bounding boxes (A10G, 4 concurrent)
- `segment` — instance segmentation (A10G, 2 concurrent)

### 3. Run the backend

```bash
uv run python -m supervisor.server
```

The supervisor starts on `http://localhost:8000`.

### 4. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

### 5. Open the app

```
http://localhost:3000
```

---

## Usage

1. **Upload** a satellite/aerial image (JPEG, PNG, GeoTIFF — max 20MB)
2. **Ask questions** in the chat sidebar:
   - "How many vehicles are in the parking lot?"
   - "Detect all buildings"
   - "Show me objects at an angle" (uses oriented bounding boxes)
   - "Highlight building boundaries" (uses segmentation)
   - "Do a full analysis" (runs both detection + segmentation)
3. **View results** overlaid on the image with colored bounding boxes or polygon masks
4. **Ask follow-ups** — "Which objects had the highest confidence?" — without re-running inference

---

## Endpoints

| Endpoint                     | Description                          |
|------------------------------|--------------------------------------|
| `GET http://localhost:8000/health` | Health check                    |
| `POST http://localhost:8000/copilotkit` | CopilotKit agent endpoint  |

Frontend proxies through:
```
http://localhost:3000/api/copilotkit → http://127.0.0.1:8000/copilotkit
```

---

## Project Structure

```
geovision/
├── supervisor/
│   ├── __init__.py
│   ├── agent.py              # LangGraph supervisor with detection + segmentation tools
│   └── server.py             # FastAPI + CopilotKit endpoint
├── detector_agent/
│   ├── __init__.py
│   ├── agent.py              # Detector agent definition
│   └── tools.py              # detect_objects, detect_oriented_objects
├── segmentation_agent/
│   ├── __init__.py
│   ├── agent.py              # Segmentation agent definition
│   └── tools.py              # segment_objects
├── modal_inference/
│   ├── __init__.py
│   ├── inference.py          # Modal GPU functions (deploy to cloud)
│   └── client.py             # Local client to call Modal functions
├── frontend/
│   ├── app/
│   │   ├── api/copilotkit/
│   │   │   └── route.ts      # Next.js → supervisor proxy
│   │   ├── globals.css
│   │   ├── layout.tsx
│   │   └── page.tsx          # Main page with state management
│   ├── components/
│   │   ├── ImageCanvas.tsx   # HTML Canvas overlay renderer
│   │   ├── ImageUpload.tsx   # Drag & drop upload with base64 conversion
│   │   └── StatsBar.tsx      # Detection summary chips
│   ├── package.json
│   └── tsconfig.json
├── pyproject.toml
├── .env.example
└── README.md
```

---

## How It Works

1. User uploads an image → converted to base64, stored in React state
2. CopilotKit makes the base64 available to the agent via `useCopilotReadable`
3. User asks a question → Supervisor agent decides which tool to call
4. Tool calls Modal GPU function remotely → YOLOv8 runs inference → returns JSON
5. Supervisor calls `displayResults` action → frontend receives detection JSON
6. `ImageCanvas` component draws bounding boxes / polygons over the original image
7. `StatsBar` shows object counts by category
8. Follow-up questions use cached results from `useCopilotReadable` — no re-inference

---

## Quick Checks

```bash
# Health check
curl http://localhost:8000/health

# Test Modal functions directly
modal run modal_inference/inference.py::detect_bbox --image-b64 "..."
```

---

## Environment Variables

| Variable          | Description                    |
|-------------------|-------------------------------|
| `NEBIUS_API_KEY`  | Nebius Token Factory API key  |
| Modal credentials | Set via `modal token set` CLI |

---
