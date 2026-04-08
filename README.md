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

The Supervisor Agent receives user queries, decides which model to invoke, calls Modal GPU inference, and stores results in a backend session store. The frontend polls for the latest results and renders overlays on the HTML Canvas.

---

## Stack

| Layer             | Technology                                      |
|-------------------|------------------------------------------------|
| Agent framework   | LangGraph (`create_agent`)                      |
| Frontend UI       | CopilotKit + Next.js 15                         |
| Agent protocol    | AG-UI (via `ag_ui_langgraph`)                   |
| LLM provider      | Nebius Token Factory (`openai/gpt-oss-20b`)     |
| GPU inference     | Modal.com (A10G)                                |
| Detection models  | YOLOv8n, YOLOv8n-obb, YOLOv8n-seg               |
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
- `detect_bbox` — axis-aligned bounding boxes (A10G, max 4 containers)
- `detect_obb` — oriented bounding boxes (A10G, max 4 containers)
- `segment` — instance segmentation (A10G, max 2 containers)

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
   - "Detect the aircraft" (uses standard bounding boxes)
   - "Give me OBB for the aircraft" (uses oriented bounding boxes — best for aerial imagery)
   - "Segment the buildings" (uses instance segmentation)
   - "How many vehicles are there?"
3. **View results** overlaid on the image with colored bounding boxes or polygon masks
4. **Ask follow-ups** — the agent answers from conversation history without re-running inference

> **Tip:** For aerial/satellite imagery, the OBB model (YOLOv8n-obb, trained on DOTA) performs dramatically better than the standard bbox model (YOLOv8n, trained on COCO). Always prefer OBB for aircraft, vehicles, and ships viewed from above.

---

## Endpoints

| Endpoint                              | Description                                   |
|---------------------------------------|-----------------------------------------------|
| `GET  /health`                        | Health check                                  |
| `POST /upload-image`                  | Stores the current session's image (base64)   |
| `GET  /latest-results`                | Returns the most recent detection/seg results |
| `POST /copilotkit`                    | AG-UI agent endpoint used by CopilotKit       |

Frontend proxies CopilotKit traffic through:
```
http://localhost:3000/api/copilotkit → http://127.0.0.1:8000/copilotkit
```

---

## Project Structure

```
geovision/
├── supervisor/
│   ├── __init__.py
│   ├── agent.py              # LangGraph supervisor + detection/segmentation tools
│   └── server.py             # FastAPI + AG-UI endpoint + session routes
├── detector_agent/           # (reserved for future standalone agent split)
├── segmentation_agent/       # (reserved for future standalone agent split)
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
│   │   └── page.tsx          # Main page with upload, canvas, and polling
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

1. User uploads an image → frontend converts it to base64 and POSTs to `/upload-image`
2. Backend stores the base64 in a module-level `SESSION_IMAGE` dict
3. User asks a question in chat → CopilotKit forwards the message through `/api/copilotkit` → AG-UI endpoint → Supervisor Agent
4. Supervisor decides which tool to call (`run_detection` with `bbox`/`obb` mode, or `run_segmentation`)
5. The tool reads the image from `SESSION_IMAGE`, calls the corresponding Modal GPU function, and stores the full result in `LATEST_RESULTS`
6. The tool returns only a small summary (counts by category) to the LLM — never raw coordinates — so the model produces a clean natural-language reply
7. Frontend watches `isLoading` from `useCopilotChat`; when the turn finishes, it fetches `/latest-results` and hands the JSON to the `ImageCanvas` component
8. `ImageCanvas` draws bounding boxes, oriented boxes, or polygon masks over the original image; `StatsBar` renders count chips

This approach keeps the (potentially multi-MB) base64 image out of LLM tool-call arguments entirely, which avoids the token bloat and base64 corruption issues you'd hit otherwise.

---

## Quick Checks

```powershell
# Health check (Windows PowerShell)
irm http://localhost:8000/health

# See the latest cached results
irm http://localhost:8000/latest-results
```

```bash
# Same, on macOS/Linux
curl http://localhost:8000/health
curl http://localhost:8000/latest-results
```

---

## Environment Variables

| Variable          | Description                                       |
|-------------------|---------------------------------------------------|
| `NEBIUS_API_KEY`  | Nebius Token Factory API key                      |
| Modal credentials | Set via `uv run modal setup` (stored by Modal CLI)|

---

## Notes on Model Choice

- **YOLOv8n (COCO)** — 80 everyday classes. Fine for ground-level photos, weak on aerial imagery.
- **YOLOv8n-obb (DOTA)** — trained specifically on aerial/satellite data. Catches aircraft, vehicles, and ships that the COCO model misses entirely. **Use this for geospatial work.**
- **YOLOv8n-seg (COCO)** — instance segmentation with 80 COCO classes.

If you need higher accuracy, swap `yolov8n*.pt` for `yolov8x*.pt` in `modal_inference/inference.py` and redeploy. The `n` (nano) variants are used by default for speed and minimal GPU memory.
