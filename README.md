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
    ├── run_segmentation()    ← Modal GPU (YOLOv8n-seg)
    ├── filter_results()      ← in-memory label filter
    └── reset_filter()        ← restore full results
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
   - "Detect the aircraft" (standard bounding boxes)
   - "Give me OBB for the aircraft" (oriented bounding boxes — best for aerial imagery)
   - "Segment the buildings" (instance segmentation)
   - "How many vehicles are there?" (answered from history, no re-inference)
3. **Refine the view** after detection without re-running the model:
   - "Show only the aircraft" → filters to planes
   - "Show just the vehicles" → filters to vehicles
   - "Show everything" / "Reset" → restores the full result set
4. **Clear the overlay** with the *Clear results* button (keeps the image) or start over with *Upload new image* (clears image, canvas, chat history, and agent memory).

> **Tip:** For aerial/satellite imagery, the OBB model (YOLOv8n-obb, trained on DOTA) performs better than the standard bbox model (YOLOv8n, trained on COCO). Always prefer OBB for aircraft, vehicles, and ships viewed from above.

---

## Endpoints

| Endpoint                              | Description                                       |
|---------------------------------------|---------------------------------------------------|
| `GET  /health`                        | Health check                                      |
| `POST /upload-image`                  | Stores the current session's image (base64)       |
| `GET  /latest-results`                | Returns `{results, updated_at}` — timestamp-gated |
| `POST /reset`                         | Wipes session image, results, and agent memory    |
| `POST /copilotkit`                    | AG-UI agent endpoint used by CopilotKit           |

Frontend proxies CopilotKit traffic through:
```
http://localhost:3000/api/copilotkit → http://127.0.0.1:8000/copilotkit
```

---

## Agent Tools

The Supervisor Agent has four tools registered with LangGraph:

| Tool               | Purpose                                                              |
|--------------------|----------------------------------------------------------------------|
| `run_detection`    | Calls Modal YOLOv8n or YOLOv8n-obb — writes to `ORIGINAL_RESULTS`    |
| `run_segmentation` | Calls Modal YOLOv8n-seg — writes to `ORIGINAL_RESULTS`               |
| `filter_results`   | Filters `ORIGINAL_RESULTS` by label, updates displayed view only     |
| `reset_filter`     | Restores the displayed view to the full `ORIGINAL_RESULTS`           |

Filter operations never mutate the original result set, so switching between filter views ("show planes" → "show vehicles" → "show planes") always works.

---

## Project Structure

```
geovision/
├── supervisor/
│   ├── __init__.py
│   ├── agent.py              # LangGraph supervisor + 4 tools + session stores
│   └── server.py             # FastAPI + AG-UI endpoint + /upload-image /reset /latest-results
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
│   │   └── page.tsx          # Upload, canvas, polling, reset flow
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

### Upload flow
1. User uploads an image → frontend resets everything first: clears canvas state, calls `resetChat()` on CopilotKit, and POSTs `/reset` to wipe backend session + agent memory
2. Frontend converts the file to base64 and POSTs to `/upload-image`
3. Backend stores the base64 in a module-level `SESSION_IMAGE` dict

### Detection flow
1. User asks a question → CopilotKit forwards it through `/api/copilotkit` → AG-UI endpoint → Supervisor Agent
2. Supervisor picks a tool (`run_detection` bbox/obb, or `run_segmentation`)
3. The tool reads the image from `SESSION_IMAGE`, calls the corresponding Modal GPU function, and stores the full result in both `ORIGINAL_RESULTS` (permanent) and `LATEST_RESULTS` (displayed view) with a fresh timestamp
4. The tool returns only a compact summary (counts by category) to the LLM — never raw coordinates — so the model writes a clean natural-language reply
5. Frontend watches `isLoading` from `useCopilotChat`; when the turn finishes, it fetches `/latest-results`. If the returned `updated_at` is newer than what the frontend last saw, it updates the canvas. Otherwise (e.g. a pure follow-up question that didn't run inference), the canvas stays as-is

### Filter flow
1. User says "show only aircraft" → agent calls `filter_results(labels=["plane"])`
2. Tool filters `ORIGINAL_RESULTS` (never the currently-displayed view) and writes the subset to `LATEST_RESULTS` with a new timestamp
3. Frontend re-polls and redraws with only the filtered boxes
4. Switching filters works freely because the original is preserved

Keeping the (potentially multi-MB) base64 image out of LLM tool-call arguments entirely avoids both token bloat and base64 corruption.

---

## Quick Checks

```powershell
# Health check (Windows PowerShell)
irm http://localhost:8000/health

# See the latest cached results
irm http://localhost:8000/latest-results

# Wipe the session
irm http://localhost:8000/reset -Method Post
```

```bash
# Same, on macOS/Linux
curl http://localhost:8000/health
curl http://localhost:8000/latest-results
curl -X POST http://localhost:8000/reset
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
