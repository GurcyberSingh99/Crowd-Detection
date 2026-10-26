"""FastAPI server for the crowd occupancy dashboard."""

import asyncio
import json
import math
import os
from collections import deque

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from utils import classify_occupancy, format_timestamp, now_utc

VIDEO_SOURCE = 0
MODEL_WEIGHTS = "yolov8n.pt"
CONF_THRESHOLD = 0.35
IMGSZ = 640
EMIT_INTERVAL = 1.0
HISTORY_SIZE = 60
USE_REAL_DETECTOR = False  # Browser camera detection is started from the dashboard.

app = FastAPI(title="Crowd Occupancy System")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_history: deque = deque(maxlen=HISTORY_SIZE)
_last_reading: dict = {
    "timestamp": format_timestamp(now_utc()),
    "people_count": 0,
    "occupancy_level": "low",
}
_clients: list[WebSocket] = []
_clients_lock = asyncio.Lock()
_browser_camera_active = False
_last_camera_update = 0.0


async def broadcast(payload: dict):
    message = json.dumps(payload)
    async with _clients_lock:
        dead = []
        for ws in _clients:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            _clients.remove(ws)


async def on_detection(payload: dict):
    global _last_reading
    payload["timestamp"] = format_timestamp(payload["timestamp"])
    _last_reading = payload
    _history.append(payload)
    await broadcast({"type": "update", "data": payload})


async def run_mock_detector():
    """Simulate readings until live browser detections arrive."""
    global _browser_camera_active
    t = 0
    while True:
        if _browser_camera_active and asyncio.get_running_loop().time() - _last_camera_update <= 5:
            await asyncio.sleep(EMIT_INTERVAL)
            continue
        _browser_camera_active = False
        count = max(0, int(10 + 8 * math.sin(t / 10) + (t % 7)))
        await on_detection({
            "timestamp": now_utc(),
            "people_count": count,
            "occupancy_level": classify_occupancy(count),
        })
        t += 1
        await asyncio.sleep(EMIT_INTERVAL)


async def run_real_detector():
    try:
        from detector import VideoDetector
        det = VideoDetector(
            source=VIDEO_SOURCE,
            model_weights=MODEL_WEIGHTS,
            imgsz=IMGSZ,
            conf=CONF_THRESHOLD,
        )
        await det.start(callback=on_detection, interval_seconds=EMIT_INTERVAL)
    except Exception as exc:
        print(f"[WARNING] Real detector unavailable ({exc}). Falling back to mock detector.")
        await run_mock_detector()


@app.on_event("startup")
async def startup():
    if not os.path.isfile(os.path.join(STATIC_DIR, "index.html")):
        raise RuntimeError(f"Dashboard file is missing from {STATIC_DIR}")
    if USE_REAL_DETECTOR:
        asyncio.create_task(run_real_detector())
    else:
        asyncio.create_task(run_mock_detector())


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/current")
async def current():
    return JSONResponse(_last_reading)


@app.get("/api/history")
async def history():
    return JSONResponse(list(_history))


@app.post("/api/camera/count")
async def browser_camera_count(request: Request):
    """Accept an on-device person count from the browser camera detector."""
    global _browser_camera_active, _last_camera_update
    try:
        data = await request.json()
        count = max(0, int(data.get("people_count", 0)))
    except (TypeError, ValueError, AttributeError):
        return JSONResponse({"error": "people_count must be an integer"}, status_code=400)

    _browser_camera_active = True
    _last_camera_update = asyncio.get_running_loop().time()
    await on_detection({
        "timestamp": now_utc(),
        "people_count": count,
        "occupancy_level": classify_occupancy(count),
    })
    return JSONResponse({"ok": True, "people_count": count})


@app.post("/api/camera/stop")
async def browser_camera_stop():
    global _browser_camera_active
    _browser_camera_active = False
    return JSONResponse({"ok": True})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    async with _clients_lock:
        _clients.append(websocket)
    await websocket.send_text(json.dumps({"type": "history", "data": list(_history)}))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        async with _clients_lock:
            if websocket in _clients:
                _clients.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
