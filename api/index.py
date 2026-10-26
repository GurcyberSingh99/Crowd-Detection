import os
from collections import deque

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from utils import classify_occupancy, format_timestamp, now_utc

app = FastAPI(title="Crowd Occupancy System")

HISTORY_SIZE = 60
_history: deque = deque(maxlen=HISTORY_SIZE)
_last_reading: dict = {
    "timestamp": format_timestamp(now_utc()),
    "people_count": 0,
    "occupancy_level": "low",
}


@app.get("/")
async def index():
    static_file = os.path.join(os.path.dirname(__file__), "..", "static", "index.html")
    return FileResponse(static_file)


@app.get("/api/current")
async def current():
    return JSONResponse(_last_reading)


@app.get("/api/history")
async def history():
    return JSONResponse(list(_history))


@app.get("/health")
async def health():
    return JSONResponse({"ok": True, "status": "healthy"})


@app.post("/api/camera/count")
async def browser_camera_count(request: Request):
    global _last_reading
    try:
        data = await request.json()
        count = max(0, int(data.get("people_count", 0)))
    except (TypeError, ValueError, AttributeError):
        return JSONResponse({"error": "people_count must be an integer"}, status_code=400)

    _last_reading = {
        "timestamp": format_timestamp(now_utc()),
        "people_count": count,
        "occupancy_level": classify_occupancy(count),
    }
    _history.append(_last_reading)
    return JSONResponse({"ok": True, "people_count": count})


@app.post("/api/camera/stop")
async def browser_camera_stop():
    return JSONResponse({"ok": True})
