"""YOLO based person detection from a webcam, video file, or stream."""

import asyncio
from typing import Callable

import cv2
import numpy as np
from ultralytics import YOLO

from utils import classify_occupancy, now_utc


class VideoDetector:
    """Read video frames and publish averaged person counts."""

    def __init__(self, source=0, model_weights="yolov8n.pt", imgsz=640, conf=0.35, person_class_idx=0):
        self.source = source
        self.model = YOLO(model_weights)
        self.imgsz = imgsz
        self.conf = conf
        self.person_class_idx = person_class_idx
        self._running = False
        self._cap = None

    async def start(self, callback: Callable[[dict], None], interval_seconds: float = 1.0):
        """Publish timestamped occupancy readings until stopped or the source ends."""
        loop = asyncio.get_running_loop()
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise RuntimeError(f"Cannot open video source: {self.source}")

        self._running = True
        last_emit = loop.time()
        accumulated_counts = []
        failed_reads = 0
        source_text = str(self.source).lower()
        is_file = isinstance(self.source, (str, bytes)) and not source_text.startswith(
            ("rtsp://", "http://", "https://")
        )
        try:
            while self._running:
                ret, frame = await asyncio.to_thread(self._cap.read)
                if not ret:
                    failed_reads += 1
                    if is_file and self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                        await asyncio.sleep(0.05)
                        continue
                    if is_file:
                        break
                    if failed_reads >= 50:
                        raise RuntimeError("Video source stopped returning frames")
                    await asyncio.sleep(0.1)
                    continue

                failed_reads = 0
                height, width = frame.shape[:2]
                scale = min(1.0, self.imgsz / max(height, width))
                if scale < 1.0:
                    frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

                results = await asyncio.to_thread(
                    self.model, frame, imgsz=self.imgsz, conf=self.conf, verbose=False
                )
                result = results[0]
                boxes = getattr(result, "boxes", None)
                if boxes is not None and len(boxes) > 0:
                    classes = boxes.cls.cpu().numpy().astype(int)
                    person_count = int(np.count_nonzero(classes == self.person_class_idx))
                else:
                    person_count = 0
                accumulated_counts.append(person_count)

                now = loop.time()
                if now - last_emit >= interval_seconds:
                    avg_count = int(round(sum(accumulated_counts) / len(accumulated_counts)))
                    payload = {
                        "timestamp": now_utc(),
                        "people_count": avg_count,
                        "occupancy_level": classify_occupancy(avg_count),
                    }
                    if asyncio.iscoroutinefunction(callback):
                        await callback(payload)
                    else:
                        await asyncio.to_thread(callback, payload)
                    accumulated_counts.clear()
                    last_emit = now
        finally:
            self._running = False
            if self._cap is not None:
                self._cap.release()
                self._cap = None

    def stop(self):
        """Request the capture loop to stop."""
        self._running = False
