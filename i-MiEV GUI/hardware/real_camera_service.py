from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Dict, List, Optional

try:
    import cv2
    import numpy as np
    import zmq
except Exception:  # pragma: no cover
    cv2 = None
    np = None
    zmq = None


@dataclass(frozen=True)
class RealCameraConfig:
    name: str
    uri: str
    width: int = 640
    height: int = 360
    fps: int = 30


class RealCameraWorker(threading.Thread):
    def __init__(self, config: RealCameraConfig, latest_frames: Dict[str, object], lock: threading.Lock):
        super().__init__(daemon=True)
        self.config = config
        self.latest_frames = latest_frames
        self.lock = lock
        self.running = False
        self.last_frame_ts = 0.0

    def run(self) -> None:
        if cv2 is None:
            print("[REAL_CAMERA] cv2 is unavailable")
            return
        self.running = True
        cap = cv2.VideoCapture(self.config.uri)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        except Exception:
            pass

        while self.running:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            self.last_frame_ts = time.monotonic()
            with self.lock:
                self.latest_frames[self.config.name] = frame
        cap.release()

    def stop(self) -> None:
        self.running = False


class RealCameraService:
    """Low-latency real-camera preview publisher.

    It publishes the same single-JPEG ZMQ stream that the current GUI already
    consumes, so the existing VideoReceiverThread does not need to change.
    AI should consume frames from latest_frames, not from the UI preview.
    """

    def __init__(self, camera_configs: List[RealCameraConfig], zmq_port: int = 5555, publish_fps: int = 15):
        if cv2 is None or np is None or zmq is None:
            raise RuntimeError("cv2, numpy and pyzmq are required for RealCameraService")
        self.camera_configs = list(camera_configs)
        self.zmq_port = int(zmq_port)
        self.publish_interval = 1.0 / max(1, int(publish_fps))
        self.latest_frames: Dict[str, object] = {}
        self._lock = threading.Lock()
        self._workers: List[RealCameraWorker] = []
        self._publisher_thread = None
        self._running = False
        self.context = None
        self.socket = None

    def start(self) -> None:
        self._running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{self.zmq_port}")

        self._workers = [RealCameraWorker(cfg, self.latest_frames, self._lock) for cfg in self.camera_configs]
        for worker in self._workers:
            worker.start()

        self._publisher_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._publisher_thread.start()
        print(f"[REAL_CAMERA] started {len(self._workers)} camera workers, ZMQ port {self.zmq_port}")

    def stop(self) -> None:
        self._running = False
        for worker in self._workers:
            worker.stop()
        for worker in self._workers:
            worker.join(timeout=1.0)
        self._workers = []
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.context is not None:
            self.context.term()
            self.context = None

    def get_latest_frame(self, name: str):
        with self._lock:
            frame = self.latest_frames.get(name)
            return None if frame is None else frame.copy()

    def _publish_loop(self) -> None:
        last = 0.0
        while self._running:
            now = time.monotonic()
            if now - last < self.publish_interval:
                time.sleep(0.002)
                continue
            last = now
            with self._lock:
                frames = {name: frame.copy() for name, frame in self.latest_frames.items() if frame is not None}
            if not frames:
                continue
            mosaic = self._compose_mosaic(frames)
            ok, buffer = cv2.imencode(".jpg", mosaic, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok and self.socket is not None:
                self.socket.send(buffer.tobytes())

    @staticmethod
    def _fit_cover(frame, width: int, height: int):
        if frame is None:
            return np.zeros((height, width, 3), dtype=np.uint8)
        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            return np.zeros((height, width, 3), dtype=np.uint8)
        scale = max(width / float(w), height / float(h))
        resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        rh, rw = resized.shape[:2]
        x = max(0, (rw - width) // 2)
        y = max(0, (rh - height) // 2)
        return resized[y:y + height, x:x + width]

    @staticmethod
    def _label(frame, text: str):
        cv2.rectangle(frame, (0, 0), (190, 32), (0, 0, 0), -1)
        cv2.putText(frame, text.upper(), (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    def _compose_mosaic(self, frames: Dict[str, object]):
        front = frames.get("front") or next(iter(frames.values()))
        canvas = self._fit_cover(front, 1280, 720)
        self._label(canvas, "front")

        thumb_w, thumb_h, margin = 390, 219, 22
        placements = [("left", margin, 720 - margin - thumb_h), ("right", 1280 - margin - thumb_w, 720 - margin - thumb_h)]
        for name, x, y in placements:
            frame = frames.get(name)
            if frame is None:
                continue
            panel = self._fit_cover(frame, thumb_w, thumb_h)
            self._label(panel, name)
            cv2.rectangle(panel, (0, 0), (thumb_w - 1, thumb_h - 1), (190, 200, 220), 1)
            canvas[y:y + thumb_h, x:x + thumb_w] = panel
        return canvas
