from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import zmq


@dataclass(frozen=True)
class CameraSpec:
    name: str
    source: object
    role: str
    lens_mm: float
    fov_deg: float
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    enabled: bool = True
    zmq_port: Optional[int] = None

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "CameraSpec":
        d = dict(data or {})
        return cls(
            name=str(d.get("name") or name),
            source=d.get("source", d.get("url", d.get("device", 0))),
            role=str(d.get("role") or name),
            lens_mm=float(d.get("lens_mm", 0.0) or 0.0),
            fov_deg=float(d.get("fov_deg", d.get("fov", 0.0)) or 0.0),
            width=int(d.get("width", 1280) or 1280),
            height=int(d.get("height", 720) or 720),
            fps=float(d.get("fps", 30.0) or 30.0),
            enabled=bool(d.get("enabled", True)),
            zmq_port=int(d["zmq_port"]) if d.get("zmq_port") is not None else None,
        )


class CameraWorker(threading.Thread):
    def __init__(self, spec: CameraSpec, latest_frames: dict, lock: threading.Lock):
        super().__init__(daemon=True)
        self.spec = spec
        self.latest_frames = latest_frames
        self.lock = lock
        self.running = False
        self.capture = None
        self.frame_count = 0
        self.last_error = ""

    def open_capture(self):
        source = self.spec.source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        cap = cv2.VideoCapture(source)
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.spec.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.spec.height)
            cap.set(cv2.CAP_PROP_FPS, self.spec.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def run(self):
        self.running = True
        reconnect_delay = 1.0
        while self.running:
            if self.capture is None or not self.capture.isOpened():
                self.capture = self.open_capture()
                if not self.capture.isOpened():
                    self.last_error = f"cannot open source {self.spec.source}"
                    time.sleep(reconnect_delay)
                    continue
                print(f"[REAL_CAMERA] {self.spec.name}: opened {self.spec.source}")

            ok, frame = self.capture.read()
            if not ok or frame is None:
                self.last_error = "read failed"
                try:
                    self.capture.release()
                except Exception:
                    pass
                self.capture = None
                time.sleep(reconnect_delay)
                continue

            with self.lock:
                self.latest_frames[self.spec.name] = {
                    "frame": frame,
                    "ts": time.monotonic(),
                    "spec": self.spec,
                }
            self.frame_count += 1

        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass

    def stop(self):
        self.running = False


class DualRealCameraService:
    """Publishes real camera streams without touching CARLA video logic.

    Outputs:
      - legacy mosaic JPEG on `zmq_port` (default 5555) for backward compatibility;
      - one JPEG stream per camera on camera-specific ports (defaults 5556/5557),
        so the GUI can show two independent cells and the real agent can consume
        exactly the same live frames.
    """

    def __init__(self, config_path: str = "config/real_cameras.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        self.zmq_port = int(self.config.get("zmq_port", 5555))
        self.publish_fps = float(self.config.get("publish_fps", self.config.get("preview_fps", 15.0)))
        self.jpeg_quality = int(self.config.get("jpeg_quality", 82))
        self.output_width = int(self.config.get("output_width", 1280))
        self.output_height = int(self.config.get("output_height", 720))
        self.latest_frames: Dict[str, dict] = {}
        self.lock = threading.Lock()
        self.workers = []
        self.running = False
        self.frame_count = 0
        self.context = zmq.Context()
        self.mosaic_socket = self.context.socket(zmq.PUB)
        self.mosaic_socket.setsockopt(zmq.LINGER, 0)
        self.camera_sockets: Dict[str, zmq.Socket] = {}

    @staticmethod
    def _load_config(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"camera config not found: {path}")
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _camera_specs(self):
        cameras = self.config.get("cameras", {}) or {}
        specs = []
        default_ports = {
            "front_wide": 5556,
            "wide": 5556,
            "wide_90": 5556,
            "front_narrow": 5557,
            "narrow": 5557,
            "narrow_50": 5557,
        }
        for name, data in cameras.items():
            spec = CameraSpec.from_dict(name, data)
            if not spec.enabled:
                continue
            if spec.zmq_port is None:
                role = spec.role.lower()
                port = None
                for token, default_port in default_ports.items():
                    if token in role:
                        port = default_port
                        break
                if port is None:
                    port = 5556 + len(specs)
                spec = CameraSpec(**{**spec.__dict__, "zmq_port": port})
            specs.append(spec)
        if not specs:
            raise RuntimeError("No enabled cameras in config")
        return specs

    def start(self):
        self.mosaic_socket.bind(f"tcp://*:{self.zmq_port}")
        for spec in self._camera_specs():
            worker = CameraWorker(spec, self.latest_frames, self.lock)
            self.workers.append(worker)
            worker.start()
            socket = self.context.socket(zmq.PUB)
            socket.setsockopt(zmq.LINGER, 0)
            socket.bind(f"tcp://*:{spec.zmq_port}")
            self.camera_sockets[spec.name] = socket
            print(f"[REAL_CAMERA] {spec.name}: publishing cell stream on tcp://127.0.0.1:{spec.zmq_port}")
        self.running = True
        print(f"[REAL_CAMERA] publishing legacy mosaic on tcp://127.0.0.1:{self.zmq_port}")
        self._publish_loop()

    def stop(self):
        self.running = False
        for worker in self.workers:
            worker.stop()
        for worker in self.workers:
            worker.join(timeout=1.0)
        for socket in list(self.camera_sockets.values()):
            try:
                socket.close(0)
            except Exception:
                pass
        self.camera_sockets.clear()
        try:
            self.mosaic_socket.close(0)
        finally:
            self.context.term()

    def _publish_loop(self):
        period = 1.0 / max(1.0, self.publish_fps)
        try:
            while self.running:
                started = time.monotonic()
                with self.lock:
                    snapshot = dict(self.latest_frames)
                self._publish_camera_cells(snapshot)
                mosaic = self._compose(snapshot)
                if mosaic is not None:
                    ok, buf = cv2.imencode(".jpg", mosaic, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                    if ok:
                        self.mosaic_socket.send(buf.tobytes())
                        self.frame_count += 1
                        if self.frame_count % 100 == 0:
                            print(f"[REAL_CAMERA] published frames: {self.frame_count}")
                elapsed = time.monotonic() - started
                time.sleep(max(0.001, period - elapsed))
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _publish_camera_cells(self, snapshot: Dict[str, dict]) -> None:
        for name, item in snapshot.items():
            socket = self.camera_sockets.get(name)
            if socket is None:
                continue
            frame = item.get("frame")
            spec = item.get("spec")
            if frame is None:
                continue
            out_w = int(getattr(spec, "width", 1280) or 1280)
            out_h = int(getattr(spec, "height", 720) or 720)
            preview = self._fit_cover(frame, min(out_w, 1280), min(out_h, 720))
            self._put_label(preview, self._label_for(item, name))
            ok, buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if ok:
                socket.send(buf.tobytes())

    def _compose(self, snapshot: Dict[str, dict]) -> Optional[np.ndarray]:
        wide = self._find_by_role(snapshot, "wide") or self._find_by_role(snapshot, "front_wide")
        narrow = self._find_by_role(snapshot, "narrow") or self._find_by_role(snapshot, "front_narrow")
        if wide is None and narrow is None:
            return None
        base = self._fit_cover(wide["frame"] if wide else narrow["frame"], self.output_width, self.output_height)
        self._put_label(base, self._label_for(wide, "2.8 мм / FOV 90°") if wide else "PRIMARY")

        if narrow is not None:
            thumb_w = int(self.output_width * 0.34)
            thumb_h = int(thumb_w * 9 / 16)
            margin = 24
            thumb = self._fit_cover(narrow["frame"], thumb_w, thumb_h)
            self._put_label(thumb, self._label_for(narrow, "6 мм / FOV 50°"))
            cv2.rectangle(thumb, (0, 0), (thumb_w - 1, thumb_h - 1), (220, 220, 220), 2)
            x = self.output_width - margin - thumb_w
            y = self.output_height - margin - thumb_h
            base[y:y + thumb_h, x:x + thumb_w] = thumb
        return base

    @staticmethod
    def _find_by_role(snapshot: Dict[str, dict], role: str) -> Optional[dict]:
        role = role.lower()
        for item in snapshot.values():
            spec = item.get("spec")
            if spec and role in str(spec.role).lower():
                return item
        return None

    @staticmethod
    def _label_for(item: Optional[dict], fallback: str) -> str:
        if not item:
            return fallback
        spec = item.get("spec")
        age_ms = (time.monotonic() - float(item.get("ts", time.monotonic()))) * 1000.0
        if spec is None:
            return f"{fallback} age={age_ms:.0f}ms"
        return f"{spec.name} {spec.lens_mm:g}мм FOV{spec.fov_deg:g} age={age_ms:.0f}ms"

    @staticmethod
    def _fit_cover(frame, slot_w, slot_h):
        if frame is None:
            return np.zeros((slot_h, slot_w, 3), dtype=np.uint8)
        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            return np.zeros((slot_h, slot_w, 3), dtype=np.uint8)
        scale = max(slot_w / float(w), slot_h / float(h))
        resized = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        rh, rw = resized.shape[:2]
        x = max(0, (rw - slot_w) // 2)
        y = max(0, (rh - slot_h) // 2)
        return np.ascontiguousarray(resized[y:y + slot_h, x:x + slot_w])

    @staticmethod
    def _put_label(frame, text):
        text = str(text)[:64]
        cv2.rectangle(frame, (0, 0), (560, 34), (0, 0, 0), -1)
        cv2.putText(frame, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Two-camera real i-MiEV ZMQ preview service")
    parser.add_argument("--config", default="config/real_cameras.example.json")
    args = parser.parse_args()
    service = DualRealCameraService(args.config)
    service.start()


if __name__ == "__main__":
    main()
