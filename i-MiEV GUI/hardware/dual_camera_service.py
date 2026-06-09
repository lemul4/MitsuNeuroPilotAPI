from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import threading
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import builtins as _builtins

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "protocol_whitelist;file,rtp,udp,crypto,data")

import cv2
import numpy as np
import zmq


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def print(*args, **kwargs):  # noqa: A001 - module-local timestamped print
    return _builtins.print(f"[{_ts()}]", *args, **kwargs)


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
    udp_port: Optional[int] = None
    zmq_port: Optional[int] = None
    relay_port: Optional[int] = None
    enabled: bool = True

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
            udp_port=int(d["udp_port"]) if d.get("udp_port") is not None else None,
            zmq_port=int(d["zmq_port"]) if d.get("zmq_port") is not None else None,
            relay_port=int(d["relay_port"]) if d.get("relay_port") is not None else None,
            enabled=bool(d.get("enabled", True)),
        )


class CameraWorker(threading.Thread):
    def __init__(self, spec: CameraSpec, latest_frames: dict, lock: threading.Lock):
        super().__init__(daemon=True)
        self.spec = spec
        self.latest_frames = latest_frames
        self.lock = lock
        self.running = False
        self.capture = None
        self.ffmpeg_process = None
        self.relay_thread = None
        self.relay_running = False
        self.frame_count = 0
        self.last_error = ""

    def open_capture(self):
        source = self.spec.source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        if isinstance(source, str) and source.lower().endswith(".sdp"):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "protocol_whitelist;file,rtp,udp,crypto,data")
        cap = cv2.VideoCapture(source)
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.spec.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.spec.height)
            cap.set(cv2.CAP_PROP_FPS, self.spec.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def start_udp_relay(self):
        relay_port = self.spec.relay_port
        if relay_port is None or not isinstance(self.spec.source, str):
            return
        if self.relay_thread is not None:
            return

        listen_port = None
        try:
            listen_port = int(getattr(self.spec, "udp_port", 0))
        except Exception:
            listen_port = None
        if not listen_port:
            return

        self.relay_running = True

        def _relay():
            sock = None
            while self.relay_running and sock is None:
                try:
                    candidate = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    if hasattr(socket, "SIO_UDP_CONNRESET"):
                        try:
                            candidate.ioctl(socket.SIO_UDP_CONNRESET, False)
                        except Exception:
                            pass
                    candidate.bind(("0.0.0.0", int(listen_port)))
                    candidate.settimeout(0.5)
                    sock = candidate
                except OSError as exc:
                    self.last_error = f"UDP relay bind failed on :{listen_port}: {exc}"
                    print(f"[REAL_CAMERA] {self.spec.name}: {self.last_error}; retrying", flush=True)
                    try:
                        candidate.close()
                    except Exception:
                        pass
                    time.sleep(1.0)
            if sock is None:
                return
            target = ("127.0.0.1", int(relay_port))
            print(f"[REAL_CAMERA] {self.spec.name}: UDP relay :{listen_port} -> 127.0.0.1:{relay_port}", flush=True)
            count = 0
            last_report = time.monotonic()
            try:
                while self.relay_running:
                    try:
                        data, _addr = sock.recvfrom(65535)
                    except socket.timeout:
                        continue
                    except ConnectionResetError:
                        continue
                    sock.sendto(data, target)
                    count += 1
                    now = time.monotonic()
                    if now - last_report >= 5.0:
                        last_report = now
                        print(f"[REAL_CAMERA] {self.spec.name}: relayed UDP packets={count}", flush=True)
            finally:
                sock.close()

        self.relay_thread = threading.Thread(target=_relay, daemon=True)
        self.relay_thread.start()

    def open_ffmpeg(self):
        try:
            import imageio_ffmpeg
        except Exception as exc:
            self.last_error = f"imageio-ffmpeg unavailable: {exc}"
            return None

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            exe,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-protocol_whitelist",
            "file,rtp,udp,crypto,data",
            "-analyzeduration",
            "30000000",
            "-probesize",
            "30000000",
            "-reorder_queue_size",
            "500",
            "-i",
            str(self.spec.source),
            "-an",
            "-vf",
            f"scale={int(self.spec.width)}:{int(self.spec.height)}",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        try:
            log_path = Path(f"camera_service.{self.spec.name}.ffmpeg.log")
            log_file = open(log_path, "ab", buffering=0)
            return subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=log_file,
                cwd=str(Path.cwd()),
                bufsize=10**7,
            )
        except Exception as exc:
            self.last_error = f"cannot start ffmpeg: {exc}"
            return None

    def read_ffmpeg_frame(self):
        process = self.ffmpeg_process
        if process is None or process.stdout is None:
            return None
        frame_size = int(self.spec.width) * int(self.spec.height) * 3
        data = process.stdout.read(frame_size)
        if len(data) != frame_size:
            return None
        return np.frombuffer(data, dtype=np.uint8).reshape((int(self.spec.height), int(self.spec.width), 3)).copy()

    def run(self):
        self.running = True
        reconnect_delay = 1.0
        use_ffmpeg = isinstance(self.spec.source, str) and self.spec.source.lower().endswith(".sdp")
        if use_ffmpeg:
            self.start_udp_relay()
        while self.running:
            if use_ffmpeg:
                if self.ffmpeg_process is None or self.ffmpeg_process.poll() is not None:
                    self.ffmpeg_process = self.open_ffmpeg()
                    if self.ffmpeg_process is None:
                        time.sleep(reconnect_delay)
                        continue
                    print(f"[REAL_CAMERA] {self.spec.name}: opened {self.spec.source} with ffmpeg", flush=True)

                frame = self.read_ffmpeg_frame()
                if frame is None:
                    self.last_error = "ffmpeg read failed"
                    try:
                        self.ffmpeg_process.kill()
                    except Exception:
                        pass
                    self.ffmpeg_process = None
                    time.sleep(reconnect_delay)
                    continue

                with self.lock:
                    self.latest_frames[self.spec.name] = {
                        "frame": frame,
                        "ts": time.monotonic(),
                        "spec": self.spec,
                    }
                self.frame_count += 1
                continue

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
        if self.ffmpeg_process is not None:
            try:
                self.ffmpeg_process.kill()
            except Exception:
                pass
        self.relay_running = False

    def stop(self):
        self.running = False
        self.relay_running = False


class DualRealCameraService:
    """Publishes two real camera frames as one JPEG mosaic over ZMQ.

    The GUI already subscribes to tcp://127.0.0.1:5555 and expects one JPEG. This
    service keeps that protocol intact. It is deliberately separate from MainWindow.
    """

    def __init__(self, config_path: str = "config/real_cameras.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        self.zmq_port = int(self.config.get("zmq_port", 5555))
        self.publish_fps = float(self.config.get("publish_fps", 15.0))
        self.jpeg_quality = int(self.config.get("jpeg_quality", 82))
        self.output_width = int(self.config.get("output_width", 1280))
        self.output_height = int(self.config.get("output_height", 720))
        self.latest_frames: Dict[str, dict] = {}
        self.lock = threading.Lock()
        self.workers = []
        self.running = False
        self.frame_count = 0
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.cell_sockets = {}

    @staticmethod
    def _load_config(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"camera config not found: {path}")
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _camera_specs(self):
        cameras = self.config.get("cameras", {}) or {}
        specs = []
        for name, data in cameras.items():
            spec = CameraSpec.from_dict(name, data)
            if spec.enabled:
                specs.append(spec)
        if not specs:
            raise RuntimeError("No enabled cameras in config")
        return specs

    def start(self):
        self.socket.bind(f"tcp://*:{self.zmq_port}")
        for spec in self._camera_specs():
            port = self._zmq_port_for(spec)
            if port is not None:
                socket = self.context.socket(zmq.PUB)
                socket.setsockopt(zmq.LINGER, 0)
                socket.bind(f"tcp://*:{port}")
                self.cell_sockets[spec.name] = socket
                print(f"[REAL_CAMERA] {spec.name}: publishing cell on tcp://127.0.0.1:{port}")
            worker = CameraWorker(spec, self.latest_frames, self.lock)
            self.workers.append(worker)
            worker.start()
        self.running = True
        print(f"[REAL_CAMERA] publishing JPEG mosaic on tcp://127.0.0.1:{self.zmq_port}")
        self._publish_loop()

    def stop(self):
        self.running = False
        for worker in self.workers:
            worker.stop()
        for worker in self.workers:
            worker.join(timeout=1.0)
        try:
            self.socket.close()
        finally:
            for socket in list(self.cell_sockets.values()):
                try:
                    socket.close(0)
                except Exception:
                    pass
            self.cell_sockets = {}
            self.context.term()

    def _publish_loop(self):
        period = 1.0 / max(1.0, self.publish_fps)
        try:
            while self.running:
                started = time.monotonic()
                with self.lock:
                    snapshot = dict(self.latest_frames)
                mosaic = self._compose(snapshot)
                self._publish_cells(snapshot)
                if mosaic is not None:
                    ok, buf = cv2.imencode(".jpg", mosaic, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                    if ok:
                        self.socket.send(buf.tobytes())
                        self.frame_count += 1
                        if self.frame_count % 100 == 0:
                            print(f"[REAL_CAMERA] published frames: {self.frame_count}")
                elapsed = time.monotonic() - started
                time.sleep(max(0.001, period - elapsed))
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _publish_cells(self, snapshot: Dict[str, dict]) -> None:
        for name, item in snapshot.items():
            socket = self.cell_sockets.get(name)
            if socket is None:
                continue
            frame = item.get("frame")
            if frame is None:
                continue
            spec = item.get("spec")
            if spec is not None:
                frame = self._fit_cover(frame, spec.width, spec.height)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if ok:
                socket.send(buf.tobytes())

    @staticmethod
    def _zmq_port_for(spec: CameraSpec) -> Optional[int]:
        value = getattr(spec, "zmq_port", None)
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    def _compose(self, snapshot: Dict[str, dict]) -> Optional[np.ndarray]:
        wide = self._find_by_role(snapshot, "wide") or self._find_by_role(snapshot, "front_wide")
        narrow = self._find_by_role(snapshot, "narrow") or self._find_by_role(snapshot, "front_narrow")
        if wide is None and narrow is None:
            return None
        base = self._fit_cover(wide["frame"] if wide else narrow["frame"], self.output_width, self.output_height)
        self._put_label(base, self._label_for(wide, "WIDE 2.8mm FOV90") if wide else "PRIMARY")

        if narrow is not None:
            thumb_w = int(self.output_width * 0.34)
            thumb_h = int(thumb_w * 9 / 16)
            margin = 24
            thumb = self._fit_cover(narrow["frame"], thumb_w, thumb_h)
            self._put_label(thumb, self._label_for(narrow, "NARROW 6mm FOV50"))
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
        return f"{spec.name} {spec.lens_mm:g}mm FOV{spec.fov_deg:g} age={age_ms:.0f}ms"

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
