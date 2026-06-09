from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_default_ffmpeg_options = "protocol_whitelist;file,udp,rtp|fflags;nobuffer|flags;low_delay|max_delay;0"
_current_ffmpeg_options = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
if "udp" not in _current_ffmpeg_options or "rtp" not in _current_ffmpeg_options:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = os.environ.get(
        "MITSU_OPENCV_FFMPEG_CAPTURE_OPTIONS",
        _default_ffmpeg_options,
    )

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    from PySide6.QtCore import QThread, Signal
    from PySide6.QtGui import QImage, QPixmap
except Exception:  # pragma: no cover
    QThread = object

    def Signal(*_args, **_kwargs):
        return None

    QImage = QPixmap = None


@dataclass(frozen=True)
class OpenCvUdpH265CameraSpec:
    name: str
    port: int
    role: str = ""
    payload: int = 96
    width: int = 1280
    height: int = 720
    host: str = "127.0.0.1"


def build_h265_rtp_sdp(spec: OpenCvUdpH265CameraSpec) -> str:
    host = str(spec.host or "127.0.0.1")
    payload = int(spec.payload or 96)
    port = int(spec.port)
    return "\n".join(
        [
            "v=0",
            "o=- 0 0 IN IP4 127.0.0.1",
            "s=MitsuNeuroPilot H265 RTP camera",
            f"c=IN IP4 {host}",
            "t=0 0",
            f"m=video {port} RTP/AVP {payload}",
            f"a=rtpmap:{payload} H265/90000",
            f"a=rtcp:{port + 1000} IN IP4 {host}",
            "a=rtcp-mux",
            "a=recvonly",
            "",
        ]
    )


def write_temp_sdp(spec: OpenCvUdpH265CameraSpec) -> str:
    path = Path(tempfile.gettempdir()) / f"mitsu_camera_{spec.name}_{int(spec.port)}.sdp"
    path.write_text(build_h265_rtp_sdp(spec), encoding="ascii")
    return str(path)


def resolve_ffmpeg_executable() -> Optional[str]:
    env_path = os.environ.get("MITSU_FFMPEG_EXE") or os.environ.get("FFMPEG_EXE")
    if env_path:
        expanded = os.path.expandvars(os.path.expanduser(env_path))
        if Path(expanded).is_file():
            return expanded

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    try:
        import imageio_ffmpeg

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_path and Path(ffmpeg_path).is_file():
            return ffmpeg_path
    except Exception:
        pass

    return None


class OpenCvUdpH265ReceiverThread(QThread):
    """Windows-friendly UDP/RTP H265 receiver.

    It uses an ffmpeg subprocess for RTP/H265 demux/decode because OpenCV's
    VideoCapture may ignore protocol_whitelist on Windows builds.
    """

    frame_received = Signal(str, QPixmap)
    model_frame_received = Signal(str, object)
    status_changed = Signal(str, bool, str)

    def __init__(self, spec: OpenCvUdpH265CameraSpec, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.is_running = True
        self.received_count = 0
        self._capture = None
        self._process = None

    def run(self):
        if cv2 is None or QImage is None or QPixmap is None:
            self.status_changed.emit(self.spec.name, False, "OpenCV/PySide unavailable")
            return

        sdp_path = write_temp_sdp(self.spec)
        width = int(self.spec.width or 1280)
        height = int(self.spec.height or 720)
        frame_size = width * height * 3
        ffmpeg_exe = resolve_ffmpeg_executable()
        if not ffmpeg_exe:
            self.status_changed.emit(
                self.spec.name,
                False,
                "ffmpeg not found; set MITSU_FFMPEG_EXE or add ffmpeg to PATH",
            )
            return

        try:
            process = subprocess.Popen(
                [
                    ffmpeg_exe,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-protocol_whitelist",
                    "file,udp,rtp",
                    "-fflags",
                    "nobuffer",
                    "-flags",
                    "low_delay",
                    "-i",
                    sdp_path,
                    "-an",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "bgr24",
                    "pipe:1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=frame_size * 2,
            )
            self._process = process
            self.status_changed.emit(self.spec.name, True, f"OpenCV UDP/H265 port={self.spec.port}")
            last_status_at = 0.0
            while self.is_running:
                if process.stdout is None:
                    break
                raw = process.stdout.read(frame_size)
                if len(raw) != frame_size:
                    if process.poll() is not None:
                        err = b""
                        try:
                            err = process.stderr.read() if process.stderr is not None else b""
                        except Exception:
                            pass
                        self.status_changed.emit(self.spec.name, False, f"ffmpeg stopped: {err.decode(errors='ignore')[:200]}")
                        break
                    time.sleep(0.01)
                    continue
                import numpy as _np

                frame_bgr = _np.frombuffer(raw, dtype=_np.uint8).reshape((height, width, 3)).copy()
                pixmap = self._frame_to_pixmap(frame_bgr)
                if pixmap is None:
                    continue
                self.received_count += 1
                self.frame_received.emit(self.spec.name, pixmap)
                self.model_frame_received.emit(self.spec.name, frame_bgr)

                now = time.monotonic()
                if now - last_status_at >= 1.0:
                    last_status_at = now
                    self.status_changed.emit(self.spec.name, True, f"frames={self.received_count}")
        finally:
            try:
                if self._process is not None:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=1.0)
                    except Exception:
                        self._process.kill()
            except Exception:
                pass
            self._process = None

    @staticmethod
    def _frame_to_pixmap(frame_bgr) -> Optional[QPixmap]:
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            height, width = frame_rgb.shape[:2]
            bytes_per_line = width * 3
            image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
            if image.isNull():
                return None
            return QPixmap.fromImage(image)
        except Exception:
            return None

    def stop(self):
        self.is_running = False
        try:
            if self._process is not None:
                self._process.terminate()
        except Exception:
            pass
        try:
            self.wait(1500)
        except Exception:
            pass
