from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

try:
    from PySide6.QtCore import QThread, Signal
    from PySide6.QtGui import QImage, QPixmap
except Exception:  # pragma: no cover
    QThread = object

    def Signal(*_args, **_kwargs):
        return None

    QImage = QPixmap = None


@dataclass(frozen=True)
class UdpH265CameraSpec:
    name: str
    port: int
    role: str = ""
    decoder: str = "avdec_h265"
    payload: int = 96
    width: int = 1280
    height: int = 720
    jitter_latency_ms: int = 0


def build_udp_h265_pipeline(spec: UdpH265CameraSpec) -> str:
    """Build a low-latency RTP/H265 pipeline ending in RGB appsink."""
    caps = (
        "application/x-rtp, "
        "media=(string)video, "
        "clock-rate=(int)90000, "
        "encoding-name=(string)H265, "
        f"payload=(int){int(spec.payload)}"
    )
    decoder = str(spec.decoder or "avdec_h265").strip()
    latency = max(0, int(spec.jitter_latency_ms))
    return (
        f'udpsrc port={int(spec.port)} caps="{caps}" ! '
        f"rtpjitterbuffer latency={latency} drop-on-latency=true ! "
        "rtph265depay ! "
        "h265parse config-interval=1 disable-passthrough=true ! "
        "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
        f"{decoder} ! "
        "videoconvert ! "
        "video/x-raw,format=RGB ! "
        "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
        "appsink name=appsink emit-signals=false max-buffers=1 drop=true sync=false"
    )


class GStreamerUdpH265ReceiverThread(QThread):
    """Receive one UDP/RTP H265 camera stream and emit the newest frame.

    The class is intentionally isolated from the rest of the vehicle stack. If
    PyGObject/GStreamer is unavailable, it emits a status error and exits.
    """

    frame_received = Signal(str, QPixmap)
    status_changed = Signal(str, bool, str)

    def __init__(self, spec: UdpH265CameraSpec, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.is_running = True
        self.received_count = 0
        self._pipeline = None

    def run(self):
        if QImage is None or QPixmap is None:
            return
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except Exception as exc:
            self.status_changed.emit(self.spec.name, False, f"GStreamer unavailable: {exc}")
            return

        Gst.init(None)
        pipeline_text = build_udp_h265_pipeline(self.spec)
        try:
            pipeline = Gst.parse_launch(pipeline_text)
            appsink = pipeline.get_by_name("appsink")
            bus = pipeline.get_bus()
            pipeline.set_state(Gst.State.PLAYING)
            self._pipeline = pipeline
            self.status_changed.emit(self.spec.name, True, f"UDP/H265 port={self.spec.port}")
        except Exception as exc:
            self.status_changed.emit(self.spec.name, False, f"pipeline error: {exc}")
            return

        last_status_at = 0.0
        try:
            while self.is_running:
                msg = bus.timed_pop_filtered(
                    0,
                    Gst.MessageType.ERROR | Gst.MessageType.EOS,
                )
                if msg is not None:
                    if msg.type == Gst.MessageType.ERROR:
                        err, debug = msg.parse_error()
                        self.status_changed.emit(self.spec.name, False, f"GStreamer error: {err}; {debug}")
                    elif msg.type == Gst.MessageType.EOS:
                        self.status_changed.emit(self.spec.name, False, "GStreamer EOS")
                    break

                sample = appsink.emit("try-pull-sample", 20 * Gst.MSECOND)
                if sample is None:
                    continue
                pixmap = self._sample_to_pixmap(sample)
                if pixmap is None:
                    continue
                self.received_count += 1
                self.frame_received.emit(self.spec.name, pixmap)

                now = time.monotonic()
                if now - last_status_at >= 1.0:
                    last_status_at = now
                    self.status_changed.emit(self.spec.name, True, f"frames={self.received_count}")
        finally:
            try:
                pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            self._pipeline = None

    def _sample_to_pixmap(self, sample) -> Optional[QPixmap]:
        caps = sample.get_caps()
        structure = caps.get_structure(0) if caps is not None and caps.get_size() else None
        if structure is None:
            return None
        width = int(structure.get_value("width") or 0)
        height = int(structure.get_value("height") or 0)
        if width <= 0 or height <= 0:
            return None

        buffer = sample.get_buffer()
        try:
            data = buffer.extract_dup(0, buffer.get_size())
        except Exception:
            return None

        bytes_per_line = width * 3
        image = QImage(data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
        if image.isNull():
            return None
        return QPixmap.fromImage(image)

    def stop(self):
        self.is_running = False
        try:
            pipeline = self._pipeline
            if pipeline is not None:
                pipeline.set_state(0)
        except Exception:
            pass
        try:
            self.wait(1500)
        except Exception:
            pass
