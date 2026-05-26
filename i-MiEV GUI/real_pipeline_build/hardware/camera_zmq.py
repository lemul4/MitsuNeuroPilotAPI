from __future__ import annotations

import importlib
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

import zmq

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover
    cv2 = None
    np = None

try:
    from PySide6.QtCore import QThread, Signal, Slot
    from PySide6.QtGui import QImage, QPixmap
except Exception:  # pragma: no cover
    QThread = object
    def Signal(*_args, **_kwargs):
        return None
    def Slot(*_args, **_kwargs):
        def _decorator(fn):
            return fn
        return _decorator
    QImage = QPixmap = None


@dataclass(frozen=True)
class CameraZmqConfig:
    wide_url: str = "tcp://127.0.0.1:5556"
    narrow_url: str = "tcp://127.0.0.1:5557"
    poll_timeout_ms: int = 100
    stale_after_ms: int = 750


class ZmqCameraCellReceiverThread(QThread):
    """Qt preview subscriber for one real camera cell.

    CARLA still uses the legacy single JPEG subscriber in main.py. This class is
    only used in REAL/MOCK modes so the two real camera streams can be displayed
    in two separate UI cells without changing the CARLA video path.
    """

    frame_received = Signal(str, QPixmap)
    status_changed = Signal(str, bool, str)

    def __init__(self, camera_name: str, url: str, parent=None):
        super().__init__(parent)
        self.camera_name = str(camera_name)
        self.url = str(url)
        self.is_running = True
        self.received_count = 0

    def run(self):
        if QImage is None or QPixmap is None:
            return
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.connect(self.url)
            socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self.status_changed.emit(self.camera_name, True, f"подключено: {self.url}")
        except Exception as exc:
            self.status_changed.emit(self.camera_name, False, f"ошибка подключения: {exc}")
            try:
                socket.close()
                context.term()
            except Exception:
                pass
            return

        last_status_at = 0.0
        try:
            while self.is_running:
                if socket.poll(250):
                    try:
                        msg = socket.recv()
                        image = QImage.fromData(msg, "JPG")
                        if image.isNull():
                            self.status_changed.emit(self.camera_name, False, "битый JPEG")
                            continue
                        pixmap = QPixmap.fromImage(image)
                        self.received_count += 1
                        self.frame_received.emit(self.camera_name, pixmap)
                        now = time.monotonic()
                        if now - last_status_at > 1.0:
                            last_status_at = now
                            self.status_changed.emit(self.camera_name, True, f"кадры: {self.received_count}")
                    except Exception as exc:
                        self.status_changed.emit(self.camera_name, False, f"ошибка кадра: {exc}")
        finally:
            try:
                socket.close(0)
                context.term()
            except Exception:
                pass

    def stop(self):
        self.is_running = False
        try:
            self.wait(1200)
        except Exception:
            pass


class RealCameraAgentAnalyzerThread(QThread):
    """Subscribes to the two real camera feeds and optionally runs a model adapter.

    The default behavior is intentionally conservative: it verifies that both
    camera streams are fresh and reports camera/agent readiness. To run a real
    neural model, set MITSU_REAL_AGENT_FACTORY to `module:function`; the factory
    must return an object with `predict(frames: dict) -> dict | None`.

    The prediction dict may contain steer/throttle/brake/target_angle/confidence;
    main.py converts it to ControlIntent and sends it to VehicleControlService.
    If no model factory is configured, VehicleControlService keeps using its
    deterministic waypoint PID fallback while the thread still verifies camera
    freshness for activation gating.
    """

    camera_status_changed = Signal(bool, str)
    prediction_ready = Signal(dict)
    analyzer_status_changed = Signal(str)

    def __init__(self, config: CameraZmqConfig | None = None, parent=None):
        super().__init__(parent)
        self.config = config or CameraZmqConfig()
        self.is_running = True
        self.ai_enabled = False
        self._model_adapter = None
        self._frame_seq = 0

    @Slot(bool)
    def set_ai_enabled(self, enabled: bool):
        self.ai_enabled = bool(enabled)

    def _load_model_adapter(self):
        spec = os.environ.get("MITSU_REAL_AGENT_FACTORY", "").strip()
        if not spec:
            self.analyzer_status_changed.emit("модель не задана; используется маршрутный PID fallback")
            return None
        try:
            module_name, func_name = spec.split(":", 1)
            module = importlib.import_module(module_name)
            factory = getattr(module, func_name)
            adapter = factory()
            self.analyzer_status_changed.emit(f"модель подключена: {spec}")
            return adapter
        except Exception as exc:
            self.analyzer_status_changed.emit(f"модель не подключена: {exc}; PID fallback")
            return None

    @staticmethod
    def _decode_jpeg(msg):
        if cv2 is None or np is None:
            return None
        array = np.frombuffer(msg, dtype=np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_COLOR)

    def run(self):
        self._model_adapter = self._load_model_adapter()
        context = zmq.Context()
        wide = context.socket(zmq.SUB)
        narrow = context.socket(zmq.SUB)
        for socket, url in ((wide, self.config.wide_url), (narrow, self.config.narrow_url)):
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt_string(zmq.SUBSCRIBE, "")
            socket.connect(url)

        poller = zmq.Poller()
        poller.register(wide, zmq.POLLIN)
        poller.register(narrow, zmq.POLLIN)
        latest: Dict[str, dict] = {}
        last_ok = False
        last_status_at = 0.0
        last_predict_at = 0.0

        try:
            while self.is_running:
                events = dict(poller.poll(self.config.poll_timeout_ms))
                now = time.monotonic()
                if wide in events:
                    msg = wide.recv()
                    latest["wide_90"] = {"jpeg": msg, "frame": self._decode_jpeg(msg), "ts": now}
                if narrow in events:
                    msg = narrow.recv()
                    latest["narrow_50"] = {"jpeg": msg, "frame": self._decode_jpeg(msg), "ts": now}

                ages = {name: (now - item["ts"]) * 1000.0 for name, item in latest.items()}
                ok = all(name in latest and ages.get(name, 999999.0) <= self.config.stale_after_ms for name in ("wide_90", "narrow_50"))
                if ok != last_ok or now - last_status_at > 1.0:
                    last_ok = ok
                    last_status_at = now
                    if ok:
                        self.camera_status_changed.emit(True, f"две камеры OK: wide={ages['wide_90']:.0f}ms narrow={ages['narrow_50']:.0f}ms")
                    else:
                        self.camera_status_changed.emit(False, f"ожидание двух камер: {list(latest.keys())}")

                # Model inference is intentionally rate-limited and only runs when
                # AI Preview is enabled. The deterministic waypoint PID fallback
                # remains active if no model adapter is configured.
                if self.ai_enabled and ok and self._model_adapter is not None and now - last_predict_at >= 0.05:
                    last_predict_at = now
                    try:
                        frames = {k: v["frame"] for k, v in latest.items()}
                        prediction = self._model_adapter.predict(frames)
                        if prediction:
                            payload = dict(prediction)
                            self._frame_seq += 1
                            payload.setdefault("frame_id", self._frame_seq)
                            payload.setdefault("timestamp_monotonic", now)
                            self.prediction_ready.emit(payload)
                    except Exception as exc:
                        self.analyzer_status_changed.emit(f"ошибка инференса: {exc}")
                        time.sleep(0.2)
        finally:
            for socket in (wide, narrow):
                try:
                    socket.close(0)
                except Exception:
                    pass
            try:
                context.term()
            except Exception:
                pass

    def stop(self):
        self.is_running = False
        try:
            self.wait(1500)
        except Exception:
            pass
