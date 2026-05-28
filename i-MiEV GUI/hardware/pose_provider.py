from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

try:
    from PySide6.QtCore import QThread, Signal
except Exception:  # pragma: no cover
    QThread = object
    def Signal(*_args, **_kwargs):
        return None


class JsonPoseProviderThread(QThread):
    """Polls a small JSON file and emits real vehicle pose.

    This gives a deterministic integration point for RTK/GNSS+IMU, MCU odometry,
    visual odometry or any external localization process before the MCU protocol
    is finalized. Expected JSON:
      {"x_m": 0.0, "y_m": 0.0, "yaw_deg": 0.0, "valid": true, "source": "rtk"}
    Optional fields: lat, lon, timestamp_ms.
    """

    pose_received = Signal(dict)
    status_changed = Signal(bool, str)

    def __init__(self, path: str | None = None, poll_interval_ms: int = 50, stale_after_ms: int = 750, parent=None):
        super().__init__(parent)
        self.path = Path(path or os.environ.get("MITSU_REAL_POSE_JSON", "config/current_pose.json"))
        self.poll_interval_ms = int(poll_interval_ms)
        self.stale_after_ms = int(stale_after_ms)
        self.running = True
        self._last_payload = None
        self._last_status_at = 0.0

    def run(self):
        while self.running:
            now = time.monotonic()
            try:
                if not self.path.exists():
                    if now - self._last_status_at > 1.0:
                        self._last_status_at = now
                        self.status_changed.emit(False, f"pose file not found: {self.path}")
                    self.msleep(self.poll_interval_ms)
                    continue
                with open(self.path, "r", encoding="utf-8") as file:
                    payload = json.load(file)
                payload = dict(payload or {})
                payload.setdefault("valid", True)
                payload.setdefault("source", "json_pose_provider")
                payload.setdefault("timestamp_monotonic", now)
                self._last_payload = payload
                self.pose_received.emit(payload)
                if now - self._last_status_at > 1.0:
                    self._last_status_at = now
                    self.status_changed.emit(bool(payload.get("valid", True)), f"pose OK: {payload.get('source')}")
            except Exception as exc:
                if now - self._last_status_at > 1.0:
                    self._last_status_at = now
                    self.status_changed.emit(False, f"pose error: {exc}")
            self.msleep(self.poll_interval_ms)

    def stop(self):
        self.running = False
        try:
            self.wait(1200)
        except Exception:
            pass
