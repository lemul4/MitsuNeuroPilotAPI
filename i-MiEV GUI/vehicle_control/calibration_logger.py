from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)
    except Exception:
        return str(value)


class RealCalibrationRecorder:
    """Append-only JSONL recorder for real-car steering/throttle calibration."""

    def __init__(self, root: Optional[Path] = None) -> None:
        configured = os.environ.get("MITSU_REAL_CALIBRATION_LOG_DIR", "").strip()
        base = Path(configured).expanduser() if configured else (_repo_root() / "outputs" / "real_calibration")
        if not base.is_absolute():
            base = (_repo_root() / base).resolve()
        run_name = os.environ.get("MITSU_REAL_CALIBRATION_RUN", "").strip()
        if not run_name:
            run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_dir = Path(root or base / run_name)
        self.enabled = os.environ.get("MITSU_REAL_CALIBRATION_LOG", "1").strip().lower() not in {"0", "false", "no", "off"}
        self._lock = threading.Lock()
        self._events_path = self.run_dir / "control_events.jsonl"
        self._manifest_path = self.run_dir / "manifest.json"
        self._started = False

    def _ensure_started(self) -> None:
        if self._started:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if not self._manifest_path.exists():
            manifest = {
                "created_at_local": datetime.now().isoformat(timespec="seconds"),
                "created_at_monotonic": time.monotonic(),
                "format": "mitsu_real_calibration_v1",
                "files": {
                    "control_events": self._events_path.name,
                },
                "notes": (
                    "Each control event stores model prediction metadata, navigator target triplet, "
                    "8th model waypoint steering metadata, arbiter output, CAN command, and current telemetry."
                ),
            }
            self._manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self._started = True

    def close(self) -> None:
        with self._lock:
            self._started = False

    def record_control_event(
        self,
        *,
        source: str,
        intent: Any,
        command: Any,
        telemetry: Any,
        goal: Any = None,
        arbiter: Any = None,
        vehicle: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        now_wall = datetime.now().isoformat(timespec="milliseconds")
        now_mono = time.monotonic()
        intent_meta: Dict[str, Any] = dict(getattr(intent, "metadata", {}) or {})
        record = {
            "event": "control_command",
            "source": source,
            "created_at_local": now_wall,
            "created_at_monotonic": now_mono,
            "intent": _json_safe(intent),
            "intent_metadata": _json_safe(intent_meta),
            "command": _json_safe(command),
            "telemetry": _json_safe(telemetry),
            "goal": _json_safe(goal),
            "vehicle": _json_safe(vehicle or {}),
            "arbiter": {
                "last_target_steering_raw": getattr(arbiter, "last_target_steering_raw", None),
                "last_steering_raw": getattr(arbiter, "last_steering_raw", None),
                "last_steering_raw_delta": getattr(arbiter, "last_steering_raw_delta", None),
                "last_steering_rate_limited": getattr(arbiter, "last_steering_rate_limited", None),
                "max_steering_raw": getattr(arbiter, "max_steering_raw", None),
                "steering_output_gain": getattr(arbiter, "steering_output_gain", None),
                "min_effective_steering_raw": getattr(arbiter, "min_effective_steering_raw", None),
                "steering_deadband_norm": getattr(arbiter, "steering_deadband_norm", None),
                "steering_center_offset_raw": getattr(arbiter, "steering_center_offset_raw", None),
            },
        }
        line = json.dumps(_json_safe(record), ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._ensure_started()
            with self._events_path.open("a", encoding="utf-8") as events_file:
                events_file.write(line + "\n")
