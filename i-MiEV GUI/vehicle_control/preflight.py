from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .safety_config import RealVehicleSafetyConfig


@dataclass
class PreflightItem:
    name: str
    ok: bool
    message: str


@dataclass
class RealVehiclePreflightReport:
    items: List[PreflightItem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(item.ok for item in self.items)

    def add(self, name: str, ok: bool, message: str) -> None:
        self.items.append(PreflightItem(name, bool(ok), str(message)))

    def to_text(self) -> str:
        lines = []
        for item in self.items:
            status = "OK" if item.ok else "FAIL"
            lines.append(f"[{status}] {item.name}: {item.message}")
        lines.append(f"Overall: {'OK' if self.ok else 'NOT READY'}")
        return "\n".join(lines)


class RealVehiclePreflightChecker:
    def __init__(self, root: str | os.PathLike[str] = "."):
        self.root = Path(root)

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def check(self) -> RealVehiclePreflightReport:
        report = RealVehiclePreflightReport()
        safety_path = self._path(os.environ.get("MITSU_REAL_SAFETY_CONFIG", "config/real_vehicle_safety.json"))
        safety = RealVehicleSafetyConfig.load(safety_path if safety_path.exists() else None)
        report.add("safety_config", True, safety.describe())
        report.add("actuation_guard", not safety.actuation_allowed, "Host actuation disabled by default" if not safety.actuation_allowed else "REAL ACTUATION ENABLED: verify physical E-stop")

        cam_path = self._path(os.environ.get("MITSU_REAL_CAMERAS_CONFIG", "config/real_cameras.json"))
        if cam_path.exists():
            try:
                payload = json.loads(cam_path.read_text(encoding="utf-8"))
                cameras = payload.get("cameras", {}) or {}
                roles = {str(v.get("role", k)).lower() for k, v in cameras.items() if isinstance(v, dict)}
                report.add("camera_config", len(cameras) >= 2, f"{len(cameras)} camera(s), roles={sorted(roles)}")
            except Exception as exc:
                report.add("camera_config", False, str(exc))
        else:
            report.add("camera_config", False, f"missing: {cam_path}")

        pose_path = self._path(os.environ.get("MITSU_REAL_POSE_JSON", "config/current_pose.json"))
        report.add("pose_provider", pose_path.exists() or not safety.require_pose, f"path={pose_path}")

        telemetry_map = self._path(os.environ.get("MITSU_MCU_TELEMETRY_MAP", "config/mcu_telemetry_map.json"))
        report.add("mcu_telemetry_map", telemetry_map.exists(), f"path={telemetry_map}")

        agent_factory = os.environ.get("MITSU_REAL_AGENT_FACTORY", "").strip()
        if agent_factory:
            try:
                mod, fn = agent_factory.split(":", 1)
                module = importlib.import_module(mod)
                getattr(module, fn)
                report.add("agent_factory", True, agent_factory)
            except Exception as exc:
                report.add("agent_factory", False, f"{agent_factory}: {exc}")
        else:
            report.add("agent_factory", False, "not set; PID fallback only")

        model_factory = os.environ.get("MITSU_REAL_MODEL_FACTORY", "").strip()
        report.add("model_factory", bool(model_factory), model_factory or "not set; adapter builds inputs but does not infer")
        return report
