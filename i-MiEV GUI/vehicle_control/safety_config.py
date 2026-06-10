from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RealVehicleSafetyConfig:
    """Runtime gates for real i-MiEV testing.

    The default profile is intentionally conservative. It permits mock/dry-run work,
    but real actuation must be explicitly enabled by config or environment variable.
    """

    profile_name: str = "bench_default"
    allow_real_actuation: bool = False
    require_heartbeat: bool = True
    require_pose: bool = True
    require_two_cameras: bool = True
    require_gear_feedback: bool = True
    max_accel_pct: int = 10
    max_brake_pct: int = 80
    max_steering_raw: int = 60
    steering_output_gain: float = 1.0
    min_effective_steering_raw: int = 0
    steering_deadband_norm: float = 0.0
    steering_center_offset_raw: int = 0
    manual_accel_rise_pct_per_sec: float = 35.0
    manual_accel_fall_pct_per_sec: float = 80.0
    manual_brake_rise_pct_per_sec: float = 120.0
    manual_brake_fall_pct_per_sec: float = 160.0
    command_timeout_ms: int = 100
    gear_confirm_timeout_sec: float = 2.0
    park_speed_threshold_kmh: float = 0.25
    dry_run: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "RealVehicleSafetyConfig":
        d = dict(data or {})
        return cls(
            profile_name=str(d.get("profile_name", d.get("name", "bench_default"))),
            allow_real_actuation=bool(d.get("allow_real_actuation", False)),
            require_heartbeat=bool(d.get("require_heartbeat", True)),
            require_pose=bool(d.get("require_pose", True)),
            require_two_cameras=bool(d.get("require_two_cameras", True)),
            require_gear_feedback=bool(d.get("require_gear_feedback", True)),
            max_accel_pct=int(d.get("max_accel_pct", 10)),
            max_brake_pct=int(d.get("max_brake_pct", 80)),
            max_steering_raw=int(d.get("max_steering_raw", 60)),
            steering_output_gain=float(d.get("steering_output_gain", 1.0)),
            min_effective_steering_raw=int(d.get("min_effective_steering_raw", 0)),
            steering_deadband_norm=float(d.get("steering_deadband_norm", 0.0)),
            steering_center_offset_raw=int(d.get("steering_center_offset_raw", 0)),
            manual_accel_rise_pct_per_sec=float(d.get("manual_accel_rise_pct_per_sec", 35.0)),
            manual_accel_fall_pct_per_sec=float(d.get("manual_accel_fall_pct_per_sec", 80.0)),
            manual_brake_rise_pct_per_sec=float(d.get("manual_brake_rise_pct_per_sec", 120.0)),
            manual_brake_fall_pct_per_sec=float(d.get("manual_brake_fall_pct_per_sec", 160.0)),
            command_timeout_ms=int(d.get("command_timeout_ms", 100)),
            gear_confirm_timeout_sec=float(d.get("gear_confirm_timeout_sec", 2.0)),
            park_speed_threshold_kmh=float(d.get("park_speed_threshold_kmh", 0.25)),
            dry_run=bool(d.get("dry_run", True)),
            metadata=dict(d.get("metadata") or {}),
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "RealVehicleSafetyConfig":
        env_path = os.environ.get("MITSU_REAL_SAFETY_CONFIG", "").strip()
        path = Path(path or env_path or "config/real_vehicle_safety.json")
        if path.exists():
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
            cfg = cls.from_dict(data)
        else:
            cfg = cls()

        # Environment override is intentionally separate from config. This prevents
        # accidental actuation when a config file is copied between laptops.
        env_actuation = os.environ.get("MITSU_REAL_ENABLE_ACTUATION", "").strip().lower()
        if env_actuation:
            cfg = cls.from_dict({**cfg.__dict__, "allow_real_actuation": env_actuation in {"1", "true", "yes", "on"}})
        env_dry_run = os.environ.get("MITSU_REAL_DRY_RUN", "").strip().lower()
        if env_dry_run:
            cfg = cls.from_dict({**cfg.__dict__, "dry_run": env_dry_run not in {"0", "false", "no", "off"}})
        env_accel_rise = os.environ.get("MITSU_MANUAL_ACCEL_RISE_PCT_PER_SEC", "").strip()
        if env_accel_rise:
            cfg = cls.from_dict({**cfg.__dict__, "manual_accel_rise_pct_per_sec": float(env_accel_rise)})
        env_accel_fall = os.environ.get("MITSU_MANUAL_ACCEL_FALL_PCT_PER_SEC", "").strip()
        if env_accel_fall:
            cfg = cls.from_dict({**cfg.__dict__, "manual_accel_fall_pct_per_sec": float(env_accel_fall)})
        env_brake_rise = os.environ.get("MITSU_MANUAL_BRAKE_RISE_PCT_PER_SEC", "").strip()
        if env_brake_rise:
            cfg = cls.from_dict({**cfg.__dict__, "manual_brake_rise_pct_per_sec": float(env_brake_rise)})
        env_brake_fall = os.environ.get("MITSU_MANUAL_BRAKE_FALL_PCT_PER_SEC", "").strip()
        if env_brake_fall:
            cfg = cls.from_dict({**cfg.__dict__, "manual_brake_fall_pct_per_sec": float(env_brake_fall)})
        return cfg

    @property
    def actuation_allowed(self) -> bool:
        return bool(self.allow_real_actuation) and not bool(self.dry_run)

    def describe(self) -> str:
        return (
            f"profile={self.profile_name}; dry_run={self.dry_run}; "
            f"actuation_allowed={self.actuation_allowed}; "
            f"max_accel={self.max_accel_pct}%; max_steer={self.max_steering_raw}; "
            f"steer_gain={self.steering_output_gain:.2f}; min_steer={self.min_effective_steering_raw}; "
            f"steer_offset={self.steering_center_offset_raw}; "
            f"manual_rates accel+={self.manual_accel_rise_pct_per_sec:.1f}%/s "
            f"brake+={self.manual_brake_rise_pct_per_sec:.1f}%/s"
        )
