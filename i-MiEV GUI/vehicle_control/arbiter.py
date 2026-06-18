from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

from .models import ControlIntent, VehicleCommand, VehicleTelemetry, Gear


def _clamp(value, lo, hi):
    try:
        value = float(value)
    except Exception:
        value = 0.0
    if math.isnan(value) or math.isinf(value):
        value = 0.0
    return max(lo, min(hi, value))


@dataclass
class ControlArbiter:
    max_accel_pct: int = 25
    max_brake_pct: int = 80
    max_steering_raw: int = 100
    max_steering_rate_raw_per_sec: float = 140.0
    steering_output_gain: float = 1.0
    min_effective_steering_raw: int = 0
    steering_deadband_norm: float = 0.0
    steering_center_offset_raw: int = 0
    stale_prediction_ms: float = 150.0
    min_confidence: float = 0.15

    _last_steering_raw: int = 0
    _last_command_time: float = 0.0
    _seq: int = 0

    @classmethod
    def from_safety_config(cls, config) -> "ControlArbiter":
        """Create conservative command limits from RealVehicleSafetyConfig."""
        arbiter = cls(
            max_accel_pct=int(getattr(config, "max_accel_pct", 10)),
            max_brake_pct=int(getattr(config, "max_brake_pct", 80)),
            max_steering_raw=int(getattr(config, "max_steering_raw", 60)),
            max_steering_rate_raw_per_sec=float(getattr(config, "max_steering_rate_raw_per_sec", 140.0)),
            steering_output_gain=float(getattr(config, "steering_output_gain", 1.0)),
            min_effective_steering_raw=int(getattr(config, "min_effective_steering_raw", 0)),
            steering_deadband_norm=float(getattr(config, "steering_deadband_norm", 0.0)),
            steering_center_offset_raw=int(getattr(config, "steering_center_offset_raw", 0)),
            stale_prediction_ms=float(getattr(config, "command_timeout_ms", 100)),
        )
        arbiter._min_steering_output_gain = float(getattr(config, "steering_output_gain", 1.0))
        return arbiter

    def build_command(
        self,
        intent: ControlIntent,
        telemetry: Optional[VehicleTelemetry] = None,
        gear: Gear = Gear.D,
        active: bool = True,
    ) -> VehicleCommand:
        now = time.monotonic()
        self._seq = (self._seq + 1) & 0x7FFFFFFF

        if intent is None:
            return VehicleCommand.safe_stop(self._seq, reason="missing_intent")

        if intent.is_expired(now) or float(intent.prediction_age_ms or 0.0) > self.stale_prediction_ms:
            return VehicleCommand.safe_stop(self._seq, reason="stale_prediction")

        if _clamp(intent.confidence, 0.0, 1.0) < self.min_confidence:
            return VehicleCommand.safe_stop(self._seq, reason="low_confidence")

        steer_norm = _clamp(intent.steer_norm, -1.0, 1.0)
        throttle_norm = _clamp(intent.throttle_norm, 0.0, 1.0)
        brake_norm = _clamp(intent.brake_norm, 0.0, 1.0)

        steering_gain = _clamp(getattr(self, "steering_output_gain", 1.0), 0.0, 3.0)
        target_steering_raw = int(round(steer_norm * steering_gain * self.max_steering_raw))
        target_steering_raw = self._apply_steering_deadband(steer_norm, target_steering_raw)
        target_steering_raw = self._apply_steering_center_offset(target_steering_raw)
        previous_steering_raw = int(self._last_steering_raw)
        steering_raw = self._limit_steering_rate(target_steering_raw, now)
        self.last_target_steering_raw = int(target_steering_raw)
        self.last_steering_raw = int(steering_raw)
        self.last_steering_raw_delta = int(steering_raw - previous_steering_raw)
        self.last_steering_rate_limited = int(steering_raw) != int(target_steering_raw)

        accel_pct = int(round(throttle_norm * self.max_accel_pct))
        brake_pct = int(round(brake_norm * self.max_brake_pct))

        # Brake has absolute priority over throttle.
        if brake_pct > 0:
            accel_pct = 0

        if not active:
            accel_pct = 0
            brake_pct = max(brake_pct, 25)

        return VehicleCommand(
            seq=self._seq,
            timestamp_monotonic=now,
            active=bool(active),
            gear_request=None,
            steering_raw=steering_raw,
            accel_pct=max(0, min(100, accel_pct)),
            brake_pct=max(0, min(100, brake_pct)),
            cruise_enabled=False,
            send_cruise_frame=False,
            valid_for_ms=min(max(int(intent.valid_for_ms or 100), 40), 250),
            reason="ai_intent" if active else "inactive",
        )

    def _apply_steering_center_offset(self, steering_raw: int) -> int:
        offset = int(round(_clamp(getattr(self, "steering_center_offset_raw", 0), -self.max_steering_raw, self.max_steering_raw)))
        if offset == 0:
            return int(steering_raw)
        return int(max(-self.max_steering_raw, min(self.max_steering_raw, int(steering_raw) + offset)))

    def _apply_steering_deadband(self, steer_norm: float, steering_raw: int) -> int:
        deadband = _clamp(getattr(self, "steering_deadband_norm", 0.0), 0.0, 0.25)
        if abs(float(steer_norm)) <= deadband:
            return 0

        min_raw = int(round(_clamp(getattr(self, "min_effective_steering_raw", 0), 0, self.max_steering_raw)))
        if min_raw <= 0 or steering_raw == 0:
            return int(max(-self.max_steering_raw, min(self.max_steering_raw, steering_raw)))

        if abs(steering_raw) < min_raw:
            steering_raw = min_raw if steering_raw > 0 else -min_raw
        return int(max(-self.max_steering_raw, min(self.max_steering_raw, steering_raw)))

    def _limit_steering_rate(self, target_raw: int, now: float) -> int:
        if self._last_command_time <= 0.0:
            self._last_command_time = now
            self._last_steering_raw = int(max(-self.max_steering_raw, min(self.max_steering_raw, target_raw)))
            return self._last_steering_raw

        dt = max(0.001, now - self._last_command_time)
        max_delta = max(1.0, self.max_steering_rate_raw_per_sec * dt)
        delta = target_raw - self._last_steering_raw
        if delta > max_delta:
            target_raw = int(round(self._last_steering_raw + max_delta))
        elif delta < -max_delta:
            target_raw = int(round(self._last_steering_raw - max_delta))

        target_raw = int(max(-self.max_steering_raw, min(self.max_steering_raw, target_raw)))
        self._last_command_time = now
        self._last_steering_raw = target_raw
        return target_raw
