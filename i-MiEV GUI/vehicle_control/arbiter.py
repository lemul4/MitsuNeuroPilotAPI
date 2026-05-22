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
    stale_prediction_ms: float = 150.0
    min_confidence: float = 0.15
    default_speed_cap_kmh: float = 3.0

    _last_steering_raw: int = 0
    _last_command_time: float = 0.0
    _seq: int = 0

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

        steering_raw = int(round(steer_norm * self.max_steering_raw))
        steering_raw = self._limit_steering_rate(steering_raw, now)

        accel_pct = int(round(throttle_norm * self.max_accel_pct))
        brake_pct = int(round(brake_norm * self.max_brake_pct))

        # Brake has absolute priority over throttle.
        if brake_pct > 0:
            accel_pct = 0

        speed_cap = _clamp(intent.speed_cap_kmh or self.default_speed_cap_kmh, 0.0, 50.0)
        if telemetry is not None and telemetry.speed_kmh > speed_cap + 0.5:
            accel_pct = 0
            brake_pct = max(brake_pct, 20)

        if not active:
            accel_pct = 0
            brake_pct = max(brake_pct, 25)

        return VehicleCommand(
            seq=self._seq,
            timestamp_monotonic=now,
            active=bool(active),
            gear_request=gear if active else None,
            steering_raw=steering_raw,
            accel_pct=max(0, min(100, accel_pct)),
            brake_pct=max(0, min(100, brake_pct)),
            cruise_enabled=bool(active),
            valid_for_ms=min(max(int(intent.valid_for_ms or 100), 40), 250),
            reason="ai_intent" if active else "inactive",
        )

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
