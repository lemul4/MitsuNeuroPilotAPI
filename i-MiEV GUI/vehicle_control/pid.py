from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from .models import ControlIntent, LocalNavigationGoal, VehicleTelemetry
from .navigation import clamp


@dataclass
class PIDController:
    kp: float
    ki: float
    kd: float
    output_min: float
    output_max: float
    integral_min: float = -1.0
    integral_max: float = 1.0
    _integral: float = 0.0
    _last_error: Optional[float] = None
    _last_ts: Optional[float] = None

    def reset(self) -> None:
        self._integral = 0.0
        self._last_error = None
        self._last_ts = None

    def step(self, error: float, now: Optional[float] = None) -> float:
        now = time.monotonic() if now is None else float(now)
        if self._last_ts is None:
            dt = 0.02
        else:
            dt = max(0.001, min(0.25, now - self._last_ts))
        self._last_ts = now

        self._integral = clamp(self._integral + float(error) * dt, self.integral_min, self.integral_max)
        if self._last_error is None:
            derivative = 0.0
        else:
            derivative = (float(error) - self._last_error) / dt
        self._last_error = float(error)

        output = self.kp * float(error) + self.ki * self._integral + self.kd * derivative
        return clamp(output, self.output_min, self.output_max)


@dataclass
class WaypointPIDController:
    """Deterministic path/speed controller for real-vehicle bench tests.

    The neural agent may later replace or augment this controller. For the first
    coordinate route tests this controller converts Navigator goals into the same
    ControlIntent structure used by the arbiter and CAN gateway.
    """

    steering_pid: PIDController = field(default_factory=lambda: PIDController(0.030, 0.000, 0.006, -1.0, 1.0, -10.0, 10.0))
    cross_track_gain: float = 0.18
    max_target_angle_deg: float = 630.0

    speed_pid: PIDController = field(default_factory=lambda: PIDController(0.20, 0.015, 0.020, -1.0, 1.0, -5.0, 5.0))
    brake_deadband_kmh: float = 0.20
    accel_deadband_kmh: float = 0.15
    stop_distance_m: float = 1.0
    min_confidence: float = 0.90
    _seq: int = 0

    def reset(self) -> None:
        self.steering_pid.reset()
        self.speed_pid.reset()
        self._seq = 0

    def build_intent(self, goal: LocalNavigationGoal, telemetry: VehicleTelemetry) -> ControlIntent:
        self._seq = (self._seq + 1) & 0x7FFFFFFF
        if goal is None or not goal.is_valid():
            return ControlIntent(seq=self._seq, brake_norm=0.5, confidence=0.0, nav_maneuver="invalid_goal", valid_for_ms=80)

        heading_error_norm = clamp(float(goal.heading_error_deg) / 45.0, -1.0, 1.0)
        xtrack_correction = clamp(float(goal.cross_track_error_m) * self.cross_track_gain, -0.5, 0.5)
        steer_error = heading_error_norm - xtrack_correction
        steer_norm = self.steering_pid.step(steer_error)
        target_angle_deg = clamp(steer_norm * self.max_target_angle_deg, -self.max_target_angle_deg, self.max_target_angle_deg)

        desired_speed = clamp(float(goal.desired_speed_kmh), 0.0, float(goal.speed_cap_kmh or 0.0))
        if goal.stop_required or goal.distance_to_goal_m <= self.stop_distance_m:
            desired_speed = 0.0

        current_speed = max(0.0, float(getattr(telemetry, "speed_kmh", 0.0) or 0.0))
        speed_error = desired_speed - current_speed
        speed_cmd = self.speed_pid.step(speed_error)

        throttle = 0.0
        brake = 0.0
        if goal.stop_required and goal.distance_to_goal_m <= self.stop_distance_m:
            brake = 0.35 if current_speed > 0.2 else 0.15
        elif speed_error > self.accel_deadband_kmh:
            throttle = clamp(speed_cmd, 0.0, 1.0)
        elif speed_error < -self.brake_deadband_kmh:
            brake = clamp(-speed_cmd, 0.0, 1.0)

        # Do not accelerate aggressively during turns or when steering is large.
        if abs(steer_norm) > 0.55:
            throttle = min(throttle, 0.25)
        if goal.maneuver in {"turn_left", "turn_right", "intersection", "slow"}:
            throttle = min(throttle, 0.35)

        return ControlIntent(
            seq=self._seq,
            steer_norm=steer_norm,
            throttle_norm=throttle,
            brake_norm=brake,
            target_angle_deg=target_angle_deg,
            confidence=self.min_confidence,
            prediction_age_ms=0.0,
            desired_speed_kmh=desired_speed,
            speed_cap_kmh=float(goal.speed_cap_kmh or desired_speed),
            nav_maneuver=str(goal.maneuver),
            nav_target_distance_m=float(goal.distance_to_target_m),
            valid_for_ms=120,
        )
