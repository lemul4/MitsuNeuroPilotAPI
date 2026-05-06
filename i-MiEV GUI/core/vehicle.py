# core/vehicle.py
# i-MiEV-like vehicle state and lightweight physics model.
#
# This file is intentionally self-contained. It keeps the public fields used by
# AppController/MainWindow:
#   speed, angle, accel, brake, gear,
#   target_angle, target_accel, target_brake, target_gear,
#   update_physics()
#
# Main fixes:
# - Brake never creates negative speed.
# - Reverse speed is possible only in R.
# - Default gear is D because the AI agent does not switch gears.
# - dt is clamped, so UI freezes do not produce speed jumps.
# - The dashboard can use real CARLA telemetry speed when it exists.

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class IMiEVPhysicsConfig:
    # Approximate Mitsubishi i-MiEV / Peugeot iOn / Citroen C-Zero parameters.
    mass_kg: float = 1080.0
    top_speed_kmh: float = 130.0
    reverse_limit_kmh: float = 18.0

    # Urban EV-like acceleration. Tuned for stable dashboard physics, not a full
    # drivetrain simulation. 1.75 m/s^2 gives about 0-50 km/h in ~8 s at 100% throttle.
    max_drive_accel_mps2: float = 1.75

    # Service brake deceleration. Brake is not reverse.
    max_brake_decel_mps2: float = 5.2

    # Coasting / regeneration.
    rolling_decel_mps2: float = 0.10
    drag_decel_at_top_speed_mps2: float = 0.55
    regen_d_decel_mps2: float = 0.18
    regen_e_decel_mps2: float = 0.10
    regen_b_decel_mps2: float = 0.55

    # Actuator smoothing.
    pedal_tau_s: float = 0.18
    brake_tau_s: float = 0.10
    steering_rate_deg_s: float = 720.0

    # Dashboard/telemetry smoothing.
    telemetry_accel_limit_mps2: float = 3.5
    telemetry_decel_limit_mps2: float = 8.0

    # Zero-speed deadband.
    stop_epsilon_mps: float = 0.035


class VehicleState:
    # Gear indexes match ui.main_window labels ["P", "R", "N", "D", "E", "B"],
    # where labels are stored as i + 1.
    GEAR_P = 1
    GEAR_R = 2
    GEAR_N = 3
    GEAR_D = 4
    GEAR_E = 5
    GEAR_B = 6

    def __init__(self, config: Optional[IMiEVPhysicsConfig] = None):
        self.config = config or IMiEVPhysicsConfig()

        self._velocity_mps = 0.0
        self._last_update_ts = time.monotonic()
        self._last_external_speed_ts = time.monotonic()

        self.angle = 0
        self.target_angle = 0

        self.accel = 0.0
        self.brake = 0.0
        self.target_accel = 0.0
        self.target_brake = 0.0

        # AI does not switch gears. Start in D to avoid "throttle does nothing".
        self.gear = self.GEAR_D
        self.target_gear = self.GEAR_D

    @property
    def speed(self) -> float:
        """Dashboard speed in km/h. Negative only when gear is R."""
        return self._velocity_mps * 3.6

    @speed.setter
    def speed(self, value: float):
        # Keeps compatibility with main.py: self.vehicle.speed = data[0]
        try:
            kmh = float(value)
        except (TypeError, ValueError):
            kmh = 0.0

        if self._gear_mode(update=False) != self.GEAR_R and kmh < 0.0:
            kmh = 0.0

        self._velocity_mps = kmh / 3.6

    @property
    def velocity_mps(self) -> float:
        return self._velocity_mps

    def reset_motion(self):
        self._velocity_mps = 0.0
        self.accel = 0.0
        self.brake = 0.0
        self.target_accel = 0.0
        self.target_brake = 0.0
        now = time.monotonic()
        self._last_update_ts = now
        self._last_external_speed_ts = now

    def force_drive_gear(self):
        """Use this when AI control is active; the agent does not emit gear commands."""
        self.gear = self.GEAR_D
        self.target_gear = self.GEAR_D

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(value)))

    @staticmethod
    def _move_towards(current: float, target: float, max_delta: float) -> float:
        if current < target:
            return min(target, current + max_delta)
        if current > target:
            return max(target, current - max_delta)
        return current

    def _resolve_dt(self, dt: Optional[float]) -> float:
        if dt is None:
            now = time.monotonic()
            dt = now - self._last_update_ts
            self._last_update_ts = now
        else:
            self._last_update_ts = time.monotonic()

        if not math.isfinite(dt):
            return 0.05

        # Prevent speed jumps after a GUI freeze or a delayed timer callback.
        return self._clamp(dt, 0.001, 0.10)

    def _external_speed_dt(self) -> float:
        now = time.monotonic()
        dt = now - self._last_external_speed_ts
        self._last_external_speed_ts = now
        if not math.isfinite(dt):
            return 0.05
        return self._clamp(dt, 0.001, 0.15)

    def _smooth_pedals(self, dt: float):
        target_accel = self._clamp(self.target_accel, 0.0, 100.0)
        target_brake = self._clamp(self.target_brake, 0.0, 100.0)

        accel_step = 100.0 * dt / max(self.config.pedal_tau_s, 1e-6)
        brake_step = 100.0 * dt / max(self.config.brake_tau_s, 1e-6)

        self.accel = self._move_towards(self.accel, target_accel, accel_step)
        self.brake = self._move_towards(self.brake, target_brake, brake_step)

    def _smooth_steering(self, dt: float):
        max_step = self.config.steering_rate_deg_s * dt
        self.angle = int(round(self._move_towards(self.angle, self.target_angle, max_step)))

    def _gear_mode(self, update: bool = True) -> int:
        try:
            target = int(self.target_gear)
        except (TypeError, ValueError):
            target = self.gear

        valid = (self.GEAR_P, self.GEAR_R, self.GEAR_N, self.GEAR_D, self.GEAR_E, self.GEAR_B)
        if target in valid:
            gear = target
        else:
            gear = self.gear if self.gear in valid else self.GEAR_D

        if update:
            self.gear = gear

        return gear

    def _aero_drag_decel(self, abs_v: float) -> float:
        top_mps = self.config.top_speed_kmh / 3.6
        if top_mps <= 0:
            return 0.0

        ratio = self._clamp(abs_v / top_mps, 0.0, 1.5)
        return self.config.drag_decel_at_top_speed_mps2 * ratio * ratio

    def _coast_decel_for_gear(self, gear: int, throttle: float) -> float:
        if throttle > 0.03:
            regen = 0.0
        elif gear == self.GEAR_B:
            regen = self.config.regen_b_decel_mps2
        elif gear == self.GEAR_E:
            regen = self.config.regen_e_decel_mps2
        elif gear == self.GEAR_D:
            regen = self.config.regen_d_decel_mps2
        else:
            regen = 0.0

        return self.config.rolling_decel_mps2 + self._aero_drag_decel(abs(self._velocity_mps)) + regen

    def _apply_decel_towards_zero(self, v: float, decel: float, dt: float) -> float:
        if abs(v) <= self.config.stop_epsilon_mps:
            return 0.0

        dv = max(0.0, decel) * dt
        if v > 0.0:
            return max(0.0, v - dv)
        return min(0.0, v + dv)

    def update_physics(self, dt: Optional[float] = None):
        """
        Update virtual/fallback physics.

        Rules:
        - P/N do not accelerate.
        - D/E/B drive forward only.
        - R drives backward only.
        - Brake reduces current speed toward zero; it never changes direction.
        """
        dt = self._resolve_dt(dt)
        cfg = self.config

        self._smooth_pedals(dt)
        self._smooth_steering(dt)

        gear = self._gear_mode(update=True)
        throttle = self._clamp(self.accel / 100.0, 0.0, 1.0)
        brake = self._clamp(self.brake / 100.0, 0.0, 1.0)

        v = self._velocity_mps

        if gear == self.GEAR_P:
            v = self._apply_decel_towards_zero(v, cfg.max_brake_decel_mps2 * 0.7, dt)
            if abs(v) < 0.25:
                v = 0.0
            self._velocity_mps = v
            return

        if gear == self.GEAR_N:
            v = self._apply_decel_towards_zero(
                v,
                cfg.rolling_decel_mps2 + self._aero_drag_decel(abs(v)),
                dt,
            )
            self._velocity_mps = v
            return

        # Brake first. It can only move speed toward 0.
        if brake > 0.001:
            v = self._apply_decel_towards_zero(v, cfg.max_brake_decel_mps2 * brake, dt)

        if abs(v) <= cfg.stop_epsilon_mps and throttle <= 0.001:
            self._velocity_mps = 0.0
            return

        if gear == self.GEAR_R:
            drive_sign = -1.0
            speed_limit_mps = cfg.reverse_limit_kmh / 3.6
            mode_factor = 0.55
        else:
            drive_sign = 1.0
            speed_limit_mps = cfg.top_speed_kmh / 3.6
            if gear == self.GEAR_E:
                mode_factor = 0.72
            elif gear == self.GEAR_B:
                mode_factor = 0.88
            else:
                mode_factor = 1.0

        abs_v = abs(v)
        speed_ratio = self._clamp(abs_v / max(speed_limit_mps, 1e-6), 0.0, 1.0)
        taper = max(0.12, 1.0 - speed_ratio ** 1.7)
        drive_accel = cfg.max_drive_accel_mps2 * mode_factor * throttle * taper

        # If throttle and brake arrive together, brake dominates.
        if brake > 0.001:
            drive_accel *= max(0.0, 1.0 - brake * 1.6)

        if throttle > 0.001:
            v += drive_sign * drive_accel * dt
        else:
            v = self._apply_decel_towards_zero(v, self._coast_decel_for_gear(gear, throttle), dt)

        # Direction guards.
        if gear in (self.GEAR_D, self.GEAR_E, self.GEAR_B) and v < 0.0:
            v = 0.0
        if gear == self.GEAR_R and v > 0.0 and throttle <= 0.001:
            v = 0.0

        # Speed limits.
        if gear == self.GEAR_R:
            v = self._clamp(v, -speed_limit_mps, speed_limit_mps)
        else:
            v = self._clamp(v, 0.0, speed_limit_mps)

        if abs(v) <= cfg.stop_epsilon_mps and throttle <= 0.001:
            v = 0.0

        self._velocity_mps = v

    def apply_external_speed_kmh(self, speed_kmh: float, smooth: bool = True) -> bool:
        """Apply real telemetry speed to the dashboard state."""
        try:
            target_kmh = float(speed_kmh)
        except (TypeError, ValueError):
            return False

        if not math.isfinite(target_kmh):
            return False

        gear = self._gear_mode(update=False)
        if gear != self.GEAR_R and target_kmh < 0.0:
            target_kmh = 0.0

        target_mps = target_kmh / 3.6

        if not smooth:
            self._velocity_mps = target_mps
            return True

        dt = self._external_speed_dt()

        delta = target_mps - self._velocity_mps
        if delta >= 0.0:
            max_delta = self.config.telemetry_accel_limit_mps2 * dt
        else:
            max_delta = self.config.telemetry_decel_limit_mps2 * dt

        if abs(delta) <= max_delta:
            self._velocity_mps = target_mps
        else:
            self._velocity_mps += math.copysign(max_delta, delta)

        if gear != self.GEAR_R and self._velocity_mps < 0.0:
            self._velocity_mps = 0.0

        return True

    def apply_telemetry(self, data: dict, smooth: bool = True) -> bool:
        """
        Use real speed from trace_log.jsonl if it exists.

        Supported keys:
        - speed_kmh / velocity_kmh / kmh: km/h
        - speed_mps / velocity_mps: m/s
        - speed / velocity:
            abs(value) <= 35 is treated as m/s,
            otherwise km/h.
        """
        if not isinstance(data, dict):
            return False

        for key in ("speed_kmh", "velocity_kmh", "kmh"):
            if key in data:
                return self.apply_external_speed_kmh(data[key], smooth=smooth)

        for key in ("speed_mps", "velocity_mps"):
            if key in data:
                try:
                    value_mps = float(data[key])
                except (TypeError, ValueError):
                    return False
                return self.apply_external_speed_kmh(value_mps * 3.6, smooth=smooth)

        for key in ("speed", "velocity"):
            if key not in data:
                continue

            try:
                value = float(data[key])
            except (TypeError, ValueError):
                return False

            if abs(value) <= 35.0:
                return self.apply_external_speed_kmh(value * 3.6, smooth=smooth)
            return self.apply_external_speed_kmh(value, smooth=smooth)

        return False
