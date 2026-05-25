from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .models import ControlIntent, LocalNavigationGoal
from .navigation import clamp


@dataclass(frozen=True)
class AgentPrediction:
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    target_angle_deg: float = 0.0
    confidence: float = 1.0
    timestamp_monotonic: float = field(default_factory=time.monotonic)
    frame_id: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentPrediction":
        d = dict(data or {})
        return cls(
            steer=float(d.get("steer", d.get("steer_norm", 0.0)) or 0.0),
            throttle=float(d.get("throttle", d.get("thr", d.get("throttle_norm", 0.0))) or 0.0),
            brake=float(d.get("brake", d.get("brk", d.get("brake_norm", 0.0))) or 0.0),
            target_angle_deg=float(d.get("target_angle_deg", d.get("target_angle", 0.0)) or 0.0),
            confidence=float(d.get("confidence", d.get("conf", 1.0)) or 1.0),
            timestamp_monotonic=float(d.get("timestamp_monotonic", time.monotonic()) or time.monotonic()),
            frame_id=int(d.get("frame_id", 0) or 0),
            metadata={k: v for k, v in d.items() if k not in {"steer", "steer_norm", "throttle", "thr", "brake", "brk", "target_angle_deg", "target_angle", "confidence", "conf", "timestamp_monotonic", "frame_id"}},
        )


@dataclass
class RealAgentBridge:
    """Combines neural output with Navigator context into ControlIntent.

    The model should not send CAN commands. It sends normalized predictions;
    this bridge attaches route context and forwards a ControlIntent to the
    control service. A deterministic PID path can be used instead of this bridge
    for early real-car tests.
    """

    default_valid_for_ms: int = 100
    max_prediction_age_ms: float = 150.0
    _seq: int = 0

    def build_intent(self, prediction: AgentPrediction, goal: Optional[LocalNavigationGoal]) -> ControlIntent:
        self._seq = (self._seq + 1) & 0x7FFFFFFF
        now = time.monotonic()
        age_ms = max(0.0, (now - float(prediction.timestamp_monotonic)) * 1000.0)
        goal_valid = goal is not None and goal.is_valid()

        desired_speed = float(goal.desired_speed_kmh) if goal_valid else 0.0
        speed_cap = float(goal.speed_cap_kmh) if goal_valid else 0.0
        maneuver = str(goal.maneuver) if goal_valid else "no_goal"
        target_distance = float(goal.distance_to_target_m) if goal_valid else 0.0

        confidence = clamp(prediction.confidence, 0.0, 1.0)
        if age_ms > self.max_prediction_age_ms or not goal_valid:
            confidence = 0.0

        return ControlIntent(
            seq=self._seq,
            timestamp_monotonic=now,
            frame_id=int(prediction.frame_id),
            steer_norm=clamp(prediction.steer, -1.0, 1.0),
            throttle_norm=clamp(prediction.throttle, 0.0, 1.0),
            brake_norm=clamp(prediction.brake, 0.0, 1.0),
            target_angle_deg=float(prediction.target_angle_deg),
            confidence=confidence,
            prediction_age_ms=age_ms,
            desired_speed_kmh=desired_speed,
            speed_cap_kmh=speed_cap,
            nav_maneuver=maneuver,
            nav_target_distance_m=target_distance,
            valid_for_ms=self.default_valid_for_ms,
        )
