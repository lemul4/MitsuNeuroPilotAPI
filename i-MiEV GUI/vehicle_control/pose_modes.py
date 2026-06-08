from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple


@dataclass(frozen=True)
class LocalPoseSeed:
    x_m: float = 0.0
    y_m: float = 0.0
    yaw_deg: float = 0.0


@dataclass
class DeadReckoningPoseState:
    x_m: float = 0.0
    y_m: float = 0.0
    yaw_deg: float = 0.0
    target_index: int = 0
    finished: bool = False


def parse_xyyaw(value: object, default: LocalPoseSeed | None = None) -> LocalPoseSeed:
    if default is None:
        default = LocalPoseSeed()
    if isinstance(value, dict):
        return LocalPoseSeed(
            x_m=float(value.get("x_m", value.get("x", default.x_m))),
            y_m=float(value.get("y_m", value.get("y", default.y_m))),
            yaw_deg=float(value.get("yaw_deg", value.get("yaw", default.yaw_deg))),
        )
    parts = [p.strip() for p in str(value or "").split(",") if p.strip()]
    if not parts:
        return default
    values = [float(p.replace(",", ".")) for p in parts]
    while len(values) < 3:
        values.append(0.0)
    return LocalPoseSeed(values[0], values[1], values[2])


def mission_targets_from_dict(mission: dict) -> Tuple[Tuple[float, float], ...]:
    points = []
    start = mission.get("start") or {}
    goal = mission.get("goal") or {}
    for hint in mission.get("hints", mission.get("via", [])) or []:
        try:
            points.append((float(hint.get("x_m", hint.get("x"))), float(hint.get("y_m", hint.get("y")))))
        except Exception:
            continue
    if goal:
        points.append((float(goal.get("x_m", goal.get("x", 0.0))), float(goal.get("y_m", goal.get("y", 0.0)))))
    if not points and start:
        points.append((float(start.get("x_m", start.get("x", 0.0))), float(start.get("y_m", start.get("y", 0.0)))))
    return tuple(points)


def advance_dead_reckoning_pose(
    state: DeadReckoningPoseState,
    targets: Sequence[Tuple[float, float]],
    speed_mps: float,
    dt_sec: float,
    arrive_radius_m: float = 0.4,
) -> DeadReckoningPoseState:
    if state.finished or not targets:
        return state

    speed_mps = max(0.0, float(speed_mps))
    dt_sec = max(0.0, float(dt_sec))
    step_m = speed_mps * dt_sec
    idx = max(0, min(int(state.target_index), len(targets) - 1))
    x = float(state.x_m)
    y = float(state.y_m)
    yaw = float(state.yaw_deg)

    while idx < len(targets):
        tx, ty = targets[idx]
        dx = float(tx) - x
        dy = float(ty) - y
        dist = math.hypot(dx, dy)
        if dist > 1e-6:
            yaw = math.degrees(math.atan2(dy, dx))
        if dist <= float(arrive_radius_m):
            if idx >= len(targets) - 1:
                return DeadReckoningPoseState(x, y, yaw, idx, True)
            idx += 1
            continue
        if step_m <= 0.0:
            return DeadReckoningPoseState(x, y, yaw, idx, False)
        move = min(step_m, dist)
        x += dx / dist * move
        y += dy / dist * move
        step_m -= move
        if move < dist:
            return DeadReckoningPoseState(x, y, yaw, idx, False)
        if idx >= len(targets) - 1:
            return DeadReckoningPoseState(x, y, yaw, idx, True)
        idx += 1

    return DeadReckoningPoseState(x, y, yaw, len(targets) - 1, True)

