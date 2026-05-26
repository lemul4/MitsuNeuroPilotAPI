from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from .models import Mission, Waypoint, Pose2D, LocalNavigationGoal, NavCommand


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def wrap_deg(angle: float) -> float:
    value = (float(angle) + 180.0) % 360.0 - 180.0
    return 180.0 if value == -180.0 else value


def distance_2d(a_x: float, a_y: float, b_x: float, b_y: float) -> float:
    return math.hypot(float(b_x) - float(a_x), float(b_y) - float(a_y))


def heading_deg(a_x: float, a_y: float, b_x: float, b_y: float) -> float:
    return math.degrees(math.atan2(float(b_y) - float(a_y), float(b_x) - float(a_x)))


@dataclass(frozen=True)
class RouteHint:
    x_m: float
    y_m: float
    command: str = "straight"
    speed_limit_kmh: Optional[float] = None
    hold_sec: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "RouteHint":
        d = dict(data or {})
        return cls(
            x_m=float(d.get("x_m", d.get("x", 0.0))),
            y_m=float(d.get("y_m", d.get("y", 0.0))),
            command=str(d.get("command", d.get("action", "straight"))),
            speed_limit_kmh=None if d.get("speed_limit_kmh", d.get("speed_kmh")) is None else float(d.get("speed_limit_kmh", d.get("speed_kmh"))),
            hold_sec=float(d.get("hold_sec", 0.0)),
        )


@dataclass(frozen=True)
class ABRouteRequest:
    mission_id: str
    name: str
    start_x_m: float
    start_y_m: float
    goal_x_m: float
    goal_y_m: float
    start_yaw_deg: Optional[float] = None
    goal_yaw_deg: Optional[float] = None
    speed_cap_kmh: float = 3.0
    spacing_m: float = 2.0
    turn_speed_kmh: float = 1.2
    stop_at_goal: bool = True
    hints: Tuple[RouteHint, ...] = ()

    @classmethod
    def from_dict(cls, data: dict) -> "ABRouteRequest":
        d = dict(data or {})
        start = d.get("start", {}) or {}
        goal = d.get("goal", {}) or {}
        hints = tuple(RouteHint.from_dict(item) for item in d.get("hints", d.get("via", [])) or [])
        return cls(
            mission_id=str(d.get("mission_id") or d.get("id") or "ab_route"),
            name=str(d.get("name") or d.get("mission_id") or "A to B route"),
            start_x_m=float(start.get("x_m", start.get("x", d.get("start_x_m", d.get("start_x", 0.0))))),
            start_y_m=float(start.get("y_m", start.get("y", d.get("start_y_m", d.get("start_y", 0.0))))),
            goal_x_m=float(goal.get("x_m", goal.get("x", d.get("goal_x_m", d.get("goal_x", 0.0))))),
            goal_y_m=float(goal.get("y_m", goal.get("y", d.get("goal_y_m", d.get("goal_y", 0.0))))),
            start_yaw_deg=None if start.get("yaw_deg", d.get("start_yaw_deg")) is None else float(start.get("yaw_deg", d.get("start_yaw_deg"))),
            goal_yaw_deg=None if goal.get("yaw_deg", d.get("goal_yaw_deg")) is None else float(goal.get("yaw_deg", d.get("goal_yaw_deg"))),
            speed_cap_kmh=float(d.get("speed_cap_kmh", 3.0)),
            spacing_m=float(d.get("spacing_m", 2.0)),
            turn_speed_kmh=float(d.get("turn_speed_kmh", 1.2)),
            stop_at_goal=bool(d.get("stop_at_goal", True)),
            hints=hints,
        )


@dataclass
class RoutePlannerConfig:
    default_spacing_m: float = 2.0
    min_spacing_m: float = 0.5
    turn_angle_threshold_deg: float = 25.0
    stop_slowdown_distance_m: float = 4.0


class CoordinateRoutePlanner:
    """Builds a waypoint mission from two coordinates and optional route hints.

    Without a road graph the planner cannot infer real intersections. It builds a
    safe local-meters polyline: A -> optional hints/turn/intersection points -> B,
    then densifies it every N meters and annotates turns by heading changes.
    """

    def __init__(self, config: Optional[RoutePlannerConfig] = None):
        self.config = config or RoutePlannerConfig()

    def build_from_ab(self, request: ABRouteRequest) -> Mission:
        spacing = max(self.config.min_spacing_m, float(request.spacing_m or self.config.default_spacing_m))
        # The START point is a marker, not a zero-speed segment.
        # Keep its speed unset so the first drivable segment inherits
        # request.speed_cap_kmh. Otherwise the first lookahead target gets
        # speed_limit_kmh=0 and the mock/real controller sits at speed=0.
        key_points: List[Tuple[float, float, str, Optional[float], float]] = [
            (float(request.start_x_m), float(request.start_y_m), NavCommand.START.value, None, 0.0)
        ]
        for hint in request.hints:
            cmd = NavCommand.normalize(hint.command, NavCommand.STRAIGHT).value
            key_points.append((hint.x_m, hint.y_m, cmd, hint.speed_limit_kmh, hint.hold_sec))
        key_points.append((float(request.goal_x_m), float(request.goal_y_m), NavCommand.GOAL.value, 0.0 if request.stop_at_goal else None, 0.0))

        waypoints: List[Waypoint] = []
        for seg_idx in range(max(0, len(key_points) - 1)):
            x0, y0, cmd0, sp0, hold0 = key_points[seg_idx]
            x1, y1, cmd1, sp1, hold1 = key_points[seg_idx + 1]
            seg_len = distance_2d(x0, y0, x1, y1)
            seg_heading = heading_deg(x0, y0, x1, y1) if seg_len > 1e-6 else 0.0
            samples = max(1, int(math.ceil(seg_len / spacing)))
            for i in range(samples):
                if seg_idx > 0 and i == 0:
                    continue
                t = float(i) / float(samples)
                x = x0 + (x1 - x0) * t
                y = y0 + (y1 - y0) * t
                command = cmd0 if i == 0 else NavCommand.STRAIGHT.value
                speed = request.speed_cap_kmh if sp0 is None else sp0
                if command in (NavCommand.TURN_LEFT.value, NavCommand.TURN_RIGHT.value, NavCommand.INTERSECTION.value):
                    speed = min(speed, request.turn_speed_kmh)
                waypoints.append(Waypoint(x, y, seg_heading, speed, command, command, hold0))

            final_speed = request.speed_cap_kmh if sp1 is None else sp1
            final_cmd = cmd1
            if final_cmd == NavCommand.GOAL.value and request.stop_at_goal:
                final_cmd = NavCommand.STOP.value
            if final_cmd in (NavCommand.TURN_LEFT.value, NavCommand.TURN_RIGHT.value, NavCommand.INTERSECTION.value):
                final_speed = min(final_speed, request.turn_speed_kmh)
            waypoints.append(Waypoint(x1, y1, seg_heading, final_speed, final_cmd, final_cmd, hold1))

        waypoints = self._annotate_implicit_turns(waypoints, request.turn_speed_kmh)
        return Mission(
            mission_id=request.mission_id,
            name=request.name,
            goal_label=f"B=({request.goal_x_m:.1f}, {request.goal_y_m:.1f})",
            speed_cap_kmh=float(request.speed_cap_kmh),
            waypoints=tuple(waypoints),
            metadata={
                "source": "ab_route_planner",
                "coordinate_frame": "local_meters",
                "spacing_m": spacing,
                "start": {"x_m": request.start_x_m, "y_m": request.start_y_m, "yaw_deg": request.start_yaw_deg},
                "goal": {"x_m": request.goal_x_m, "y_m": request.goal_y_m, "yaw_deg": request.goal_yaw_deg},
            },
        )

    def _annotate_implicit_turns(self, waypoints: Sequence[Waypoint], turn_speed_kmh: float) -> List[Waypoint]:
        if len(waypoints) < 3:
            return list(waypoints)
        out: List[Waypoint] = [waypoints[0]]
        threshold = float(self.config.turn_angle_threshold_deg)
        for idx in range(1, len(waypoints) - 1):
            prev_wp = out[-1]
            wp = waypoints[idx]
            next_wp = waypoints[idx + 1]
            incoming = heading_deg(prev_wp.x_m, prev_wp.y_m, wp.x_m, wp.y_m)
            outgoing = heading_deg(wp.x_m, wp.y_m, next_wp.x_m, next_wp.y_m)
            delta = wrap_deg(outgoing - incoming)
            command = wp.command
            speed = wp.speed_limit_kmh
            if NavCommand.normalize(command) in (NavCommand.STRAIGHT, NavCommand.LANE_FOLLOW) and abs(delta) >= threshold:
                command = NavCommand.TURN_LEFT.value if delta > 0 else NavCommand.TURN_RIGHT.value
                speed = min(speed, float(turn_speed_kmh))
            yaw = outgoing if abs(delta) >= threshold else wp.yaw_deg
            out.append(Waypoint(wp.x_m, wp.y_m, yaw, speed, command, command, wp.hold_sec, wp.metadata))
        out.append(waypoints[-1])
        return out


@dataclass
class NavigatorService:
    lookahead_m: float = 3.0
    waypoint_reached_m: float = 0.8
    goal_reached_m: float = 0.8
    stale_pose_ms: float = 500.0
    _active_index: int = 0
    _last_goal: LocalNavigationGoal = field(default_factory=LocalNavigationGoal)

    def reset(self) -> None:
        self._active_index = 0
        self._last_goal = LocalNavigationGoal()

    def update(self, mission: Optional[Mission], pose: Pose2D, speed_kmh: float = 0.0) -> LocalNavigationGoal:
        if mission is None or not mission.waypoints:
            return LocalNavigationGoal(maneuver="no_mission", valid_until_monotonic=time.monotonic() + 0.1)
        if not pose.valid or pose.age_ms() > self.stale_pose_ms:
            return LocalNavigationGoal(mission_id=mission.mission_id, maneuver="pose_lost", valid_until_monotonic=time.monotonic() + 0.1)

        wps = list(mission.waypoints)
        self._active_index = max(0, min(self._active_index, len(wps) - 1))

        while self._active_index < len(wps) - 1:
            current = wps[self._active_index]
            if distance_2d(pose.x_m, pose.y_m, current.x_m, current.y_m) <= self.waypoint_reached_m:
                self._active_index += 1
            else:
                break

        target_idx = self._pick_lookahead_index(wps, pose)
        target = wps[target_idx]
        prev = wps[max(0, min(target_idx - 1, len(wps) - 1))]
        final = wps[-1]

        target_dist = distance_2d(pose.x_m, pose.y_m, target.x_m, target.y_m)
        goal_dist = distance_2d(pose.x_m, pose.y_m, final.x_m, final.y_m)
        desired_heading = heading_deg(pose.x_m, pose.y_m, target.x_m, target.y_m) if target_dist > 1e-6 else target.yaw_deg
        heading_error = wrap_deg(desired_heading - pose.yaw_deg)
        xtrack = self._cross_track_error(prev, target, pose)
        speed_cap = max(0.0, min(float(mission.speed_cap_kmh), float(target.speed_limit_kmh)))
        stop_required = target.nav_command in (NavCommand.STOP, NavCommand.GOAL) or goal_dist <= self.goal_reached_m
        desired_speed = 0.0 if stop_required else speed_cap
        progress = 100.0 * float(target_idx) / max(1.0, float(len(wps) - 1))

        goal = LocalNavigationGoal(
            mission_id=mission.mission_id,
            route_progress=progress,
            target_x_m=target.x_m,
            target_y_m=target.y_m,
            target_heading_deg=desired_heading,
            distance_to_target_m=target_dist,
            distance_to_goal_m=goal_dist,
            cross_track_error_m=xtrack,
            heading_error_deg=heading_error,
            waypoint_index=target_idx,
            maneuver=target.nav_command.value,
            desired_speed_kmh=desired_speed,
            speed_cap_kmh=speed_cap,
            stop_required=stop_required,
            valid_until_monotonic=time.monotonic() + 0.25,
        )
        self._last_goal = goal
        return goal

    def _pick_lookahead_index(self, wps: Sequence[Waypoint], pose: Pose2D) -> int:
        best_idx = self._active_index
        acc = 0.0
        last_x = pose.x_m
        last_y = pose.y_m
        for idx in range(self._active_index, len(wps)):
            wp = wps[idx]
            step = distance_2d(last_x, last_y, wp.x_m, wp.y_m)
            acc += step
            best_idx = idx
            if acc >= self.lookahead_m:
                break
            last_x, last_y = wp.x_m, wp.y_m
        return best_idx

    @staticmethod
    def _cross_track_error(a: Waypoint, b: Waypoint, pose: Pose2D) -> float:
        ax, ay = a.x_m, a.y_m
        bx, by = b.x_m, b.y_m
        px, py = pose.x_m, pose.y_m
        dx = bx - ax
        dy = by - ay
        denom = math.hypot(dx, dy)
        if denom < 1e-6:
            return distance_2d(px, py, bx, by)
        return ((px - ax) * dy - (py - ay) * dx) / denom
