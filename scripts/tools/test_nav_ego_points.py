#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke-test navigator prev/target/next -> ego-local model format."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
GUI = ROOT / "i-MiEV GUI"
sys.path.insert(0, str(GUI))
sys.path.insert(0, str(ROOT))

from vehicle_control.models import Mission, Waypoint, Pose2D  # noqa: E402
from vehicle_control.navigation import NavigatorService  # noqa: E402
from real_agent_adapters.lead_real_model_0011_adapter import RealModel0011InputBuilder  # noqa: E402


def assert_close_pair(name, value, expected, eps=1e-4):
    got = [float(value[0]), float(value[1])]
    exp = [float(expected[0]), float(expected[1])]
    if abs(got[0] - exp[0]) > eps or abs(got[1] - exp[1]) > eps:
        raise AssertionError(f"{name}: got={got}, expected={exp}")


def main() -> int:
    mission = Mission(
        mission_id="nav_ego_test",
        name="Navigator ego-local test",
        speed_cap_kmh=5.0,
        waypoints=(
            Waypoint(0.0, 0.0, 0.0, 5.0, "start", "start"),
            Waypoint(5.0, 0.0, 0.0, 5.0, "lane_follow", "straight"),
            Waypoint(10.0, 0.0, 0.0, 5.0, "lane_follow", "straight"),
            Waypoint(15.0, 0.0, 0.0, 5.0, "lane_follow", "straight"),
        ),
    )
    nav = NavigatorService(lookahead_m=9.0, waypoint_reached_m=0.1, stale_pose_ms=5000.0)
    pose = Pose2D(x_m=0.0, y_m=0.0, yaw_deg=0.0, valid=True, timestamp_monotonic=time.monotonic())
    goal = nav.update(mission, pose, speed_kmh=0.0)

    for attr in ("previous_x_m", "previous_y_m", "next_x_m", "next_y_m"):
        if not hasattr(goal, attr):
            raise AssertionError(f"LocalNavigationGoal missing {attr}")

    builder = RealModel0011InputBuilder()
    telemetry = SimpleNamespace(x_m=0.0, y_m=0.0, yaw_deg=0.0)
    previous, target, next_point = builder._target_points({"telemetry": telemetry, "goal": goal})

    print("goal:")
    print("  previous_world:", goal.previous_x_m, goal.previous_y_m)
    print("  target_world:  ", goal.target_x_m, goal.target_y_m)
    print("  next_world:    ", goal.next_x_m, goal.next_y_m)
    print("model points:")
    print("  previous:", previous.tolist())
    print("  target:  ", target.tolist())
    print("  next:    ", next_point.tolist())

    if not (previous[0] <= target[0] <= next_point[0]):
        raise AssertionError("Expected previous.x <= target.x <= next.x for straight route")

    fake_goal = SimpleNamespace(
        previous_x_m=0.0,
        previous_y_m=5.0,
        target_x_m=0.0,
        target_y_m=10.0,
        next_x_m=0.0,
        next_y_m=15.0,
    )
    telemetry90 = SimpleNamespace(x_m=0.0, y_m=0.0, yaw_deg=90.0)
    previous90, target90, next90 = builder._target_points({"telemetry": telemetry90, "goal": fake_goal})
    assert_close_pair("yaw90 previous", previous90, [5.0, 0.0])
    assert_close_pair("yaw90 target", target90, [10.0, 0.0])
    assert_close_pair("yaw90 next", next90, [15.0, 0.0])

    previous_e, target_e, next_e = builder._target_points({
        "telemetry": telemetry90,
        "goal": fake_goal,
        "target_point_previous_ego": [1.0, -1.0],
        "target_point_ego": [2.0, -2.0],
        "target_point_next_ego": [3.0, -3.0],
    })
    assert_close_pair("explicit previous", previous_e, [1.0, -1.0])
    assert_close_pair("explicit target", target_e, [2.0, -2.0])
    assert_close_pair("explicit next", next_e, [3.0, -3.0])

    print("OK: navigator prev/target/next are converted to ego-local model points.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
