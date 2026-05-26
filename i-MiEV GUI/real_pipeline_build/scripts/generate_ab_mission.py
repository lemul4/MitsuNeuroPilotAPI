from __future__ import annotations

import argparse
import json
from pathlib import Path

from vehicle_control.navigation import ABRouteRequest, CoordinateRoutePlanner
from vehicle_control.mission_store import JsonMissionStore


def main():
    parser = argparse.ArgumentParser(description="Generate densified A-to-B i-MiEV mission")
    parser.add_argument("--start", required=True, help="start x,y,yaw_deg, e.g. 0,0,0")
    parser.add_argument("--goal", required=True, help="goal x,y,yaw_deg, e.g. 25,8,15")
    parser.add_argument("--speed", type=float, default=3.0)
    parser.add_argument("--spacing", type=float, default=2.0)
    parser.add_argument("--mission-id", default="generated_ab_route")
    parser.add_argument("--name", default="Generated A-B Route")
    parser.add_argument("--output", default="missions/generated_ab_route.json")
    args = parser.parse_args()

    sx, sy, *syaw = [float(v.strip()) for v in args.start.split(",")]
    gx, gy, *gyaw = [float(v.strip()) for v in args.goal.split(",")]
    req = ABRouteRequest(
        mission_id=args.mission_id,
        name=args.name,
        start_x_m=sx,
        start_y_m=sy,
        goal_x_m=gx,
        goal_y_m=gy,
        start_yaw_deg=syaw[0] if syaw else None,
        goal_yaw_deg=gyaw[0] if gyaw else None,
        speed_cap_kmh=args.speed,
        spacing_m=args.spacing,
    )
    mission = CoordinateRoutePlanner().build_from_ab(req)
    out = Path(args.output)
    JsonMissionStore(out).save_all([mission])
    print(f"wrote {out} with {len(mission.waypoints)} waypoints")


if __name__ == "__main__":
    main()
