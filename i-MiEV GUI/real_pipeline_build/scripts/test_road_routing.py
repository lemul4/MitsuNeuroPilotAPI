from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from scripts/ without installing package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vehicle_control.geo import GeoPoint
from vehicle_control.road_routing import RoadRouteRequest, OsrmRoadRouteProvider


def _wp_dict(wp):
    return {
        "x_m": wp.x_m,
        "y_m": wp.y_m,
        "yaw_deg": wp.yaw_deg,
        "speed_limit_kmh": wp.speed_limit_kmh,
        "action": wp.action,
        "command": wp.command,
        "hold_sec": wp.hold_sec,
        "metadata": dict(wp.metadata or {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка road-routing OSRM для маршрута A -> B.")
    parser.add_argument("--a-lat", type=float, required=True)
    parser.add_argument("--a-lon", type=float, required=True)
    parser.add_argument("--b-lat", type=float, required=True)
    parser.add_argument("--b-lon", type=float, required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--spacing", type=float, default=2.0)
    parser.add_argument("--osrm", default="https://router.project-osrm.org")
    parser.add_argument("--out", default="outputs/road_route_debug.json")
    args = parser.parse_args()

    request = RoadRouteRequest(
        mission_id="road_debug",
        name="Road routing debug",
        start=GeoPoint(args.a_lat, args.a_lon),
        goal=GeoPoint(args.b_lat, args.b_lon),
        speed_cap_kmh=args.speed,
        spacing_m=args.spacing,
        osrm_base_url=args.osrm,
    )
    mission = OsrmRoadRouteProvider().build_mission(request)
    turns = [wp for wp in mission.waypoints if "turn" in str(wp.command)]
    payload = {
        "mission_id": mission.mission_id,
        "name": mission.name,
        "goal_label": mission.goal_label,
        "metadata": mission.metadata,
        "waypoint_count": len(mission.waypoints),
        "turn_count": len(turns),
        "first_waypoints": [_wp_dict(wp) for wp in mission.waypoints[:8]],
        "last_waypoints": [_wp_dict(wp) for wp in mission.waypoints[-8:]],
        "turn_waypoints": [_wp_dict(wp) for wp in turns[:20]],
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("OSRM road-routing OK")
    print(f"raw_route_points: {mission.metadata.get('raw_route_points')}")
    print(f"waypoints:        {len(mission.waypoints)}")
    print(f"turn waypoints:   {len(turns)}")
    print(f"debug file:       {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
