from __future__ import annotations

import json
from pathlib import Path
from typing import List, Iterable, Optional
from dataclasses import replace

from .models import Mission, Waypoint
from .navigation import ABRouteRequest, CoordinateRoutePlanner


class JsonMissionStore:
    """Loads/saves real vehicle coordinate missions.

    Supported formats:
    1. {"missions": [{"waypoints": [...]}]}
    2. {"missions": [{"start": {...}, "goal": {...}, "hints": [...]}]}
    3. a single mission object with either waypoints or start/goal.
    """

    def __init__(self, path, planner: Optional[CoordinateRoutePlanner] = None):
        self.path = Path(path)
        self.planner = planner or CoordinateRoutePlanner()

    def load_all(self) -> List[Mission]:
        if not self.path.exists():
            return [Mission.default_test_mission()]
        with open(self.path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        items = payload.get("missions") if isinstance(payload, dict) else None
        if items is None:
            items = [payload]
        missions = [self._mission_from_dict(item) for item in items if isinstance(item, dict)]
        return [m for m in missions if m is not None] or [Mission.default_test_mission()]

    def load_one(self, mission_id: str) -> Mission:
        for mission in self.load_all():
            if mission.mission_id == mission_id or mission.name == mission_id:
                return mission
        raise KeyError(f"Mission not found: {mission_id}")

    def _mission_from_dict(self, item: dict) -> Optional[Mission]:
        if "start" in item and "goal" in item and not item.get("waypoints"):
            mission = self.planner.build_from_ab(ABRouteRequest.from_dict(item))
            metadata = dict(mission.metadata or {})
            metadata.update(dict(item.get("metadata") or {}))
            return replace(mission, metadata=metadata)

        default_speed = float(item.get("speed_cap_kmh", 3.0))
        waypoints = tuple(Waypoint.from_dict(wp, default_speed) for wp in item.get("waypoints", item.get("points", [])) or [])
        if not waypoints and "start" in item and "goal" in item:
            mission = self.planner.build_from_ab(ABRouteRequest.from_dict(item))
            metadata = dict(mission.metadata or {})
            metadata.update(dict(item.get("metadata") or {}))
            return replace(mission, metadata=metadata)
        return Mission(
            mission_id=str(item.get("mission_id") or item.get("id") or item.get("name") or "mission"),
            name=str(item.get("name") or item.get("mission_id") or "Mission"),
            goal_label=str(item.get("goal_label") or item.get("goal") or ""),
            speed_cap_kmh=default_speed,
            waypoints=waypoints,
            metadata=dict(item.get("metadata") or {}),
        )

    def save_all(self, missions: Iterable[Mission]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"missions": [self._mission_to_dict(mission) for mission in missions]}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        tmp.replace(self.path)

    @staticmethod
    def _mission_to_dict(mission: Mission) -> dict:
        return {
            "mission_id": mission.mission_id,
            "name": mission.name,
            "goal_label": mission.goal_label,
            "speed_cap_kmh": mission.speed_cap_kmh,
            "waypoints": [
                {
                    "x_m": wp.x_m,
                    "y_m": wp.y_m,
                    "yaw_deg": wp.yaw_deg,
                    "speed_limit_kmh": wp.speed_limit_kmh,
                    "action": wp.action,
                    "command": wp.command,
                    "hold_sec": wp.hold_sec,
                    "metadata": wp.metadata,
                }
                for wp in mission.waypoints
            ],
            "metadata": mission.metadata,
        }
