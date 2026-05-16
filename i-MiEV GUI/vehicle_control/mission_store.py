from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .models import Mission, Waypoint


class JsonMissionStore:
    def __init__(self, path):
        self.path = Path(path)

    def load_all(self) -> List[Mission]:
        if not self.path.exists():
            return [Mission.default_test_mission()]
        with open(self.path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        missions = []
        for item in payload.get("missions", []):
            waypoints = tuple(Waypoint(**wp) for wp in item.get("waypoints", []))
            missions.append(Mission(
                mission_id=str(item.get("mission_id") or item.get("id") or item.get("name")),
                name=str(item.get("name") or item.get("mission_id") or "Mission"),
                goal_label=str(item.get("goal_label") or item.get("goal") or ""),
                speed_cap_kmh=float(item.get("speed_cap_kmh", 3.0)),
                waypoints=waypoints,
                metadata=dict(item.get("metadata") or {}),
            ))
        return missions or [Mission.default_test_mission()]

    def save_all(self, missions: List[Mission]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"missions": []}
        for mission in missions:
            payload["missions"].append({
                "mission_id": mission.mission_id,
                "name": mission.name,
                "goal_label": mission.goal_label,
                "speed_cap_kmh": mission.speed_cap_kmh,
                "waypoints": [wp.__dict__ for wp in mission.waypoints],
                "metadata": mission.metadata,
            })
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        tmp.replace(self.path)
