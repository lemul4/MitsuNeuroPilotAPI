from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from .geo import GeoPoint, GeoReference, latlon_to_local_m
from .models import Mission, NavCommand, Waypoint
from .navigation import distance_2d, heading_deg, wrap_deg


@dataclass(frozen=True)
class RoadRouteRequest:
    mission_id: str
    name: str
    start: GeoPoint
    goal: GeoPoint
    speed_cap_kmh: float = 3.0
    spacing_m: float = 2.0
    turn_speed_kmh: float = 1.2
    provider: str = "osrm"
    osrm_base_url: str = "https://router.project-osrm.org"
    timeout_sec: float = 4.0

    @classmethod
    def from_mission_dict(cls, data: dict) -> "RoadRouteRequest":
        d = dict(data or {})
        metadata = dict(d.get("metadata") or {})
        start_geo = metadata.get("start_geo") or d.get("start_geo")
        goal_geo = metadata.get("goal_geo") or d.get("goal_geo")
        if not start_geo or not goal_geo:
            raise ValueError("Road route requires start_geo and goal_geo metadata")
        return cls(
            mission_id=str(d.get("mission_id") or "road_route"),
            name=str(d.get("name") or "Road route"),
            start=GeoPoint.from_dict(start_geo),
            goal=GeoPoint.from_dict(goal_geo),
            speed_cap_kmh=float(d.get("speed_cap_kmh", 3.0)),
            spacing_m=float(d.get("spacing_m", 2.0)),
            turn_speed_kmh=float(d.get("turn_speed_kmh", min(1.2, float(d.get("speed_cap_kmh", 3.0))))),
            provider=str(metadata.get("routing_provider") or d.get("routing_provider") or "osrm"),
            osrm_base_url=str(metadata.get("osrm_base_url") or d.get("osrm_base_url") or "https://router.project-osrm.org"),
            timeout_sec=float(metadata.get("routing_timeout_sec", d.get("routing_timeout_sec", 4.0))),
        )


class OsrmRoadRouteProvider:
    """Builds a road-following mission from A/B coordinates via OSRM.

    The provider requests GeoJSON geometry so no polyline dependency is needed.
    It is intended for planning before arming, not for high-rate control.
    """

    def build_mission(self, request: RoadRouteRequest) -> Mission:
        if request.provider.lower() not in {"osrm", "openstreetmap", "osm"}:
            raise ValueError(f"Unsupported road routing provider: {request.provider}")
        coordinates = self._fetch_osrm_geojson(request)
        if len(coordinates) < 2:
            raise RuntimeError("OSRM returned an empty geometry")
        local_points = self._to_local_points(request.start, coordinates)
        waypoints = self._densify_and_annotate(
            local_points,
            spacing_m=max(0.5, float(request.spacing_m)),
            speed_cap_kmh=max(0.0, float(request.speed_cap_kmh)),
            turn_speed_kmh=max(0.0, float(request.turn_speed_kmh)),
        )
        if waypoints:
            last = waypoints[-1]
            waypoints[-1] = Waypoint(
                last.x_m,
                last.y_m,
                last.yaw_deg,
                0.0,
                NavCommand.STOP.value,
                NavCommand.STOP.value,
                last.hold_sec,
                dict(last.metadata or {}),
            )
        return Mission(
            mission_id=request.mission_id,
            name=request.name,
            goal_label=f"Дорожный маршрут B=({request.goal.lat:.6f}, {request.goal.lon:.6f})",
            speed_cap_kmh=float(request.speed_cap_kmh),
            waypoints=tuple(waypoints),
            metadata={
                "source": "road_route_provider",
                "routing_provider": "osrm",
                "coordinate_frame": "wgs84_local_tangent",
                "origin_geo": {"lat": request.start.lat, "lon": request.start.lon},
                "start_geo": {"lat": request.start.lat, "lon": request.start.lon},
                "goal_geo": {"lat": request.goal.lat, "lon": request.goal.lon},
                "raw_route_points": len(coordinates),
            },
        )

    def _fetch_osrm_geojson(self, request: RoadRouteRequest) -> List[Tuple[float, float]]:
        base = request.osrm_base_url.rstrip("/")
        coords = f"{request.start.lon:.8f},{request.start.lat:.8f};{request.goal.lon:.8f},{request.goal.lat:.8f}"
        query = urllib.parse.urlencode({
            "overview": "full",
            "geometries": "geojson",
            "steps": "false",
            "alternatives": "false",
        })
        url = f"{base}/route/v1/driving/{coords}?{query}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "MitsuNeuroPilot/real-vehicle-test (route planning)",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=float(request.timeout_sec)) as response:
            payload = json.loads(response.read().decode("utf-8"))
        routes = payload.get("routes") or []
        if not routes:
            message = payload.get("message") or payload.get("code") or "no routes"
            raise RuntimeError(f"OSRM route failed: {message}")
        geometry = routes[0].get("geometry") or {}
        coords_lon_lat = geometry.get("coordinates") or []
        out: List[Tuple[float, float]] = []
        for item in coords_lon_lat:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                lon = float(item[0])
                lat = float(item[1])
                out.append((lat, lon))
        return out

    def _to_local_points(self, origin: GeoPoint, coordinates_lat_lon: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
        points: List[Tuple[float, float]] = []
        for lat, lon in coordinates_lat_lon:
            x_m, y_m = latlon_to_local_m(GeoPoint(float(lat), float(lon)), GeoReference(origin))
            points.append((float(x_m), float(y_m)))
        return points

    def _densify_and_annotate(
        self,
        points: Sequence[Tuple[float, float]],
        spacing_m: float,
        speed_cap_kmh: float,
        turn_speed_kmh: float,
    ) -> List[Waypoint]:
        if not points:
            return []
        dense: List[Tuple[float, float]] = [points[0]]
        for a, b in zip(points, points[1:]):
            x0, y0 = a
            x1, y1 = b
            seg_len = distance_2d(x0, y0, x1, y1)
            if seg_len <= 1e-6:
                continue
            samples = max(1, int(math.ceil(seg_len / spacing_m)))
            for i in range(1, samples + 1):
                t = i / float(samples)
                dense.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))

        waypoints: List[Waypoint] = []
        for idx, (x, y) in enumerate(dense):
            if idx < len(dense) - 1:
                yaw = heading_deg(x, y, dense[idx + 1][0], dense[idx + 1][1])
            elif idx > 0:
                yaw = heading_deg(dense[idx - 1][0], dense[idx - 1][1], x, y)
            else:
                yaw = 0.0
            command = NavCommand.STRAIGHT.value if idx > 0 else NavCommand.START.value
            speed = float(speed_cap_kmh)
            if 0 < idx < len(dense) - 1:
                incoming = heading_deg(dense[idx - 1][0], dense[idx - 1][1], x, y)
                outgoing = heading_deg(x, y, dense[idx + 1][0], dense[idx + 1][1])
                delta = wrap_deg(outgoing - incoming)
                if abs(delta) >= 28.0:
                    command = NavCommand.TURN_LEFT.value if delta > 0 else NavCommand.TURN_RIGHT.value
                    speed = min(speed, float(turn_speed_kmh))
            waypoints.append(Waypoint(x, y, yaw, speed, command, command, 0.0, {"route_source": "road"}))
        return waypoints
