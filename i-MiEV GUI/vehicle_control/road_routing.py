from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from .geo import GeoPoint, GeoReference, latlon_to_local_m, local_m_to_latlon
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
    lane_policy: str = "right_side"
    lane_offset_m: float = 1.7
    traffic_side: str = "right"

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
            lane_policy=str(metadata.get("lane_policy", d.get("lane_policy", "right_side"))),
            lane_offset_m=float(metadata.get("lane_offset_m", d.get("lane_offset_m", 1.7))),
            traffic_side=str(metadata.get("traffic_side", d.get("traffic_side", "right"))),
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
        centerline_points = self._to_local_points(request.start, coordinates)
        local_points = self._apply_lane_policy(centerline_points, request)
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
                "lane_policy": request.lane_policy,
                "lane_offset_m": float(request.lane_offset_m),
                "traffic_side": request.traffic_side,
                "trajectory_geometry": "centerline" if self._is_centerline_policy(request) else "right_side_offset_approximation",
            },
        )


    @staticmethod
    def _is_centerline_policy(request: RoadRouteRequest) -> bool:
        return str(request.lane_policy or "").lower() in {"center", "centerline", "osm_centerline", "none", "0"} or abs(float(request.lane_offset_m or 0.0)) < 1e-6

    def _apply_lane_policy(self, points: Sequence[Tuple[float, float]], request: RoadRouteRequest) -> List[Tuple[float, float]]:
        """Shift OSRM/OSM centerline to the drivable side for the travel direction.

        OSRM returns a road centerline, not lane geometry. For low-speed proving
        ground tests we can create a deterministic control trajectory offset to
        the right-hand side of travel. This is an approximation, not an HD-map
        lane boundary. Use lane_offset_m=0 or lane_policy=centerline to keep the
        original OSRM geometry.
        """
        pts = [(float(x), float(y)) for x, y in (points or [])]
        if len(pts) < 2 or self._is_centerline_policy(request):
            return pts
        offset_m = max(0.0, min(4.0, abs(float(request.lane_offset_m or 0.0))))
        if offset_m <= 0.0:
            return pts
        traffic_side = str(request.traffic_side or "right").lower()
        side_sign = 1.0 if traffic_side not in {"left", "lhs", "left_hand"} else -1.0
        normals: List[Tuple[float, float]] = []
        for idx, (x, y) in enumerate(pts):
            candidates: List[Tuple[float, float]] = []
            if idx > 0:
                px, py = pts[idx - 1]
                dx, dy = x - px, y - py
                length = math.hypot(dx, dy)
                if length > 1e-6:
                    # Right-of-travel normal in local ENU: (dy, -dx).
                    candidates.append((dy / length * side_sign, -dx / length * side_sign))
            if idx < len(pts) - 1:
                nx, ny = pts[idx + 1]
                dx, dy = nx - x, ny - y
                length = math.hypot(dx, dy)
                if length > 1e-6:
                    candidates.append((dy / length * side_sign, -dx / length * side_sign))
            if not candidates:
                normals.append((0.0, 0.0))
                continue
            sx = sum(n[0] for n in candidates)
            sy = sum(n[1] for n in candidates)
            length = math.hypot(sx, sy)
            if length <= 1e-6:
                normals.append(candidates[-1])
            else:
                normals.append((sx / length, sy / length))
        return [(x + nx * offset_m, y + ny * offset_m) for (x, y), (nx, ny) in zip(pts, normals)]

    def route_preview_latlon(self, request: RoadRouteRequest) -> dict:
        """Return centerline and control-trajectory geometry for UI/debug tools."""
        coordinates = self._fetch_osrm_geojson(request)
        center_local = self._to_local_points(request.start, coordinates)
        control_local = self._apply_lane_policy(center_local, request)
        ref = GeoReference(request.start)
        center_geo = [local_m_to_latlon(x, y, ref).to_dict() for x, y in center_local]
        control_geo = [local_m_to_latlon(x, y, ref).to_dict() for x, y in control_local]
        return {
            "centerline": center_geo,
            "control_trajectory": control_geo,
            "lane_policy": request.lane_policy,
            "lane_offset_m": float(request.lane_offset_m),
            "traffic_side": request.traffic_side,
        }

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
            waypoints.append(Waypoint(x, y, yaw, speed, command, command, 0.0, {"route_source": "road", "trajectory": "control_side"}))
        return waypoints
