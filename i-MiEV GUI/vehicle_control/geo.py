from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple

EARTH_RADIUS_M = 6378137.0


@dataclass(frozen=True)
class GeoPoint:
    """WGS84 coordinate used by the map picker.

    The vehicle navigation core still works in a local metric frame. This class
    is only the map-facing representation. Convert to local meters with
    latlon_to_local_m() before building a Mission.
    """

    lat: float
    lon: float
    yaw_deg: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeoPoint":
        d = dict(data or {})
        return cls(
            lat=float(d.get("lat", d.get("latitude", 0.0))),
            lon=float(d.get("lon", d.get("lng", d.get("longitude", 0.0)))),
            yaw_deg=float(d.get("yaw_deg", d.get("yaw", 0.0))),
        )

    def to_dict(self) -> Dict[str, float]:
        return {"lat": float(self.lat), "lon": float(self.lon), "yaw_deg": float(self.yaw_deg)}


@dataclass(frozen=True)
class GeoReference:
    """Local tangent-plane reference for a short A/B vehicle mission.

    For proving-ground and low-speed tests this equirectangular approximation is
    enough and intentionally simple. For long public-road routes replace this
    with a proper projection/road graph in the navigator layer.
    """

    origin: GeoPoint

    @classmethod
    def from_origin(cls, lat: float, lon: float, yaw_deg: float = 0.0) -> "GeoReference":
        return cls(GeoPoint(float(lat), float(lon), float(yaw_deg)))

    def to_dict(self) -> Dict[str, float]:
        return self.origin.to_dict()


def latlon_to_local_m(point: GeoPoint, ref: GeoReference) -> Tuple[float, float]:
    lat0 = math.radians(ref.origin.lat)
    lon0 = math.radians(ref.origin.lon)
    lat = math.radians(point.lat)
    lon = math.radians(point.lon)
    x = (lon - lon0) * math.cos((lat + lat0) * 0.5) * EARTH_RADIUS_M
    y = (lat - lat0) * EARTH_RADIUS_M
    return x, y


def local_m_to_latlon(x_m: float, y_m: float, ref: GeoReference, yaw_deg: float = 0.0) -> GeoPoint:
    lat0 = math.radians(ref.origin.lat)
    lon0 = math.radians(ref.origin.lon)
    lat = lat0 + float(y_m) / EARTH_RADIUS_M
    lon = lon0 + float(x_m) / (EARTH_RADIUS_M * math.cos((lat + lat0) * 0.5))
    return GeoPoint(math.degrees(lat), math.degrees(lon), float(yaw_deg))


def bearing_deg(a: GeoPoint, b: GeoPoint) -> float:
    lat1 = math.radians(a.lat)
    lat2 = math.radians(b.lat)
    dlon = math.radians(b.lon - a.lon)
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def geo_points_to_local_ab(start: GeoPoint, goal: GeoPoint) -> Dict[str, Dict[str, float]]:
    ref = GeoReference(start)
    gx, gy = latlon_to_local_m(goal, ref)
    heading = bearing_deg(start, goal)
    return {
        "origin_geo": ref.to_dict(),
        "start": {"x_m": 0.0, "y_m": 0.0, "yaw_deg": float(start.yaw_deg or heading)},
        "goal": {"x_m": gx, "y_m": gy, "yaw_deg": float(goal.yaw_deg or heading)},
        "start_geo": start.to_dict(),
        "goal_geo": goal.to_dict(),
    }
