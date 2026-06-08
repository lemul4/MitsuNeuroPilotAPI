"""CARLA ego-vehicle sensor profile override for MitsuNeuroPilot.

This module lets the CARLA/Leaderboard SensorAgent keep its default sensor
setup unless an explicit profile is provided.  When the profile is enabled, RGB
camera sensor specs returned by ``av_sensor_setup(...)`` are replaced with the
camera specs from a JSON file, while non-camera sensors (LiDAR, radar, GNSS,
IMU, speedometer) are kept unchanged.

The intended use is to spawn the same dual-front camera layout in CARLA that is
used on the real Mitsubishi i-MiEV test vehicle: wide 90 deg camera + narrow
50 deg camera attached to the ego/hero vehicle.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

PROFILE_ENV = "MITSU_CARLA_SENSOR_PROFILE"
AUTO_ENV = "MITSU_CARLA_SENSOR_PROFILE_AUTO"
DISABLE_ENV = "MITSU_CARLA_SENSOR_PROFILE_DISABLE"
DEFAULT_PROFILE_RELATIVE_PATH = "config/carla_vehicle_sensors.imiev_dual_front_rgb_1_rgb_3.json"

_REQUIRED_CAMERA_FIELDS = ("id", "x", "y", "z", "width", "height", "fov")
_FLOAT_CAMERA_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw", "fov")
_INT_CAMERA_FIELDS = ("width", "height")


def _project_root_from_here() -> Path:
    # lead/common/carla_vehicle_sensor_profile.py -> repository root
    return Path(__file__).resolve().parents[2]


def _resolve_profile_path(profile_path: str | os.PathLike[str]) -> Path:
    path = Path(profile_path)
    if path.is_absolute():
        return path
    candidates = [
        Path.cwd() / path,
        _project_root_from_here() / path,
        _project_root_from_here() / "i-MIEV GUI" / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Return the project-root candidate for a clear error message.
    return candidates[1]


def _env_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_active_profile_path(config: Any | None = None) -> Path | None:
    """Return the active sensor profile path, or None if override is disabled.

    Priority:
      1. MITSU_CARLA_SENSOR_PROFILE_DISABLE=1 disables override explicitly.
      2. Explicit MITSU_CARLA_SENSOR_PROFILE path.
      3. Auto profile for dual_front_camera_mode checkpoints.  Auto mode is ON
         by default for this project so the new dual-front model can run without
         extra PowerShell environment variables.
      4. No override for non-dual-front checkpoints.
    """

    if _env_enabled(DISABLE_ENV, default=False):
        return None

    explicit = os.environ.get(PROFILE_ENV, "").strip()
    if explicit and explicit.lower() not in {"0", "false", "off", "none", "disable", "disabled"}:
        return _resolve_profile_path(explicit)
    if explicit.lower() in {"0", "false", "off", "none", "disable", "disabled"}:
        return None

    if not _env_enabled(AUTO_ENV, default=True):
        return None

    if config is not None and not bool(getattr(config, "dual_front_camera_mode", False)):
        return None

    path = _resolve_profile_path(DEFAULT_PROFILE_RELATIVE_PATH)
    if not path.exists():
        LOG.warning("[CARLA sensor profile] Auto profile is enabled but file is missing: %s", path)
        return None
    return path


def load_sensor_profile(profile_path: str | os.PathLike[str]) -> dict[str, Any]:
    resolved = _resolve_profile_path(profile_path)
    if not resolved.exists():
        raise FileNotFoundError(f"CARLA sensor profile not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as fh:
        profile = json.load(fh)
    validate_sensor_profile(profile, source=str(resolved))
    return profile


def validate_sensor_profile(profile: dict[str, Any], source: str = "<memory>") -> None:
    if not isinstance(profile, dict):
        raise TypeError(f"CARLA sensor profile must be a JSON object: {source}")
    cameras = profile.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError(f"CARLA sensor profile has no cameras[] list: {source}")

    ids: set[str] = set()
    for index, camera in enumerate(cameras):
        if not isinstance(camera, dict):
            raise TypeError(f"camera #{index} is not an object in {source}")
        missing = [field for field in _REQUIRED_CAMERA_FIELDS if field not in camera]
        if missing:
            raise ValueError(f"camera #{index} missing fields {missing} in {source}")
        camera_id = str(camera["id"])
        if camera_id in ids:
            raise ValueError(f"duplicated CARLA camera id '{camera_id}' in {source}")
        ids.add(camera_id)
        if not camera_id.startswith("rgb_"):
            raise ValueError(
                f"camera id '{camera_id}' should use CARLA/LEAD RGB id style, e.g. rgb_1"
            )


def build_camera_sensor_specs(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert profile JSON cameras to Leaderboard sensor specs."""

    validate_sensor_profile(profile)
    specs: list[dict[str, Any]] = []
    for camera in profile["cameras"]:
        spec: dict[str, Any] = {
            "type": str(camera.get("type", "sensor.camera.rgb")),
            "id": str(camera["id"]),
            "role": str(camera.get("role", camera["id"])),
        }
        for field in _FLOAT_CAMERA_FIELDS:
            spec[field] = float(camera.get(field, 0.0))
        for field in _INT_CAMERA_FIELDS:
            spec[field] = int(camera[field])
        # Preserve optional CARLA sensor attributes if the profile defines them.
        for optional_key in ("sensor_tick", "lens_circle_falloff", "lens_circle_multiplier"):
            if optional_key in camera:
                spec[optional_key] = camera[optional_key]
        specs.append(spec)
    return specs


def is_rgb_camera_sensor(sensor_spec: dict[str, Any]) -> bool:
    return str(sensor_spec.get("type", "")).strip().lower() == "sensor.camera.rgb"


def apply_camera_sensor_profile(
    sensors: list[dict[str, Any]],
    config: Any | None = None,
    profile_path: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Replace CARLA RGB camera specs with an active JSON sensor profile.

    If no profile is active, the original sensor list is returned unchanged.
    """

    active_profile_path = Path(profile_path) if profile_path is not None else get_active_profile_path(config)
    if active_profile_path is None:
        return sensors

    profile = load_sensor_profile(active_profile_path)
    profile_cameras = build_camera_sensor_specs(profile)

    keep_non_camera_sensors = bool(profile.get("keep_non_camera_sensors", True))
    if keep_non_camera_sensors:
        non_camera_sensors = [sensor for sensor in sensors if not is_rgb_camera_sensor(sensor)]
    else:
        non_camera_sensors = []

    result = profile_cameras + non_camera_sensors
    LOG.info(
        "[CARLA sensor profile] Using %s: %d RGB camera(s), %d non-camera sensor(s)",
        active_profile_path,
        len(profile_cameras),
        len(non_camera_sensors),
    )
    return result
