"""Dual-front CARLA camera preview service for MitsuNeuroPilot.

This service is intentionally separate from the model sensor path.  The model
sensors are spawned by Leaderboard from SensorAgent.sensors().  This service is
only an operator preview channel for the GUI.

Default behavior:
  1. Attach to the current ego/hero vehicle.
  2. Reuse the model RGB cameras already attached to that vehicle when possible:
       wide 90 deg and narrow 50.03 deg.
  3. If those cameras are not present, optionally spawn preview-only cameras.
  4. Publish a two-camera JPEG mosaic to ZMQ tcp://*:5555.

Environment variables:
  MITSU_CARLA_HOST                  default 127.0.0.1
  MITSU_CARLA_PORT                  default 2000
  MITSU_CARLA_ZMQ_BIND              default tcp://*:5555
  MITSU_CARLA_PREVIEW_FPS           default 20
  MITSU_CARLA_PREVIEW_JPEG_QUALITY  default 85
  MITSU_CARLA_PREVIEW_WIDTH         default 800
  MITSU_CARLA_PREVIEW_HEIGHT        default 450
  MITSU_CARLA_PREVIEW_USE_EXISTING  default 1
  MITSU_CARLA_PREVIEW_SPAWN_IF_MISSING default 1
  MITSU_CARLA_PREVIEW_WAIT_TIMEOUT_S default 0 (0 = wait forever)
  MITSU_CARLA_PREVIEW_PREFER_EXISTING default 0 (0 = spawn GUI copies with same 90/50 layout)
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import zmq

try:
    import carla
except Exception as exc:  # pragma: no cover - depends on CARLA runtime
    carla = None
    _CARLA_IMPORT_ERROR = exc
else:
    _CARLA_IMPORT_ERROR = None


@dataclass(frozen=True)
class CameraProfile:
    role: str
    fov: float
    x: float = 0.90
    y: float = 0.0
    z: float = 1.55
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


WIDE_90 = CameraProfile(role="gui_wide_90", fov=90.0, y=-0.0675)
NARROW_50 = CameraProfile(role="gui_narrow_50", fov=50.03, y=0.0675)


class LatestFrameBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: dict[str, np.ndarray] = {}
        self._timestamps: dict[str, float] = {}

    def put(self, key: str, frame: np.ndarray) -> None:
        with self._lock:
            self._frames[key] = frame
            self._timestamps[key] = time.monotonic()

    def get_pair(self) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, float]]:
        with self._lock:
            return (
                self._frames.get("wide_90"),
                self._frames.get("narrow_50"),
                dict(self._timestamps),
            )


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def _actor_role(actor: Any) -> str:
    try:
        return str(actor.attributes.get("role_name", ""))
    except Exception:
        return ""


def _actor_fov(actor: Any) -> float | None:
    try:
        return float(actor.attributes.get("fov", "nan"))
    except Exception:
        return None


def _find_hero_vehicle(client: Any, timeout_s: float = 0.0) -> tuple[Any, Any]:
    """Wait for CARLA ego/hero vehicle.

    timeout_s <= 0 means wait indefinitely.  The service is often started before
    Leaderboard finishes spawning the scenario actor, and CARLA may reload the
    world while the route is being prepared.  Therefore this function refreshes
    client.get_world() on every iteration and does not abort by default.
    """

    deadline = None if timeout_s <= 0 else time.monotonic() + timeout_s
    last_status_at = 0.0

    while deadline is None or time.monotonic() < deadline:
        try:
            world = client.get_world()
            vehicles = list(world.get_actors().filter("vehicle.*"))
        except Exception as exc:
            now = time.monotonic()
            if now - last_status_at > 2.0:
                print(f"[CAMERA_SERVICE] CARLA world/actors еще не готовы: {exc}", flush=True)
                last_status_at = now
            time.sleep(0.5)
            continue

        # Prefer the actor explicitly controlled by Leaderboard/ScenarioRunner.
        for vehicle in vehicles:
            role = _actor_role(vehicle).strip().lower()
            if role in {"hero", "ego", "ego_vehicle"}:
                print(
                    f"[CAMERA_SERVICE] Успешно прицепились к: id={vehicle.id}, "
                    f"type={vehicle.type_id}, role={_actor_role(vehicle)}",
                    flush=True,
                )
                return world, vehicle

        # Fallback: sometimes the controlled vehicle appears before role_name is
        # propagated.  Use a single available vehicle only after it has been
        # visible for several cycles; this keeps the service from attaching to
        # a random traffic actor too early.
        if len(vehicles) == 1:
            vehicle = vehicles[0]
            print(
                f"[CAMERA_SERVICE] hero role еще не найден; временно используем единственный vehicle: "
                f"id={vehicle.id}, type={vehicle.type_id}, role={_actor_role(vehicle)}",
                flush=True,
            )
            return world, vehicle

        now = time.monotonic()
        if now - last_status_at > 2.0:
            if vehicles:
                roles = ", ".join(
                    f"id={v.id}:role={_actor_role(v) or '-'}:type={v.type_id}" for v in vehicles[:8]
                )
                print(f"[CAMERA_SERVICE] Ждем ego/hero vehicle; vehicles={len(vehicles)} [{roles}]", flush=True)
            else:
                print("[CAMERA_SERVICE] Ждем ego/hero vehicle...", flush=True)
            last_status_at = now
        time.sleep(0.5)

    raise RuntimeError("Cannot find CARLA ego/hero vehicle before timeout")


def _attached_camera_actors(world: Any, vehicle: Any) -> list[Any]:
    result = []
    for actor in world.get_actors():
        if not str(actor.type_id).startswith("sensor.camera"):
            continue
        parent = getattr(actor, "parent", None)
        if parent is not None and getattr(parent, "id", None) == vehicle.id:
            result.append(actor)
    return result


def _is_rgb_camera(actor: Any) -> bool:
    return str(actor.type_id) == "sensor.camera.rgb"


def _select_existing_dual_front_cameras(world: Any, vehicle: Any) -> tuple[Any | None, Any | None]:
    cameras = _attached_camera_actors(world, vehicle)
    print(f"[CAMERA_SERVICE] Найдены штатные камеры на ego-авто: {len(cameras)}", flush=True)
    for cam in cameras:
        attrs = {
            key: cam.attributes.get(key)
            for key in ["image_size_x", "image_size_y", "fov", "sensor_tick", "role_name"]
            if key in cam.attributes
        }
        print(
            f"[CAMERA_SERVICE]   existing camera id={cam.id}, type={cam.type_id}, "
            f"role={_actor_role(cam)}, attrs={attrs}",
            flush=True,
        )

    rgb = [cam for cam in cameras if _is_rgb_camera(cam)]
    # Prefer model cameras, not preview gui cameras from older service runs.
    rgb = [cam for cam in rgb if not _actor_role(cam).startswith("gui_")] or rgb

    def nearest(target: float, exclude_id: int | None = None) -> Any | None:
        candidates = []
        for cam in rgb:
            if exclude_id is not None and cam.id == exclude_id:
                continue
            fov = _actor_fov(cam)
            if fov is None:
                continue
            candidates.append((abs(fov - target), cam))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        if candidates[0][0] <= 2.0:
            return candidates[0][1]
        return None

    wide = nearest(90.0)
    narrow = nearest(50.03, exclude_id=getattr(wide, "id", None))
    if wide is not None and narrow is not None:
        print(
            f"[CAMERA_SERVICE] Используем существующие камеры модели: "
            f"wide_90 id={wide.id} fov={_actor_fov(wide)}, "
            f"narrow_50 id={narrow.id} fov={_actor_fov(narrow)}",
            flush=True,
        )
    else:
        print(
            "[CAMERA_SERVICE] Не удалось выбрать обе существующие камеры 90/50 для preview",
            flush=True,
        )
    return wide, narrow


def _spawn_preview_camera(world: Any, vehicle: Any, profile: CameraProfile, width: int, height: int, fps: int) -> Any:
    blueprint_library = world.get_blueprint_library()
    blueprint = blueprint_library.find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", str(width))
    blueprint.set_attribute("image_size_y", str(height))
    blueprint.set_attribute("fov", str(profile.fov))
    blueprint.set_attribute("sensor_tick", str(1.0 / max(fps, 1)))
    blueprint.set_attribute("role_name", profile.role)

    transform = carla.Transform(
        carla.Location(x=profile.x, y=profile.y, z=profile.z),
        carla.Rotation(roll=profile.roll, pitch=profile.pitch, yaw=profile.yaw),
    )
    camera = world.spawn_actor(blueprint, transform, attach_to=vehicle)
    print(
        f"[CAMERA_SERVICE] Preview camera '{profile.role}' запущена: "
        f"id={camera.id}, fov={profile.fov}",
        flush=True,
    )
    return camera


def _carla_image_to_bgr(image: Any) -> np.ndarray:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    # CARLA gives BGRA by default. Drop alpha.
    return array[:, :, :3].copy()


def _listen(camera: Any, key: str, buffer: LatestFrameBuffer) -> None:
    def callback(image: Any) -> None:
        try:
            buffer.put(key, _carla_image_to_bgr(image))
        except Exception as exc:
            print(f"[CAMERA_SERVICE] Ошибка кадра {key}: {exc}", flush=True)

    camera.listen(callback)


def _tile(frame: np.ndarray | None, label: str, width: int, height: int, stale: bool = False) -> np.ndarray:
    if frame is None:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(canvas, "NO FRAME", (28, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 255), 2)
    else:
        canvas = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(canvas, (0, 0), (width, 34), (0, 0, 0), -1)
    color = (0, 220, 255) if not stale else (0, 120, 255)
    cv2.putText(canvas, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2)
    return canvas


def _build_mosaic(wide: np.ndarray | None, narrow: np.ndarray | None, ages: dict[str, float], width: int, height: int) -> np.ndarray:
    now = time.monotonic()
    wide_stale = (now - ages.get("wide_90", 0.0)) > 1.0
    narrow_stale = (now - ages.get("narrow_50", 0.0)) > 1.0
    wide_tile = _tile(wide, "CARLA WIDE 90 deg / real 2.8 mm analog", width, height, wide_stale)
    narrow_tile = _tile(narrow, "CARLA NARROW 50 deg / real 6 mm analog", width, height, narrow_stale)
    return np.concatenate([wide_tile, narrow_tile], axis=1)


class CarlaDualFrontCameraService:
    def __init__(self) -> None:
        if carla is None:  # pragma: no cover - depends on runtime
            raise RuntimeError(f"CARLA Python API import failed: {_CARLA_IMPORT_ERROR}")
        self.host = os.environ.get("MITSU_CARLA_HOST", "127.0.0.1")
        self.port = _env_int("MITSU_CARLA_PORT", 2000)
        self.zmq_bind = os.environ.get("MITSU_CARLA_ZMQ_BIND", "tcp://*:5555")
        self.fps = _env_int("MITSU_CARLA_PREVIEW_FPS", 20)
        self.jpeg_quality = _env_int("MITSU_CARLA_PREVIEW_JPEG_QUALITY", 85)
        self.tile_width = _env_int("MITSU_CARLA_PREVIEW_WIDTH", 800)
        self.tile_height = _env_int("MITSU_CARLA_PREVIEW_HEIGHT", 450)
        self.use_existing = _env_bool("MITSU_CARLA_PREVIEW_USE_EXISTING", True)
        self.prefer_existing = _env_bool("MITSU_CARLA_PREVIEW_PREFER_EXISTING", False)
        self.spawn_if_missing = _env_bool("MITSU_CARLA_PREVIEW_SPAWN_IF_MISSING", True)
        self.wait_timeout_s = _env_float("MITSU_CARLA_PREVIEW_WAIT_TIMEOUT_S", 0.0)
        self._stop = threading.Event()
        self._spawned: list[Any] = []
        self.buffer = LatestFrameBuffer()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> int:
        context = zmq.Context.instance()
        socket = context.socket(zmq.PUB)
        socket.bind(self.zmq_bind)
        print(f"[CAMERA_SERVICE] ZMQ preview bind: {self.zmq_bind}", flush=True)

        client = carla.Client(self.host, self.port)
        client.set_timeout(10.0)
        world, vehicle = _find_hero_vehicle(client, timeout_s=self.wait_timeout_s)

        wide = narrow = None
        existing_wide = existing_narrow = None
        if self.use_existing:
            existing_wide, existing_narrow = _select_existing_dual_front_cameras(world, vehicle)

        # By default we spawn two GUI-only cameras with the same i-MiEV dual-front
        # geometry.  This avoids interfering with model sensors that Leaderboard
        # is already listening to, while the operator preview still matches the
        # real vehicle layout.  Set MITSU_CARLA_PREVIEW_PREFER_EXISTING=1 to reuse
        # the already attached model cameras directly.
        if self.prefer_existing and existing_wide is not None and existing_narrow is not None:
            wide, narrow = existing_wide, existing_narrow
            print("[CAMERA_SERVICE] Preview читает существующие модельные камеры 90/50", flush=True)
        elif self.spawn_if_missing:
            print("[CAMERA_SERVICE] Спавним две GUI-preview камеры с layout i-MiEV: wide_90 + narrow_50", flush=True)
            wide = _spawn_preview_camera(world, vehicle, WIDE_90, self.tile_width, self.tile_height, self.fps)
            narrow = _spawn_preview_camera(world, vehicle, NARROW_50, self.tile_width, self.tile_height, self.fps)
            self._spawned.extend([wide, narrow])
        else:
            wide, narrow = existing_wide, existing_narrow

        if wide is None or narrow is None:
            raise RuntimeError("Cannot start dual-front preview: wide/narrow cameras are missing")

        _listen(wide, "wide_90", self.buffer)
        _listen(narrow, "narrow_50", self.buffer)
        print("[CAMERA_SERVICE] Двухкамерный поток видео запущен: wide_90 + narrow_50", flush=True)

        frame_period = 1.0 / max(self.fps, 1)
        try:
            while not self._stop.is_set():
                wide_frame, narrow_frame, stamps = self.buffer.get_pair()
                mosaic = _build_mosaic(wide_frame, narrow_frame, stamps, self.tile_width, self.tile_height)
                ok, encoded = cv2.imencode(".jpg", mosaic, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                if ok:
                    socket.send(encoded.tobytes())
                time.sleep(frame_period)
        finally:
            for camera in [wide, narrow]:
                try:
                    camera.stop()
                except Exception:
                    pass
            for camera in self._spawned:
                try:
                    camera.destroy()
                except Exception:
                    pass
            try:
                socket.close(0)
            except Exception:
                pass
        return 0


# Compatibility aliases for code that imports the old service by a class name.
CarlaCameraService = CarlaDualFrontCameraService
CameraService = CarlaDualFrontCameraService


def run_service() -> int:
    return CarlaDualFrontCameraService().run()


def main() -> int:
    service = CarlaDualFrontCameraService()

    def _signal_handler(_signum, _frame):
        service.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    return service.run()


if __name__ == "__main__":
    raise SystemExit(main())
