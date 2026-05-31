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

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()
            self._timestamps.clear()

    def get_pair(self) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, float]]:
        with self._lock:
            return (
                self._frames.get("wide_90"),
                self._frames.get("narrow_50"),
                dict(self._timestamps),
            )

    def fresh_count(self, max_age_s: float) -> int:
        now = time.monotonic()
        with self._lock:
            return sum(1 for ts in self._timestamps.values() if now - ts <= max_age_s)

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
    # Do not show a stale spawn frame as if it were live road video.
    # When a camera stream dies, publish a diagnostic tile instead.
    if frame is None or stale:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        text = "NO LIVE FRAME" if frame is None else "STALE FRAME - RECONNECTING"
        cv2.putText(canvas, text, (28, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (80, 80, 255), 2)
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

# --- MITSU_CAMERA_OPERATOR_PREVIEW_PATCH_BEGIN ---
# Clean CARLA operator preview:
# - wide_90 fills the frame;
# - narrow_50 is picture-in-picture;
# - no debug side-by-side labels during normal operation.

def _mitsu_fit_cover(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = frame.shape[:2]
    if src_w <= 0 or src_h <= 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    scale = max(width / src_w, height / src_h)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    x0 = max(0, (resized_w - width) // 2)
    y0 = max(0, (resized_h - height) // 2)
    return resized[y0:y0 + height, x0:x0 + width].copy()


def _mitsu_draw_status_overlay(canvas: np.ndarray, text: str, color=(80, 80, 255)) -> None:
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w, 38), (0, 0, 0), -1)
    cv2.putText(canvas, text, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)


def _mitsu_build_operator_preview(wide: np.ndarray | None, narrow: np.ndarray | None, ages: dict[str, float], width: int, height: int) -> np.ndarray:
    now = time.monotonic()
    wide_stale = wide is None or (now - ages.get("wide_90", 0.0)) > 1.0
    narrow_stale = narrow is None or (now - ages.get("narrow_50", 0.0)) > 1.0

    if wide_stale:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        _mitsu_draw_status_overlay(canvas, "CARLA preview: waiting for live wide camera")
    else:
        canvas = _mitsu_fit_cover(wide, width, height)

    if not narrow_stale:
        inset_w = max(220, int(width * 0.34))
        inset_h = max(120, int(inset_w * height / max(width, 1)))
        inset = _mitsu_fit_cover(narrow, inset_w, inset_h)

        margin = max(12, int(width * 0.018))
        x1 = width - margin
        y1 = height - margin
        x0 = x1 - inset_w
        y0 = y1 - inset_h

        cv2.rectangle(canvas, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), (0, 0, 0), -1)
        canvas[y0:y1, x0:x1] = inset
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (72, 76, 88), 2)
    elif not wide_stale:
        _mitsu_draw_status_overlay(canvas, "CARLA preview: narrow camera reconnecting", color=(0, 160, 255))

    return canvas


_build_mosaic = _mitsu_build_operator_preview
# --- MITSU_CAMERA_OPERATOR_PREVIEW_PATCH_END ---

# --- MITSU_CAMERA_BUSINESS_PREVIEW_V2_BEGIN ---
# Product CARLA operator preview.
# The GUI receives one JPEG, but it looks like a clean dual-camera product view.

def _mitsu_fit_cover(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = frame.shape[:2]
    if src_w <= 0 or src_h <= 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    scale = max(width / src_w, height / src_h)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    x0 = max(0, (resized_w - width) // 2)
    y0 = max(0, (resized_h - height) // 2)
    return resized[y0:y0 + height, x0:x0 + width].copy()


def _mitsu_overlay_text_box(img: np.ndarray, text: str, x: int, y: int, color=(255, 255, 255), bg=(0, 0, 0), alpha=0.60):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.54
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad_x, pad_y = 10, 7
    x0, y0 = x, y
    x1, y1 = x + tw + pad_x * 2, y + th + pad_y * 2
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), bg, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, (x + pad_x, y + pad_y + th), font, scale, color, thickness, cv2.LINE_AA)


def _mitsu_business_preview(wide: np.ndarray | None, narrow: np.ndarray | None, ages: dict[str, float], width: int, height: int) -> np.ndarray:
    now = time.monotonic()
    wide_age = now - ages.get("wide_90", 0.0) if ages.get("wide_90") else 10000.0
    narrow_age = now - ages.get("narrow_50", 0.0) if ages.get("narrow_50") else 10000.0
    wide_live = wide is not None and wide_age <= 1.0
    narrow_live = narrow is not None and narrow_age <= 1.0

    last_ts = getattr(_mitsu_business_preview, "_last_ts", 0.0)
    fps = getattr(_mitsu_business_preview, "_fps", 0.0)
    if last_ts:
        dt = max(1e-3, now - last_ts)
        instant = 1.0 / dt
        fps = instant if fps <= 0 else (fps * 0.88 + instant * 0.12)
    _mitsu_business_preview._last_ts = now
    _mitsu_business_preview._fps = fps

    if not wide_live:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        _mitsu_overlay_text_box(canvas, "CARLA preview: ожидание live-кадра 2.8 мм / 90°", 18, 18, color=(80, 80, 255))
        return canvas

    canvas = _mitsu_fit_cover(wide, width, height)

    live_age_ms = int(min(wide_age, narrow_age if narrow_live else wide_age) * 1000)
    status = f"LIVE · {fps:.0f} FPS · {live_age_ms} ms" if fps > 0 else f"LIVE · {live_age_ms} ms"
    _mitsu_overlay_text_box(canvas, "Симулятор CARLA", 16, 16, color=(255, 255, 255), bg=(0, 0, 0), alpha=0.54)
    _mitsu_overlay_text_box(canvas, status, max(16, width - 210), 16, color=(20, 230, 80), bg=(0, 0, 0), alpha=0.54)
    _mitsu_overlay_text_box(canvas, "2.8 мм · широкий обзор 90°", max(16, width - 294), max(16, height - 42), color=(255, 255, 255), bg=(0, 0, 0), alpha=0.58)

    if narrow_live:
        inset_w = max(210, int(width * 0.28))
        inset_h = max(118, int(inset_w * 9 / 16))
        margin = max(14, int(width * 0.018))
        x0 = margin
        y0 = height - margin - inset_h
        x1 = x0 + inset_w
        y1 = y0 + inset_h

        inset = _mitsu_fit_cover(narrow, inset_w, inset_h)

        cv2.rectangle(canvas, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (0, 0, 0), -1)
        canvas[y0:y1, x0:x1] = inset
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (94, 100, 114), 2)

        _mitsu_overlay_text_box(canvas, "6 мм · дальний обзор 50°", x0 + 8, y0 + 8, color=(255, 255, 255), bg=(0, 0, 0), alpha=0.62)
    else:
        _mitsu_overlay_text_box(canvas, "6 мм · 50°: ожидание", 16, max(16, height - 42), color=(0, 180, 255), bg=(0, 0, 0), alpha=0.58)

    return canvas


_build_mosaic = _mitsu_business_preview
# --- MITSU_CAMERA_BUSINESS_PREVIEW_V2_END ---

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

        # MITSU_REALTIME_CAMERA_REACQUIRE_PATCH
        # If CARLA reloads the world or Leaderboard respawns the hero actor,
        # sensor callbacks can stop. Re-acquire instead of publishing old frames forever.
        self.reacquire_after_stale_s = _env_float("MITSU_CARLA_PREVIEW_REACQUIRE_AFTER_STALE_S", 2.5)

        self._stop = threading.Event()
        self._spawned: list[Any] = []
        self.buffer = LatestFrameBuffer()

    def stop(self) -> None:
        self._stop.set()

    def _safe_stop_camera(self, camera: Any, destroy: bool = False) -> None:
        if camera is None:
            return
        try:
            camera.stop()
        except Exception:
            pass
        if destroy:
            try:
                camera.destroy()
            except Exception:
                pass

    def _release_preview_cameras(self, wide: Any, narrow: Any) -> None:
        spawned_ids = {getattr(camera, "id", None) for camera in self._spawned}

        for camera in (wide, narrow):
            if camera is None:
                continue
            self._safe_stop_camera(camera, destroy=getattr(camera, "id", None) in spawned_ids)

        self._spawned.clear()

    def _acquire_preview_cameras(self, client: Any) -> tuple[Any, Any]:
        world, vehicle = _find_hero_vehicle(client, timeout_s=self.wait_timeout_s)

        existing_wide = existing_narrow = None
        if self.use_existing:
            existing_wide, existing_narrow = _select_existing_dual_front_cameras(world, vehicle)

        # Default: spawn GUI-only preview copies. This does not interfere with
        # model sensors that Leaderboard/agent may already listen to.
        if self.prefer_existing and existing_wide is not None and existing_narrow is not None:
            print("[CAMERA_SERVICE] Preview читает существующие модельные камеры 90/50", flush=True)
            return existing_wide, existing_narrow

        if self.spawn_if_missing:
            print("[CAMERA_SERVICE] Спавним две GUI-preview камеры с layout i-MiEV: wide_90 + narrow_50", flush=True)
            wide = _spawn_preview_camera(world, vehicle, WIDE_90, self.tile_width, self.tile_height, self.fps)
            narrow = _spawn_preview_camera(world, vehicle, NARROW_50, self.tile_width, self.tile_height, self.fps)
            self._spawned.extend([wide, narrow])
            return wide, narrow

        wide, narrow = existing_wide, existing_narrow
        if wide is None or narrow is None:
            raise RuntimeError("Cannot start dual-front preview: wide/narrow cameras are missing")

        return wide, narrow

    def _publish_diagnostic_frame(self, socket: Any) -> None:
        mosaic = _build_mosaic(None, None, {}, self.tile_width, self.tile_height)
        ok, encoded = cv2.imencode(".jpg", mosaic, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if ok:
            socket.send(encoded.tobytes())

    def run(self) -> int:
        context = zmq.Context.instance()
        socket = context.socket(zmq.PUB)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDHWM, 1)
        socket.bind(self.zmq_bind)
        print(f"[CAMERA_SERVICE] ZMQ preview bind: {self.zmq_bind}", flush=True)

        client = carla.Client(self.host, self.port)
        client.set_timeout(10.0)

        wide = None
        narrow = None
        frame_period = 1.0 / max(self.fps, 1)
        last_reacquire_at = 0.0

        try:
            while not self._stop.is_set():
                now = time.monotonic()

                need_acquire = wide is None or narrow is None

                if not need_acquire and self.buffer.fresh_count(self.reacquire_after_stale_s) == 0:
                    if now - last_reacquire_at >= self.reacquire_after_stale_s:
                        print(
                            "[CAMERA_SERVICE] Нет свежих CARLA-кадров; "
                            "переподключаем preview-камеры к текущему hero actor",
                            flush=True,
                        )
                        need_acquire = True

                if need_acquire:
                    last_reacquire_at = now
                    self._release_preview_cameras(wide, narrow)
                    self.buffer.clear()
                    wide = None
                    narrow = None

                    try:
                        wide, narrow = self._acquire_preview_cameras(client)
                        _listen(wide, "wide_90", self.buffer)
                        _listen(narrow, "narrow_50", self.buffer)
                        print("[CAMERA_SERVICE] Realtime preview подключён: wide_90 + narrow_50", flush=True)
                    except Exception as exc:
                        print(f"[CAMERA_SERVICE] Realtime preview ждёт hero/cameras: {exc}", flush=True)
                        self._publish_diagnostic_frame(socket)
                        time.sleep(min(1.0, frame_period))
                        continue

                wide_frame, narrow_frame, stamps = self.buffer.get_pair()
                mosaic = _build_mosaic(wide_frame, narrow_frame, stamps, self.tile_width, self.tile_height)
                ok, encoded = cv2.imencode(".jpg", mosaic, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                if ok:
                    socket.send(encoded.tobytes())

                time.sleep(frame_period)

        finally:
            self._release_preview_cameras(wide, narrow)
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

# --- MITSU_CAMERA_CLEAN_COMPOSITE_AND_RECONNECT_V3_BEGIN ---
# Clean CARLA camera composite:
# - no text inside JPEG frame; Qt renders all labels, FPS and latency;
# - smaller PiP fully inside the image;
# - no aggressive mid-scenario respawn loop;
# - reuse existing gui_wide_90/gui_narrow_50 sensors if they are already attached.

def _mitsu_v3_fit_cover(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = frame.shape[:2]
    if src_w <= 0 or src_h <= 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    scale = max(width / src_w, height / src_h)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    x0 = max(0, (resized_w - width) // 2)
    y0 = max(0, (resized_h - height) // 2)
    return resized[y0:y0 + height, x0:x0 + width].copy()


def _mitsu_v3_build_clean_composite(
    wide: np.ndarray | None,
    narrow: np.ndarray | None,
    ages: dict[str, float],
    width: int,
    height: int,
) -> np.ndarray:
    now = time.monotonic()
    wide_age = now - ages.get("wide_90", 0.0) if ages.get("wide_90") else 10000.0
    narrow_age = now - ages.get("narrow_50", 0.0) if ages.get("narrow_50") else 10000.0

    wide_live = wide is not None and wide_age <= 1.0
    narrow_live = narrow is not None and narrow_age <= 1.0

    if not wide_live:
        # Diagnostic frame without Cyrillic text: Qt side will show WAIT/STALE status.
        return np.zeros((height, width, 3), dtype=np.uint8)

    canvas = _mitsu_v3_fit_cover(wide, width, height)

    if narrow_live:
        inset_w = max(150, min(int(width * 0.22), 260))
        inset_h = max(84, int(inset_w * 9 / 16))
        margin = max(10, int(width * 0.014))

        # Bottom-left, fully inside the video frame.
        x0 = margin
        y0 = height - margin - inset_h
        x1 = min(width - margin, x0 + inset_w)
        y1 = min(height - margin, y0 + inset_h)

        inset = _mitsu_v3_fit_cover(narrow, x1 - x0, y1 - y0)

        cv2.rectangle(canvas, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), (0, 0, 0), -1)
        canvas[y0:y1, x0:x1] = inset
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (72, 76, 88), 2)

    return canvas


_build_mosaic = _mitsu_v3_build_clean_composite


_mitsu_v3_original_init = CarlaDualFrontCameraService.__init__


def _mitsu_v3_init(self, *args, **kwargs):
    _mitsu_v3_original_init(self, *args, **kwargs)

    # 2.5 s caused repeated reconnect/spawn loops during valid scenarios.
    # If a user explicitly sets the env variable, keep their value. Otherwise use a conservative value.
    if "MITSU_CARLA_PREVIEW_REACQUIRE_AFTER_STALE_S" not in os.environ:
        self.reacquire_after_stale_s = 30.0


CarlaDualFrontCameraService.__init__ = _mitsu_v3_init


def _mitsu_v3_acquire_preview_cameras(self, client: Any) -> tuple[Any, Any]:
    world, vehicle = _find_hero_vehicle(client, timeout_s=self.wait_timeout_s)

    existing_wide = existing_narrow = None
    if self.use_existing:
        existing_wide, existing_narrow = _select_existing_dual_front_cameras(world, vehicle)

    # Critical stability fix:
    # If both GUI/model preview cameras already exist, reuse them. Do not spawn another pair.
    if existing_wide is not None and existing_narrow is not None:
        print(
            f"[CAMERA_SERVICE] Reusing existing preview cameras: "
            f"wide_90 id={getattr(existing_wide, 'id', '-')}, "
            f"narrow_50 id={getattr(existing_narrow, 'id', '-')}",
            flush=True,
        )
        return existing_wide, existing_narrow

    if self.spawn_if_missing:
        print("[CAMERA_SERVICE] Спавним две GUI-preview камеры с layout i-MiEV: wide_90 + narrow_50", flush=True)
        wide = _spawn_preview_camera(world, vehicle, WIDE_90, self.tile_width, self.tile_height, self.fps)
        narrow = _spawn_preview_camera(world, vehicle, NARROW_50, self.tile_width, self.tile_height, self.fps)
        self._spawned.extend([wide, narrow])
        return wide, narrow

    if existing_wide is None or existing_narrow is None:
        raise RuntimeError("Cannot start dual-front preview: wide/narrow cameras are missing")
    return existing_wide, existing_narrow


CarlaDualFrontCameraService._acquire_preview_cameras = _mitsu_v3_acquire_preview_cameras
# --- MITSU_CAMERA_CLEAN_COMPOSITE_AND_RECONNECT_V3_END ---

# --- MITSU_CARLA_CAMERA_BUSINESS_LAYOUT_V4_BEGIN ---
# CARLA preview composite for business UI.
# No cv2.putText labels; Qt draws all Russian labels/status.
# Default layout: wide camera + separate narrow camera rail, not PiP overlay.

def _mitsu_v4_fit_cover(frame, width, height):
    src_h, src_w = frame.shape[:2]
    if src_w <= 0 or src_h <= 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    scale = max(width / src_w, height / src_h)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    x0 = max(0, (resized_w - width) // 2)
    y0 = max(0, (resized_h - height) // 2)
    return resized[y0:y0 + height, x0:x0 + width].copy()


def _mitsu_v4_build_business_composite(wide, narrow, ages, width, height):
    now = time.monotonic()
    wide_age = now - ages.get("wide_90", 0.0) if ages.get("wide_90") else 10000.0
    narrow_age = now - ages.get("narrow_50", 0.0) if ages.get("narrow_50") else 10000.0

    wide_live = wide is not None and wide_age <= 1.0
    narrow_live = narrow is not None and narrow_age <= 1.0

    if not wide_live:
        return np.zeros((height, width, 3), dtype=np.uint8)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if narrow_live:
        gap = max(8, int(width * 0.010))
        rail_w = int(width * 0.28)
        main_w = width - rail_w - gap

        canvas[:, :main_w] = _mitsu_v4_fit_cover(wide, main_w, height)

        rail_x = main_w + gap
        rail = np.zeros((height, rail_w, 3), dtype=np.uint8)
        cv2.rectangle(rail, (0, 0), (rail_w - 1, height - 1), (18, 20, 27), -1)

        margin = max(8, int(rail_w * 0.045))
        card_w = rail_w - margin * 2
        card_h = min(int(card_w * 9 / 16), int(height * 0.42))
        x0 = margin
        y0 = margin
        x1 = x0 + card_w
        y1 = y0 + card_h

        rail[y0:y1, x0:x1] = _mitsu_v4_fit_cover(narrow, card_w, card_h)
        cv2.rectangle(rail, (x0, y0), (x1, y1), (72, 76, 88), 2)

        canvas[:, rail_x:rail_x + rail_w] = rail
    else:
        canvas[:, :] = _mitsu_v4_fit_cover(wide, width, height)

    return canvas


_build_mosaic = _mitsu_v4_build_business_composite


_mitsu_v4_original_service_init = CarlaDualFrontCameraService.__init__


def _mitsu_v4_service_init(self, *args, **kwargs):
    _mitsu_v4_original_service_init(self, *args, **kwargs)
    if "MITSU_CARLA_PREVIEW_REACQUIRE_AFTER_STALE_S" not in os.environ:
        self.reacquire_after_stale_s = 30.0


CarlaDualFrontCameraService.__init__ = _mitsu_v4_service_init


def _mitsu_v4_acquire_preview_cameras(self, client):
    world, vehicle = _find_hero_vehicle(client, timeout_s=self.wait_timeout_s)

    existing_wide = existing_narrow = None
    if self.use_existing:
        existing_wide, existing_narrow = _select_existing_dual_front_cameras(world, vehicle)

    if existing_wide is not None and existing_narrow is not None:
        print(
            f"[CAMERA_SERVICE] Reusing existing preview cameras: "
            f"wide_90 id={getattr(existing_wide, 'id', '-')}, "
            f"narrow_50 id={getattr(existing_narrow, 'id', '-')}",
            flush=True,
        )
        return existing_wide, existing_narrow

    if self.spawn_if_missing:
        print("[CAMERA_SERVICE] Спавним две GUI-preview камеры: wide_90 + narrow_50", flush=True)
        wide = _spawn_preview_camera(world, vehicle, WIDE_90, self.tile_width, self.tile_height, self.fps)
        narrow = _spawn_preview_camera(world, vehicle, NARROW_50, self.tile_width, self.tile_height, self.fps)
        self._spawned.extend([wide, narrow])
        return wide, narrow

    if existing_wide is None or existing_narrow is None:
        raise RuntimeError("Cannot start dual-front preview: wide/narrow cameras are missing")
    return existing_wide, existing_narrow


CarlaDualFrontCameraService._acquire_preview_cameras = _mitsu_v4_acquire_preview_cameras
# --- MITSU_CARLA_CAMERA_BUSINESS_LAYOUT_V4_END ---

# --- MITSU_CARLA_FULL_PRIMARY_COMPOSITE_V5_BEGIN ---
# Full-primary CARLA composite.
#
# Default: wide 90° fills the entire video frame.
# Secondary camera is not permanently shown because it steals horizontal FOV.
# Optional debug/service-side layouts:
#   MITSU_CARLA_CAMERA_LAYOUT=wide|pip|split|narrow
#
# Qt draws all labels/status; no cv2.putText here.

def _mitsu_v5_fit_cover(frame, width, height):
    src_h, src_w = frame.shape[:2]
    if src_w <= 0 or src_h <= 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    scale = max(width / src_w, height / src_h)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    x0 = max(0, (resized_w - width) // 2)
    y0 = max(0, (resized_h - height) // 2)
    return resized[y0:y0 + height, x0:x0 + width].copy()


def _mitsu_v5_fit_contain(frame, width, height):
    src_h, src_w = frame.shape[:2]
    if src_w <= 0 or src_h <= 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(width / src_w, height / src_h)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    x0 = max(0, (width - resized_w) // 2)
    y0 = max(0, (height - resized_h) // 2)
    canvas[y0:y0 + resized_h, x0:x0 + resized_w] = resized
    return canvas


def _mitsu_v5_build_full_primary_composite(wide, narrow, ages, width, height):
    now = time.monotonic()
    wide_age = now - ages.get("wide_90", 0.0) if ages.get("wide_90") else 10000.0
    narrow_age = now - ages.get("narrow_50", 0.0) if ages.get("narrow_50") else 10000.0

    wide_live = wide is not None and wide_age <= 1.0
    narrow_live = narrow is not None and narrow_age <= 1.0

    layout = os.environ.get("MITSU_CARLA_CAMERA_LAYOUT", "wide").strip().lower()

    if layout == "narrow" and narrow_live:
        return _mitsu_v5_fit_cover(narrow, width, height)

    if not wide_live:
        return np.zeros((height, width, 3), dtype=np.uint8)

    if layout == "pip" and narrow_live:
        canvas = _mitsu_v5_fit_cover(wide, width, height)

        # Small optional PiP. Manual debug/preview layout only, not default.
        inset_w = max(170, min(int(width * 0.22), 260))
        inset_h = max(96, int(inset_w * 9 / 16))
        margin = max(12, int(width * 0.016))
        x0 = margin
        y0 = height - margin - inset_h
        x1 = min(width - margin, x0 + inset_w)
        y1 = min(height - margin, y0 + inset_h)

        inset = _mitsu_v5_fit_cover(narrow, x1 - x0, y1 - y0)
        cv2.rectangle(canvas, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (0, 0, 0), -1)
        canvas[y0:y1, x0:x1] = inset
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (72, 76, 88), 2)
        return canvas

    if layout == "split" and narrow_live:
        # Split debug layout: keep aspect, do not stretch.
        gap = max(8, int(width * 0.010))
        left_w = int((width - gap) * 0.68)
        right_w = width - gap - left_w
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:, :left_w] = _mitsu_v5_fit_cover(wide, left_w, height)
        canvas[:, left_w + gap:] = _mitsu_v5_fit_contain(narrow, right_w, height)
        return canvas

    # Default production/demo layout: primary wide camera full viewport.
    return _mitsu_v5_fit_cover(wide, width, height)


_build_mosaic = _mitsu_v5_build_full_primary_composite


# Keep previous anti-respawn stability if this block is applied after older patches.
if "CarlaDualFrontCameraService" in globals():
    _mitsu_v5_original_service_init = CarlaDualFrontCameraService.__init__

    def _mitsu_v5_service_init(self, *args, **kwargs):
        _mitsu_v5_original_service_init(self, *args, **kwargs)
        if "MITSU_CARLA_PREVIEW_REACQUIRE_AFTER_STALE_S" not in os.environ:
            self.reacquire_after_stale_s = 30.0

    CarlaDualFrontCameraService.__init__ = _mitsu_v5_service_init

    def _mitsu_v5_acquire_preview_cameras(self, client):
        world, vehicle = _find_hero_vehicle(client, timeout_s=self.wait_timeout_s)

        existing_wide = existing_narrow = None
        if self.use_existing:
            existing_wide, existing_narrow = _select_existing_dual_front_cameras(world, vehicle)

        if existing_wide is not None and existing_narrow is not None:
            print(
                f"[CAMERA_SERVICE] Reusing existing preview cameras: "
                f"wide_90 id={getattr(existing_wide, 'id', '-')}, "
                f"narrow_50 id={getattr(existing_narrow, 'id', '-')}",
                flush=True,
            )
            return existing_wide, existing_narrow

        if self.spawn_if_missing:
            print("[CAMERA_SERVICE] Спавним две GUI-preview камеры: wide_90 + narrow_50", flush=True)
            wide = _spawn_preview_camera(world, vehicle, WIDE_90, self.tile_width, self.tile_height, self.fps)
            narrow = _spawn_preview_camera(world, vehicle, NARROW_50, self.tile_width, self.tile_height, self.fps)
            self._spawned.extend([wide, narrow])
            return wide, narrow

        if existing_wide is None or existing_narrow is None:
            raise RuntimeError("Cannot start dual-front preview: wide/narrow cameras are missing")
        return existing_wide, existing_narrow

    CarlaDualFrontCameraService._acquire_preview_cameras = _mitsu_v5_acquire_preview_cameras
# --- MITSU_CARLA_FULL_PRIMARY_COMPOSITE_V5_END ---

# --- MITSU_CARLA_RUNTIME_CAMERA_LAYOUT_V6_BEGIN ---
# Runtime camera layout switching for CARLA preview.
# Reads temp-file written by GUI:
#   %TEMP%/mitsu_carla_camera_layout.txt
# or custom path from MITSU_CARLA_CAMERA_LAYOUT_FILE.

def _mitsu_v6_layout_file_path():
    import tempfile
    from pathlib import Path
    value = os.environ.get("MITSU_CARLA_CAMERA_LAYOUT_FILE", "")
    if value:
        return Path(value)
    return Path(tempfile.gettempdir()) / "mitsu_carla_camera_layout.txt"


def _mitsu_v6_current_layout():
    aliases = {
        "90": "wide",
        "90°": "wide",
        "primary": "wide",
        "main": "wide",
        "50": "narrow",
        "50°": "narrow",
        "secondary": "narrow",
        "tele": "narrow",
        "2x": "split",
    }

    path = _mitsu_v6_layout_file_path()
    mode = ""
    try:
        if path.exists():
            mode = path.read_text(encoding="utf-8", errors="replace").strip().lower()
    except Exception:
        mode = ""

    if not mode:
        mode = os.environ.get("MITSU_CARLA_CAMERA_LAYOUT", "wide").strip().lower()

    mode = aliases.get(mode, mode)
    if mode not in {"wide", "narrow", "split", "pip"}:
        mode = "wide"
    return mode


def _mitsu_v6_fit_cover(frame, width, height):
    src_h, src_w = frame.shape[:2]
    if src_w <= 0 or src_h <= 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    scale = max(width / src_w, height / src_h)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    x0 = max(0, (resized_w - width) // 2)
    y0 = max(0, (resized_h - height) // 2)
    return resized[y0:y0 + height, x0:x0 + width].copy()


def _mitsu_v6_fit_contain(frame, width, height):
    src_h, src_w = frame.shape[:2]
    if src_w <= 0 or src_h <= 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(width / src_w, height / src_h)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    x0 = max(0, (width - resized_w) // 2)
    y0 = max(0, (height - resized_h) // 2)
    canvas[y0:y0 + resized_h, x0:x0 + resized_w] = resized
    return canvas


def _mitsu_v6_build_runtime_layout_composite(wide, narrow, ages, width, height):
    now = time.monotonic()
    wide_age = now - ages.get("wide_90", 0.0) if ages.get("wide_90") else 10000.0
    narrow_age = now - ages.get("narrow_50", 0.0) if ages.get("narrow_50") else 10000.0

    wide_live = wide is not None and wide_age <= 1.0
    narrow_live = narrow is not None and narrow_age <= 1.0

    mode = _mitsu_v6_current_layout()

    if mode == "narrow" and narrow_live:
        return _mitsu_v6_fit_cover(narrow, width, height)

    if not wide_live:
        return np.zeros((height, width, 3), dtype=np.uint8)

    if mode == "pip" and narrow_live:
        canvas = _mitsu_v6_fit_cover(wide, width, height)

        inset_w = max(170, min(int(width * 0.22), 260))
        inset_h = max(96, int(inset_w * 9 / 16))
        margin = max(12, int(width * 0.016))
        x0 = margin
        y0 = height - margin - inset_h
        x1 = min(width - margin, x0 + inset_w)
        y1 = min(height - margin, y0 + inset_h)

        inset = _mitsu_v6_fit_cover(narrow, x1 - x0, y1 - y0)
        cv2.rectangle(canvas, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (0, 0, 0), -1)
        canvas[y0:y1, x0:x1] = inset
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (72, 76, 88), 2)
        return canvas

    if mode == "split" and narrow_live:
        gap = max(8, int(width * 0.010))
        left_w = int((width - gap) * 0.68)
        right_w = width - gap - left_w
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:, :left_w] = _mitsu_v6_fit_cover(wide, left_w, height)
        canvas[:, left_w + gap:] = _mitsu_v6_fit_contain(narrow, right_w, height)
        return canvas

    return _mitsu_v6_fit_cover(wide, width, height)


_build_mosaic = _mitsu_v6_build_runtime_layout_composite


# Preserve anti-respawn behavior.
if "CarlaDualFrontCameraService" in globals():
    _mitsu_v6_original_service_init = CarlaDualFrontCameraService.__init__

    def _mitsu_v6_service_init(self, *args, **kwargs):
        _mitsu_v6_original_service_init(self, *args, **kwargs)
        if "MITSU_CARLA_PREVIEW_REACQUIRE_AFTER_STALE_S" not in os.environ:
            self.reacquire_after_stale_s = 30.0

    CarlaDualFrontCameraService.__init__ = _mitsu_v6_service_init

    def _mitsu_v6_acquire_preview_cameras(self, client):
        world, vehicle = _find_hero_vehicle(client, timeout_s=self.wait_timeout_s)

        existing_wide = existing_narrow = None
        if self.use_existing:
            existing_wide, existing_narrow = _select_existing_dual_front_cameras(world, vehicle)

        if existing_wide is not None and existing_narrow is not None:
            print(
                f"[CAMERA_SERVICE] Reusing existing preview cameras: "
                f"wide_90 id={getattr(existing_wide, 'id', '-')}, "
                f"narrow_50 id={getattr(existing_narrow, 'id', '-')}",
                flush=True,
            )
            return existing_wide, existing_narrow

        if self.spawn_if_missing:
            print("[CAMERA_SERVICE] Спавним две GUI-preview камеры: wide_90 + narrow_50", flush=True)
            wide = _spawn_preview_camera(world, vehicle, WIDE_90, self.tile_width, self.tile_height, self.fps)
            narrow = _spawn_preview_camera(world, vehicle, NARROW_50, self.tile_width, self.tile_height, self.fps)
            self._spawned.extend([wide, narrow])
            return wide, narrow

        if existing_wide is None or existing_narrow is None:
            raise RuntimeError("Cannot start dual-front preview: wide/narrow cameras are missing")
        return existing_wide, existing_narrow

    CarlaDualFrontCameraService._acquire_preview_cameras = _mitsu_v6_acquire_preview_cameras
# --- MITSU_CARLA_RUNTIME_CAMERA_LAYOUT_V6_END ---

def main() -> int:
    service = CarlaDualFrontCameraService()

    def _signal_handler(_signum, _frame):
        service.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    return service.run()


if __name__ == "__main__":
    raise SystemExit(main())
