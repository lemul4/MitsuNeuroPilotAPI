# -*- coding: utf-8 -*-
"""Capture visible, real GUI window screenshots.

Unlike capture_test_screenshots.py, this script does not use Qt offscreen mode,
does not draw explanatory overlays, and does not synthesize road images. It
shows the real MainWindow on the desktop and captures the window frame.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


GUI_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = GUI_ROOT / "outputs" / "real_gui_screenshots"

if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))

from PySide6.QtCore import QEventLoop, QTimer, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.main_window import MainWindow  # noqa: E402


FONT_FAMILY = "Arial"
WINDOW_W = 980
WINDOW_H = 540


def install_fonts(app: QApplication) -> None:
    global FONT_FAMILY
    for font_path in (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/tahoma.ttf"),
    ):
        if font_path.exists():
            font_id = QFontDatabase.addApplicationFont(str(font_path))
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                FONT_FAMILY = families[0]
                app.setFont(QFont(FONT_FAMILY, 9))
                return


def wait_ms(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def safe_call(obj, name: str, *args, **kwargs) -> None:
    fn = getattr(obj, name, None)
    if callable(fn):
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            print(f"[real-capture] {name} failed: {exc}")


def route_payload(route_id: str) -> dict:
    return {
        "id": route_id,
        "name": route_id,
        "city": "Town10HD",
        "town": "Town10HD",
        "scenario": "Accident",
        "split": "lead",
        "path": str(
            GUI_ROOT
            / "data"
            / "data_routes"
            / "lead"
            / "Accident"
            / f"{route_id}.xml"
        ),
    }


def set_device(window: MainWindow, value: str) -> None:
    combo = getattr(window, "combo_ports", None)
    if combo is None:
        return
    for idx in range(combo.count()):
        data = combo.itemData(idx)
        text = combo.itemText(idx)
        if value in str(data) or value in str(text):
            combo.setCurrentIndex(idx)
            return


def gear_id(gear: str) -> int:
    return {"P": 1, "R": 2, "N": 3, "D": 4, "E": 5, "B": 6}.get(str(gear).upper(), 3)


def vehicle_state(speed: float, accel: float, brake: float, gear: str, angle: float = 3.0):
    return type(
        "VehicleState",
        (),
        {
            "speed": float(speed),
            "accel": float(accel),
            "brake": float(brake),
            "angle": float(angle),
            "target_angle": float(angle),
            "gear": gear_id(gear),
        },
    )()


def feed_history(window: MainWindow, final_state) -> None:
    final_speed = float(getattr(final_state, "speed", 0.0) or 0.0)
    final_accel = float(getattr(final_state, "accel", 0.0) or 0.0)
    final_brake = float(getattr(final_state, "brake", 0.0) or 0.0)
    gear = int(getattr(final_state, "gear", 3) or 3)
    speed_steps = [
        0.0, 0.0, 0.0, 0.0,
        final_speed * 0.20,
        final_speed * 0.20,
        final_speed * 0.45,
        final_speed * 0.45,
        final_speed * 0.65,
        final_speed * 0.65,
        final_speed * 0.85,
        final_speed * 0.85,
        final_speed,
        final_speed,
    ]
    if final_speed <= 0.0 and final_brake > 0.0:
        speed_steps = [8.0, 8.0, 8.0, 4.0, 4.0, 1.0, 1.0, 0.0, 0.0]
    accel_steps = [
        0.0, 0.0, 0.0,
        final_accel,
        final_accel,
        final_accel * 0.5,
        final_accel * 0.5,
        0.0,
        0.0,
    ]
    if final_brake > 0.0:
        accel_steps = [0.0, final_brake, final_brake, final_brake * 0.5, final_brake * 0.5, 0.0, 0.0]

    for idx in range(80):
        speed = speed_steps[min(len(speed_steps) - 1, idx // 6)]
        accel_or_brake = accel_steps[min(len(accel_steps) - 1, idx // 8)]
        accel = accel_or_brake if final_brake <= 0.0 else 0.0
        brake = accel_or_brake if final_brake > 0.0 else 0.0
        safe_call(
            window,
            "update_dashboard",
            vehicle_state(speed, accel, brake, "N", angle=float(getattr(final_state, "angle", 0.0))),
        )
        # update_dashboard receives a temporary gear above; preserve the target
        # gear on the final sample without fabricating UI outside its own method.
        getattr(window, "gear_btns", {}).get(gear, None)
    safe_call(window, "update_dashboard", final_state)


def set_real_state(window: MainWindow, mode: str, state: str, msg: str, final_state) -> None:
    safe_call(window, "set_ui_mode", "real")
    set_device(window, mode)
    safe_call(window, "set_connection_status", True, f"Connected: {mode}")
    safe_call(
        window,
        "set_real_mission_card",
        raw_mission={"name": "A → B", "goal_label": "A → B по координатам"},
        mode_text=mode,
    )
    safe_call(window, "set_real_readiness", route=True, pose=True, cameras=True, ai=True, vehicle=True)
    safe_call(window, "set_real_agent_status", True, "test/mock камеры активны")
    safe_call(window, "set_real_vehicle_state", state, msg)
    feed_history(window, final_state)


def set_carla_frame_from_simulator(window: MainWindow) -> bool:
    try:
        import carla
    except Exception as exc:
        print(f"[real-capture] carla import failed: {exc}")
        return False

    camera = None
    image_holder = {}
    try:
        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(60.0)
        world = client.get_world()
        blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", "1280")
        blueprint.set_attribute("image_size_y", "720")
        blueprint.set_attribute("fov", "90")
        spectator = world.get_spectator()
        transform = spectator.get_transform()
        camera = world.spawn_actor(blueprint, transform)
        camera.listen(lambda image: image_holder.setdefault("image", image))
        deadline = time.time() + 10.0
        while "image" not in image_holder and time.time() < deadline:
            world.wait_for_tick(1.0)
        image = image_holder.get("image")
        if image is None:
            return False
        qimage = QImage(
            image.raw_data,
            image.width,
            image.height,
            image.width * 4,
            QImage.Format_ARGB32,
        ).copy()
        safe_call(window, "update_camera_frame", QPixmap.fromImage(qimage))
        return True
    except Exception as exc:
        print(f"[real-capture] carla frame failed: {exc}")
        return False
    finally:
        if camera is not None:
            try:
                camera.stop()
                camera.destroy()
            except Exception:
                pass


def capture_window(window: MainWindow, name: str) -> None:
    app = QApplication.instance()
    window.showNormal()
    window.raise_()
    window.activateWindow()
    for _ in range(5):
        app.processEvents()
        wait_ms(80)
    frame = window.frameGeometry()
    screen = app.primaryScreen()
    pix = screen.grabWindow(0, frame.x(), frame.y(), frame.width(), frame.height())
    pix.save(str(OUT_DIR / name))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    install_fonts(app)

    window = MainWindow()
    window.setWindowTitle("MitsuNeuroPilot — интерфейс управления")
    window.setGeometry(0, 0, WINDOW_W, WINDOW_H)
    window.show()
    wait_ms(300)

    routes = [route_payload("route_001816"), route_payload("route_001817")]

    set_real_state(window, "TEST_MOCK_VEHICLE", "READY", "mock route ready", vehicle_state(0.0, 0.0, 0.0, "P", 0.0))
    safe_call(window, "set_manual_checkbox", False)
    capture_window(window, "01_mock_ready_real_window.png")

    set_real_state(window, "TEST_MOCK_VEHICLE", "READY", "AI Preview enabled", vehicle_state(8.0, 0.0, 0.0, "D", 0.0))
    safe_call(window, "set_ai_checkbox", True)
    capture_window(window, "02_ai_preview_real_window.png")

    set_real_state(window, "TEST_MOCK_VEHICLE", "AI_ACTIVE", "Автономное управление активно", vehicle_state(18.4, 28.0, 0.0, "D", 3.0))
    capture_window(window, "03_ai_active_real_window.png")

    set_real_state(window, "TEST_MOCK_VEHICLE", "MANUAL_ACTIVE", "Перехват ИИ: ручное управление активно", vehicle_state(0.0, 0.0, 35.0, "P", 0.0))
    safe_call(window, "set_manual_checkbox", True)
    capture_window(window, "04_manual_takeover_real_window.png")

    set_real_state(window, "TEST_MOCK_VEHICLE", "DISENGAGING", "Safe stop completed", vehicle_state(0.0, 0.0, 35.0, "P", 0.0))
    capture_window(window, "05_disengaging_park_real_window.png")

    set_real_state(window, "TEST_SERIAL_LOOPBACK", "READY", "COM dry-run / loopback", vehicle_state(0.0, 0.0, 0.0, "P", 0.0))
    safe_call(window, "_append_log", "COM dry-run: physical command execution disabled")
    capture_window(window, "06_com_dry_run_real_window.png")

    safe_call(window, "set_ui_mode", "carla")
    set_device(window, "VIRTUAL_DEMO_MODE")
    safe_call(window, "set_connection_status", True, "CARLA RPC OK: Town10HD")
    safe_call(window, "set_route_queue", routes)
    safe_call(window, "set_active_route", routes[0], 0, 2, "running", "route_001816.xml running")
    safe_call(window, "set_route_queue_position", 1, 2)
    safe_call(window, "_append_log", "AI ROUTES: старт маршрута 1/2: route_001816.xml")
    safe_call(window, "_append_log", "AI ROUTES: следующий маршрут: route_001817.xml")
    feed_history(window, vehicle_state(11.5, 0.0, 0.0, "D", 2.0))
    capture_window(window, "07_carla_route_queue_real_window.png")

    set_carla_frame_from_simulator(window)
    safe_call(window, "_append_log", "CARLA frame received from simulator RPC")
    capture_window(window, "08_carla_video_real_window.png")

    safe_call(window, "set_ui_mode", "real")
    set_device(window, "COM4")
    safe_call(window, "set_connection_status", True, "COM4 open; waiting heartbeat")
    safe_call(window, "set_real_readiness", route=False, pose=False, cameras=False, ai=True, vehicle=False)
    safe_call(window, "set_real_agent_status", False, "activation blocked")
    safe_call(window, "set_real_vehicle_state", "BLOCKED", "Нет маршрута / pose / heartbeat / cameras")
    feed_history(window, vehicle_state(0.0, 0.0, 0.0, "P", 0.0))
    capture_window(window, "09_activation_blocked_real_window.png")

    manifest = {
        "output_dir": str(OUT_DIR),
        "screenshots": sorted(p.name for p in OUT_DIR.glob("*.png")),
        "note": "Visible desktop captures of the real Qt MainWindow; no explanatory overlays.",
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    window.close()
    app.processEvents()


if __name__ == "__main__":
    main()
