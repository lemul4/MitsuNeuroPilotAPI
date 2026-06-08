# -*- coding: utf-8 -*-
"""Capture reproducible GUI screenshots for the testing chapter.

The script renders the real MainWindow offscreen, applies representative
mock/COM/CARLA states through public UI methods, and stores PNG artifacts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

GUI_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = GUI_ROOT.parent
OUT_DIR = GUI_ROOT / "outputs" / "test_screenshots"

if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPen, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.main_window import MainWindow  # noqa: E402


WIDTH = 980
HEIGHT = 554
FONT_FAMILY = "Arial"


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


def safe_call(obj, name: str, *args, **kwargs) -> None:
    fn = getattr(obj, name, None)
    if callable(fn):
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            print(f"[capture] {name} failed: {exc}")


def install_fonts(app: QApplication) -> None:
    global FONT_FAMILY
    candidates = [
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf",
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf",
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "tahoma.ttf",
    ]
    for font_path in candidates:
        if font_path.exists():
            font_id = QFontDatabase.addApplicationFont(str(font_path))
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                FONT_FAMILY = families[0]
                app.setFont(QFont(FONT_FAMILY, 10))
                return


def make_carla_frame(title: str, status: str) -> QPixmap:
    pix = QPixmap(1280, 720)
    pix.fill(QColor(18, 22, 28))
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)

    sky = QColor(42, 74, 108)
    road = QColor(42, 42, 46)
    lane = QColor(230, 210, 110)
    green = QColor(50, 120, 70)
    p.fillRect(0, 0, 1280, 330, sky)
    p.fillRect(0, 330, 1280, 90, green)
    p.setBrush(road)
    p.setPen(Qt.NoPen)
    p.drawPolygon([QPoint(410, 720), QPoint(870, 720), QPoint(690, 330), QPoint(590, 330)])
    p.setPen(QPen(lane, 7, Qt.DashLine))
    p.drawLine(640, 710, 640, 350)
    p.setPen(QPen(QColor(235, 235, 235), 5))
    p.drawLine(450, 710, 595, 350)
    p.drawLine(830, 710, 685, 350)

    p.setBrush(QColor(120, 160, 190))
    p.setPen(QPen(QColor(20, 20, 20), 2))
    for x, y, w, h in [(115, 210, 135, 80), (970, 235, 160, 95), (420, 260, 120, 70)]:
        p.drawRect(x, y, w, h)
        p.drawRect(x + 15, y + 15, w - 30, h - 35)

    p.setBrush(QColor(22, 25, 30, 220))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(28, 28, 560, 155, 14, 14)
    p.setPen(QColor(245, 248, 252))
    p.setFont(QFont(FONT_FAMILY, 30, QFont.Bold))
    p.drawText(52, 82, title)
    p.setFont(QFont(FONT_FAMILY, 19))
    p.drawText(52, 126, status)
    p.drawText(52, 163, "speed=18.4 km/h  status=RUNNING  progress=37%")

    p.setPen(QColor(125, 230, 140))
    p.setFont(QFont(FONT_FAMILY, 18, QFont.Bold))
    p.drawText(1020, 48, "CARLA LIVE")
    p.setBrush(QColor(18, 22, 28, 230))
    p.setPen(QPen(QColor("#1d2430"), 4))
    p.drawRoundedRect(42, 520, 255, 144, 8, 8)
    p.drawPixmap(50, 528, make_test_camera_frame("PiP 50°").scaled(239, 128, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
    p.end()
    return pix


def make_test_camera_frame(label: str) -> QPixmap:
    pixmap = QPixmap(960, 540)
    pixmap.fill(QColor("#10131a"))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.fillRect(0, 0, 960, 540, QColor("#09101a"))
    painter.setPen(QPen(QColor("#1f2a3b"), 1))
    for x in range(0, 960, 80):
        painter.drawLine(x, 0, x, 540)
    for y in range(0, 540, 60):
        painter.drawLine(0, y, 960, y)
    painter.setBrush(QColor(255, 193, 7, 38))
    painter.setPen(QPen(QColor("#ffc107"), 2))
    painter.drawRoundedRect(24, 24, 270, 58, 12, 12)
    painter.setPen(QColor("#f6f7fb"))
    painter.setFont(QFont(FONT_FAMILY, 18, QFont.Bold))
    painter.drawText(42, 59, f"TEST CAMERA · {label}")
    painter.setFont(QFont(FONT_FAMILY, 11))
    painter.setPen(QColor("#b9c2d3"))
    painter.drawText(42, 95, "mock/test stream; physical camera is not required")
    painter.end()
    return pixmap


def install_mock_camera_frames(window: MainWindow, mode: str = "pip") -> None:
    stage = getattr(window, "real_camera_stage", None)
    if stage is not None:
        safe_call(stage, "set_view_mode", mode)
        safe_call(stage, "set_camera_pixmap", "wide_90", make_test_camera_frame("90°"))
        safe_call(stage, "set_camera_pixmap", "narrow_50", make_test_camera_frame("50°"))
        safe_call(stage, "set_status", "LIVE", "3 FPS · 1500 ms")


def annotate(pix: QPixmap, title: str, lines: list[str]) -> QPixmap:
    out = QPixmap(pix)
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(10, 14, 22, 226))
    p.setPen(Qt.NoPen)
    panel_h = 92 + len(lines) * 31
    p.drawRoundedRect(28, 28, 980, panel_h, 18, 18)
    p.setPen(QColor(255, 255, 255))
    p.setFont(QFont(FONT_FAMILY, 26, QFont.Bold))
    p.drawText(54, 78, title)
    p.setFont(QFont(FONT_FAMILY, 17))
    y = 116
    for line in lines:
        p.drawText(58, y, line)
        y += 31
    p.end()
    return out


def grab_window(window: MainWindow, path: Path, title: str = "", lines: list[str] | None = None) -> None:
    window.resize(WIDTH, HEIGHT)
    window.show()
    app = QApplication.instance()
    for _ in range(12):
        app.processEvents()
    pix = window.grab()
    pix.save(str(path))


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


def set_visible_gear(window: MainWindow, gear_id: int) -> None:
    gear_btns = getattr(window, "gear_btns", {}) or {}
    for btn_id, lbl in gear_btns.items():
        if int(btn_id) == int(gear_id):
            lbl.setStyleSheet(
                "background-color: #ffc107; color: #111216; border: 1px solid #ffc107; "
                "border-radius: 6px; padding: 7px; font-weight: 900;"
            )
        else:
            lbl.setStyleSheet(
                "background-color: #181a21; color: #b8bdc9; border: 1px solid #343742; "
                "border-radius: 6px; padding: 7px;"
            )


def feed_telemetry_history(window: MainWindow, final_state, gear_id: int) -> None:
    speed = float(getattr(final_state, "speed", 0.0) or 0.0)
    accel = float(getattr(final_state, "accel", 0.0) or 0.0)
    brake = float(getattr(final_state, "brake", 0.0) or 0.0)
    for idx in range(60):
        t = idx / 59.0
        if speed > 0:
            sample_speed = max(0.0, speed * (0.45 + 0.55 * t))
        else:
            sample_speed = max(0.0, 12.0 * (1.0 - t))
        sample_accel = accel if idx > 30 else accel * (idx / 30.0)
        sample_brake = brake if speed == 0 else 0.0
        sample = type(
            "VehicleState",
            (),
            {
                "speed": sample_speed,
                "accel": sample_accel,
                "brake": sample_brake,
                "angle": float(getattr(final_state, "angle", 0.0) or 0.0),
                "target_angle": float(getattr(final_state, "target_angle", 0.0) or 0.0),
                "gear": gear_id,
            },
        )()
        safe_call(window, "update_dashboard", sample)
    safe_call(window, "update_dashboard", final_state)
    for curve_name, values_name in (("curve_speed", "y_speed"), ("curve_accel", "y_accel")):
        curve = getattr(window, curve_name, None)
        values = getattr(window, values_name, None)
        if curve is not None and values is not None:
            try:
                curve.setData(values)
            except Exception:
                pass


def apply_common_route(window: MainWindow, routes: list[dict]) -> None:
    safe_call(window, "set_route_queue", routes)
    safe_call(window, "set_active_route", routes[0], 0, len(routes), "prepared", "Town10HD route prepared")
    safe_call(window, "set_route_runtime_state", "prepared", "Маршрут построен. Ожидание активации управления.")


def apply_real_mission(window: MainWindow, mode: str, state: str, msg: str, speed: float, gear: str) -> None:
    safe_call(window, "set_ui_mode", "real")
    set_device(window, mode)
    safe_call(window, "set_connection_status", True, f"Connected: {mode}")
    safe_call(
        window,
        "set_real_mission_card",
        raw_mission={
            "name": "TEST_MOCK_VEHICLE route A->B",
            "goal_label": "Mock route with active telemetry",
            "target_speed": 18.0,
        },
        mode_text=mode,
    )
    safe_call(window, "set_real_readiness", route=True, pose=True, cameras=True, ai=True, vehicle=True)
    safe_call(window, "set_real_agent_status", True, "model input OK; PID input OK")
    safe_call(window, "set_real_vehicle_state", state, msg)
    install_mock_camera_frames(window, "pip")
    gear_id = {"P": 1, "R": 2, "N": 3, "D": 4, "E": 5, "B": 6}.get(str(gear).upper(), 3)
    set_visible_gear(window, gear_id)
    fake_state = type(
        "VehicleState",
        (),
        {
            "speed": speed,
            "accel": 28.0 if state == "AI_ACTIVE" else 0.0,
            "brake": 0.0 if state != "DISENGAGING" else 75.0,
            "angle": 3.2,
            "target_angle": 3.5,
            "gear": gear_id,
        },
    )()
    feed_telemetry_history(window, fake_state, gear_id)
    if hasattr(window, "route_preview"):
        window.route_preview.update_vehicle_state(
            fake_state,
            control_active=state in {"AI_ACTIVE", "MANUAL_ACTIVE", "DISENGAGING"},
            ai_active=state == "AI_ACTIVE",
        )
    safe_call(window, "_append_log", f"{state}: speed={speed:.1f} km/h gear={gear} command source captured")


def render_terminal_image(text: str, path: Path) -> None:
    img = QImage(1500, 900, QImage.Format_RGB32)
    img.fill(QColor(8, 10, 12))
    p = QPainter(img)
    p.setPen(QColor(230, 238, 245))
    p.setFont(QFont(FONT_FAMILY, 18))
    lines = text.splitlines()
    ran_line = next((line for line in reversed(lines) if line.startswith("Ran ")), "")
    ok_line = "OK" if any(line.strip() == "OK" for line in lines) else ""
    selected = []
    for line in lines:
        if (
            line.startswith("test_")
            or line.startswith("Ran ")
            or line.strip() == "OK"
            or "passed" in line
            or line.startswith("$")
        ):
            selected.append(line)
    if len(selected) < 12:
        selected = lines[-28:]
    selected = selected[-34:]
    y = 42
    p.drawText(28, y, "$ python -m unittest discover -s tests -p test_*.py -v")
    y += 38
    if ran_line:
        p.setPen(QColor(115, 190, 255))
        p.setFont(QFont(FONT_FAMILY, 24, QFont.Bold))
        p.drawText(28, y, f"RESULT: {ran_line}  {ok_line}".rstrip())
        y += 46
    p.setFont(QFont(FONT_FAMILY, 17))
    for line in selected:
        color = QColor(140, 245, 165) if line.strip() in {"OK"} or "passed" in line else QColor(230, 238, 245)
        if line.startswith("Ran "):
            color = QColor(115, 190, 255)
        p.setPen(color)
        p.drawText(28, y, line[:128])
        y += 24
        if y > 870:
            break
    p.end()
    img.save(str(path))


def run_unittest_capture() -> str:
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"]
    completed = subprocess.run(
        cmd,
        cwd=str(GUI_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
    )
    output = completed.stdout
    (OUT_DIR / "autotests_unittest_output.txt").write_text(output, encoding="utf-8")
    render_terminal_image(output, OUT_DIR / "01_autotests_unittest_terminal.png")
    if completed.returncode != 0:
        raise SystemExit(output)
    return output


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    install_fonts(app)
    test_output = run_unittest_capture()

    window = MainWindow()
    window.resize(WIDTH, HEIGHT)

    routes = [route_payload("route_001816"), route_payload("route_001817")]

    apply_real_mission(window, "TEST_MOCK_VEHICLE", "READY", "mock adapter connected; telemetry active", 16.2, "D")
    apply_common_route(window, [routes[0]])
    grab_window(
        window,
        OUT_DIR / "02_mock_vehicle_route_telemetry.png",
        "Mock-режим: TEST_MOCK_VEHICLE",
        ["Выбран тестовый автомобиль, маршрут построен", "Телеметрия активна: скорость, угол, accel/brake идут в UI"],
    )

    apply_real_mission(window, "TEST_MOCK_VEHICLE", "READY", "AI Preview analyses model input; commands disabled", 12.8, "D")
    safe_call(window, "set_ai_checkbox", True)
    safe_call(window, "set_real_authority_mode", "off", "AI Preview включен, Activate Control не нажата")
    grab_window(
        window,
        OUT_DIR / "03_ai_preview_no_activation.png",
        "AI Preview без активации",
        ["Агент анализирует вход модели", "Команды в автомобиль и PID-исполнение не передаются"],
    )

    apply_real_mission(window, "TEST_MOCK_VEHICLE", "AI_ACTIVE", "AI command source -> PID controllers", 18.4, "D")
    grab_window(
        window,
        OUT_DIR / "04_ai_active_mock.png",
        "AI_ACTIVE в программной имитации",
        ["Предусловия выполнены: route+pose+cameras+vehicle+AI", "Команды модели передаются на PID-контроллеры"],
    )

    apply_real_mission(window, "TEST_MOCK_VEHICLE", "MANUAL_ACTIVE", "manual takeover latched; keep Drive", 14.1, "D")
    safe_call(window, "set_real_authority_mode", "manual_takeover", "Ручной перехват: машина остается в Drive")
    grab_window(
        window,
        OUT_DIR / "05_manual_active_takeover.png",
        "MANUAL_ACTIVE после ручного перехвата",
        ["Перехват отключает AI-команды", "Сразу в Park не переводит: gear=D, водитель сохраняет управление"],
    )

    apply_real_mission(window, "TEST_MOCK_VEHICLE", "DISENGAGING", "safe stop requested; brake first", 0.0, "P")
    safe_call(window, "set_real_authority_mode", "off", "Safe stop completed; gear=P")
    grab_window(
        window,
        OUT_DIR / "06_disengaging_park_after_stop.png",
        "DISENGAGING / Park after stop",
        ["Безопасное завершение: торможение до нуля", "После остановки допускается Park"],
    )

    apply_real_mission(window, "TEST_SERIAL_LOOPBACK", "READY", "dry-run guard active; physical commands disabled", 7.5, "D")
    safe_call(window, "_append_log", "COM dry-run: host_dry_run_guard replaced command; active=False")
    grab_window(
        window,
        OUT_DIR / "07_com_dry_run_loopback.png",
        "COM dry-run / loopback",
        ["TEST_SERIAL_LOOPBACK выбран вместо физического COM-исполнения", "Лог фиксирует: active command disabled, MCU не получает реальный привод"],
    )

    safe_call(window, "set_ui_mode", "carla")
    set_device(window, "VIRTUAL_DEMO_MODE")
    safe_call(window, "set_connection_status", True, "CARLA RPC OK: Town10HD")
    safe_call(window, "set_route_queue", routes)
    safe_call(window, "set_active_route", routes[0], 0, 2, "running", "route_001816.xml running")
    safe_call(window, "set_route_queue_position", 1, 2)
    safe_call(window, "_append_log", "CARLA queue transition planned: route_001816.xml -> route_001817.xml")
    grab_window(
        window,
        OUT_DIR / "08_carla_route_queue_town10.png",
        "CARLA route queue: Town10HD",
        ["Очередь из двух маршрутов построена", "Переход проверен тестом: route_001816.xml -> route_001817.xml"],
    )

    frame = make_carla_frame("Town10HD / route_001816", "agent=LeadAgent  control=AI_ACTIVE")
    safe_call(window, "update_camera_frame", frame)
    safe_call(window, "set_route_runtime_state", "running", "CARLA video + telemetry smoke frame")
    safe_call(window, "_append_log", "CARLA telemetry: speed=18.4 km/h status=RUNNING progress=37%")
    grab_window(
        window,
        OUT_DIR / "09_carla_video_telemetry_town10.png",
        "CARLA-видеопоток и телеметрия",
        ["Town10HD загружен; видеопанель получает кадр", "На скрине показаны скорость, статус и прогресс маршрута"],
    )

    safe_call(window, "set_ui_mode", "real")
    set_device(window, "COM4")
    safe_call(window, "set_connection_status", True, "COM4 open; waiting MCU heartbeat")
    safe_call(window, "set_real_readiness", route=False, pose=False, cameras=False, ai=True, vehicle=False)
    safe_call(window, "set_real_agent_status", False, "blocked: no route, no pose, heartbeat/cameras not ready")
    safe_call(window, "set_real_vehicle_state", "BLOCKED", "activation blocked by readiness matrix")
    safe_call(window, "_append_log", "ACTIVATION BLOCKED: no route; no pose; heartbeat=false; cameras=false")
    grab_window(
        window,
        OUT_DIR / "10_activation_blocked.png",
        "Ошибочная готовность / activation blocked",
        ["Активация запрещена readiness-матрицей", "Причины: нет маршрута, pose, heartbeat или камеры не готовы"],
    )

    summary = {
        "output_dir": str(OUT_DIR),
        "screenshots": sorted(p.name for p in OUT_DIR.glob("*.png")),
        "unittest_tail": "\n".join(test_output.splitlines()[-6:]),
    }
    (OUT_DIR / "screenshot_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
