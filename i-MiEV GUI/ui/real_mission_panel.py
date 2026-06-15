from __future__ import annotations

import json
import os

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QFrame,
    QSizePolicy,
    QToolButton,
    QMenu,
    QWidget,
    QWidgetAction,
    QLineEdit,
    QTextEdit,
)


try:
    from ui.marquee_label import ScrollingLabel, MarqueeButton
except Exception:  # pragma: no cover
    ScrollingLabel = QLabel
    MarqueeButton = QPushButton

try:
    from ui.map_picker_dialog import MapPickerDialog
except Exception:  # pragma: no cover
    MapPickerDialog = None

try:
    from vehicle_control.geo import GeoPoint, geo_points_to_local_ab
except Exception:  # pragma: no cover
    GeoPoint = None
    geo_points_to_local_ab = None


class RealMissionPanel(QGroupBox):
    """Compact navigator panel for REAL/MOCK vehicle modes.

    The visible row stays the same height as the old Route Launcher. Coordinate
    route settings live in the Details drop-down, so the main layout does not
    shift when switching between CARLA and real vehicle testing.
    """

    mission_validated = Signal(dict)
    mission_record_requested = Signal()

    FIXED_PANEL_HEIGHT = 104

    def __init__(self, parent=None):
        super().__init__("Навигатор / миссия", parent)
        self._mission_ready = False
        self._last_state = "idle"
        self._last_message = ""
        self._map_start_geo = None
        self._map_goal_geo = None
        self._setup_ui()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(self.FIXED_PANEL_HEIGHT)

    def _setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 18, 8, 8)
        root.setSpacing(8)

        root.addWidget(QLabel("Маршрут"))
        self.combo_mission = QComboBox()
        self.combo_mission.addItem("A → B по координатам", {
            "mission_id": "ab_route_ui",
            "name": "Маршрут A → B",
            "goal_label": "Точка B",
            "mode": "ab",
            "speed_cap_kmh": 3.0,
        })
        self.combo_mission.addItem("Тестовый круг 01", {
            "mission_id": "test_loop_01",
            "name": "Тестовый круг 01",
            "goal_label": "Локальный тестовый круг",
            "mode": "default_test",
            "speed_cap_kmh": 3.0,
        })
        self.combo_mission.addItem("Тест удержания полосы", {
            "mission_id": "lane_follow_test",
            "name": "Тест удержания полосы",
            "goal_label": "Только видимая полоса",
            "mode": "lane_follow",
            "speed_cap_kmh": 1.0,
        })
        self.combo_mission.currentIndexChanged.connect(self._sync_preview_from_controls)
        root.addWidget(self.combo_mission, stretch=2)

        self.btn_map_picker_main = MarqueeButton("Карта и маршрут")
        self.btn_map_picker_main.clicked.connect(self._open_map_picker)
        root.addWidget(self.btn_map_picker_main)

        self.btn_validate = MarqueeButton("Проверить маршрут")
        self.btn_validate.setObjectName("PrimaryButton")
        self.btn_validate.clicked.connect(self._emit_validated)
        root.addWidget(self.btn_validate)

        # Readiness used to occupy the space between "Проверить маршрут" and
        # "Настройки". It is now kept as a hidden state holder/tooltip so the
        # production row stays visually clean and fixed-width.
        self.readiness = ScrollingLabel("Маршрут --  Поза --  Камеры --  ИИ --  Авто --", self)
        self.readiness.setVisible(False)

        self.btn_details = MarqueeButton("Настройки")
        self.btn_details.setMenu(self._build_details_menu())
        root.addWidget(self.btn_details)
        self._sync_preview_from_controls()

    def _build_details_menu(self):
        menu = QMenu(self)
        menu.setObjectName("MissionDetailsMenu")
        holder = QWidget(menu)
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        info = QGridLayout()
        info.setHorizontalSpacing(8)
        info.setVerticalSpacing(6)
        self.lbl_goal = self._value_box("Цель", "Точка B")
        self.lbl_next = self._value_box("Далее", "не проверено")
        self.lbl_maneuver = self._value_box("Маневр", "ожидание")
        info.addWidget(self.lbl_goal, 0, 0)
        info.addWidget(self.lbl_next, 0, 1)
        info.addWidget(self.lbl_maneuver, 1, 0)
        layout.addLayout(info)

        coord_grid = QGridLayout()
        coord_grid.setHorizontalSpacing(8)
        coord_grid.setVerticalSpacing(6)
        coord_grid.addWidget(QLabel("Старт A x,y,yaw"), 0, 0)
        self.input_start = QLineEdit("0,0,0")
        self.input_start.setPlaceholderText("0,0,0")
        coord_grid.addWidget(self.input_start, 0, 1)
        coord_grid.addWidget(QLabel("Финиш B x,y,yaw"), 1, 0)
        self.input_goal = QLineEdit("22,8,20")
        self.input_goal.setPlaceholderText("22,8,20")
        coord_grid.addWidget(self.input_goal, 1, 1)
        coord_grid.addWidget(QLabel("Шаг waypoint, м"), 2, 0)
        self.input_spacing = QLineEdit("2.0")
        coord_grid.addWidget(self.input_spacing, 2, 1)
        coord_grid.addWidget(QLabel("Построение"), 3, 0)
        self.combo_routing_provider = QComboBox()
        self.combo_routing_provider.addItem("Прямая A→B", "direct")
        self.combo_routing_provider.addItem("По дорогам OpenStreetMap/OSRM", "osrm")
        coord_grid.addWidget(self.combo_routing_provider, 3, 1)
        coord_grid.addWidget(QLabel("Траектория"), 4, 0)
        self.combo_lane_policy = QComboBox()
        self.combo_lane_policy.addItem("По нужной стороне дороги", "right_side")
        self.combo_lane_policy.addItem("По центру линии OSM", "centerline")
        coord_grid.addWidget(self.combo_lane_policy, 4, 1)
        coord_grid.addWidget(QLabel("Смещение от центра, м"), 5, 0)
        self.input_lane_offset = QLineEdit("1.7")
        self.input_lane_offset.setPlaceholderText("1.5–2.0")
        coord_grid.addWidget(self.input_lane_offset, 5, 1)
        coord_grid.addWidget(QLabel("Pose автомобиля"), 6, 0)
        self.combo_pose_mode = QComboBox()
        self.combo_pose_mode.addItem("GPS / внешний pose", "external")
        self.combo_pose_mode.addItem("GPS NMEA-0183 через COM", "nmea_serial")
        self.combo_pose_mode.addItem("Без GPS: A→B по скорости", "dead_reckoning_ab")
        coord_grid.addWidget(self.combo_pose_mode, 6, 1)
        coord_grid.addWidget(QLabel("GPS COM-порт"), 7, 0)
        self.input_gps_com_port = QLineEdit(os.environ.get("MITSU_GPS_COM_PORT", ""))
        self.input_gps_com_port.setPlaceholderText("например COM7")
        coord_grid.addWidget(self.input_gps_com_port, 7, 1)
        coord_grid.addWidget(QLabel("GPS baudrate"), 8, 0)
        self.input_gps_baudrate = QLineEdit(os.environ.get("MITSU_GPS_BAUDRATE", "115200"))
        coord_grid.addWidget(self.input_gps_baudrate, 8, 1)
        layout.addLayout(coord_grid)

        map_row = QHBoxLayout()
        self.btn_map_picker = MarqueeButton("Выбрать на карте")
        self.btn_map_picker.clicked.connect(self._open_map_picker)
        map_row.addWidget(self.btn_map_picker)
        self.lbl_map_points = ScrollingLabel("Точки A/B: не заданы")
        self.lbl_map_points.setObjectName("MutedText")
        self.lbl_map_points.setWordWrap(False)
        map_row.addWidget(self.lbl_map_points, stretch=1)
        layout.addLayout(map_row)

        self.hints_json = QTextEdit()
        self.hints_json.setPlaceholderText("Дополнительные точки JSON: [{\"x_m\": 8, \"y_m\": 0, \"command\": \"turn_left\"}]")
        self.hints_json.setFixedHeight(74)
        self.hints_json.setPlainText('[{"x_m": 8.0, "y_m": 0.0, "command": "straight"}, {"x_m": 12.0, "y_m": 2.5, "command": "turn_left"}]')
        layout.addWidget(self.hints_json)

        self.lbl_runtime = ScrollingLabel("Состояние: ожидание")
        self.lbl_runtime.setObjectName("MutedText")
        self.lbl_runtime.setWordWrap(False)
        layout.addWidget(self.lbl_runtime)

        buttons = QHBoxLayout()
        self.btn_record = MarqueeButton("Записать")
        self.btn_record.clicked.connect(self.mission_record_requested.emit)
        buttons.addWidget(self.btn_record)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        action = QWidgetAction(menu)
        action.setDefaultWidget(holder)
        menu.addAction(action)
        return menu

    def _value_box(self, title, value):
        box = QFrame()
        box.setObjectName("StatBox")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 8, 10, 8)
        title_lbl = QLabel(str(title))
        title_lbl.setObjectName("MutedText")
        value_lbl = ScrollingLabel(str(value))
        value_lbl.setObjectName("StatValue")
        layout.addWidget(title_lbl)
        layout.addWidget(value_lbl)
        box.value_label = value_lbl
        return box

    @staticmethod
    def _parse_xyz(text):
        parts = [p.strip() for p in str(text or "").split(",") if p.strip()]
        values = [float(p) for p in parts]
        while len(values) < 3:
            values.append(0.0)
        return values[0], values[1], values[2]

    def _open_map_picker(self):
        if MapPickerDialog is None:
            self.lbl_map_points.setText("Карта недоступна: не удалось импортировать ui.map_picker_dialog")
            return
        dialog = MapPickerDialog(self, start=self._map_start_geo, goal=self._map_goal_geo)
        dialog.points_selected.connect(self._apply_map_points)
        dialog.exec()

    def _apply_map_points(self, payload):
        if not isinstance(payload, dict):
            return
        start = payload.get("start_geo")
        goal = payload.get("goal_geo")
        if not start or not goal:
            return
        self._map_start_geo = dict(start)
        self._map_goal_geo = dict(goal)
        try:
            local = geo_points_to_local_ab(GeoPoint.from_dict(start), GeoPoint.from_dict(goal)) if GeoPoint and geo_points_to_local_ab else None
            if local:
                s = local["start"]
                g = local["goal"]
                self.input_start.setText(f"{s['x_m']:.2f},{s['y_m']:.2f},{s['yaw_deg']:.1f}")
                self.input_goal.setText(f"{g['x_m']:.2f},{g['y_m']:.2f},{g['yaw_deg']:.1f}")
        except Exception:
            pass
        self.lbl_map_points.setText(
            f"A=({float(start['lat']):.6f},{float(start['lon']):.6f})  "
            f"B=({float(goal['lat']):.6f},{float(goal['lon']):.6f})"
        )
        self._sync_preview_from_controls()

    def _current_mission(self):
        data = dict(self.combo_mission.currentData() or {})
        speed = float(data.get("speed_cap_kmh", 3.0))
        data["speed_cap_kmh"] = speed
        if data.get("mode") == "ab":
            sx, sy, syaw = self._parse_xyz(self.input_start.text())
            gx, gy, gyaw = self._parse_xyz(self.input_goal.text())
            try:
                hints = json.loads(self.hints_json.toPlainText().strip() or "[]")
                if not isinstance(hints, list):
                    hints = []
            except Exception:
                hints = []
            start_payload = {"x_m": sx, "y_m": sy, "yaw_deg": syaw}
            goal_payload = {"x_m": gx, "y_m": gy, "yaw_deg": gyaw}
            lane_policy = self.combo_lane_policy.currentData() if hasattr(self, "combo_lane_policy") else "right_side"
            try:
                lane_offset_m = float(str(self.input_lane_offset.text() or "1.7").replace(",", "."))
            except Exception:
                lane_offset_m = 1.7
            metadata_payload = {
                "lane_policy": lane_policy,
                "lane_offset_m": lane_offset_m,
                "traffic_side": "right",
            }
            pose_mode = self.combo_pose_mode.currentData() if hasattr(self, "combo_pose_mode") else "external"
            gps_com_port = self.input_gps_com_port.text().strip() if hasattr(self, "input_gps_com_port") else ""
            try:
                gps_baudrate = int(self.input_gps_baudrate.text().strip() or 115200)
            except Exception:
                gps_baudrate = 115200
            goal_label = f"B=({gx:.1f}, {gy:.1f})"

            if self._map_start_geo and self._map_goal_geo and GeoPoint is not None and geo_points_to_local_ab is not None:
                local = geo_points_to_local_ab(
                    GeoPoint.from_dict(self._map_start_geo),
                    GeoPoint.from_dict(self._map_goal_geo),
                )
                start_payload = local["start"]
                goal_payload = local["goal"]
                metadata_payload.update({
                    "origin_geo": local["origin_geo"],
                    "start_geo": local["start_geo"],
                    "goal_geo": local["goal_geo"],
                    "coordinate_frame": "wgs84_local_tangent",
                    "routing_provider": self.combo_routing_provider.currentData() if hasattr(self, "combo_routing_provider") else "direct",
                })
                goal_label = f"B geo=({self._map_goal_geo['lat']:.6f}, {self._map_goal_geo['lon']:.6f})"

            data.update({
                "mission_id": "ab_route_ui",
                "name": "Маршрут A → B",
                "start": start_payload,
                "goal": goal_payload,
                "goal_label": goal_label,
                "pose_mode": pose_mode,
                "gps_com_port": gps_com_port,
                "gps_baudrate": gps_baudrate,
                "spacing_m": float(self.input_spacing.text() or 2.0),
                "turn_speed_kmh": speed,
                "hints": hints,
                "metadata": metadata_payload,
            })
        return data

    def _sync_preview_from_controls(self):
        mission = self._current_mission()
        goal = str(mission.get("goal_label", "-"))
        if hasattr(self, "lbl_goal"):
            self.lbl_goal.value_label.setText(goal)

    def _emit_validated(self):
        mission = self._current_mission()
        self._mission_ready = True
        self.lbl_goal.value_label.setText(str(mission.get("goal_label", "-")))
        if mission.get("start") and mission.get("goal"):
            self.lbl_next.value_label.setText(f"A→B шаг {mission.get('spacing_m', 2.0)}м")
            self.lbl_maneuver.value_label.setText("координатный маршрут")
        else:
            self.lbl_next.value_label.setText("WP 1 / test")
            self.lbl_maneuver.value_label.setText("ready")
        self.set_readiness(route=True, pose=True, cameras=True, ai=False, vehicle=False)
        self.mission_validated.emit(mission)

    def set_readiness(self, route=False, pose=False, cameras=False, ai=False, vehicle=False):
        def mark(ok):
            return "OK" if ok else "--"
        text = f"Маршрут {mark(route)}  Поза {mark(pose)}  Камеры {mark(cameras)}  ИИ {mark(ai)}  Авто {mark(vehicle)}"
        self.readiness.setText(text)
        self.setToolTip(text)
        if hasattr(self, "btn_validate"):
            self.btn_validate.setToolTip(text)

    def set_mission_summary(self, mission):
        try:
            count = len(getattr(mission, "waypoints", []) or [])
            goal = getattr(mission, "goal_label", "-") or "-"
            self.lbl_goal.value_label.setText(str(goal))
            self.lbl_next.value_label.setText(f"{count} контрольных точек")
            if count:
                last = mission.waypoints[-1]
                self.lbl_maneuver.value_label.setText(str(getattr(last, "command", getattr(last, "action", "ready"))))
        except Exception:
            pass

    def set_nav_goal(self, goal):
        try:
            self.lbl_next.value_label.setText(f"WP {goal.waypoint_index} · {goal.distance_to_target_m:.1f}m")
            self.lbl_maneuver.value_label.setText(str(goal.maneuver))
            self.lbl_runtime.setText(
                f"Состояние: {self._last_state}\n"
                f"target=({goal.target_x_m:.1f}, {goal.target_y_m:.1f}) "
                f"heading_err={goal.heading_error_deg:.1f}° "
                f"xtrack={goal.cross_track_error_m:.2f}m"
            )
        except Exception:
            pass

    def set_runtime_status(self, state: str, message: str = ""):
        state = str(state or "idle")
        self._last_state = state
        self._last_message = str(message or "")
        if state == "AI_ACTIVE":
            self.lbl_maneuver.value_label.setText("ИИ активно")
        elif state == "ARMING":
            self.lbl_maneuver.value_label.setText("подготовка")
        elif state == "DISENGAGING":
            self.lbl_maneuver.value_label.setText("остановка")
        elif state == "FAULT":
            self.lbl_maneuver.value_label.setText("ошибка")
        elif state == "CONNECTED_MANUAL":
            self.lbl_maneuver.value_label.setText("ручной режим")
        if message:
            self.lbl_next.value_label.setText(str(message)[:44])
            self.lbl_runtime.setText(f"Состояние: {state} · {message}")
        else:
            self.lbl_runtime.setText(f"Состояние: {state}")
