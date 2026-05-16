from __future__ import annotations

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
)


class RealMissionPanel(QGroupBox):
    """Compact navigator panel for REAL/MOCK vehicle modes.

    The main window keeps the same vertical footprint as the previous CARLA
    Route Launcher. Extra navigator/status controls are exposed through a
    drop-down details menu, so the center layout does not jump when the user
    switches from VIRTUAL_DEMO_MODE to TEST_MOCK_VEHICLE/real COM mode.
    """

    mission_validated = Signal(dict)
    mission_record_requested = Signal()
    speed_cap_changed = Signal(float)

    FIXED_PANEL_HEIGHT = 104

    def __init__(self, parent=None):
        super().__init__("Navigator / Mission", parent)
        self._mission_ready = False
        self._last_state = "idle"
        self._last_message = ""
        self._setup_ui()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(self.FIXED_PANEL_HEIGHT)

    def _setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 18, 8, 8)
        root.setSpacing(8)

        root.addWidget(QLabel("Mission"))

        self.combo_mission = QComboBox()
        self.combo_mission.addItem("Test Loop 01", {
            "mission_id": "test_loop_01",
            "name": "Test Loop 01",
            "goal_label": "Local test loop",
            "speed_cap_kmh": 3.0,
        })
        self.combo_mission.addItem("Lane Follow Test", {
            "mission_id": "lane_follow_test",
            "name": "Lane Follow Test",
            "goal_label": "Visible lane only",
            "speed_cap_kmh": 1.0,
        })
        self.combo_mission.currentIndexChanged.connect(self._sync_preview_from_controls)
        root.addWidget(self.combo_mission, stretch=2)

        root.addWidget(QLabel("Speed"))
        self.speed_cap_combo = QComboBox()
        for value in (1.0, 3.0, 5.0, 10.0):
            self.speed_cap_combo.addItem(f"{value:.0f} km/h", value)
        self.speed_cap_combo.currentIndexChanged.connect(self._on_speed_cap_changed)
        root.addWidget(self.speed_cap_combo)

        self.btn_validate = QPushButton("Validate")
        self.btn_validate.setObjectName("PrimaryButton")
        self.btn_validate.clicked.connect(self._emit_validated)
        root.addWidget(self.btn_validate)

        self.readiness = QLabel("Route --  Pose --  Cam --  AI --  Veh --")
        self.readiness.setObjectName("MutedText")
        self.readiness.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        root.addWidget(self.readiness, stretch=2)

        self.btn_details = QToolButton()
        self.btn_details.setText("Details")
        self.btn_details.setPopupMode(QToolButton.InstantPopup)
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
        self.lbl_goal = self._value_box("Goal", "Local test loop")
        self.lbl_next = self._value_box("Next", "not validated")
        self.lbl_maneuver = self._value_box("Maneuver", "idle")
        self.lbl_speed_cap = self._value_box("Speed cap", "3.0 km/h")
        info.addWidget(self.lbl_goal, 0, 0)
        info.addWidget(self.lbl_next, 0, 1)
        info.addWidget(self.lbl_maneuver, 1, 0)
        info.addWidget(self.lbl_speed_cap, 1, 1)
        layout.addLayout(info)

        self.lbl_runtime = QLabel("State: idle")
        self.lbl_runtime.setObjectName("MutedText")
        self.lbl_runtime.setWordWrap(True)
        layout.addWidget(self.lbl_runtime)

        buttons = QHBoxLayout()
        self.btn_record = QPushButton("Record")
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
        value_lbl = QLabel(str(value))
        value_lbl.setObjectName("StatValue")
        layout.addWidget(title_lbl)
        layout.addWidget(value_lbl)
        box.value_label = value_lbl
        return box

    def _current_mission(self):
        data = self.combo_mission.currentData() or {}
        data = dict(data)
        data["speed_cap_kmh"] = float(self.speed_cap_combo.currentData() or data.get("speed_cap_kmh", 3.0))
        return data

    def _sync_preview_from_controls(self):
        mission = self._current_mission()
        goal = str(mission.get("goal_label", "-"))
        speed = float(mission.get("speed_cap_kmh", 3.0))
        if hasattr(self, "lbl_goal"):
            self.lbl_goal.value_label.setText(goal)
        if hasattr(self, "lbl_speed_cap"):
            self.lbl_speed_cap.value_label.setText(f"{speed:.1f} km/h")

    def _emit_validated(self):
        mission = self._current_mission()
        self._mission_ready = True
        self.lbl_goal.value_label.setText(str(mission.get("goal_label", "-")))
        self.lbl_next.value_label.setText("WP 1 / test")
        self.lbl_maneuver.value_label.setText("ready")
        self.lbl_speed_cap.value_label.setText(f"{float(mission.get('speed_cap_kmh', 3.0)):.1f} km/h")
        self.set_readiness(route=True, pose=True, cameras=True, ai=False, vehicle=False)
        self.mission_validated.emit(mission)

    def _on_speed_cap_changed(self):
        value = float(self.speed_cap_combo.currentData() or 3.0)
        self.lbl_speed_cap.value_label.setText(f"{value:.1f} km/h")
        self.speed_cap_changed.emit(value)

    def set_readiness(self, route=False, pose=False, cameras=False, ai=False, vehicle=False):
        def mark(ok):
            return "OK" if ok else "--"
        self.readiness.setText(
            f"Route {mark(route)}  Pose {mark(pose)}  Cam {mark(cameras)}  "
            f"AI {mark(ai)}  Veh {mark(vehicle)}"
        )

    def set_runtime_status(self, state: str, message: str = ""):
        state = str(state or "idle")
        self._last_state = state
        self._last_message = str(message or "")
        if state == "AI_ACTIVE":
            self.lbl_maneuver.value_label.setText("AI active")
        elif state == "ARMING":
            self.lbl_maneuver.value_label.setText("arming")
        elif state == "DISENGAGING":
            self.lbl_maneuver.value_label.setText("stopping")
        elif state == "FAULT":
            self.lbl_maneuver.value_label.setText("fault")
        elif state == "CONNECTED_MANUAL":
            self.lbl_maneuver.value_label.setText("manual")

        if message:
            self.lbl_next.value_label.setText(str(message)[:44])
            self.lbl_runtime.setText(f"State: {state}\n{message}")
        else:
            self.lbl_runtime.setText(f"State: {state}")
