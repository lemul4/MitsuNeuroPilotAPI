# ui/main_window.py

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QComboBox, QGroupBox, QGridLayout, 
    QProgressBar, QTableWidget, QTableWidgetItem, 
    QHeaderView, QDoubleSpinBox, QCheckBox
)
from PySide6.QtCore import Qt, QTimer, Signal, QEvent
from datetime import datetime
import pyqtgraph as pg

# Импортируем кастомный виджет руля (предполагается, что он в ui/widgets.py)
# Если вы не выносили его в отдельный файл, просто вставьте класс SteeringWidget сюда
from ui.widgets import SteeringWidget 

class MainWindow(QMainWindow):
    # --- Сигналы для связи с Контроллером (main.py) ---
    connect_requested = Signal(str)
    disconnect_requested = Signal()
    control_toggled = Signal(bool)
    gear_requested = Signal(int)
    pid_update_requested = Signal(float, float, float)
    ai_toggled = Signal(bool)
    telemetry_toggled = Signal(bool)
    
    # Сигнал для ручного управления (angle, accel, brake)
    manual_input_updated = Signal(int, int, int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("i-MiEV Auto Control [MVC Refactored]")
        self.resize(1200, 800)

        # Локальные переменные для перехвата клавиатуры
        self.pressed_keys = set()
        self.local_target_angle = 0
        self.local_target_accel = 0
        self.local_target_brake = 0
        self.control_active = False

        # Буферы для графиков
        self.y_speed = [0] * 100
        self.y_accel = [0] * 100

        self.setup_ui()

        # Таймер опроса клавиатуры (чтобы интерфейс плавно реагировал на удержание кнопок)
        self.key_poll_timer = QTimer()
        self.key_poll_timer.timeout.connect(self.process_held_keys)
        self.key_poll_timer.start(50)

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ================= ЛЕВАЯ КОЛОНКА (Дашборд) =================
        left_col = QVBoxLayout()
        
        self.wheel_widget = SteeringWidget()
        left_col.addWidget(self.wheel_widget, stretch=2)
        
        self.lbl_speed = QLabel("0.00 km/h")
        self.lbl_speed.setStyleSheet("font-size: 32px; font-weight: bold; color: orange;")
        self.lbl_speed.setAlignment(Qt.AlignCenter)
        left_col.addWidget(self.lbl_speed)
        
        self.lbl_angle_text = QLabel("Angle: 0°")
        left_col.addWidget(self.lbl_angle_text)

        gb_gauges = QGroupBox("Pedals")
        v_gauges = QVBoxLayout()
        self.pb_accel = QProgressBar()
        self.pb_accel.setStyleSheet("QProgressBar::chunk { background-color: green; }")
        self.pb_accel.setFormat("Accel: %v%")
        v_gauges.addWidget(self.pb_accel)
        
        self.pb_brake = QProgressBar()
        self.pb_brake.setStyleSheet("QProgressBar::chunk { background-color: red; }")
        self.pb_brake.setFormat("Brake: %v%")
        v_gauges.addWidget(self.pb_brake)
        gb_gauges.setLayout(v_gauges)
        left_col.addWidget(gb_gauges)

        gb_gears = QGroupBox("Transmission")
        grid_gears = QGridLayout()
        self.gear_btns = {}
        gears = ["P", "R", "N", "D", "E", "B"]
        for i, g in enumerate(gears):
            btn = QLabel(g)
            btn.setAlignment(Qt.AlignCenter)
            btn.setStyleSheet("border: 1px solid gray; padding: 10px; border-radius: 5px;")
            self.gear_btns[i + 1] = btn
            grid_gears.addWidget(btn, 0, i)
        gb_gears.setLayout(grid_gears)
        left_col.addWidget(gb_gears)

        self.btn_control = QPushButton("Activate Control (Space)")
        self.btn_control.setCheckable(True)
        self.btn_control.setStyleSheet("background-color: #444; color: white; padding: 10px;")
        self.btn_control.toggled.connect(self.on_control_toggled)
        left_col.addWidget(self.btn_control)

        main_layout.addLayout(left_col, stretch=1)

        # ================= СРЕДНЯЯ КОЛОНКА (Настройки) =================
        mid_col = QVBoxLayout()
        
        conn_layout = QHBoxLayout()
        self.combo_ports = QComboBox()
        import serial.tools.list_ports
        self.combo_ports.addItem("VIRTUAL_DEMO_MODE")
        for p in serial.tools.list_ports.comports():
            self.combo_ports.addItem(p.device)
        conn_layout.addWidget(self.combo_ports)
        
        self.btn_connect = QPushButton("Connect")
        self.btn_connect.clicked.connect(self.on_connect_clicked)
        conn_layout.addWidget(self.btn_connect)
        mid_col.addLayout(conn_layout)

        gb_pid = QGroupBox("PID Controller Tuning")
        form_pid = QGridLayout()
        self.spin_kp = self._create_pid_spinbox()
        self.spin_ki = self._create_pid_spinbox()
        self.spin_kd = self._create_pid_spinbox()
        
        form_pid.addWidget(QLabel("Kp:"), 0, 0); form_pid.addWidget(self.spin_kp, 0, 1)
        form_pid.addWidget(QLabel("Ki:"), 1, 0); form_pid.addWidget(self.spin_ki, 1, 1)
        form_pid.addWidget(QLabel("Kd:"), 2, 0); form_pid.addWidget(self.spin_kd, 2, 1)
        
        btn_send_pid = QPushButton("Update PID")
        btn_send_pid.clicked.connect(lambda: self.pid_update_requested.emit(
            self.spin_kp.value(), self.spin_ki.value(), self.spin_kd.value()
        ))
        form_pid.addWidget(btn_send_pid, 3, 0, 1, 2)
        gb_pid.setLayout(form_pid)
        mid_col.addWidget(gb_pid)

        gb_extra = QGroupBox("Modules")
        v_extra = QVBoxLayout()
        self.chk_telemetry = QCheckBox("Record Telemetry (CSV)")
        self.chk_telemetry.stateChanged.connect(lambda: self.telemetry_toggled.emit(self.chk_telemetry.isChecked()))
        v_extra.addWidget(self.chk_telemetry)
        
        self.chk_ai = QCheckBox("Enable AI Simulation")
        self.chk_ai.stateChanged.connect(lambda: self.ai_toggled.emit(self.chk_ai.isChecked()))
        v_extra.addWidget(self.chk_ai)
        gb_extra.setLayout(v_extra)
        mid_col.addWidget(gb_extra)

        self.table_can = QTableWidget()
        self.table_can.setColumnCount(3)
        self.table_can.setHorizontalHeaderLabels(["ID", "Data", "Time"])
        self.table_can.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        mid_col.addWidget(QLabel("CAN Monitor:"))
        mid_col.addWidget(self.table_can)

        main_layout.addLayout(mid_col, stretch=1)

        # ================= ПРАВАЯ КОЛОНКА (Графики) =================
        right_col = QVBoxLayout()
        self.plot_widget = pg.PlotWidget(title="Speed & Accel History")
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True)
        self.curve_speed = self.plot_widget.plot(pen=pg.mkPen("orange", width=2), name="Speed")
        self.curve_accel = self.plot_widget.plot(pen=pg.mkPen("g", width=2), name="Accel")
        right_col.addWidget(self.plot_widget)
        main_layout.addLayout(right_col, stretch=2)

        # Защита фокуса клавиатуры (чтобы стрелки работали всегда)
        for widget in [self.combo_ports, self.btn_connect, self.btn_control, 
                       self.spin_kp, self.spin_ki, self.spin_kd, 
                       self.chk_telemetry, self.chk_ai, self.table_can]:
            widget.setFocusPolicy(Qt.NoFocus)
            
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    def _create_pid_spinbox(self):
        spin = QDoubleSpinBox()
        spin.setRange(0, 100)
        spin.setDecimals(2)
        spin.setSingleStep(0.05)
        return spin

    # --- Обработчики UI ---
    def on_connect_clicked(self):
        if self.btn_connect.text() == "Connect":
            port = self.combo_ports.currentText()
            if port:
                self.connect_requested.emit(port)
        else:
            self.disconnect_requested.emit()

    def set_connection_status(self, is_connected, message):
        self.statusBar().showMessage(message)
        if is_connected:
            self.combo_ports.setDisabled(True)
            self.btn_connect.setText("Disconnect")
        else:
            self.combo_ports.setDisabled(False)
            self.btn_connect.setText("Connect")

    def on_control_toggled(self, checked):
        self.control_active = checked
        self.btn_control.setText("Control ACTIVE" if checked else "Activate Control (Space)")
        self.btn_control.setStyleSheet(f"background-color: {'green' if checked else '#444'}; color: white;")
        self.control_toggled.emit(checked)

    def set_ai_checkbox(self, state):
        self.chk_ai.setChecked(state)

    def update_can_table(self, can_id_str, data):
        self.table_can.insertRow(0)
        self.table_can.setItem(0, 0, QTableWidgetItem(can_id_str))
        self.table_can.setItem(0, 1, QTableWidgetItem(str(data)))
        self.table_can.setItem(0, 2, QTableWidgetItem(datetime.now().strftime("%H:%M:%S.%f")[:-3]))
        if self.table_can.rowCount() > 20:
            self.table_can.removeRow(20)

    def update_dashboard(self, vehicle_state):
        """Единственный метод, обновляющий UI данными из физической модели"""
        self.wheel_widget.set_angle(vehicle_state.angle)
        self.lbl_speed.setText(f"{vehicle_state.speed:.2f} km/h")
        self.lbl_angle_text.setText(f"Angle: {vehicle_state.angle}° (T: {vehicle_state.target_angle})")

        self.pb_accel.setValue(int(vehicle_state.accel))
        self.pb_brake.setValue(int(vehicle_state.brake))

        for k, lbl in self.gear_btns.items():
            lbl.setStyleSheet("background-color: yellow; border: 2px solid black;" if k == vehicle_state.gear else "border: 1px solid gray;")

        self.y_speed = self.y_speed[1:] + [vehicle_state.speed]
        self.y_accel = self.y_accel[1:] + [vehicle_state.accel]
        self.curve_speed.setData(self.y_speed)
        self.curve_accel.setData(self.y_accel)

    # --- Обработка клавиатуры ---
    def event(self, event):
        if event.type() == QEvent.KeyPress:
            # Разрешаем перехват только нужных кнопок, чтобы не блочить весь UI
            driving_keys = (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down, 
                            Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5, Qt.Key_6, Qt.Key_Space, Qt.Key_W, Qt.Key_S)
            if event.key() in driving_keys:
                self.keyPressEvent(event)
                return True
        return super().event(event)

    def keyPressEvent(self, event):
        if event.isAutoRepeat(): return
        
        if event.key() == Qt.Key_Space:
            self.btn_control.toggle()
            return
            
        self.pressed_keys.add(event.key())

        gear_map = {Qt.Key_1: 1, Qt.Key_2: 2, Qt.Key_3: 3, Qt.Key_4: 4, Qt.Key_5: 5, Qt.Key_6: 6}
        if event.key() in gear_map:
            self.gear_requested.emit(gear_map[event.key()])

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat(): return
        if event.key() in self.pressed_keys:
            self.pressed_keys.remove(event.key())

    def process_held_keys(self):
        if not self.control_active: return

        MAX_ANGLE = 630 # Из config.py
        changed = False

        if Qt.Key_Up in self.pressed_keys:
            self.local_target_accel = min(100, self.local_target_accel + 3)
            changed = True
        elif self.local_target_accel > 0:
            self.local_target_accel = max(0, self.local_target_accel - 4)
            changed = True

        # Можно использовать W/S или Стрелку вниз для тормоза
        if Qt.Key_Down in self.pressed_keys or Qt.Key_S in self.pressed_keys:
            self.local_target_brake = min(100, self.local_target_brake + 10)
            self.local_target_accel = 0
            changed = True
        elif self.local_target_brake > 0:
            self.local_target_brake = max(0, self.local_target_brake - 15)
            changed = True

        if Qt.Key_Left not in self.pressed_keys and Qt.Key_Right not in self.pressed_keys:
            if abs(self.local_target_angle) > 15:
                self.local_target_angle += -20 if self.local_target_angle > 0 else 20
                changed = True
            elif self.local_target_angle != 0:
                self.local_target_angle = 0
                changed = True

        if Qt.Key_Left in self.pressed_keys:
            self.local_target_angle = max(-MAX_ANGLE, self.local_target_angle - 15)
            changed = True
        elif Qt.Key_Right in self.pressed_keys:
            self.local_target_angle = min(MAX_ANGLE, self.local_target_angle + 15)
            changed = True

        if changed:
            self.manual_input_updated.emit(self.local_target_angle, self.local_target_accel, self.local_target_brake)