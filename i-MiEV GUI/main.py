import sys
import asyncio
import time
import queue
import os
from datetime import datetime

from regex import T
from lead_integration import LeadAgentThread
# GUI Imports
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QComboBox, 
                               QGroupBox, QGridLayout, QProgressBar, QTableWidget, 
                               QTableWidgetItem, QHeaderView, QDoubleSpinBox, QCheckBox, QFrame)
from PySide6.QtCore import Qt, QTimer, Slot, Signal, QThread, QObject, QEvent
from PySide6.QtGui import QPixmap, QPainter, QTransform, QColor, QFont, QPen, QBrush
from PySide6.QtCore import Slot
import numpy as np
# Plotting
import pyqtgraph as pg

# Async & Hardware
import qasync
import serial
import serial_asyncio
import serial.tools.list_ports

# Custom Modules
import utils
from core.telemetry import TelemetryRecorder
from core.ai_pilot import AiDriverSim

# --- Constants ---
MAX_ANGLE = 630
BUFFER_SIZE = 1024 # Buffer for incoming bytes

# --- Steering Wheel Widget ---
class SteeringWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.angle = 0
        self.setMinimumSize(200, 200)
        # Загружаем картинку, если есть, иначе рисуем вектор
        try:
            self.pixmap = QPixmap("steeringWheelR.png")
            self.has_image = not self.pixmap.isNull()
        except:
            self.has_image = False

    def set_angle(self, angle):
        self.angle = angle
        self.update() # Trigger paintEvent

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        center_x, center_y = w / 2, h / 2
        
        painter.translate(center_x, center_y)
        painter.rotate(self.angle)
        
        if self.has_image:
            # Масштабируем картинку под размер виджета
            s = min(w, h) * 0.8
            scaled = self.pixmap.scaled(int(s), int(s), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(-scaled.width()//2, -scaled.height()//2, scaled)
        else:
            # Fallback drawing (Vector Wheel)
            radius = min(w, h) * 0.4
            painter.setPen(QPen(Qt.black, 10))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(int(-radius), int(-radius), int(radius*2), int(radius*2))
            painter.setPen(QPen(Qt.gray, 15))
            painter.drawLine(0, 0, 0, int(-radius)) # Спица вверх
            painter.drawLine(0, 0, int(-radius*0.8), int(radius*0.6))
            painter.drawLine(0, 0, int(radius*0.8), int(radius*0.6))

# --- Worker for Serial Communication (Asyncio Wrapper) ---
class SerialManager(QObject):
    data_received = Signal(object) # Emits decoded packet
    connection_status = Signal(bool, str)
    
    def __init__(self):
        super().__init__()
        self.transport = None
        self.protocol = None
        self.running = False
        self.buffer = utils.CircularBuffer(BUFFER_SIZE)
        self.cmd_queue = asyncio.Queue()

    async def connect_serial(self, port):
        if self.running:
            self.close()
            
        try:
            # Using serial_asyncio to create a connection
            self.transport, self.protocol = await serial_asyncio.create_serial_connection(
                asyncio.get_event_loop(),
                lambda: SerialProtocol(self),
                url=port,
                baudrate=1000000
            )
            self.running = True
            self.connection_status.emit(True, f"Connected to {port}")
            asyncio.create_task(self.process_commands())
        except Exception as e:
            self.connection_status.emit(False, str(e))

    def close(self):
        self.running = False
        if self.transport:
            self.transport.close()
            self.transport = None

    def handle_data(self, data):
        # Raw bytes from protocol
        for b in data:
            self.buffer.add(b)
            if self.buffer.count >= utils.PACKET_SIZE:
                ret = self.buffer.check_buffer()
                if ret:
                    try:
                        pkt = utils.Serial_Data(ret)
                        self.data_received.emit(pkt)
                        self.buffer.remove(utils.PACKET_SIZE)
                    except Exception as e:
                        print(f"Decode error: {e}")
                else:
                    # Если буфер полон, а пакета нет - сдвигаем на 1
                    if self.buffer.count == self.buffer.size:
                         self.buffer.remove(1)

    async def process_commands(self):
        while self.running:
            try:
                # Ждем команду из очереди
                cmd = await self.cmd_queue.get()
                if self.transport:
                    self.transport.write(cmd.bytes())
                self.cmd_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Write error: {e}")

    def send_command(self, cmd_obj):
        """Безопасно кладем команду в очередь. Без await и без call_soon."""
        try:
            # put_nowait работает мгновенно и не трогает Event Loop Qt
            self.cmd_queue.put_nowait(cmd_obj)
        except Exception as e:
            print(f"[SerialManager] Ошибка очереди: {e}")

class SerialProtocol(asyncio.Protocol):
    def __init__(self, manager):
        self.manager = manager

    def connection_made(self, transport):
        self.transport = transport
        print("Port opened", transport)

    def data_received(self, data):
        self.manager.handle_data(data)

    def connection_lost(self, exc):
        print("Port closed")
        self.manager.connection_status.emit(False, "Connection Lost")

# --- Main Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("i-MiEV Auto Control [PySide6 + Asyncio]")
        self.resize(1200, 800)

        
        
        self.agent_thread = None
        
        # State Variables
        self.speed = 0
        self.angle = 0
        self.accel = 0
        self.brake = 0
        self.gear = 1 # P=1
        self.is_virtual = False
        self.target_angle = 0
        self.target_accel = 0
        self.target_brake = 0
        self.target_gear = 1
        self.control_active = False
        self.physics_timer = QTimer()
        self.physics_timer.timeout.connect(self.update_physics_sim)
        self.key_check_timer = QTimer()
        self.key_check_timer.timeout.connect(self.process_held_keys)
        self.key_check_timer.start(50)
        self.pressed_keys = set()
        # Init Helpers
        self.serial_manager = SerialManager()
        self.recorder = TelemetryRecorder()
        self.ai = AiDriverSim()
        
        # UI Setup
        self.setup_ui()
        self.setup_connections()
        
        # Plot Data Buffers
        self.x_data = list(range(100))
        self.y_speed = [0]*100
        self.y_accel = [0]*100
        
        # Timers
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui_periodic)
        self.update_timer.start(50) # 20 FPS UI updates

        # AI Timer
        self.ai_timer = QTimer()
        self.ai_timer.timeout.connect(self.process_ai)
        
        # Init Commands (Pre-build objects to reuse ID/Struct)
        self.init_commands()

    def event(self, event):
        if event.type() == QEvent.KeyPress:
            # Добавляем цифры 1-6 в список "разрешенных" для ручного перехвата
            driving_keys = (
                Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down,
                Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5, Qt.Key_6,
                Qt.Key_Space
            )
            if event.key() in driving_keys:
                self.keyPressEvent(event)
                return True 
        return super().event(event)
    
    def init_commands(self):
        # Create base command objects like in original code
        self.cmd_gear = utils.Serial_Data(bytearray.fromhex("AA 00000000 3300 00 02 01 00 00 00 01 00 00"))
        self.cmd_accel = utils.Serial_Data(bytearray.fromhex("AA 00000000 3800 00 02 01 00 00 00 01 00 00"))
        self.cmd_brake = utils.Serial_Data(bytearray.fromhex("AA 00000000 3700 00 02 01 00 00 00 01 00 00"))
        self.cmd_angle = utils.Serial_Data(bytearray.fromhex("AA 00000000 3200 00 02 01 00 00 00 01 00 00"))
        self.cmd_cruise = utils.Serial_Data(bytearray.fromhex("AA 00000000 7700 00 02 01 00 00 00 01 00 00"))
        self.cmd_pid = utils.Serial_Data(bytearray.fromhex("AA 00000000 7800 00 02 01 00 00 00 01 00 00")) # Generic for PID

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # === LEFT COLUMN: Visuals ===
        left_col = QVBoxLayout()
        
        # Steering Wheel
        self.wheel_widget = SteeringWidget()
        left_col.addWidget(self.wheel_widget, stretch=2)
        
        # Info Labels
        self.lbl_speed = QLabel("0 km/h")
        self.lbl_speed.setStyleSheet("font-size: 32px; font-weight: bold; color: orange;")
        self.lbl_speed.setAlignment(Qt.AlignCenter)
        left_col.addWidget(self.lbl_speed)
        
        self.lbl_angle_text = QLabel("Angle: 0°")
        left_col.addWidget(self.lbl_angle_text)

        # Gauges (Progress Bars)
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
        
        # Gears
        gb_gears = QGroupBox("Transmission")
        grid_gears = QGridLayout()
        self.gear_btns = {}
        gears = ["P", "R", "N", "D", "E", "B"]
        for i, g in enumerate(gears):
            btn = QLabel(g) # Using Label as indicator, keys control it
            btn.setAlignment(Qt.AlignCenter)
            btn.setStyleSheet("border: 1px solid gray; padding: 10px; border-radius: 5px;")
            self.gear_btns[i+1] = btn
            grid_gears.addWidget(btn, 0, i)
        gb_gears.setLayout(grid_gears)
        left_col.addWidget(gb_gears)

        # Control Button
        self.btn_control = QPushButton("Activate Control (Space)")
        self.btn_control.setCheckable(True)
        self.btn_control.setStyleSheet("background-color: #444; color: white; padding: 10px;")
        self.btn_control.toggled.connect(self.toggle_control)
        left_col.addWidget(self.btn_control)

        main_layout.addLayout(left_col, stretch=1)

        # === MIDDLE COLUMN: Settings & CAN ===
        mid_col = QVBoxLayout()
        
        # Connection
        conn_layout = QHBoxLayout()
        self.combo_ports = QComboBox()
        self.refresh_ports()
        conn_layout.addWidget(self.combo_ports)
        
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self.refresh_ports)
        conn_layout.addWidget(btn_refresh)
        
        self.btn_connect = QPushButton("Connect") # Сделали self, чтобы менять текст
        self.btn_connect.clicked.connect(self.on_connect_click)
        conn_layout.addWidget(self.btn_connect)
        mid_col.addLayout(conn_layout)

        # PID Tuning
        gb_pid = QGroupBox("PID Controller Tuning")
        form_pid = QGridLayout()
        self.spin_kp = QDoubleSpinBox(); self.spin_kp.setRange(0, 100); self.spin_kp.setSingleStep(0.1)
        self.spin_ki = QDoubleSpinBox(); self.spin_ki.setRange(0, 100); self.spin_ki.setSingleStep(0.1)
        self.spin_kd = QDoubleSpinBox(); self.spin_kd.setRange(0, 100); self.spin_kd.setSingleStep(0.1)
        
        form_pid.addWidget(QLabel("Kp:"), 0, 0); form_pid.addWidget(self.spin_kp, 0, 1)
        form_pid.addWidget(QLabel("Ki:"), 1, 0); form_pid.addWidget(self.spin_ki, 1, 1)
        form_pid.addWidget(QLabel("Kd:"), 2, 0); form_pid.addWidget(self.spin_kd, 2, 1)
        
        btn_send_pid = QPushButton("Update PID")
        btn_send_pid.clicked.connect(self.send_pid_params)
        form_pid.addWidget(btn_send_pid, 3, 0, 1, 2)
        
        gb_pid.setLayout(form_pid)
        mid_col.addWidget(gb_pid)

        # Telemetry & AI
        gb_extra = QGroupBox("Modules")
        v_extra = QVBoxLayout()
        self.chk_telemetry = QCheckBox("Record Telemetry (CSV)")
        self.chk_telemetry.stateChanged.connect(self.toggle_telemetry)
        v_extra.addWidget(self.chk_telemetry)
        
        self.chk_ai = QCheckBox("Enable AI Simulation")
        self.chk_ai.stateChanged.connect(self.toggle_ai)
        v_extra.addWidget(self.chk_ai)
        gb_extra.setLayout(v_extra)
        mid_col.addWidget(gb_extra)

        # CAN Monitor
        self.table_can = QTableWidget()
        self.table_can.setColumnCount(3)
        self.table_can.setHorizontalHeaderLabels(["ID", "Data", "Time"])
        self.table_can.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        mid_col.addWidget(QLabel("CAN Monitor:"))
        mid_col.addWidget(self.table_can)

        main_layout.addLayout(mid_col, stretch=1)

        # === RIGHT COLUMN: Plots ===
        right_col = QVBoxLayout()
        
        self.plot_widget = pg.PlotWidget(title="Speed & Accel History")
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True)
        self.curve_speed = self.plot_widget.plot(pen=pg.mkPen('orange', width=2), name="Speed")
        self.curve_accel = self.plot_widget.plot(pen=pg.mkPen('g', width=2), name="Accel")
        
        right_col.addWidget(self.plot_widget)
        main_layout.addLayout(right_col, stretch=2)

        # Список всех элементов, которые могут перехватить фокус
        interactive_widgets = [
            self.combo_ports, self.btn_connect, self.btn_control,
            self.spin_kp, self.spin_ki, self.spin_kd,
            self.chk_telemetry, self.chk_ai, self.table_can
        ]

        for widget in interactive_widgets:
            widget.setFocusPolicy(Qt.NoFocus)

        # Явно разрешаем главному окну принимать фокус
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setFocus()

    def setup_connections(self):
        self.serial_manager.data_received.connect(self.on_packet_received)
        self.serial_manager.connection_status.connect(self.on_conn_status)

    def refresh_ports(self):
        self.combo_ports.clear()
        # Добавляем виртуальный режим первым
        self.combo_ports.addItem("VIRTUAL_DEMO_MODE") 
        ports = serial.tools.list_ports.comports()
        for p in ports:
            self.combo_ports.addItem(p.device)

    def on_connect_click(self):
        # Если уже подключены — отключаемся
        if self.is_virtual or self.serial_manager.running:
            self.disconnect_hardware()
            return

        port = self.combo_ports.currentText()
        if not port: return

        if port == "VIRTUAL_DEMO_MODE":
            self.is_virtual = True
            self.on_conn_status(True, "Connected to Virtual Car")
            self.physics_timer.start(50)
            self.btn_control.setChecked(True)
        else:
            self.is_virtual = False
            asyncio.create_task(self.serial_manager.connect_serial(port))
    
    def disconnect_hardware(self):
        self.is_virtual = False
        self.physics_timer.stop()
        self.serial_manager.close()
        self.on_conn_status(False, "Disconnected")
        self.btn_connect.setText("Connect")
        self.btn_control.setChecked(False)

    def on_conn_status(self, connected, msg):
        self.statusBar().showMessage(msg)
        if connected:
            self.combo_ports.setDisabled(True)
            self.btn_connect.setText("Disconnect")
        else:
            self.combo_ports.setDisabled(False)
            self.btn_connect.setText("Connect")

    def on_packet_received(self, pkt):
        # Update internal state based on Packet ID (Logic imported from old code)
        can_id = pkt.CAN_ID
        data = pkt.CAN_DATA.DATA
        
        # --- CAN PARSING ---
        if can_id == 0x0001: # Angle
            val = int.from_bytes(data[0:2], 'little', signed=True)
            self.angle = int(val * MAX_ANGLE / 0x500)
        elif can_id == 0x0003: # Speed
            self.speed = data[0]
        elif can_id == 0x0017: # Brake
            self.brake = data[0]
        elif can_id == 0x0018: # Accel
            self.accel = data[0]
        elif can_id == 0x0004: # Gear
            self.gear = data[0]
        
        # Update Table Monitor (Show only last 10 packets)
        self.update_can_table(hex(can_id), data)
        
        # Log to telemetry
        self.recorder.log(self.speed, self.angle, self.accel, self.brake, self.gear)

    def update_can_table(self, can_id_str, data):
        self.table_can.insertRow(0)
        self.table_can.setItem(0, 0, QTableWidgetItem(can_id_str))
        self.table_can.setItem(0, 1, QTableWidgetItem(str(data)))
        self.table_can.setItem(0, 2, QTableWidgetItem(datetime.now().strftime("%H:%M:%S.%f")[:-3]))
        if self.table_can.rowCount() > 20:
            self.table_can.removeRow(20)

    def update_physics_sim(self):
        if not self.is_virtual:
            return

        # Константы для i-MiEV
        max_speed_forward = 130.0
        max_speed_reverse = -20.0
        
        # Плавность педалей
        self.accel += (self.target_accel - self.accel) * 0.2
        self.brake += (self.target_brake - self.brake) * 0.3
        self.angle = int(self.target_angle)
        
        acceleration = 0
        # Коэффициент инерции (чем меньше, тем "тяжелее" машина и дольше катится)
        mass_factor = 0.02 
        # Реалистичное сопротивление (воздух + трение колес)
        rolling_resistance = self.speed * 0.005 
        # Базовое трение покоя (чтобы машина в итоге остановилась, а не катилась вечно 0.01 км/ч)
        static_friction = 0.03 if abs(self.speed) > 0.1 else 0

        # Логика передач
        if self.gear == 4 or self.gear == 5 or self.gear == 6: # D, E, B (Вперед)
            if self.brake > 5:
                acceleration = -(self.brake * 0.15)
            else:
                motor_force = (self.accel * mass_factor)
                # Creep mode: i-MiEV медленно ползет вперед сам (отпускаем тормоз)
                creep = 0.05 if self.speed < 5 else 0
                acceleration = motor_force + creep - rolling_resistance - static_friction

        elif self.gear == 2: # Reverse (Назад)
            if self.brake > 5:
                acceleration = (self.brake * 0.15) # Тормозим (ускорение в сторону 0)
            else:
                # Тяга назад работает только если скорость не слишком большая вперед
                motor_force = -(self.accel * mass_factor * 0.5) 
                acceleration = motor_force - rolling_resistance + static_friction

        else: # Neutral (N) или Park (P)
            # В нейтрали только тормоз и естественные силы
            if self.brake > 5:
                # Если нажали тормоз в нейтрали
                brake_dir = -1 if self.speed > 0 else 1
                acceleration = brake_dir * (self.brake * 0.15)
            else:
                # Просто катимся
                drag_dir = -1 if self.speed > 0 else 1
                acceleration = drag_dir * (rolling_resistance + static_friction)

        # Применяем ускорение
        self.speed += acceleration

        # Мягкий стоп (чтобы не дергалась около нуля)
        if abs(self.speed) < 0.05 and self.accel < 1:
            self.speed = 0

        # Общий лимит скорости
        self.speed = max(max_speed_reverse, min(max_speed_forward, self.speed))

        # Обновляем таблицу CAN
        self.update_can_table("SIM_VSS", [int(abs(self.speed))])        
        if self.recorder.enabled:
            self.recorder.log(self.speed, self.angle, self.accel, self.brake, self.gear)

    def update_ui_periodic(self):
        # Update Visuals
        self.wheel_widget.set_angle(self.angle)
        self.lbl_speed.setText(f"{self.speed:.2f} km/h")
        self.lbl_angle_text.setText(f"Angle: {self.angle}° (T: {self.target_angle})")
        
        self.pb_accel.setValue(self.accel)
        self.pb_brake.setValue(self.brake)
        
        # Highlight Gear
        for k, lbl in self.gear_btns.items():
            if k == self.gear:
                lbl.setStyleSheet("background-color: yellow; border: 2px solid black;")
            else:
                lbl.setStyleSheet("border: 1px solid gray;")

        # Update Plot
        self.y_speed = self.y_speed[1:] + [self.speed]
        self.y_accel = self.y_accel[1:] + [self.accel]
        self.curve_speed.setData(self.y_speed)
        self.curve_accel.setData(self.y_accel)

    # --- INPUT HANDLERS ---
    def keyPressEvent(self, event):
        if event.isAutoRepeat(): return
        self.pressed_keys.add(event.key())
        
        # --- ПЕРЕДАЧИ ДОЛЖНЫ БЫТЬ ТУТ (ВНЕ УСЛОВИЯ) ---
        gear_map = {
            Qt.Key_1: 1, Qt.Key_2: 2, Qt.Key_3: 3,
            Qt.Key_4: 4, Qt.Key_5: 5, Qt.Key_6: 6
        }
        
        if event.key() in gear_map:
            new_gear = gear_map[event.key()]
            self.target_gear = new_gear
            self.gear = new_gear  # Для вирт. режима меняем сразу
            self.send_gear(new_gear)
            print(f"Gear changed to: {new_gear}") # Для отладки в консоли
            return

        # Остальное управление (газ/тормоз) оставляем под защитой
        if not self.control_active: return
        
        # Пробел для активации (если не через кнопку)
        if event.key() == Qt.Key_Space:
            self.btn_control.toggle()

            
    def keyReleaseEvent(self, event):
        if event.isAutoRepeat(): return
        if event.key() in self.pressed_keys:
            self.pressed_keys.remove(event.key())

    def process_held_keys(self):
        if not self.control_active: return

        changed = False
        
       # УПРАВЛЕНИЕ ГАЗОМ (Стрелки)
        if Qt.Key_Up in self.pressed_keys:
            self.target_accel = min(100, self.target_accel + 3)
            changed = True
        else:
            if self.target_accel > 0:
                self.target_accel = max(0, self.target_accel - 4) # Самоотвод газа
                changed = True

        # УПРАВЛЕНИЕ ТОРМОЗОМ (Клавиша W)
        if Qt.Key_Down in self.pressed_keys:
            self.target_brake = min(100, self.target_brake + 10)
            self.target_accel = 0 # Безопасность: тормоз отменяет газ
            changed = True
        elif self.target_brake > 0:
            self.target_brake = max(0, self.target_brake - 15) # Быстрый отпуск тормоза
            changed = True
        
        # Логика автовозврата руля (Spring effect)
        if Qt.Key_Left not in self.pressed_keys and Qt.Key_Right not in self.pressed_keys:
            if abs(self.target_angle) > 15:
                # Возвращаем к нулю со скоростью 20 градусов за такт
                step = 20
                if self.target_angle > 0: self.target_angle -= step
                else: self.target_angle += step
                changed = True
            else:
                if self.target_angle != 0:
                    self.target_angle = 0
                    changed = True
        
        # Руль (Стрелки влево/вправо)
        if Qt.Key_Left in self.pressed_keys:
            self.target_angle = max(-MAX_ANGLE, self.target_angle - 15)
            changed = True
        elif Qt.Key_Right in self.pressed_keys:
            self.target_angle = min(MAX_ANGLE, self.target_angle + 15)
            changed = True
        
        if changed:
            self.send_accel()
            self.send_brake()
            self.send_angle()


    def toggle_control(self, checked):
        self.control_active = checked
        self.btn_control.setText("Control ACTIVE" if checked else "Activate Control (Space)")
        self.btn_control.setStyleSheet(f"background-color: {'green' if checked else '#444'}; color: white;")
        
        # Send Cruise packet
        self.cmd_cruise.TIME = int(time.time())
        self.cmd_cruise.CAN_DATA.DATA[1] = 1 if checked else 0
        self.cmd_cruise.store_crc8()
        self.serial_manager.send_command(self.cmd_cruise)

    # --- SENDERS ---
    def _prepare_send(self, cmd_obj, val, data_idx=1):
        cmd_obj.TIME = int(time.time())
        cmd_obj.CAN_DATA.CNC = (cmd_obj.CAN_DATA.CNC + 1) & 0xFF
        cmd_obj.CAN_DATA.DATA[0] = 0x01
        cmd_obj.CAN_DATA.DATA[data_idx] = val
        cmd_obj.store_crc8()
        self.serial_manager.send_command(cmd_obj)

    def send_gear(self, gear_idx):
        self._prepare_send(self.cmd_gear, gear_idx)

    def send_accel(self):
        self._prepare_send(self.cmd_accel, self.target_accel)

    def send_brake(self):
        self._prepare_send(self.cmd_brake, self.target_brake)

    def send_angle(self):
        # Convert degrees back to raw byte mapping if needed, 
        # or simplified mapping as per old code logic
        raw_angle = max(-100, min(100, int(self.target_angle / MAX_ANGLE * 100)))
        # Needs signed byte conversion
        byte_val = raw_angle.to_bytes(1, 'little', signed=True)[0]
        self._prepare_send(self.cmd_angle, byte_val)

    def send_pid_params(self):
        # Собираем все коэффициенты
        kp = int(self.spin_kp.value() * 10) # Умножаем на 10 для передачи целым числом
        ki = int(self.spin_ki.value() * 10)
        kd = int(self.spin_kd.value() * 10)
        
        self.cmd_pid.TIME = int(time.time())
        self.cmd_pid.CAN_DATA.DATA[0] = 0x01 # Настройка активна
        self.cmd_pid.CAN_DATA.DATA[1] = kp
        self.cmd_pid.CAN_DATA.DATA[2] = ki
        self.cmd_pid.CAN_DATA.DATA[3] = kd
        self.cmd_pid.store_crc8()
        
        self.serial_manager.send_command(self.cmd_pid)
        self.statusBar().showMessage(f"PID Sent: P={kp/10}, I={ki/10}, D={kd/10}")
        
    # --- MODULES ---
    def toggle_telemetry(self, checked):
        if checked: self.recorder.start()
        else: self.recorder.stop()


    def process_ai(self):
        if not self.control_active: return
        
        # Если Lead Agent запущен, он сам управляет процессом через wrapper.
        # Этот метод можно оставить пустым или использовать для отрисовки
        # предсказаний модели на графиках.
        pass


    # --- ОСНОВНОЙ МЕТОД ДЛЯ AI ---
    def toggle_ai(self, checked):
        if checked:
            # 1. Проверяем наличие папки ПЕРЕД запуском
            base_path = r"E:\основы программирования\MitsuNeuroPilotAPI"
        
            # Исправленный путь к маршрутам (без папки lead)
            routes_path = os.path.normpath(os.path.join(base_path, "data", "data_routes", "leaderboard1", "ControlLoss", "Town01_Scenario1_0.xml"))
            checkpoint_path = os.path.normpath(os.path.join(base_path, "model_2.pth"))

            if not os.path.exists(routes_path):
                print(f"!!! ФАЙЛ НЕ НАЙДЕН ПО ПУТИ: {routes_path}")
                # На всякий случай проверим вариант с маленькой буквой town13
                routes_path = os.path.normpath(os.path.join(base_path, "data", "benchmark_routes", "town13", "0.xml"))

            config = {
                "project_root": base_path,
                "checkpoint_path": checkpoint_path,
                "routes": routes_path,
                #"port": 3000,
                "host": "localhost",
                "expert_mode": True
                }

            print(f"[GUI] Попытка запуска Lead Agent с конфигом: {config['checkpoint_path']}")
            
            try:
                self.agent_thread = LeadAgentThread(config)
                # Важно: Сначала коннектим сигналы, потом старт!
                self.agent_thread.log_received.connect(self.handle_agent_log)
                self.agent_thread.status_changed.connect(lambda s: self.statusBar().showMessage(s))
                self.agent_thread.error_occurred.connect(self.handle_agent_error)
                
                self.agent_thread.start()
                self.statusBar().showMessage("Lead Agent: Starting thread...")
            except Exception as e:
                print(f"!!! Ошибка при создании потока: {e}")
                self.chk_ai.setChecked(False)
        else:
            if hasattr(self, 'agent_thread') and self.agent_thread:
                print("[GUI] Остановка Lead Agent...")
                self.agent_thread.stop()
                self.agent_thread = None
            self.statusBar().showMessage("Lead Agent: Offline")

    @Slot(str)
    def handle_agent_log(self, message):
        # Печатаем ВООБЩЕ ВСЁ, что приходит от агента
        print(f"[AGENT_OUT]: {message}")
        self.statusBar().showMessage(f"AI: {message[:40]}")

    @Slot(str)
    def handle_agent_error(self, error):
        print(f"[AGENT_ERR]: {error}")
        self.statusBar().showMessage(f"AI ERROR: {error}")
        self.chk_ai.setChecked(False)


# --- BOOTSTRAP ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    w = MainWindow()
    w.show()

    with loop:
        loop.run_forever()