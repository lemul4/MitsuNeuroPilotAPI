import sys
import asyncio
import qasync
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, QObject, Slot, QThread, Signal 
from datetime import datetime
from config import PHYSICS_UPDATE_RATE_MS
from ui.main_window import MainWindow
from core.vehicle import VehicleState
from hardware.serial_comm import SerialManager
import utils 
from core.telemetry import TelemetryRecorder, RawTelemetryJsonlReader
from lead_integration import LeadAgentThread
from PySide6.QtGui import QPixmap, QImage

import zmq 


class VideoReceiverThread(QThread):
    frame_received = Signal(QPixmap)

    def __init__(self):
        super().__init__()
        self.is_running = True
        self.received_count = 0

    def run(self):
        print("[GUI_VIDEO_THREAD] Запуск потока приема видео...")
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        
        # Важно: используем адрес отправителя
        try:
            socket.connect("tcp://127.0.0.1:5555")
            socket.setsockopt_string(zmq.SUBSCRIBE, "")
            print("[GUI_VIDEO_THREAD] Подключено к сокету 127.0.0.1:5555")
        except Exception as e:
            print(f"[GUI_VIDEO_THREAD] Ошибка подключения к сокету: {e}")
            return

        while self.is_running:
            # Проверяем наличие данных (таймаут 500мс)
            if socket.poll(500):
                try:
                    msg = socket.recv()
                    self.received_count += 1
                    
                    # Логируем каждые 50 полученных кадров
                    if self.received_count % 50 == 0:
                        print(f"[GUI_VIDEO_THREAD] Принято кадров от сервиса: {self.received_count}")

                    image = QImage.fromData(msg, "JPG")
                    if not image.isNull():
                        pixmap = QPixmap.fromImage(image)
                        self.frame_received.emit(pixmap)
                    else:
                        print("[GUI_VIDEO_THREAD] Ошибка: Получены битые данные (не удалось собрать QImage)")
                except Exception as e:
                    print(f"[GUI_VIDEO_THREAD] Ошибка при чтении сообщения: {e}")
            else:
                # Если данных нет долгое время
                pass 

    def stop(self):
        self.is_running = False
        self.wait()

class AppController(QObject):
    def __init__(self, view: MainWindow):
        super().__init__()
        self.view = view
        self.video_receiver = VideoReceiverThread()
        self.video_receiver.frame_received.connect(self.view.update_camera_frame)
        self.video_receiver.start()
        self.vehicle = VehicleState()
        self.serial = SerialManager()
        self.telemetry = TelemetryRecorder()
        
        self.agent_thread = None
        self.raw_telemetry_reader = None
        self.raw_telemetry_timer = QTimer()
        self.raw_telemetry_timer.timeout.connect(self.poll_ai_telemetry)

        self.is_virtual = False
        self.control_active = False
        self.ai_control_requested = False
        self.ai_agent_loading = False
        self.ai_agent_running = False
        self.ui_update_timer = QTimer()
        self.ui_update_timer.timeout.connect(self.update_view)
        self.ui_update_timer.start(50)

        self.physics_timer = QTimer()
        self.physics_timer.timeout.connect(self.step_virtual_physics)
        self.raw_telemetry_timer = QTimer()
        self.raw_telemetry_timer.timeout.connect(self.poll_ai_telemetry)
        self.view.connect_requested.connect(self.handle_connect)
        self.view.disconnect_requested.connect(self.handle_disconnect)
        self.view.control_toggled.connect(self.handle_control_toggle)
        self.view.gear_requested.connect(self.handle_gear_change)
        self.view.manual_input_updated.connect(self.handle_manual_input)
        self.view.telemetry_toggled.connect(self.handle_telemetry_toggle)
        self.view.ai_toggled.connect(self.handle_ai_toggle)
        
        self.serial.connection_status.connect(self.view.set_connection_status)
        self.serial.data_received.connect(self.handle_can_packet)
        self.latest_frame_path = os.path.join("outputs", "visualizations", "latest_front_cam.png")
        self.init_commands()


    def init_commands(self):
        self.cmd_gear = utils.Serial_Data(bytearray.fromhex("AA 00000000 3300 00 02 01 00 00 00 01 00 00"))
        self.cmd_accel = utils.Serial_Data(bytearray.fromhex("AA 00000000 3800 00 02 01 00 00 00 01 00 00"))
        self.cmd_brake = utils.Serial_Data(bytearray.fromhex("AA 00000000 3700 00 02 01 00 00 00 01 00 00"))
        self.cmd_angle = utils.Serial_Data(bytearray.fromhex("AA 00000000 3200 00 02 01 00 00 00 01 00 00"))
        self.cmd_cruise = utils.Serial_Data(bytearray.fromhex("AA 00000000 7700 00 02 01 00 00 00 01 00 00"))

    def handle_connect(self, port_name):
        self.stop_ai_agent()
        if port_name == "VIRTUAL_DEMO_MODE":
            self.is_virtual = True
            self.physics_timer.start(PHYSICS_UPDATE_RATE_MS)
            self.view.set_connection_status(True, "Virtual Mode Active")
        else:
            self.is_virtual = False
            asyncio.create_task(self.serial.connect_serial(port_name))

    def handle_disconnect(self):
        self.is_virtual = False
        self.physics_timer.stop()
        self.serial.close()
        self.view.set_connection_status(False, "Disconnected")

    def handle_control_toggle(self, is_active):
        self.control_active = is_active
        if not self.is_virtual:
            import time
            self.cmd_cruise.TIME = int(time.time())
            self.cmd_cruise.CAN_DATA.DATA[1] = 1 if is_active else 0
            self.cmd_cruise.store_crc8()
            self.serial.send_command(self.cmd_cruise)

        if is_active:
            if self.ai_control_requested:
                self.start_ai_agent()
        else:
            if self.agent_thread or self.raw_telemetry_reader:
                self.stop_ai_agent()


    def handle_ai_toggle(self, is_active):
        self.ai_control_requested = bool(is_active)
        if is_active:
            self.view.statusBar().showMessage("AI Control armed: press Activate Control to start scenario")
            if hasattr(self.view, "set_route_runtime_state"):
                self.view.set_route_runtime_state("armed")
        else:
            self.stop_ai_agent()

    def start_ai_agent(self):
        if self.agent_thread or self.ai_agent_loading:
            return

        selected_route_path = None
        if hasattr(self.view, "get_selected_route"):
            selected_route_path = self.view.get_selected_route()

        if not selected_route_path:
            message = "AI Control: маршрут не выбран, запуск отменен"
            print(message)
            self.view.statusBar().showMessage(message)
            if hasattr(self.view, "set_route_error_status"):
                self.view.set_route_error_status("Выберите маршрут перед Activate Control")
            self.view.set_ai_checkbox(False)
            return

        gui_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(gui_dir)

        telemetry_file = os.path.join(project_root, "trace_log.jsonl")
        try:
            if os.path.exists(telemetry_file):
                os.remove(telemetry_file)
        except OSError:
            pass

        config = {
            "project_root": project_root,
            "routes": selected_route_path,
            "checkpoint_path": os.path.join(project_root, "model_2.pth"),
            "telemetry_file": telemetry_file,
            "expert_mode": True,
            "host": "localhost"
        }

        try:
            self.ai_agent_loading = True
            self.ai_agent_running = False
            if hasattr(self.view, "set_route_loading_status"):
                self.view.set_route_loading_status("Загрузка мира и сценария")
            self.view.statusBar().showMessage("AI Status: loading scenario")

            self.agent_thread = LeadAgentThread(config)
            self.agent_thread.log_received.connect(self.handle_agent_log)
            self.agent_thread.status_changed.connect(self.handle_agent_status)
            self.agent_thread.error_occurred.connect(self.handle_agent_error)
            self.agent_thread.start()

            self.raw_telemetry_reader = RawTelemetryJsonlReader(config["telemetry_file"])
            self.raw_telemetry_timer.start(50)

        except Exception as e:
            self.ai_agent_loading = False
            self.ai_agent_running = False
            print(f"Error starting AI thread: {e}")
            if hasattr(self.view, "set_route_error_status"):
                self.view.set_route_error_status(str(e))
            self.view.set_ai_checkbox(False)

    def stop_ai_agent(self):
        if self.agent_thread:
            self.agent_thread.stop()
            self.agent_thread = None
        self.raw_telemetry_timer.stop()
        self.raw_telemetry_reader = None
        was_active = self.ai_agent_loading or self.ai_agent_running
        self.ai_agent_loading = False
        self.ai_agent_running = False
        self.view.statusBar().showMessage("Lead Agent: Offline")
        if was_active and hasattr(self.view, "set_route_stopped_status"):
            self.view.set_route_stopped_status("Сценарий остановлен")

    def poll_ai_telemetry(self):
        if not self.raw_telemetry_reader: 
            return
            
        data = self.raw_telemetry_reader.get_latest_data()
        
        if data is not None:
            if not self.ai_agent_running:
                self.ai_agent_loading = False
                self.ai_agent_running = True
                if hasattr(self.view, "set_route_running_status"):
                    self.view.set_route_running_status("Сценарий выполняется: телеметрия получена")
                self.view.statusBar().showMessage("AI Status: scenario running")
            # Извлекаем значения
            steer = data.get("steer", 0.0)
            throttle = data.get("throttle", 0.0)
            brake = data.get("brake", 0.0)
            
            # Преобразуем для модели
            target_angle = int(steer * 630)
            target_accel = int(throttle * 100)
            target_brake = int(brake * 100)
            
            # --- ВЫВОД ЛОГОВ В КОНСОЛЬ (как на скриншоте) ---
            # Используем f-строки для выравнивания
            log_line = (
                f"[AI TELEMETRY] "
                f"Steer: {steer:>6.2f} | "
                f"Thr: {throttle:>5.2f} | "
                f"Brk: {brake:>5.2f} | "
                f"Target Angle: {target_angle:>4}°"
            )
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] {log_line}")  

            # Обновляем состояние модели
            self.vehicle.target_angle = target_angle
            self.vehicle.target_accel = target_accel
            self.vehicle.target_brake = target_brake
            if os.path.exists(self.latest_frame_path):
                try:
                    pixmap = QPixmap(self.latest_frame_path)
                    self.view.update_camera_frame(pixmap)
                except Exception as e:
                    print(f"Ошибка загрузки кадра камеры: {e}")

            # Если управление активно, отправляем в CAN
            if self.control_active:
                self.handle_manual_input(target_angle, target_accel, target_brake)

    @Slot(str)
    def handle_agent_log(self, message):
        text = str(message)
        self.view.statusBar().showMessage(f"AI: {text[:40]}")
        self._update_route_loading_from_text(text)


    @Slot(str)
    def handle_agent_status(self, status):
        text = str(status)
        self.view.statusBar().showMessage(f"AI Status: {text}")
        self._update_route_loading_from_text(text)


    def _update_route_loading_from_text(self, text):
        low = str(text or "").lower()
        if any(token in low for token in ("error", "exception", "failed", "traceback")):
            if hasattr(self.view, "set_route_error_status"):
                self.view.set_route_error_status(str(text))
            return
        if not self.ai_agent_running and self.ai_agent_loading:
            if any(token in low for token in ("spawn", "world", "map", "route", "scenario", "carla", "load", "loading", "connect")):
                if hasattr(self.view, "set_route_loading_status"):
                    self.view.set_route_loading_status(str(text))


    @Slot(str)
    def handle_agent_error(self, error):
        self.ai_agent_loading = False
        self.ai_agent_running = False
        self.view.statusBar().showMessage(f"AI ERROR: {error}")
        if hasattr(self.view, "set_route_error_status"):
            self.view.set_route_error_status(str(error))
        self.view.set_ai_checkbox(False)

    def handle_manual_input(self, angle, accel, brake):
        if not self.control_active: return
        self.vehicle.target_angle = angle
        self.vehicle.target_accel = accel
        self.vehicle.target_brake = brake
        self._send_command_to_car(self.cmd_accel, accel)
        self._send_command_to_car(self.cmd_brake, brake)
        raw_angle = max(-100, min(100, int(angle / 630 * 100)))
        byte_val = raw_angle.to_bytes(1, "little", signed=True)[0]
        self._send_command_to_car(self.cmd_angle, byte_val)

    def handle_gear_change(self, gear_idx):
        if not self.control_active: return
        self.vehicle.target_gear = gear_idx
        if self.is_virtual:
            self.vehicle.gear = gear_idx
        self._send_command_to_car(self.cmd_gear, gear_idx)

    def handle_telemetry_toggle(self, is_active):
        if is_active: self.telemetry.start()
        else: self.telemetry.stop()

    def _send_command_to_car(self, cmd_obj, val, data_idx=1):
        import time
        cmd_obj.TIME = int(time.time())
        cmd_obj.CAN_DATA.CNC = (cmd_obj.CAN_DATA.CNC + 1) & 0xFF
        cmd_obj.CAN_DATA.DATA[0] = 0x01
        cmd_obj.CAN_DATA.DATA[data_idx] = val
        cmd_obj.store_crc8()
        self.serial.send_command(cmd_obj)

    def step_virtual_physics(self):
        if self.is_virtual:
            self.vehicle.update_physics()
            if self.telemetry.enabled:
                self.telemetry.log(self.vehicle.speed, self.vehicle.angle, 
                                   self.vehicle.accel, self.vehicle.brake, self.vehicle.gear)

    def handle_can_packet(self, pkt):
        can_id = pkt.CAN_ID
        data = pkt.CAN_DATA.DATA
        if can_id == 0x0001:
            val = int.from_bytes(data[0:2], "little", signed=True)
            self.vehicle.angle = int(val * 630 / 0x500)
        elif can_id == 0x0003:
            self.vehicle.speed = data[0]
        elif can_id == 0x0017:
            self.vehicle.brake = data[0]
        elif can_id == 0x0018:
            self.vehicle.accel = data[0]
        elif can_id == 0x0004:
            self.vehicle.gear = data[0]
        # self.view.update_can_table(hex(can_id), data)
        self.telemetry.log(self.vehicle.speed, self.vehicle.angle, 
                           self.vehicle.accel, self.vehicle.brake, self.vehicle.gear)

    def update_view(self):
        self.view.update_dashboard(self.vehicle)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    main_window = MainWindow()
    controller = AppController(main_window)
    main_window.show()
    with loop:
        loop.run_forever()