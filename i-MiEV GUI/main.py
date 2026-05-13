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
        self.is_connected = False
        self.control_active = False
        self.ai_control_requested = False
        self.ai_agent_loading = False
        self.ai_agent_running = False

        # Очередь маршрутов запускается кооперативно: один LeadAgentThread на один XML.
        # Второй маршрут стартует только после finished текущего thread. Принудительный
        # переход к следующему маршруту убран, потому что он рвал QThread/CARLA lifecycle.
        self.pending_route_queue = []
        self.current_route_index = -1
        self.queue_mode = None  # None / "single" / "queue"
        self.queue_stop_requested = False
        self.route_plan_prepared = False
        self.route_transition_delay_ms = 6000
        self._agent_stop_requested = False

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

        # Новый UI отправляет один список: 1 маршрут => одиночный запуск, N маршрутов => очередь.
        # Старые сигналы single/queue оставлены как fallback, но основной путь — route_launch_requested.
        if hasattr(self.view, "route_launch_requested"):
            self.view.route_launch_requested.connect(self.handle_route_launch_requested)
        else:
            if hasattr(self.view, "route_single_launch_requested"):
                self.view.route_single_launch_requested.connect(lambda route: self.handle_route_launch_requested([route] if route else []))
            if hasattr(self.view, "route_queue_launch_requested"):
                self.view.route_queue_launch_requested.connect(self.handle_route_launch_requested)
        if hasattr(self.view, "route_queue_updated"):
            self.view.route_queue_updated.connect(self.handle_route_queue_updated)
        
        self.serial.connection_status.connect(self.view.set_connection_status)
        self.serial.connection_status.connect(self.handle_serial_connection_status)
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
        # Подготовленная очередь не сбрасывается при подключении: пользователь может
        # выбрать/подготовить маршруты до Connect, после Connect или после AI Control.
        if port_name == "VIRTUAL_DEMO_MODE":
            self.is_virtual = True
            self.is_connected = True
            self.physics_timer.start(PHYSICS_UPDATE_RATE_MS)
            self.view.set_connection_status(True, "Virtual Mode Active")
        else:
            self.is_virtual = False
            self.is_connected = False
            asyncio.create_task(self.serial.connect_serial(port_name))

    @Slot(bool, str)
    def handle_serial_connection_status(self, is_connected, message):
        self.is_connected = bool(is_connected)

    def handle_disconnect(self):
        self.abort_active_scenario("Disconnected", keep_plan=True)
        self.is_virtual = False
        self.is_connected = False
        self.physics_timer.stop()
        self.serial.close()
        self.view.set_connection_status(False, "Disconnected")

    def _set_control_button_checked_later(self, checked):
        btn = getattr(self.view, "btn_control", None)
        if btn is not None and btn.isChecked() != checked:
            QTimer.singleShot(0, lambda: btn.setChecked(checked))

    def _reject_control_activation(self, message):
        self.control_active = False
        print(f"AI CONTROL: запуск отклонен: {message}")
        self.view.statusBar().showMessage(message)
        if hasattr(self.view, "set_route_runtime_state"):
            state = "prepared" if self.route_plan_prepared else "selected"
            self.view.set_route_runtime_state(state, message)
        elif hasattr(self.view, "set_route_error_status"):
            self.view.set_route_error_status(message)
        self._set_control_button_checked_later(False)

    def _validate_control_start_requirements(self):
        if not self.is_connected:
            return False, "Сначала нажмите Connect"
        if not self.ai_control_requested:
            return False, "Включите AI Control"
        if not self.pending_route_queue:
            return False, "Выберите маршрут и подготовьте запуск"
        if not self.route_plan_prepared:
            count = len(self.pending_route_queue)
            if count > 1:
                return False, "Нажмите «Подготовить очередь» перед Activate Control"
            return False, "Нажмите «Подготовить маршрут» перед Activate Control"
        if self.agent_thread is not None or self.ai_agent_loading:
            return False, "Текущий сценарий еще запускается или выполняется"
        return True, ""

    def _send_cruise_state(self, is_active):
        if self.is_virtual:
            return
        try:
            import time
            self.cmd_cruise.TIME = int(time.time())
            self.cmd_cruise.CAN_DATA.DATA[1] = 1 if is_active else 0
            self.cmd_cruise.store_crc8()
            self.serial.send_command(self.cmd_cruise)
        except Exception as exc:
            print(f"AI CONTROL: ошибка отправки cruise state: {exc}")

    def handle_control_toggle(self, is_active):
        if is_active:
            ok, message = self._validate_control_start_requirements()
            if not ok:
                self._reject_control_activation(message)
                return

            self.control_active = True
            self.queue_stop_requested = False
            self._agent_stop_requested = False
            self._send_cruise_state(True)

            if hasattr(self.vehicle, "force_drive_gear"):
                self.vehicle.force_drive_gear()

            print("AI CONTROL: Activate Control включен; запуск подготовленного сценария")
            self.start_pending_route()
            return

        # Повторное нажатие Activate Control выключает активный сценарий. План маршрутов
        # остается подготовленным, чтобы можно было повторно стартовать с первого маршрута.
        self.control_active = False
        self._send_cruise_state(False)
        self.abort_active_scenario("Control выключен: сценарий остановлен", keep_plan=True)

    def handle_ai_toggle(self, is_active):
        self.ai_control_requested = bool(is_active)
        if is_active:
            if self.route_plan_prepared:
                message = "AI Control включен. Нажмите Activate Control для запуска"
                state = "prepared"
            else:
                message = "AI Control включен. Подготовьте маршрут/очередь"
                state = "armed"
            self.view.statusBar().showMessage(message)
            if hasattr(self.view, "set_route_runtime_state"):
                self.view.set_route_runtime_state(state, message)
        else:
            self.abort_active_scenario("AI Control выключен", keep_plan=True)
            self.view.statusBar().showMessage("AI Control выключен")
            if hasattr(self.view, "set_route_runtime_state"):
                self.view.set_route_runtime_state("selected", "AI Control выключен")

    def _route_path_from_record(self, route):
        if route is None:
            return None
        if isinstance(route, (str, os.PathLike)):
            return os.fspath(route)
        if isinstance(route, dict):
            return (
                route.get("path")
                or route.get("xml_path")
                or route.get("file_path")
                or route.get("route_path")
                or route.get("relative_path")
            )
        return getattr(route, "path", None) or getattr(route, "xml_path", None) or getattr(route, "file_path", None)

    def _fallback_route_queue_from_view(self):
        if hasattr(self.view, "get_route_queue"):
            queue = self.view.get_route_queue()
            if queue:
                return list(queue)
        if hasattr(self.view, "get_selected_route_data"):
            route = self.view.get_selected_route_data()
            if route:
                return [route]
        if hasattr(self.view, "get_selected_route"):
            route = self.view.get_selected_route()
            if route:
                return [route]
        return []

    def handle_route_queue_updated(self, routes):
        # Изменение выбора не запускает и не готовит сценарий. Оно только делает
        # предыдущий план устаревшим: пользователь должен явно нажать кнопку подготовки.
        if self.ai_agent_loading or self.ai_agent_running:
            return
        queue = list(routes or [])
        self.pending_route_queue = [route for route in queue if route]
        self.current_route_index = -1
        self.queue_mode = "queue" if len(self.pending_route_queue) > 1 else ("single" if self.pending_route_queue else None)
        self.route_plan_prepared = False
        self.queue_stop_requested = False
        self._agent_stop_requested = False

    def handle_route_launch_requested(self, routes):
        # Кнопка маршрута/очереди только подготавливает план. CARLA/LeadAgentThread
        # стартуют строго по событию Activate Control ON.
        queue = list(routes or self._fallback_route_queue_from_view())
        queue = [route for route in queue if route]
        self.pending_route_queue = queue
        self.current_route_index = -1
        self.queue_mode = "queue" if len(queue) > 1 else ("single" if queue else None)
        self.queue_stop_requested = False
        self._agent_stop_requested = False
        self.route_plan_prepared = bool(queue)

        if not queue:
            message = "Маршрут не выбран"
            print(f"AI ROUTES: {message}")
            self.view.statusBar().showMessage(message)
            if hasattr(self.view, "set_route_error_status"):
                self.view.set_route_error_status(message)
            return

        label = f"очередь из {len(queue)} маршрутов" if len(queue) > 1 else "маршрут"
        message = f"Подготовлено: {label}. Нажмите Activate Control для запуска"
        print(f"AI ROUTES: {message}")
        self.view.statusBar().showMessage(message)
        if hasattr(self.view, "set_route_runtime_state"):
            self.view.set_route_runtime_state("prepared", message)
        elif hasattr(self.view, "set_route_loading_status"):
            self.view.set_route_loading_status(message)

    def start_pending_route(self):
        if self.agent_thread is not None or self.ai_agent_loading:
            self.view.statusBar().showMessage("AI Status: текущий сценарий еще выполняется")
            return

        ok, message = self._validate_control_start_requirements()
        if not ok:
            self._reject_control_activation(message)
            return

        # Каждый новый запуск очереди начинается с первого подготовленного маршрута.
        self.current_route_index = 0
        self.queue_stop_requested = False
        self._agent_stop_requested = False
        self._start_current_route()

    # Старое имя оставлено для совместимости с прежним flow.
    def start_ai_agent(self):
        self.start_pending_route()

    def _start_current_route(self):
        if self.queue_stop_requested:
            return
        if self.agent_thread is not None or self.ai_agent_loading:
            return
        if not (0 <= self.current_route_index < len(self.pending_route_queue)):
            self._finish_route_queue("Очередь завершена")
            return

        route_record = self.pending_route_queue[self.current_route_index]
        selected_route_path = self._route_path_from_record(route_record)
        if not selected_route_path:
            self.handle_agent_error("У выбранного маршрута нет XML path")
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
            self._agent_stop_requested = False
            total = len(self.pending_route_queue)
            idx = self.current_route_index

            if hasattr(self.view, "set_active_route"):
                self.view.set_active_route(
                    route_record,
                    index=idx,
                    total=total,
                    state="loading",
                    message=f"Загрузка сценария {idx + 1} / {total}",
                )
            elif hasattr(self.view, "set_route_loading_status"):
                self.view.set_route_loading_status(f"Загрузка сценария {idx + 1} / {total}")

            self.view.statusBar().showMessage(f"AI Status: loading route {idx + 1}/{total}")
            print(f"AI ROUTES: старт маршрута {idx + 1}/{total}: {selected_route_path}")

            self.agent_thread = LeadAgentThread(config)
            self.agent_thread.log_received.connect(self.handle_agent_log)
            self.agent_thread.status_changed.connect(self.handle_agent_status)
            self.agent_thread.error_occurred.connect(self.handle_agent_error)
            self.agent_thread.finished.connect(self.handle_agent_finished)
            self.agent_thread.start()

            self.raw_telemetry_reader = RawTelemetryJsonlReader(config["telemetry_file"])
            self.raw_telemetry_timer.start(50)

        except Exception as e:
            self.ai_agent_loading = False
            self.ai_agent_running = False
            self.agent_thread = None
            print(f"Error starting AI thread: {e}")
            if hasattr(self.view, "set_route_error_status"):
                self.view.set_route_error_status(str(e))
            self.view.set_ai_checkbox(False)

    def _reset_local_vehicle_state(self):
        try:
            if hasattr(self.vehicle, "reset_motion"):
                self.vehicle.reset_motion()
            self.vehicle.target_angle = 0
            self.vehicle.angle = 0
            self.vehicle.target_accel = 0
            self.vehicle.target_brake = 0
            self.vehicle.accel = 0
            self.vehicle.brake = 0
        except Exception as exc:
            print(f"AI CONTROL: ошибка сброса локального состояния: {exc}")

    def _request_stop_current_agent(self, message="Lead Agent: stopping"):
        self.raw_telemetry_timer.stop()
        self.raw_telemetry_reader = None
        self.ai_agent_loading = False
        self.ai_agent_running = False
        self._agent_stop_requested = True
        self.view.statusBar().showMessage(message)

        thread = self.agent_thread
        if thread is None:
            return True

        try:
            thread.stop()
        except Exception as exc:
            print(f"AI ROUTES: ошибка stop(): {exc}")

        # Не обнуляем self.agent_thread здесь. Иначе Qt может уничтожить QThread,
        # пока он еще завершает CARLA/ScenarioRunner: это и давало
        # 'QThread: Destroyed while thread is still running'.
        try:
            if hasattr(thread, "isRunning") and thread.isRunning():
                print("AI ROUTES: ожидаем штатного завершения LeadAgentThread")
                return False
        except RuntimeError:
            pass

        self.agent_thread = None
        return True

    def stop_ai_agent(self):
        self.abort_active_scenario("Lead Agent: stopping", keep_plan=True)

    def abort_active_scenario(self, message="Сценарий остановлен", keep_plan=True):
        # Безопасная остановка: просим LeadAgentThread завершиться, но не уничтожаем
        # объект QThread до сигнала finished.
        self.queue_stop_requested = True
        self.current_route_index = -1
        if not keep_plan:
            self.pending_route_queue = []
            self.queue_mode = None
            self.route_plan_prepared = False
        self._reset_local_vehicle_state()
        self._request_stop_current_agent(message)
        if hasattr(self.view, "set_route_stopped_status"):
            suffix = " Очередь остается подготовленной." if keep_plan and self.route_plan_prepared else ""
            self.view.set_route_stopped_status(f"{message}.{suffix}")

    # Старое имя оставлено для совместимости.
    def stop_route_queue(self, message="Очередь остановлена"):
        self.abort_active_scenario(message, keep_plan=True)

    @Slot()
    def handle_agent_finished(self):
        self.raw_telemetry_timer.stop()
        self.raw_telemetry_reader = None
        self.agent_thread = None
        self.ai_agent_loading = False
        self.ai_agent_running = False

        if self._agent_stop_requested or self.queue_stop_requested:
            self._agent_stop_requested = False
            self.queue_stop_requested = False
            self.current_route_index = -1
            if hasattr(self.view, "set_route_runtime_state"):
                if self.route_plan_prepared and self.pending_route_queue:
                    self.view.set_route_runtime_state(
                        "prepared",
                        "Сценарий прерван. CARLA очищена, можно снова нажать Activate Control",
                    )
                else:
                    self.view.set_route_runtime_state("stopped", "Сценарий прерван")
            return

        total = len(self.pending_route_queue)
        finished_index = self.current_route_index
        if total <= 0 or finished_index < 0:
            return

        print(f"AI ROUTES: маршрут {finished_index + 1}/{total} завершен")

        if (
            self.queue_mode == "queue"
            and self.control_active
            and self.ai_control_requested
            and self.route_plan_prepared
            and finished_index + 1 < total
        ):
            self.current_route_index += 1
            if hasattr(self.view, "set_route_queue_position"):
                self.view.set_route_queue_position(self.current_route_index + 1, total)
            self.view.statusBar().showMessage(
                f"AI Status: next route in {self.route_transition_delay_ms / 1000:.1f}s"
            )
            QTimer.singleShot(self.route_transition_delay_ms, self._start_current_route)
        else:
            self._finish_route_queue("Очередь завершена" if self.queue_mode == "queue" else "Маршрут завершен")

    def _finish_route_queue(self, message):
        self.pending_route_queue = []
        self.current_route_index = -1
        self.queue_mode = None
        self.queue_stop_requested = False
        self.route_plan_prepared = False
        self._agent_stop_requested = False
        self.control_active = False
        self._send_cruise_state(False)
        self._set_control_button_checked_later(False)
        if hasattr(self.view, "set_route_queue_finished"):
            self.view.set_route_queue_finished(message)
        self.view.statusBar().showMessage(message)

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
            # Обновляем состояние модели
            self.vehicle.target_angle = target_angle
            self.vehicle.target_accel = target_accel
            self.vehicle.target_brake = target_brake

            # Агент не переключает передачи. Для AI-сценария держим локальную модель в D.
            if hasattr(self.vehicle, "force_drive_gear"):
                self.vehicle.force_drive_gear()
            else:
                self.vehicle.target_gear = 4
                self.vehicle.gear = 4

            # Если trace_log.jsonl содержит реальную скорость CARLA, используем ее.
            # Если скорости нет, обновляем fallback-физику.
            if hasattr(self.vehicle, "apply_telemetry") and self.vehicle.apply_telemetry(data):
                pass
            else:
                self.vehicle.update_physics(dt=0.05)
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
        self.queue_stop_requested = True
        self.route_plan_prepared = False
        self.control_active = False
        self._set_control_button_checked_later(False)
        self.view.statusBar().showMessage(f"AI ERROR: {error}")
        if hasattr(self.view, "set_route_error_status"):
            self.view.set_route_error_status(str(error))
        print(f"AI ERROR: {error}")
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