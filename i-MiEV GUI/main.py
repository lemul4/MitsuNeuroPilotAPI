import sys
import asyncio
import concurrent.futures
import threading
import os
import time
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, QObject, Slot, QThread, Signal 
from datetime import datetime
from config import PHYSICS_UPDATE_RATE_MS
from ui.main_window import MainWindow
from core.vehicle import VehicleState
from hardware.serial_comm import SerialManager

try:
    from vehicle_control import (
        DeviceDescriptor, DeviceKind, Gear, ControlIntent, Mission, Waypoint,
        VehicleGateway, RealSerialVehicleAdapter, MockVehicleAdapter,
        VehicleAdapterFactory, VehicleControlService, ControlArbiter,
        ABRouteRequest, CoordinateRoutePlanner, RoadRouteRequest, OsrmRoadRouteProvider,
    )
    VEHICLE_CONTROL_AVAILABLE = True
except Exception as _vehicle_control_import_error:
    VEHICLE_CONTROL_AVAILABLE = False
    DeviceDescriptor = DeviceKind = Gear = ControlIntent = Mission = Waypoint = None
    VehicleGateway = RealSerialVehicleAdapter = MockVehicleAdapter = None
    VehicleAdapterFactory = VehicleControlService = ControlArbiter = None
    ABRouteRequest = CoordinateRoutePlanner = RoadRouteRequest = OsrmRoadRouteProvider = None

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


class CarlaMotionMonitorThread(QThread):
    """Background CARLA motion probe for startup watchdog.

    It does not spawn or destroy anything. It only reads the hero/ego vehicle
    actor position and velocity from CARLA. This is deliberately independent
    from agent commands, because a stuck agent can still output changing
    steer/throttle/brake values while the vehicle is not moving.
    """

    motion_update = Signal(object)
    status_update = Signal(str)

    def __init__(self, host="127.0.0.1", port=2000, roles=None, interval_ms=1000, parent=None):
        super().__init__(parent)
        self.host = host
        self.port = int(port)
        self.roles = tuple(roles or ("hero", "ego_vehicle", "ego", "player"))
        self.interval_ms = int(interval_ms)
        self._running = True

    def stop(self):
        self._running = False
        try:
            self.wait(1500)
        except RuntimeError:
            pass

    def _pick_vehicle(self, world):
        vehicles = list(world.get_actors().filter("vehicle.*"))
        for vehicle in vehicles:
            role = vehicle.attributes.get("role_name", "")
            if role in self.roles:
                return vehicle, vehicles
        if len(vehicles) == 1:
            return vehicles[0], vehicles
        return None, vehicles

    def run(self):
        try:
            import carla
        except Exception as exc:
            self.status_update.emit(f"CARLA watchdog unavailable: cannot import carla: {exc}")
            self.motion_update.emit({"found": False, "error": f"cannot import carla: {exc}"})
            return

        try:
            client = carla.Client(self.host, self.port)
            client.set_timeout(0.7)
            self.status_update.emit(f"CARLA watchdog connected to {self.host}:{self.port}")
        except Exception as exc:
            self.status_update.emit(f"CARLA watchdog cannot create client: {exc}")
            self.motion_update.emit({"found": False, "error": str(exc)})
            return

        while self._running:
            try:
                world = client.get_world()
                vehicle, vehicles = self._pick_vehicle(world)
                map_name = world.get_map().name

                if vehicle is None:
                    self.motion_update.emit({
                        "found": False,
                        "map": map_name,
                        "vehicles": len(vehicles),
                        "reason": "ego vehicle not found",
                    })
                else:
                    loc = vehicle.get_location()
                    vel = vehicle.get_velocity()
                    speed_kmh = ((vel.x * vel.x + vel.y * vel.y + vel.z * vel.z) ** 0.5) * 3.6
                    self.motion_update.emit({
                        "found": True,
                        "map": map_name,
                        "actor_id": vehicle.id,
                        "type_id": vehicle.type_id,
                        "role": vehicle.attributes.get("role_name", ""),
                        "x": float(loc.x),
                        "y": float(loc.y),
                        "z": float(loc.z),
                        "speed_kmh": float(speed_kmh),
                        "vehicles": len(vehicles),
                    })
            except Exception as exc:
                self.motion_update.emit({"found": False, "error": str(exc)})

            steps = max(1, self.interval_ms // 100)
            for _ in range(steps):
                if not self._running:
                    break
                self.msleep(100)


class AsyncRuntime:
    """Dedicated standard asyncio loop for serial/real-vehicle coroutines.

    The GUI uses the normal Qt event loop. Real-vehicle async work is kept off
    qasync to avoid qasync/QTimer assertion failures on Windows when Qt signals
    schedule coroutines from slots.
    """

    def __init__(self):
        self._loop = None
        self._ready = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._run_loop, name="MitsuAsyncRuntime", daemon=True)
        self._thread.start()

    def _run_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    def submit(self, coro):
        if self._closed:
            try:
                coro.close()
            except Exception:
                pass
            raise RuntimeError("AsyncRuntime is closed")
        if not self._ready.wait(timeout=5.0) or self._loop is None:
            try:
                coro.close()
            except Exception:
                pass
            raise RuntimeError("AsyncRuntime event loop did not start")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass
        try:
            self._thread.join(timeout=2.0)
        except RuntimeError:
            pass


class AppController(QObject):
    async_task_completed = Signal(str, object, str)

    def __init__(self, view: MainWindow):
        super().__init__()
        self.async_runtime = AsyncRuntime()
        self.async_task_completed.connect(self._handle_async_task_completed)
        self.view = view
        self.video_receiver = VideoReceiverThread()
        self.video_receiver.frame_received.connect(self.view.update_camera_frame)
        self.video_receiver.start()
        self.vehicle = VehicleState()
        self.serial = SerialManager()
        self.telemetry = TelemetryRecorder()

        self.runtime_mode = "carla"
        self.real_mission_prepared = False
        self.real_mission = None
        self.vehicle_control = None
        self._last_real_gear_ignore_log_at = 0.0
        self.vehicle_adapter_factory = None
        self.vehicle_gateway = None
        if VEHICLE_CONTROL_AVAILABLE:
            try:
                self.vehicle_gateway = VehicleGateway(serial_manager=self.serial)
                self.vehicle_adapter_factory = VehicleAdapterFactory(
                    real_adapter=RealSerialVehicleAdapter(self.serial, self.vehicle_gateway),
                    mock_adapter=MockVehicleAdapter(),
                    loopback_adapter=MockVehicleAdapter(label="TEST_SERIAL_LOOPBACK"),
                )
                self.vehicle_control = VehicleControlService(
                    adapter_factory=self.vehicle_adapter_factory,
                    arbiter=ControlArbiter(),
                )
                self.vehicle_control.telemetry_changed.connect(self.handle_vehicle_control_telemetry)
                self.vehicle_control.state_changed.connect(self.handle_vehicle_control_state)
                self.vehicle_control.event_logged.connect(self.handle_vehicle_control_event)
                self.vehicle_control.activation_blocked.connect(self.handle_vehicle_activation_blocked)
                if hasattr(self.vehicle_control, "nav_goal_changed"):
                    self.vehicle_control.nav_goal_changed.connect(self.handle_vehicle_nav_goal)
            except Exception as exc:
                print(f"REAL CONTROL: init failed: {exc}")
                self.vehicle_control = None
        
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

        # Startup watchdog. It checks real movement, not agent commands.
        # A stuck route can still produce changing steer/throttle/brake, so those
        # values are deliberately ignored for timeout decisions.
        self.route_startup_timeout_ms = 90000
        self.route_min_progress_speed_kmh = 1.0
        self.route_min_progress_distance_m = 2.0
        self.route_watchdog_interval_ms = 1000
        self.carla_watchdog_host = "127.0.0.1"
        self.carla_watchdog_port = 2000
        self._carla_motion_monitor = None
        self._route_started_monotonic = None
        self._route_last_telemetry_monotonic = None
        self._route_progress_seen = False
        self._route_progress_reason = ""
        self._route_last_physical_progress_monotonic = None
        self._route_last_progress_location = None
        self._route_has_motion_measurement = False
        self._route_start_location = None
        self._route_last_location = None
        self._route_carla_actor_id = None
        self._route_carla_start_location = None
        self._route_carla_last_location = None
        self._route_carla_last_speed_kmh = 0.0
        self._route_carla_vehicle_seen = False
        self._route_carla_last_update_monotonic = None
        self._route_carla_last_error = ""
        self._route_skip_after_stop = False
        self._route_skip_reason = ""

        # A scenario can output telemetry forever while the ego vehicle is blocked
        # at the spawn point. Startup pass/fail is therefore based on total CARLA
        # displacement from the first observed ego location, not on commands and
        # not on dashboard fallback physics.
        self.route_startup_min_distance_m = 5.0
        self.route_stop_grace_ms = 15000
        self._route_total_distance_m = 0.0
        self._route_debug_last_log_monotonic = None
        self._route_force_continue_timer = QTimer()
        self._route_force_continue_timer.setSingleShot(True)
        self._route_force_continue_timer.timeout.connect(self._force_continue_after_skip_if_needed)

        self.ui_update_timer = QTimer()
        self.ui_update_timer.timeout.connect(self.update_view)
        self.ui_update_timer.start(50)

        self.physics_timer = QTimer()
        self.physics_timer.timeout.connect(self.step_virtual_physics)

        self.route_watchdog_timer = QTimer()
        self.route_watchdog_timer.setInterval(1000)
        self.route_watchdog_timer.timeout.connect(self.check_route_startup_watchdog)
        self.route_watchdog_timer.start()

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
        if hasattr(self.view, "real_mission_validated"):
            self.view.real_mission_validated.connect(self.handle_real_mission_validated)
        if hasattr(self.view, "real_speed_cap_changed"):
            self.view.real_speed_cap_changed.connect(self.handle_real_speed_cap_changed)
        
        self.serial.connection_status.connect(self.view.set_connection_status)
        self.serial.connection_status.connect(self.handle_serial_connection_status)
        self.serial.data_received.connect(self.handle_can_packet)
        self.latest_frame_path = os.path.join("outputs", "visualizations", "latest_front_cam.png")
        self.init_commands()


    def _schedule_async(self, coro, label="async task"):
        """Run a coroutine on the dedicated asyncio worker, not on qasync.

        Do not touch Qt widgets from the coroutine itself. Return a small dict
        result and update the UI from _handle_async_task_completed(), which runs
        in the Qt main thread through a queued signal.
        """
        try:
            future = self.async_runtime.submit(coro)
        except Exception as exc:
            try:
                coro.close()
            except Exception:
                pass
            self.async_task_completed.emit(label, None, str(exc))
            return None

        def _done(done_future):
            result = None
            error = ""
            try:
                result = done_future.result()
            except (asyncio.CancelledError, concurrent.futures.CancelledError):
                error = "cancelled"
            except Exception as exc:
                error = str(exc)
            self.async_task_completed.emit(label, result, error)

        future.add_done_callback(_done)
        return future

    @Slot(str, object, str)
    def _handle_async_task_completed(self, label, result, error):
        result = result or {}
        error = str(error or "")

        if error and error != "cancelled":
            print(f"ASYNC WORKER: {label} failed: {error}")

        if label == "connect_vehicle_control_device":
            ok = bool(result.get("ok")) and not error
            device_text = result.get("device_text", "vehicle")
            message = result.get("message") or (f"Vehicle connected: {device_text}" if ok else "Vehicle connect error")
            self.is_connected = ok
            self.is_virtual = False
            self.view.set_connection_status(ok, message)
            if hasattr(self.view, "set_real_readiness"):
                self.view.set_real_readiness(
                    route=self.real_mission_prepared,
                    pose=True,
                    cameras=True,
                    ai=self.ai_control_requested,
                    vehicle=ok,
                )
            return

        if label == "disconnect_vehicle_control_device":
            self.is_connected = False
            self.control_active = False
            if hasattr(self.view, "set_real_readiness"):
                self.view.set_real_readiness(route=self.real_mission_prepared, pose=True, cameras=True, ai=self.ai_control_requested, vehicle=False)
            return

        if label == "real_control_toggle":
            requested = bool(result.get("requested"))
            active = bool(result.get("active")) and not error
            self.control_active = active
            if requested and not active:
                self._set_control_button_checked_later(False)
            if not requested:
                self._set_control_button_checked_later(False)
            message = result.get("message")
            if message:
                self.view.statusBar().showMessage(message)
            return

        if label == "real_ai_preview_disable":
            self.control_active = False
            self._set_control_button_checked_later(False)
            if self.vehicle_control is not None:
                self.vehicle_control.set_ai_preview_enabled(False)
            if hasattr(self.view, "set_real_readiness"):
                self.view.set_real_readiness(
                    route=self.real_mission_prepared,
                    pose=True,
                    cameras=True,
                    ai=False,
                    vehicle=self.is_connected,
                )
            message = result.get("message") or "AI Preview disabled"
            self.view.statusBar().showMessage(message)
            return

        if label == "serial_connect":
            # SerialManager emits connection_status itself.
            return

        if label in {"real_manual_command", "real_gear_request"}:
            return

    def shutdown(self):
        """Stop background workers before QApplication destroys QThreads.

        Without this, closing the app while VideoReceiverThread/CARLA watchdog/
        LeadAgentThread is still alive can produce:
            QThread: Destroyed while thread '' is still running
        """
        if getattr(self, "_shutdown_done", False):
            return
        self._shutdown_done = True

        for timer_name in (
            "raw_telemetry_timer",
            "ui_update_timer",
            "physics_timer",
            "route_watchdog_timer",
            "_route_force_continue_timer",
        ):
            timer = getattr(self, timer_name, None)
            try:
                if timer is not None:
                    timer.stop()
            except Exception:
                pass

        try:
            self._stop_carla_motion_monitor()
        except Exception:
            pass

        thread = getattr(self, "agent_thread", None)
        if thread is not None:
            try:
                if hasattr(thread, "stop"):
                    thread.stop()
            except Exception:
                pass
            try:
                if hasattr(thread, "isRunning") and thread.isRunning():
                    thread.wait(1500)
                if hasattr(thread, "isRunning") and thread.isRunning():
                    thread.terminate()
                    thread.wait(1500)
            except Exception:
                pass
            self.agent_thread = None

        try:
            if getattr(self, "video_receiver", None) is not None:
                self.video_receiver.stop()
        except Exception:
            pass

        try:
            self.serial.close()
        except Exception:
            pass

        try:
            self.async_runtime.shutdown()
        except Exception:
            pass

    def _descriptor_from_text(self, device_text):
        if VEHICLE_CONTROL_AVAILABLE:
            return DeviceDescriptor.from_combo_text(device_text)
        return None

    def _is_real_vehicle_mode_text(self, device_text):
        descriptor = self._descriptor_from_text(device_text)
        return bool(descriptor and descriptor.is_real_control_like)

    def _set_runtime_mode_from_device(self, device_text):
        descriptor = self._descriptor_from_text(device_text)
        if descriptor is None or descriptor.kind == DeviceKind.VIRTUAL_DEMO:
            self.runtime_mode = "carla"
            if hasattr(self.view, "set_ui_mode"):
                self.view.set_ui_mode("carla")
            return
        self.runtime_mode = "real"
        if hasattr(self.view, "set_ui_mode"):
            self.view.set_ui_mode("real")

    async def _connect_vehicle_control_device(self, device_text):
        if self.vehicle_control is None:
            return {
                "ok": False,
                "device_text": device_text,
                "message": "Real vehicle control modules unavailable",
            }
        try:
            await self.vehicle_control.connect_device(device_text)
            return {
                "ok": True,
                "device_text": device_text,
                "message": f"Vehicle connected: {device_text}",
            }
        except Exception as exc:
            print(f"REAL CONTROL: connect error: {exc}")
            return {
                "ok": False,
                "device_text": device_text,
                "message": f"Vehicle connect error: {exc}",
            }

    async def _disconnect_vehicle_control_device(self):
        if self.vehicle_control is not None:
            try:
                await self.vehicle_control.disconnect()
            except Exception as exc:
                print(f"REAL CONTROL: disconnect error: {exc}")
                return {"ok": False, "message": str(exc)}
        return {"ok": True, "message": "Disconnected"}

    @Slot(object)
    def handle_vehicle_control_telemetry(self, telemetry):
        try:
            self.vehicle.speed = float(getattr(telemetry, "speed_kmh", 0.0) or 0.0)
            self.vehicle.angle = int(float(getattr(telemetry, "angle_deg", 0.0) or 0.0))
            self.vehicle.target_angle = int(float(getattr(telemetry, "target_angle_deg", 0.0) or 0.0))
            self.vehicle.accel = int(float(getattr(telemetry, "accel_pct", 0.0) or 0.0))
            self.vehicle.brake = int(float(getattr(telemetry, "brake_pct", 0.0) or 0.0))
            gear = getattr(telemetry, "gear", None)
            self.vehicle.gear = int(gear.value) if hasattr(gear, "value") else int(gear or 1)
            if hasattr(self.view, "set_real_readiness"):
                self.view.set_real_readiness(
                    route=self.real_mission_prepared,
                    pose=bool(getattr(telemetry, "pose_valid", False)),
                    cameras=True,
                    ai=self.ai_control_requested,
                    vehicle=bool(getattr(telemetry, "connected", False)) and bool(getattr(telemetry, "heartbeat_ok", False)),
                )
        except Exception as exc:
            print(f"REAL CONTROL: telemetry mapping error: {exc}")

    @Slot(str)
    def handle_vehicle_control_state(self, state):
        if hasattr(self.view, "set_real_vehicle_state"):
            self.view.set_real_vehicle_state(state)

    @Slot(str)
    def handle_vehicle_control_event(self, message):
        print(f"REAL CONTROL: {message}")

    @Slot(object)
    def handle_vehicle_nav_goal(self, goal):
        if hasattr(self.view, "set_real_nav_goal"):
            self.view.set_real_nav_goal(goal)

    @Slot(str)
    def handle_vehicle_activation_blocked(self, message):
        self.control_active = False
        self._set_control_button_checked_later(False)
        self.view.statusBar().showMessage(f"Activation blocked: {message}")
        if hasattr(self.view, "set_real_vehicle_state"):
            self.view.set_real_vehicle_state("BLOCKED", message)

    @Slot(dict)
    def handle_real_mission_validated(self, mission):
        if self.runtime_mode != "real":
            return
        self.real_mission_prepared = True
        self.real_mission = dict(mission or {})
        if self.vehicle_control is not None and Mission is not None:
            try:
                metadata = dict(self.real_mission.get("metadata") or {})
                routing_provider = str(metadata.get("routing_provider") or self.real_mission.get("routing_provider") or "direct").lower()
                if (
                    routing_provider in {"osrm", "osm", "openstreetmap"}
                    and RoadRouteRequest is not None
                    and OsrmRoadRouteProvider is not None
                    and (metadata.get("start_geo") or self.real_mission.get("start_geo"))
                    and (metadata.get("goal_geo") or self.real_mission.get("goal_geo"))
                ):
                    try:
                        print("REAL CONTROL: road routing requested: provider=OSRM")
                        m = OsrmRoadRouteProvider().build_mission(RoadRouteRequest.from_mission_dict(self.real_mission))
                        road_meta = dict(getattr(m, "metadata", {}) or {})
                        raw_points = road_meta.get("raw_route_points", "?")
                        lane_policy = road_meta.get("lane_policy", "-")
                        lane_offset = road_meta.get("lane_offset_m", "-")
                        print(
                            f"REAL CONTROL: road routing OK: raw_points={raw_points}, "
                            f"waypoints={len(getattr(m, 'waypoints', []) or [])}, "
                            f"trajectory={road_meta.get('trajectory_geometry', 'road')}, "
                            f"lane_policy={lane_policy}, offset={lane_offset}m"
                        )
                    except Exception as route_exc:
                        print(f"REAL CONTROL: road routing failed, fallback to direct A→B: {route_exc}")
                        m = CoordinateRoutePlanner().build_from_ab(ABRouteRequest.from_dict(self.real_mission))
                elif self.real_mission.get("start") and self.real_mission.get("goal") and CoordinateRoutePlanner is not None:
                    m = CoordinateRoutePlanner().build_from_ab(ABRouteRequest.from_dict(self.real_mission))
                elif self.real_mission.get("waypoints") and Waypoint is not None:
                    speed_cap = float(self.real_mission.get("speed_cap_kmh", 3.0))
                    m = Mission(
                        mission_id=str(self.real_mission.get("mission_id") or "real_mission"),
                        name=str(self.real_mission.get("name") or "Real Mission"),
                        goal_label=str(self.real_mission.get("goal_label") or ""),
                        speed_cap_kmh=speed_cap,
                        waypoints=tuple(Waypoint.from_dict(wp, speed_cap) for wp in self.real_mission.get("waypoints", [])),
                        metadata=dict(self.real_mission.get("metadata") or {}),
                    )
                else:
                    m = Mission.default_test_mission()
                self.vehicle_control.set_mission(m)
                if hasattr(self.view, "set_real_mission_summary"):
                    self.view.set_real_mission_summary(m)
            except Exception as exc:
                print(f"REAL CONTROL: mission build error: {exc}")
                self.vehicle_control.set_mission(Mission.default_test_mission())
        self.view.statusBar().showMessage("Mission validated. Enable AI Preview, then Activate Control")
        if hasattr(self.view, "set_real_readiness"):
            self.view.set_real_readiness(route=True, pose=True, cameras=True, ai=self.ai_control_requested, vehicle=self.is_connected)

    @Slot(float)
    def handle_real_speed_cap_changed(self, value):
        if self.real_mission is not None:
            self.real_mission["speed_cap_kmh"] = float(value)

    async def _handle_real_control_toggle(self, is_active):
        if self.vehicle_control is None:
            return {
                "ok": False,
                "requested": bool(is_active),
                "active": False,
                "message": "Real vehicle control modules unavailable",
            }
        if is_active:
            ok = await self.vehicle_control.activate_control()
            return {
                "ok": bool(ok),
                "requested": True,
                "active": bool(ok),
                "message": "AI control active" if ok else "Activation blocked",
            }
        await self.vehicle_control.deactivate_control(reason="user_requested")
        return {
            "ok": True,
            "requested": False,
            "active": False,
            "message": "AI control disabled",
        }

    async def _disable_real_ai_preview_after_disengage(self):
        if self.vehicle_control is None:
            return {"ok": False, "message": "Real vehicle control modules unavailable"}
        if self.control_active:
            await self.vehicle_control.deactivate_control(reason="ai_preview_disabled")
        return {"ok": True, "message": "AI Preview disabled after disengage"}

    def init_commands(self):
        self.cmd_gear = utils.Serial_Data(bytearray.fromhex("AA 00000000 3300 00 02 01 00 00 00 01 00 00"))
        self.cmd_accel = utils.Serial_Data(bytearray.fromhex("AA 00000000 3800 00 02 01 00 00 00 01 00 00"))
        self.cmd_brake = utils.Serial_Data(bytearray.fromhex("AA 00000000 3700 00 02 01 00 00 00 01 00 00"))
        self.cmd_angle = utils.Serial_Data(bytearray.fromhex("AA 00000000 3200 00 02 01 00 00 00 01 00 00"))
        self.cmd_cruise = utils.Serial_Data(bytearray.fromhex("AA 00000000 7700 00 02 01 00 00 00 01 00 00"))

    def handle_connect(self, port_name):
        # Подготовленная очередь не сбрасывается при подключении: пользователь может
        # выбрать/подготовить маршруты до Connect, после Connect или после AI Control.
        self._set_runtime_mode_from_device(port_name)
        if port_name == "VIRTUAL_DEMO_MODE":
            self.is_virtual = True
            self.is_connected = True
            self.physics_timer.start(PHYSICS_UPDATE_RATE_MS)
            self.view.set_connection_status(True, "Virtual Mode Active")
        elif self.runtime_mode == "real":
            self.is_virtual = False
            self.is_connected = False
            self._schedule_async(self._connect_vehicle_control_device(port_name), "connect_vehicle_control_device")
        else:
            self.is_virtual = False
            self.is_connected = False
            self._schedule_async(self.serial.connect_serial(port_name), "serial_connect")

    @Slot(bool, str)
    def handle_serial_connection_status(self, is_connected, message):
        self.is_connected = bool(is_connected)

    def handle_disconnect(self):
        if self.runtime_mode == "real":
            self._schedule_async(self._disconnect_vehicle_control_device(), "disconnect_vehicle_control_device")
        else:
            self.abort_active_scenario("Disconnected", keep_plan=True)
        self.is_virtual = False
        self.is_connected = False
        self.control_active = False
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
        if self.runtime_mode == "real":
            self._schedule_async(self._handle_real_control_toggle(is_active), "real_control_toggle")
            return
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
        if self.runtime_mode == "real":
            if bool(is_active):
                if self.vehicle_control is not None:
                    self.vehicle_control.set_ai_preview_enabled(True)
                if hasattr(self.view, "set_real_readiness"):
                    self.view.set_real_readiness(route=self.real_mission_prepared, pose=True, cameras=True, ai=True, vehicle=self.is_connected)
                self.view.statusBar().showMessage("AI Preview включен")
                return

            # Turning AI Preview off while AI has authority must first disengage
            # control. Do not let the service remain in AI_ACTIVE with preview off.
            if self.control_active:
                self.view.statusBar().showMessage("AI Preview будет выключен после безопасного отключения управления")
                self._schedule_async(self._disable_real_ai_preview_after_disengage(), "real_ai_preview_disable")
                return

            if self.vehicle_control is not None:
                self.vehicle_control.set_ai_preview_enabled(False)
            if hasattr(self.view, "set_real_readiness"):
                self.view.set_real_readiness(route=self.real_mission_prepared, pose=True, cameras=True, ai=False, vehicle=self.is_connected)
            self.view.statusBar().showMessage("AI Preview выключен")
            return
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

    def _start_carla_motion_monitor(self):
        self._stop_carla_motion_monitor()
        monitor = CarlaMotionMonitorThread(
            host=self.carla_watchdog_host,
            port=self.carla_watchdog_port,
            interval_ms=self.route_watchdog_interval_ms if hasattr(self, "route_watchdog_interval_ms") else 1000,
        )
        monitor.motion_update.connect(self.handle_carla_motion_update)
        monitor.status_update.connect(lambda message: print(f"AI WATCHDOG: {message}"))
        self._carla_motion_monitor = monitor
        monitor.start()

    def _stop_carla_motion_monitor(self):
        monitor = self._carla_motion_monitor
        self._carla_motion_monitor = None
        if monitor is not None:
            try:
                monitor.stop()
            except Exception as exc:
                print(f"AI WATCHDOG: ошибка остановки CARLA monitor: {exc}")

    @Slot(object)
    def handle_carla_motion_update(self, info):
        if self._route_started_monotonic is None or self._route_skip_after_stop:
            return
        if not isinstance(info, dict):
            return

        now = time.monotonic()
        self._route_carla_last_update_monotonic = now

        if not info.get("found"):
            error = info.get("error") or info.get("reason") or "ego vehicle not found"
            self._route_carla_last_error = str(error)
            return

        self._route_carla_vehicle_seen = True
        self._route_carla_last_error = ""
        self._route_has_motion_measurement = True

        actor_id = info.get("actor_id")
        location = (
            self._safe_float(info.get("x")),
            self._safe_float(info.get("y")),
            self._safe_float(info.get("z", 0.0)) or 0.0,
        )
        if location[0] is None or location[1] is None:
            location = None

        if actor_id != self._route_carla_actor_id:
            self._route_carla_actor_id = actor_id
            self._route_carla_start_location = location
            self._route_last_progress_location = location
            self._route_total_distance_m = 0.0
            print(
                f"AI WATCHDOG: найден ego actor id={actor_id}, "
                f"type={info.get('type_id', '-')}, role={info.get('role', '-')}, "
                f"start={location}"
            )

        self._route_carla_last_location = location

        if self._route_carla_start_location is not None and location is not None:
            self._route_total_distance_m = self._distance_2d(self._route_carla_start_location, location)
            self._update_location_progress(location, "CARLA position")

        speed_kmh = self._safe_float(info.get("speed_kmh"))
        if speed_kmh is not None:
            self._route_carla_last_speed_kmh = speed_kmh
            try:
                if hasattr(self.vehicle, "apply_external_speed_kmh"):
                    self.vehicle.apply_external_speed_kmh(speed_kmh, smooth=True)
                else:
                    self.vehicle.speed = speed_kmh
            except Exception:
                pass

        # Diagnostic heartbeat so it is clear that the watchdog is alive. It is
        # intentionally sparse to avoid flooding the UI log.
        if self._route_debug_last_log_monotonic is None or now - self._route_debug_last_log_monotonic >= 15.0:
            self._route_debug_last_log_monotonic = now
            elapsed = self._route_elapsed_seconds()
            print(
                f"AI WATCHDOG: route {self.current_route_index + 1}/{len(self.pending_route_queue)} "
                f"elapsed={elapsed:.0f}s, speed={self._route_carla_last_speed_kmh:.1f} km/h, "
                f"dist={self._route_total_distance_m:.1f} m"
            )

    def _reset_route_watchdog(self):
        self._stop_carla_motion_monitor()
        self._route_started_monotonic = time.monotonic()
        self._route_last_telemetry_monotonic = None
        self._route_progress_seen = False
        self._route_progress_reason = ""
        self._route_last_physical_progress_monotonic = None
        self._route_last_progress_location = None
        self._route_has_motion_measurement = False
        self._route_start_location = None
        self._route_last_location = None
        self._route_carla_actor_id = None
        self._route_carla_start_location = None
        self._route_carla_last_location = None
        self._route_carla_last_speed_kmh = 0.0
        self._route_carla_vehicle_seen = False
        self._route_carla_last_update_monotonic = None
        self._route_carla_last_error = ""
        self._route_skip_after_stop = False
        self._route_skip_reason = ""
        self._route_total_distance_m = 0.0
        self._route_debug_last_log_monotonic = None
        if hasattr(self, "_route_force_continue_timer"):
            self._route_force_continue_timer.stop()
        self._start_carla_motion_monitor()

    def _clear_route_watchdog(self):
        self._stop_carla_motion_monitor()
        self._route_started_monotonic = None
        self._route_last_telemetry_monotonic = None
        self._route_progress_seen = False
        self._route_progress_reason = ""
        self._route_last_physical_progress_monotonic = None
        self._route_last_progress_location = None
        self._route_has_motion_measurement = False
        self._route_start_location = None
        self._route_last_location = None
        self._route_carla_actor_id = None
        self._route_carla_start_location = None
        self._route_carla_last_location = None
        self._route_carla_last_speed_kmh = 0.0
        self._route_carla_vehicle_seen = False
        self._route_carla_last_update_monotonic = None
        self._route_carla_last_error = ""
        self._route_total_distance_m = 0.0
        self._route_debug_last_log_monotonic = None
        if hasattr(self, "_route_force_continue_timer"):
            self._route_force_continue_timer.stop()

    def _mark_route_progress(self, reason):
        if not self._route_progress_seen:
            print(f"AI WATCHDOG: маршрут начал движение/прогресс: {reason}")
        self._route_progress_seen = True
        self._route_progress_reason = str(reason or "progress")
        self._route_last_physical_progress_monotonic = time.monotonic()

    @staticmethod
    def _safe_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _extract_speed_kmh_from_telemetry(self, data):
        if not isinstance(data, dict):
            return None

        for key in ("speed_kmh", "velocity_kmh", "kmh"):
            if key in data:
                return self._safe_float(data.get(key))

        for key in ("speed_mps", "velocity_mps"):
            if key in data:
                value = self._safe_float(data.get(key))
                return None if value is None else value * 3.6

        for key in ("speed", "velocity"):
            if key in data:
                value = self._safe_float(data.get(key))
                if value is None:
                    return None
                # Same heuristic as VehicleState.apply_telemetry: small values are m/s.
                return value * 3.6 if abs(value) <= 35.0 else value

        return None

    def _extract_location_from_telemetry(self, data):
        if not isinstance(data, dict):
            return None

        key_sets = (
            ("x", "y", "z"),
            ("pos_x", "pos_y", "pos_z"),
            ("position_x", "position_y", "position_z"),
            ("location_x", "location_y", "location_z"),
            ("ego_x", "ego_y", "ego_z"),
        )
        for keys in key_sets:
            if keys[0] in data and keys[1] in data:
                x = self._safe_float(data.get(keys[0]))
                y = self._safe_float(data.get(keys[1]))
                z = self._safe_float(data.get(keys[2], 0.0))
                if x is not None and y is not None:
                    return (x, y, z or 0.0)

        for key in ("location", "position", "loc", "gps", "ego_location"):
            value = data.get(key)
            if isinstance(value, dict):
                x = self._safe_float(value.get("x") or value.get("lat") or value.get("latitude"))
                y = self._safe_float(value.get("y") or value.get("lon") or value.get("longitude"))
                z = self._safe_float(value.get("z") or value.get("alt") or value.get("altitude") or 0.0)
                if x is not None and y is not None:
                    return (x, y, z or 0.0)
            elif isinstance(value, (list, tuple)) and len(value) >= 2:
                x = self._safe_float(value[0])
                y = self._safe_float(value[1])
                z = self._safe_float(value[2]) if len(value) >= 3 else 0.0
                if x is not None and y is not None:
                    return (x, y, z or 0.0)

        return None

    @staticmethod
    def _distance_2d(a, b):
        if not a or not b:
            return 0.0
        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        return (dx * dx + dy * dy) ** 0.5

    def _update_location_progress(self, location, source):
        if location is None:
            return
        self._route_has_motion_measurement = True
        if self._route_last_progress_location is None:
            self._route_last_progress_location = location
            return
        distance = self._distance_2d(self._route_last_progress_location, location)
        if distance >= self.route_min_progress_distance_m:
            self._route_last_progress_location = location
            self._mark_route_progress(f"{source} moved {distance:.1f} m")

    def _update_route_watchdog_from_telemetry(self, data):
        if self._route_started_monotonic is None:
            return

        # Only record that the agent is producing telemetry. Do not use telemetry
        # speed/position as route progress here: in this project trace_log.jsonl
        # can keep changing even when the CARLA ego actor is physically stuck.
        self._route_last_telemetry_monotonic = time.monotonic()

    def _route_elapsed_seconds(self):
        if self._route_started_monotonic is None:
            return 0.0
        return max(0.0, time.monotonic() - self._route_started_monotonic)

    def check_route_startup_watchdog(self):
        if self._route_started_monotonic is None:
            return
        if self._route_skip_after_stop or self.queue_stop_requested:
            return
        if not (self.control_active and self.route_plan_prepared and self.current_route_index >= 0):
            return
        if self.agent_thread is None and not self.ai_agent_loading and not self.ai_agent_running:
            return

        now = time.monotonic()
        timeout_s = self.route_startup_timeout_ms / 1000.0
        elapsed_ms = (now - self._route_started_monotonic) * 1000.0

        # Startup-deadlock rule: after N seconds the ego actor must have actually
        # displaced from its first observed CARLA position. Commands, telemetry
        # updates and small steering/brake oscillations do not count.
        if elapsed_ms >= self.route_startup_timeout_ms:
            if not self._route_carla_vehicle_seen:
                detail = self._route_carla_last_error or "ego vehicle not found"
                reason = f"CARLA ego vehicle не найден {timeout_s:.0f} секунд ({detail})"
                self._skip_current_route_due_to_timeout(reason)
                return

            if self._route_total_distance_m < self.route_startup_min_distance_m:
                reason = (
                    f"за {timeout_s:.0f} секунд ego actor проехал только "
                    f"{self._route_total_distance_m:.1f} м < {self.route_startup_min_distance_m:.1f} м; "
                    "сценарий считается зависшим"
                )
                self._skip_current_route_due_to_timeout(reason)
                return

        # After the route has clearly started, also catch long hard-stalls. This
        # uses CARLA displacement deltas only, not agent commands.
        if self._route_last_physical_progress_monotonic is not None:
            stalled_ms = (now - self._route_last_physical_progress_monotonic) * 1000.0
            if stalled_ms >= self.route_startup_timeout_ms:
                reason = (
                    f"нет нового CARLA-смещения {timeout_s:.0f} секунд "
                    f"после последнего прогресса ({self._route_progress_reason}); "
                    f"dist_from_start={self._route_total_distance_m:.1f} м"
                )
                self._skip_current_route_due_to_timeout(reason)

    def _skip_current_route_due_to_timeout(self, reason):
        if self._route_skip_after_stop:
            return

        total = len(self.pending_route_queue)
        idx = self.current_route_index
        if idx < 0:
            return

        message = f"Таймаут запуска маршрута {idx + 1}/{total}: {reason}"
        print(f"AI WATCHDOG: {message}")
        self.view.statusBar().showMessage(message)
        if hasattr(self.view, "set_route_loading_status"):
            self.view.set_route_loading_status(message)
        elif hasattr(self.view, "set_route_runtime_state"):
            self.view.set_route_runtime_state("loading", message)

        self._route_skip_after_stop = True
        self._route_skip_reason = reason
        self._agent_stop_requested = True

        stopped_now = self._request_stop_current_agent(message)
        if stopped_now:
            self._continue_after_route_skip()
        else:
            print(
                f"AI WATCHDOG: ждем завершения LeadAgentThread до "
                f"{self.route_stop_grace_ms / 1000:.1f}s; затем будет принудительный переход"
            )
            self._route_force_continue_timer.start(self.route_stop_grace_ms)

    def _force_continue_after_skip_if_needed(self):
        if not self._route_skip_after_stop:
            return

        thread = self.agent_thread
        if thread is not None:
            try:
                if hasattr(thread, "isRunning") and thread.isRunning():
                    print(
                        "AI WATCHDOG: LeadAgentThread не завершился после stop(); "
                        "принудительно завершаем QThread перед переходом к следующему маршруту"
                    )
                    thread.terminate()
                    thread.wait(5000)
            except RuntimeError:
                pass
            except Exception as exc:
                print(f"AI WATCHDOG: ошибка принудительного завершения LeadAgentThread: {exc}")
            self.agent_thread = None

        self._continue_after_route_skip()

    def _continue_after_route_skip(self):
        total = len(self.pending_route_queue)
        skipped_index = self.current_route_index
        reason = self._route_skip_reason or "таймаут запуска"
        if hasattr(self, "_route_force_continue_timer"):
            self._route_force_continue_timer.stop()

        self._clear_route_watchdog()
        self._route_skip_after_stop = False
        self._route_skip_reason = ""
        self._agent_stop_requested = False
        self.ai_agent_loading = False
        self.ai_agent_running = False
        self.raw_telemetry_timer.stop()
        self.raw_telemetry_reader = None

        if (
            self.queue_mode == "queue"
            and self.control_active
            and self.ai_control_requested
            and self.route_plan_prepared
            and skipped_index + 1 < total
        ):
            self.current_route_index = skipped_index + 1
            msg = f"Маршрут {skipped_index + 1}/{total} пропущен: {reason}. Запуск следующего"
            print(f"AI WATCHDOG: {msg}")
            self.view.statusBar().showMessage(msg)
            if hasattr(self.view, "set_route_queue_position"):
                self.view.set_route_queue_position(self.current_route_index + 1, total)
            QTimer.singleShot(self.route_transition_delay_ms, self._start_current_route)
        else:
            self._finish_route_queue(
                f"Маршрут остановлен по таймауту: {reason}. Следующего маршрута нет"
            )

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
        self._route_skip_after_stop = False
        self._route_skip_reason = ""

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
        self._route_skip_after_stop = False
        self._route_skip_reason = ""
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
            self._reset_route_watchdog()

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
            self._clear_route_watchdog()
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
        self._route_skip_after_stop = False
        self._route_skip_reason = ""
        self._clear_route_watchdog()
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

        if self._route_skip_after_stop:
            self._continue_after_route_skip()
            return

        self._clear_route_watchdog()

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
        self._route_skip_after_stop = False
        self._route_skip_reason = ""
        self._clear_route_watchdog()
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
            self._update_route_watchdog_from_telemetry(data)
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
        # During watchdog skip we intentionally stop the current LeadAgentThread.
        # Any shutdown noise from ScenarioRunner/CARLA must not cancel the prepared
        # queue; handle_agent_finished() will continue with the next route.
        if self._route_skip_after_stop:
            print(f"AI ERROR during watchdog skip, ignored for queue continuation: {error}")
            self.view.statusBar().showMessage(f"AI watchdog: stopping timed-out route")
            return

        self.ai_agent_loading = False
        self.ai_agent_running = False
        self._clear_route_watchdog()
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
        if self.runtime_mode == "real":
            if self.vehicle_control is None or not self.is_connected:
                return
            # In real/mock mode manual commands are available after AI authority
            # has been disengaged. While AI is active, keyboard input is ignored
            # rather than mixed into the autonomy command stream.
            if self.control_active:
                return
            self._schedule_async(self.vehicle_control.submit_manual_command(angle, accel, brake), "real_manual_command")
            return
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
        if self.runtime_mode == "real":
            if self.vehicle_control is None or not self.is_connected:
                return
            if self.control_active:
                now = time.monotonic()
                if now - self._last_real_gear_ignore_log_at > 1.0:
                    self._last_real_gear_ignore_log_at = now
                    print("REAL CONTROL: ручная передача заблокирована: сначала отключите Activate Control")
                return
            self._schedule_async(self.vehicle_control.request_manual_gear(gear_idx), "real_gear_request")
            return
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
    # Use the plain Qt event loop for the GUI. Async serial/real-control work
    # runs in AppController.async_runtime on a dedicated standard asyncio loop.
    # This avoids qasync QTimer assertion failures on Windows while preserving
    # the existing CARLA/QThread simulation flow.
    app = QApplication(sys.argv)
    main_window = MainWindow()
    controller = AppController(main_window)
    app.aboutToQuit.connect(controller.shutdown)
    main_window.show()
    sys.exit(app.exec())
