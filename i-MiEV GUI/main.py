import sys
import argparse
import asyncio
import concurrent.futures
import threading
import subprocess
import json
import os
import time
import importlib
import xml.etree.ElementTree as ET
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, QObject, Slot, QThread, Signal, qInstallMessageHandler 
from datetime import datetime
from config import PHYSICS_UPDATE_RATE_MS
from ui.main_window import MainWindow
from core.vehicle import VehicleState
from hardware.serial_comm import SerialManager

try:
    from vehicle_control import (
        DeviceDescriptor, DeviceKind, Gear, ControlIntent, Mission, Waypoint, Pose2D,
        VehicleGateway, RealSerialVehicleAdapter, MockVehicleAdapter,
        VehicleAdapterFactory, VehicleControlService, ControlArbiter,
        ABRouteRequest, CoordinateRoutePlanner, RoadRouteRequest, OsrmRoadRouteProvider,
        AgentPrediction, RealAgentBridge, RealVehicleSafetyConfig,
    )
    VEHICLE_CONTROL_AVAILABLE = True
except Exception as _vehicle_control_import_error:
    VEHICLE_CONTROL_AVAILABLE = False
    DeviceDescriptor = DeviceKind = Gear = ControlIntent = Mission = Waypoint = Pose2D = None
    VehicleGateway = RealSerialVehicleAdapter = MockVehicleAdapter = None
    VehicleAdapterFactory = VehicleControlService = ControlArbiter = None
    ABRouteRequest = CoordinateRoutePlanner = RoadRouteRequest = OsrmRoadRouteProvider = None
    AgentPrediction = RealAgentBridge = RealVehicleSafetyConfig = None

import utils 
from core.telemetry import TelemetryRecorder, RawTelemetryJsonlReader
from lead_integration import LeadAgentThread
from PySide6.QtGui import QPixmap, QImage

import zmq
# --- MITSU_QT_FONT_WARNING_FILTER_BEGIN ---
def _mitsu_qt_message_handler(mode, context, message):
    msg = str(message)
    if msg.startswith("QFont::setPointSize: Point size <= 0"):
        return
    try:
        sys.__stderr__.write(msg + "\\n")
    except Exception:
        pass

qInstallMessageHandler(_mitsu_qt_message_handler)
# --- MITSU_QT_FONT_WARNING_FILTER_END ---
 

try:
    from hardware.camera_zmq import (
        CameraZmqConfig,
        ZmqCameraCellReceiverThread,
        RealCameraAgentAnalyzerThread,
    )
except Exception as _camera_zmq_import_error:
    CameraZmqConfig = None
    ZmqCameraCellReceiverThread = None
    RealCameraAgentAnalyzerThread = None

try:
    from hardware.gstreamer_udp_camera import (
        GStreamerUdpH265ReceiverThread,
        UdpH265CameraSpec,
    )
except Exception as _gstreamer_udp_import_error:
    GStreamerUdpH265ReceiverThread = None
    UdpH265CameraSpec = None

try:
    from hardware.opencv_udp_camera import (
        OpenCvUdpH265ReceiverThread,
        OpenCvUdpH265CameraSpec,
    )
except Exception as _opencv_udp_import_error:
    OpenCvUdpH265ReceiverThread = None
    OpenCvUdpH265CameraSpec = None

try:
    from hardware.pose_provider import JsonPoseProviderThread
except Exception:
    JsonPoseProviderThread = None

try:
    from vehicle_control.pose_modes import (
        DeadReckoningPoseState,
        advance_dead_reckoning_pose,
        mission_targets_from_dict,
        parse_xyyaw,
    )
except Exception:
    DeadReckoningPoseState = None
    advance_dead_reckoning_pose = mission_targets_from_dict = parse_xyyaw = None

# --- MITSU_SAFE_MOCK_READY_PRINT_FILTER_V11B_BEGIN ---
# Safe duplicate-log filter for mock readiness.
#
# This does not modify readiness/control logic. It only suppresses repeated
# identical console messages in this module.

import builtins as _mitsu_v11b_builtins
from datetime import datetime as _mitsu_log_datetime

if not hasattr(_mitsu_v11b_builtins, "_mitsu_original_print"):
    _mitsu_v11b_builtins._mitsu_original_print = _mitsu_v11b_builtins.print

_mitsu_v11b_seen_mock_ready_logs = set()


def print(*args, **kwargs):  # noqa: A001 - intentional module-local print wrapper
    try:
        message = " ".join(str(arg) for arg in args)
    except Exception:
        message = ""

    suppress_prefixes = (
        "MOCK MODE: camera stream marked ready",
        "MOCK MODE: camera readiness forced at service level",
    )

    for prefix in suppress_prefixes:
        if message.startswith(prefix):
            if prefix in _mitsu_v11b_seen_mock_ready_logs:
                return
            _mitsu_v11b_seen_mock_ready_logs.add(prefix)
            break

    timestamp = _mitsu_log_datetime.now().strftime("%H:%M:%S.%f")[:-3]
    return _mitsu_v11b_builtins._mitsu_original_print(f"[{timestamp}]", *args, **kwargs)

# --- MITSU_SAFE_MOCK_READY_PRINT_FILTER_V11B_END ---

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
        # MITSU_REALTIME_ZMQ_RECEIVER_PATCH
        # Operator preview must show the newest road frame, not a queued backlog.
        # If the UI thread is slower than the camera service, drop old JPEGs.
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVHWM, 1)
        try:
            socket.setsockopt(zmq.CONFLATE, 1)
        except Exception:
            pass
        
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

        # MITSU_REALTIME_ZMQ_RECEIVER_CLOSE
        try:
            socket.close(0)
        except Exception:
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
    real_direct_model_prediction_ready = Signal(dict)

    def __init__(self, view: MainWindow):
        super().__init__()
        self.async_runtime = AsyncRuntime()
        self.async_task_completed.connect(self._handle_async_task_completed)
        self.real_direct_model_prediction_ready.connect(self.handle_real_agent_prediction)
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
        self.real_camera_process = None
        self.real_camera_receivers = []
        self.real_agent_analyzer = None
        self.real_direct_model_adapter = None
        self.real_direct_model_frame_parts = {}
        self.real_direct_model_latest_sample = None
        self.real_direct_model_sample_seq = 0
        self.real_direct_model_consumed_sample_seq = 0
        self.real_direct_model_part_seq = 0
        self.real_direct_model_sample_generation = 0
        self.real_direct_model_condition = threading.Condition()
        self.real_direct_model_worker_thread = None
        self.real_direct_model_worker_stop = False
        self.real_direct_model_auto_worker = True
        self.real_direct_model_frame_seq = 0
        self.real_camera_status = {"wide_90": False, "narrow_50": False}
        self.real_agent_bridge = RealAgentBridge() if RealAgentBridge is not None else None
        self.real_pose_provider = None
        self.real_pose_mode = os.environ.get("MITSU_REAL_POSE_MODE", "external").strip().lower() or "external"
        self.real_dead_reckoning_pose = None
        self.real_dead_reckoning_targets = ()
        self.real_dead_reckoning_last_ts = None
        self.real_safety_config = RealVehicleSafetyConfig.load() if RealVehicleSafetyConfig is not None else None
        if self.real_safety_config is not None:
            print(f"REAL SAFETY: {self.real_safety_config.describe()}")
        self.vehicle_adapter_factory = None
        self.vehicle_gateway = None
        if VEHICLE_CONTROL_AVAILABLE:
            try:
                self.vehicle_gateway = VehicleGateway(serial_manager=self.serial)
                self.vehicle_adapter_factory = VehicleAdapterFactory(
                    real_adapter=RealSerialVehicleAdapter(self.serial, self.vehicle_gateway, safety_config=self.real_safety_config),
                    mock_adapter=MockVehicleAdapter(),
                    loopback_adapter=MockVehicleAdapter(label="TEST_SERIAL_LOOPBACK"),
                )
                arbiter = ControlArbiter.from_safety_config(self.real_safety_config) if self.real_safety_config is not None else ControlArbiter()
                self.vehicle_control = VehicleControlService(
                    adapter_factory=self.vehicle_adapter_factory,
                    arbiter=arbiter,
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

        # РћС‡РµСЂРµРґСЊ РјР°СЂС€СЂСѓС‚РѕРІ Р·Р°РїСѓСЃРєР°РµС‚СЃСЏ РєРѕРѕРїРµСЂР°С‚РёРІРЅРѕ: РѕРґРёРЅ LeadAgentThread РЅР° РѕРґРёРЅ XML.
        # Р’С‚РѕСЂРѕР№ РјР°СЂС€СЂСѓС‚ СЃС‚Р°СЂС‚СѓРµС‚ С‚РѕР»СЊРєРѕ РїРѕСЃР»Рµ finished С‚РµРєСѓС‰РµРіРѕ thread. РџСЂРёРЅСѓРґРёС‚РµР»СЊРЅС‹Р№
        # РїРµСЂРµС…РѕРґ Рє СЃР»РµРґСѓСЋС‰РµРјСѓ РјР°СЂС€СЂСѓС‚Сѓ СѓР±СЂР°РЅ, РїРѕС‚РѕРјСѓ С‡С‚Рѕ РѕРЅ СЂРІР°Р» QThread/CARLA lifecycle.
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

        self.real_dead_reckoning_timer = QTimer()
        self.real_dead_reckoning_timer.setInterval(100)
        self.real_dead_reckoning_timer.timeout.connect(self._tick_real_dead_reckoning_pose)

        self.view.connect_requested.connect(self.handle_connect)
        self.view.disconnect_requested.connect(self.handle_disconnect)
        self.view.control_toggled.connect(self.handle_control_toggle)
        self.view.gear_requested.connect(self.handle_gear_change)
        self.view.manual_input_updated.connect(self.handle_manual_input)
        self.view.telemetry_toggled.connect(self.handle_telemetry_toggle)
        self.view.ai_toggled.connect(self.handle_ai_toggle)
        if hasattr(self.view, "manual_toggled"):
            self.view.manual_toggled.connect(self.handle_manual_toggle)

        # РќРѕРІС‹Р№ UI РѕС‚РїСЂР°РІР»СЏРµС‚ РѕРґРёРЅ СЃРїРёСЃРѕРє: 1 РјР°СЂС€СЂСѓС‚ => РѕРґРёРЅРѕС‡РЅС‹Р№ Р·Р°РїСѓСЃРє, N РјР°СЂС€СЂСѓС‚РѕРІ => РѕС‡РµСЂРµРґСЊ.
        # РЎС‚Р°СЂС‹Рµ СЃРёРіРЅР°Р»С‹ single/queue РѕСЃС‚Р°РІР»РµРЅС‹ РєР°Рє fallback, РЅРѕ РѕСЃРЅРѕРІРЅРѕР№ РїСѓС‚СЊ вЂ” route_launch_requested.
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
            if ok:
                self._start_real_camera_stack()
                self._start_real_pose_provider()
            if hasattr(self.view, "set_real_readiness"):
                self.view.set_real_readiness(
                    route=self.real_mission_prepared,
                    pose=False,
                    cameras=False,
                    ai=self.ai_control_requested,
                    vehicle=ok,
                )
            return

        if label == "disconnect_vehicle_control_device":
            self.is_connected = False
            self.control_active = False
            self._stop_real_camera_stack()
            self._stop_real_pose_provider()
            if hasattr(self.view, "set_real_readiness"):
                self.view.set_real_readiness(route=self.real_mission_prepared, pose=False, cameras=False, ai=self.ai_control_requested, vehicle=False)
            return

        if label == "real_control_toggle":
            requested = bool(result.get("requested"))
            active = bool(result.get("active")) and not error
            self.control_active = active
            if requested and not active:
                self._set_control_button_checked_later(False)
            if not requested:
                self._set_control_button_checked_later(False)
                if self.real_agent_analyzer is not None:
                    self.real_agent_analyzer.set_ai_enabled(False)
                self._stop_real_direct_model_worker()
                self._reset_real_direct_model_sample()
            if requested and active:
                if self.real_agent_analyzer is not None:
                    self.real_agent_analyzer.set_ai_enabled(True)
                self._reset_real_direct_model_sample()
                self._ensure_real_direct_model_worker()
            message = result.get("message")
            if message:
                self.view.statusBar().showMessage(message)
            return

        if label == "real_ai_preview_disable":
            self.control_active = False
            self._set_control_button_checked_later(False)
            self._stop_real_direct_model_worker()
            self._reset_real_direct_model_sample()
            if self.real_agent_analyzer is not None:
                self.real_agent_analyzer.set_ai_enabled(False)
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

        if label in {"real_manual_command", "real_gear_request", "real_agent_intent"}:
            return

    def _real_camera_config_path(self):
        candidates = [
            os.environ.get("MITSU_REAL_CAMERAS_CONFIG", ""),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "real_cameras.json"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "real_cameras.example.json"),
        ]
        for value in candidates:
            if value and os.path.exists(value):
                return value
        return ""

    def _load_real_camera_config(self):
        config_path = self._real_camera_config_path()
        if not config_path:
            return {}
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as exc:
            print(f"REAL CAMERA: cannot read config: {exc}")
            return {}

    def _real_camera_backend(self):
        env_value = os.environ.get("MITSU_REAL_CAMERA_BACKEND", "").strip().lower()
        if env_value:
            return env_value
        payload = self._load_real_camera_config()
        value = str(payload.get("backend") or payload.get("camera_backend") or "").strip().lower()
        return value or "zmq"

    def _load_real_camera_ports(self):
        ports = {"wide_90": 5556, "narrow_50": 5557}
        payload = self._load_real_camera_config()
        for name, data in (payload.get("cameras") or {}).items():
            role = str(data.get("role") or name).lower()
            port = data.get("zmq_port")
            if port is None:
                continue
            if "wide" in role:
                ports["wide_90"] = int(port)
            elif "narrow" in role:
                ports["narrow_50"] = int(port)
        return ports

    def _load_udp_h265_camera_specs(self):
        if UdpH265CameraSpec is None:
            return []
        return self._build_udp_h265_camera_specs(UdpH265CameraSpec)

    def _load_opencv_udp_h265_camera_specs(self):
        if OpenCvUdpH265CameraSpec is None:
            return []
        return self._build_udp_h265_camera_specs(OpenCvUdpH265CameraSpec)

    def _build_udp_h265_camera_specs(self, spec_cls):
        payload = self._load_real_camera_config()
        decoder = str(
            os.environ.get("MITSU_GSTREAMER_H265_DECODER")
            or payload.get("decoder")
            or payload.get("h265_decoder")
            or "avdec_h265"
        ).strip()
        default_payload = int(payload.get("rtp_payload", 96) or 96)
        jitter_latency_ms = int(payload.get("jitter_latency_ms", 0) or 0)

        specs = []
        cameras = payload.get("cameras") or {}
        if cameras:
            for index, (name, data) in enumerate(cameras.items(), start=1):
                if not bool(data.get("enabled", True)):
                    continue
                port = data.get("udp_port", data.get("port"))
                if port is None:
                    continue
                role = str(data.get("role") or name)
                role_l = role.lower()
                if "wide" in role_l:
                    ui_name = "wide_90"
                elif "narrow" in role_l:
                    ui_name = "narrow_50"
                else:
                    ui_name = str(data.get("ui_name") or f"camera_{index}")
                source_path = str(data.get("source") or "").strip()
                if source_path and not os.path.isabs(source_path):
                    source_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), source_path)
                sdp_video = self._read_sdp_video(source_path) if source_path else {}
                sdp_port = sdp_video.get("port")
                sdp_payload = sdp_video.get("payload")
                kwargs = dict(
                    name=ui_name,
                    port=int(sdp_port or port),
                    role=role,
                    payload=int(data.get("payload", sdp_payload or default_payload) or default_payload),
                    width=int(data.get("width", 1280) or 1280),
                    height=int(data.get("height", 720) or 720),
                )
                if spec_cls is UdpH265CameraSpec:
                    kwargs["decoder"] = str(data.get("decoder") or decoder)
                    kwargs["jitter_latency_ms"] = int(data.get("jitter_latency_ms", jitter_latency_ms) or 0)
                if spec_cls is OpenCvUdpH265CameraSpec:
                    kwargs["host"] = str(data.get("listen_ip") or data.get("host") or payload.get("listen_ip") or "0.0.0.0")
                specs.append(spec_cls(**kwargs))
        if specs:
            return specs
        if spec_cls is OpenCvUdpH265CameraSpec:
            return [
                spec_cls("wide_90", 5601, role="front_wide"),
                spec_cls("narrow_50", 5602, role="front_narrow"),
                spec_cls("camera_3", 5603, role="aux_3"),
                spec_cls("camera_4", 5604, role="aux_4"),
            ]
        return [
            spec_cls("wide_90", 5601, role="front_wide", decoder=decoder),
            spec_cls("narrow_50", 5602, role="front_narrow", decoder=decoder),
            spec_cls("camera_3", 5603, role="aux_3", decoder=decoder),
            spec_cls("camera_4", 5604, role="aux_4", decoder=decoder),
        ]

    @staticmethod
    def _read_sdp_video(path):
        try:
            with open(path, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if line.startswith("m=video "):
                        parts = line.split()
                        if len(parts) >= 4:
                            return {"port": int(parts[1]), "payload": int(parts[3])}
        except Exception:
            return {}
        return {}

    def _start_real_camera_stack(self):
        backend = self._real_camera_backend()
        if backend in {"opencv_udp", "opencv_udp_h265", "windows_udp_h265", "cv2_udp_h265"}:
            if OpenCvUdpH265ReceiverThread is None or OpenCvUdpH265CameraSpec is None:
                print("REAL CAMERA: opencv_udp module unavailable")
                return
            specs = self._load_opencv_udp_h265_camera_specs()
            if not specs:
                print("REAL CAMERA: no OpenCV UDP/H265 camera specs")
                return
            if not self.real_camera_receivers:
                for spec in specs:
                    receiver = OpenCvUdpH265ReceiverThread(spec)
                    if str(spec.name) in {"wide_90", "narrow_50"}:
                        receiver.frame_received.connect(self.view.update_real_camera_cell)
                        if hasattr(receiver, "model_frame_received"):
                            receiver.model_frame_received.connect(self.handle_real_direct_model_frame)
                    receiver.status_changed.connect(self.handle_real_camera_cell_status)
                    self.real_camera_receivers.append(receiver)
                    receiver.start()
                print(
                    "REAL CAMERA: OpenCV UDP/H265 receivers started: "
                    + ", ".join(f"{spec.name}@:{spec.port}" for spec in specs)
                )
                self._ensure_real_direct_model_adapter()
            return

        if backend in {"gstreamer_udp", "udp_h265", "gst_udp_h265", "gstreamer"}:
            if GStreamerUdpH265ReceiverThread is None or UdpH265CameraSpec is None:
                print("REAL CAMERA: gstreamer_udp module unavailable")
                return
            specs = self._load_udp_h265_camera_specs()
            if not specs:
                print("REAL CAMERA: no UDP/H265 camera specs")
                return
            if not self.real_camera_receivers:
                for spec in specs:
                    receiver = GStreamerUdpH265ReceiverThread(spec)
                    if str(spec.name) in {"wide_90", "narrow_50"}:
                        receiver.frame_received.connect(self.view.update_real_camera_cell)
                        if hasattr(receiver, "model_frame_received"):
                            receiver.model_frame_received.connect(self.handle_real_direct_model_frame)
                    receiver.status_changed.connect(self.handle_real_camera_cell_status)
                    self.real_camera_receivers.append(receiver)
                    receiver.start()
                print(
                    "REAL CAMERA: GStreamer UDP/H265 receivers started: "
                    + ", ".join(f"{spec.name}@:{spec.port}" for spec in specs)
                )
                self._ensure_real_direct_model_adapter()
            return

        if ZmqCameraCellReceiverThread is None or CameraZmqConfig is None:
            print("REAL CAMERA: camera_zmq module unavailable")
            return

        config_path = self._real_camera_config_path()
        external = os.environ.get("MITSU_REAL_CAMERA_EXTERNAL", "0").lower() in {"1", "true", "yes", "on"}
        if config_path and self.real_camera_process is None and not external:
            try:
                script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hardware", "dual_camera_service.py")
                self.real_camera_process = subprocess.Popen([sys.executable, script, "--config", config_path])
                print(f"REAL CAMERA: dual camera service started: {config_path}")
            except Exception as exc:
                print(f"REAL CAMERA: cannot start dual camera service: {exc}")
        elif external:
            print("REAL CAMERA: external camera service mode; not spawning dual_camera_service")
        else:
            print("REAL CAMERA: config not found; receivers will wait for external streams")

        ports = self._load_real_camera_ports()
        if not self.real_camera_receivers:
            for camera_name, port in (("wide_90", ports["wide_90"]), ("narrow_50", ports["narrow_50"])):
                receiver = ZmqCameraCellReceiverThread(camera_name, f"tcp://127.0.0.1:{port}")
                receiver.frame_received.connect(self.view.update_real_camera_cell)
                receiver.status_changed.connect(self.handle_real_camera_cell_status)
                self.real_camera_receivers.append(receiver)
                receiver.start()

        if RealCameraAgentAnalyzerThread is not None and self.real_agent_analyzer is None:
            cfg = CameraZmqConfig(
                wide_url=f"tcp://127.0.0.1:{ports['wide_90']}",
                narrow_url=f"tcp://127.0.0.1:{ports['narrow_50']}",
            )
            analyzer = RealCameraAgentAnalyzerThread(cfg, context_provider=self._build_real_agent_context)
            analyzer.camera_status_changed.connect(self.handle_real_agent_camera_status)
            analyzer.prediction_ready.connect(self.handle_real_agent_prediction)
            analyzer.analyzer_status_changed.connect(self.handle_real_agent_analyzer_status)
            self.real_agent_analyzer = analyzer
            analyzer.start()

    def _start_real_pose_provider(self):
        if str(getattr(self, "real_pose_mode", "external") or "external").lower() == "dead_reckoning_ab":
            self._start_real_dead_reckoning_pose()
            return
        if JsonPoseProviderThread is None or self.real_pose_provider is not None:
            return
        pose_path = os.environ.get("MITSU_REAL_POSE_JSON", "").strip()
        if not pose_path:
            return
        provider = JsonPoseProviderThread(pose_path)
        provider.pose_received.connect(self.handle_real_pose_payload)
        provider.status_changed.connect(lambda ok, msg: print(f"REAL POSE: {msg}"))
        self.real_pose_provider = provider
        provider.start()
        print(f"REAL POSE: provider started: {pose_path}")

    def _stop_real_pose_provider(self):
        self._stop_real_dead_reckoning_pose()
        provider = self.real_pose_provider
        self.real_pose_provider = None
        if provider is not None:
            try:
                provider.stop()
            except Exception:
                pass

    def _configure_real_dead_reckoning_pose(self, mission):
        if (
            DeadReckoningPoseState is None
            or mission_targets_from_dict is None
            or parse_xyyaw is None
        ):
            return False
        mission = dict(mission or {})
        try:
            seed = parse_xyyaw(mission.get("start") or "0,0,0")
            targets = mission_targets_from_dict(mission)
            if not targets:
                return False
            self.real_dead_reckoning_pose = DeadReckoningPoseState(seed.x_m, seed.y_m, seed.yaw_deg)
            self.real_dead_reckoning_targets = tuple(targets)
            self.real_dead_reckoning_last_ts = None
            return True
        except Exception as exc:
            print(f"REAL POSE: dead-reckoning setup failed: {exc}")
            return False

    def _start_real_dead_reckoning_pose(self):
        if (
            self.real_dead_reckoning_pose is None
            and self.real_mission is not None
            and not self._configure_real_dead_reckoning_pose(self.real_mission)
        ):
            return
        if self.real_dead_reckoning_pose is None:
            return
        self.real_dead_reckoning_last_ts = time.monotonic()
        if not self.real_dead_reckoning_timer.isActive():
            self.real_dead_reckoning_timer.start()
        self._submit_real_dead_reckoning_pose()
        print("REAL POSE: local A->B dead-reckoning provider started")

    def _stop_real_dead_reckoning_pose(self):
        timer = getattr(self, "real_dead_reckoning_timer", None)
        try:
            if timer is not None:
                timer.stop()
        except Exception:
            pass
        self.real_dead_reckoning_last_ts = None

    def _submit_real_dead_reckoning_pose(self):
        if self.vehicle_control is None or Pose2D is None or self.real_dead_reckoning_pose is None:
            return
        state = self.real_dead_reckoning_pose
        self.vehicle_control.submit_pose(Pose2D(
            x_m=float(state.x_m),
            y_m=float(state.y_m),
            yaw_deg=float(state.yaw_deg),
            valid=True,
            source="dead_reckoning_ab_no_gps",
        ))
        if hasattr(self.vehicle_control, "update_navigation_preview"):
            try:
                self.vehicle_control.update_navigation_preview()
            except Exception as exc:
                print(f"REAL POSE: navigation preview update failed: {exc}")

    def _tick_real_dead_reckoning_pose(self):
        if (
            self.real_dead_reckoning_pose is None
            or advance_dead_reckoning_pose is None
            or not self.real_dead_reckoning_targets
        ):
            return
        now = time.monotonic()
        last = self.real_dead_reckoning_last_ts or now
        self.real_dead_reckoning_last_ts = now
        telemetry = self.vehicle_control.get_telemetry() if self.vehicle_control is not None else None
        speed_kmh = float(getattr(telemetry, "speed_kmh", 0.0) or 0.0)
        self.real_dead_reckoning_pose = advance_dead_reckoning_pose(
            self.real_dead_reckoning_pose,
            self.real_dead_reckoning_targets,
            speed_kmh / 3.6,
            now - last,
        )
        self._submit_real_dead_reckoning_pose()

    @Slot(dict)
    def handle_real_pose_payload(self, payload):
        if self.vehicle_control is None or Pose2D is None:
            return
        try:
            pose = Pose2D(
                x_m=float(payload.get("x_m", 0.0)),
                y_m=float(payload.get("y_m", 0.0)),
                yaw_deg=float(payload.get("yaw_deg", 0.0)),
                valid=bool(payload.get("valid", True)),
                source=str(payload.get("source", "json_pose_provider")),
            )
            self.vehicle_control.submit_pose(pose)
            if hasattr(self.vehicle_control, "update_navigation_preview"):
                self.vehicle_control.update_navigation_preview()
        except Exception as exc:
            print(f"REAL POSE: payload rejected: {exc}")

    def _stop_real_camera_stack(self):
        analyzer = self.real_agent_analyzer
        self.real_agent_analyzer = None
        if analyzer is not None:
            try:
                analyzer.stop()
            except Exception:
                pass
        for receiver in list(self.real_camera_receivers):
            try:
                receiver.stop()
            except Exception:
                pass
        self.real_camera_receivers = []
        self._stop_real_direct_model_worker()
        self._reset_real_direct_model_sample()
        adapter = self.real_direct_model_adapter
        self.real_direct_model_adapter = None
        if adapter is not None:
            close = getattr(adapter, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    print(f"REAL MODEL PREVIEW: cleanup failed: {exc}")
        proc = self.real_camera_process
        self.real_camera_process = None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    @Slot(str, bool, str)
    def handle_real_camera_cell_status(self, camera_name, ok, message):
        self.real_camera_status[str(camera_name)] = bool(ok)
        all_ok = self.real_camera_status.get("wide_90", False) and self.real_camera_status.get("narrow_50", False)
        if self.vehicle_control is not None:
            self.vehicle_control.set_camera_status(all_ok)
        if hasattr(self.view, "set_real_camera_status"):
            self.view.set_real_camera_status(str(camera_name), bool(ok), str(message or ""))

    @Slot(bool, str)
    def handle_real_agent_camera_status(self, ok, message):
        if self.vehicle_control is not None:
            self.vehicle_control.set_camera_status(bool(ok))
        if hasattr(self.view, "set_real_agent_status"):
            self.view.set_real_agent_status(bool(ok), str(message or ""))

    @Slot(str)
    def handle_real_agent_analyzer_status(self, message):
        message = str(message or "")
        print(f"REAL AGENT: {message}")

    def _ensure_real_direct_model_adapter(self):
        if self.real_direct_model_adapter is not None:
            return self.real_direct_model_adapter
        spec = os.environ.get(
            "MITSU_REAL_AGENT_FACTORY",
            "real_agent_adapters.lead_real_model_0011_adapter:create_agent",
        ).strip()
        if not spec:
            print("REAL MODEL PREVIEW: model factory disabled; PID fallback remains active")
            return None
        try:
            module_name, func_name = spec.split(":", 1)
            module = importlib.import_module(module_name)
            factory = getattr(module, func_name)
            self.real_direct_model_adapter = factory()
            print(f"REAL MODEL PREVIEW: model connected: {spec}")
        except Exception as exc:
            self.real_direct_model_adapter = None
            print(f"REAL MODEL PREVIEW: model unavailable: {exc}; PID fallback remains active")
        return self.real_direct_model_adapter

    def _ensure_real_direct_model_sample_state(self):
        if not hasattr(self, "real_direct_model_condition"):
            self.real_direct_model_condition = threading.Condition()
        if not hasattr(self, "real_direct_model_frame_parts"):
            self.real_direct_model_frame_parts = {}
        if not hasattr(self, "real_direct_model_latest_sample"):
            self.real_direct_model_latest_sample = None
        if not hasattr(self, "real_direct_model_sample_seq"):
            self.real_direct_model_sample_seq = 0
        if not hasattr(self, "real_direct_model_consumed_sample_seq"):
            self.real_direct_model_consumed_sample_seq = 0
        if not hasattr(self, "real_direct_model_part_seq"):
            self.real_direct_model_part_seq = 0
        if not hasattr(self, "real_direct_model_sample_generation"):
            self.real_direct_model_sample_generation = 0
        if not hasattr(self, "real_direct_model_worker_thread"):
            self.real_direct_model_worker_thread = None
        if not hasattr(self, "real_direct_model_worker_stop"):
            self.real_direct_model_worker_stop = False
        if not hasattr(self, "real_direct_model_auto_worker"):
            self.real_direct_model_auto_worker = True

    def _real_direct_model_inference_enabled(self):
        return bool(getattr(self, "control_active", False)) and bool(getattr(self, "ai_control_requested", False))

    def _reset_real_direct_model_sample(self):
        self._ensure_real_direct_model_sample_state()
        with self.real_direct_model_condition:
            self.real_direct_model_frame_parts = {}
            self.real_direct_model_latest_sample = None
            self.real_direct_model_sample_seq = 0
            self.real_direct_model_consumed_sample_seq = 0
            self.real_direct_model_part_seq = 0
            self.real_direct_model_sample_generation += 1
            self.real_direct_model_condition.notify_all()

    def _ensure_real_direct_model_worker(self):
        self._ensure_real_direct_model_sample_state()
        if self._ensure_real_direct_model_adapter() is None:
            return None
        worker = self.real_direct_model_worker_thread
        if worker is not None and worker.is_alive():
            return worker
        with self.real_direct_model_condition:
            self.real_direct_model_worker_stop = False
        worker = threading.Thread(
            target=self._real_direct_model_worker_loop,
            name="MitsuRealDirectModelWorker",
            daemon=True,
        )
        self.real_direct_model_worker_thread = worker
        worker.start()
        return worker

    def _stop_real_direct_model_worker(self):
        self._ensure_real_direct_model_sample_state()
        worker = self.real_direct_model_worker_thread
        with self.real_direct_model_condition:
            self.real_direct_model_worker_stop = True
            self.real_direct_model_condition.notify_all()
        if worker is not None and worker.is_alive():
            try:
                worker.join(timeout=2.0)
            except RuntimeError:
                pass
        self.real_direct_model_worker_thread = None

    def _publish_real_direct_model_sample(self, camera_name, frame, captured_at=None):
        self._ensure_real_direct_model_sample_state()
        name = str(camera_name)
        if name not in {"wide_90", "narrow_50"}:
            return None
        now = time.monotonic()
        try:
            frame_ts = float(captured_at)
        except Exception:
            frame_ts = now
        if frame_ts <= 0.0:
            frame_ts = now
        with self.real_direct_model_condition:
            self.real_direct_model_part_seq += 1
            self.real_direct_model_frame_parts[name] = {
                "frame": frame,
                "ts": frame_ts,
                "received_ts": now,
                "part_seq": self.real_direct_model_part_seq,
            }
            parts = self.real_direct_model_frame_parts
            if not all(key in parts for key in ("wide_90", "narrow_50")):
                return None
            self.real_direct_model_sample_seq += 1
            sample = {
                "seq": self.real_direct_model_sample_seq,
                "frames": {key: parts[key]["frame"] for key in ("wide_90", "narrow_50")},
                "timestamps": {key: float(parts[key]["ts"]) for key in ("wide_90", "narrow_50")},
                "received_timestamps": {key: float(parts[key]["received_ts"]) for key in ("wide_90", "narrow_50")},
                "part_sequences": {key: int(parts[key]["part_seq"]) for key in ("wide_90", "narrow_50")},
                "created_at": now,
                "generation": int(self.real_direct_model_sample_generation),
            }
            self.real_direct_model_latest_sample = sample
            self.real_direct_model_condition.notify_all()
            return sample

    def _next_real_direct_model_sample(self, block=True):
        self._ensure_real_direct_model_sample_state()
        with self.real_direct_model_condition:
            while True:
                sample = self.real_direct_model_latest_sample
                sample_seq = int(sample.get("seq", 0)) if sample else 0
                consumed_seq = int(self.real_direct_model_consumed_sample_seq or 0)
                if sample is not None and sample_seq > consumed_seq and self._real_direct_model_inference_enabled():
                    self.real_direct_model_consumed_sample_seq = sample_seq
                    return dict(sample)
                if not block:
                    return None
                if bool(self.real_direct_model_worker_stop):
                    return None
                self.real_direct_model_condition.wait()

    def _predict_real_direct_model_sample(self, sample):
        adapter = self._ensure_real_direct_model_adapter()
        if adapter is None or not self._real_direct_model_inference_enabled():
            return None
        now = time.monotonic()
        try:
            frames = dict(sample.get("frames") or {})
            context_payload = self._build_real_agent_context()
            context_payload["input_sample_seq"] = int(sample.get("seq", 0))
            context_payload["input_sample_generation"] = int(sample.get("generation", 0))
            context_payload["input_sample_created_at_monotonic"] = float(sample.get("created_at", 0.0) or 0.0)
            context_payload["input_frame_timestamps_monotonic"] = dict(sample.get("timestamps") or {})
            context_payload["input_frame_part_sequences"] = dict(sample.get("part_sequences") or {})
            prediction = adapter.predict(frames, context_payload)
            if not prediction or not self._real_direct_model_inference_enabled():
                return None
            current_generation = int(getattr(self, "real_direct_model_sample_generation", 0) or 0)
            if int(sample.get("generation", -1)) != current_generation:
                return None
            payload = dict(prediction)
            self.real_direct_model_frame_seq += 1
            payload.setdefault("frame_id", self.real_direct_model_frame_seq)
            payload.setdefault("timestamp_monotonic", now)
            payload.setdefault("input_sample_seq", int(sample.get("seq", 0)))
            payload.setdefault("input_sample_generation", int(sample.get("generation", 0)))
            payload.setdefault("input_sample_created_at_monotonic", float(sample.get("created_at", 0.0) or 0.0))
            payload.setdefault("input_frame_timestamps_monotonic", dict(sample.get("timestamps") or {}))
            payload.setdefault("input_frame_part_sequences", dict(sample.get("part_sequences") or {}))
            if os.environ.get("MITSU_DEBUG_REAL_MODEL_PREVIEW", "0").lower() in {"1", "true", "yes", "on"}:
                print(
                    "REAL MODEL PREVIEW: "
                    f"sample={payload.get('input_sample_seq')} "
                    f"frame={payload.get('frame_id')} "
                    f"steer={float(payload.get('steer', payload.get('steer_norm', 0.0))):.3f} "
                    f"thr={float(payload.get('throttle', payload.get('thr', 0.0))):.3f} "
                    f"brk={float(payload.get('brake', payload.get('brk', 0.0))):.3f} "
                    f"pose={context_payload.get('pose_source', 'none')} "
                    f"target={context_payload.get('target_point')}"
                )
            return payload
        except Exception as exc:
            last = float(getattr(self, "_last_real_direct_model_error_at", 0.0) or 0.0)
            if now - last >= 1.0:
                self._last_real_direct_model_error_at = now
                print(f"REAL MODEL PREVIEW: inference error: {exc}")
            return None

    def _real_direct_model_worker_loop(self):
        while True:
            sample = self._next_real_direct_model_sample(block=True)
            if sample is None:
                return
            payload = self._predict_real_direct_model_sample(sample)
            if payload:
                try:
                    self.real_direct_model_prediction_ready.emit(payload)
                except Exception:
                    self.handle_real_agent_prediction(payload)

    def _consume_real_direct_model_sample_once_for_test(self):
        sample = self._next_real_direct_model_sample(block=False)
        if sample is None:
            return False
        payload = self._predict_real_direct_model_sample(sample)
        if payload:
            self.handle_real_agent_prediction(payload)
        return payload is not None

    @Slot(str, object)
    @Slot(str, object, float)
    def handle_real_direct_model_frame(self, camera_name, frame, captured_at=None):
        sample = self._publish_real_direct_model_sample(camera_name, frame, captured_at)
        if not self._real_direct_model_inference_enabled():
            return
        if sample is not None and bool(getattr(self, "real_direct_model_auto_worker", True)):
            self._ensure_real_direct_model_worker()

    def _build_real_agent_context(self):
        if self.vehicle_control is None or RealAgentBridge is None:
            return {}

        telemetry = self.vehicle_control.get_telemetry()
        goal = self.vehicle_control.get_current_goal() if hasattr(self.vehicle_control, "get_current_goal") else None
        payload = RealAgentBridge.build_model_command_payload(goal)

        def _as_float(value, default=0.0):
            try:
                return float(value)
            except Exception:
                return float(default)

        def _world_to_ego(point_x, point_y):
            # Model input convention:
            #   ego/current vehicle position == [0, 0]
            #   x = forward from vehicle
            #   y = lateral from vehicle
            import math

            ego_x = _as_float(getattr(telemetry, "x_m", 0.0))
            ego_y = _as_float(getattr(telemetry, "y_m", 0.0))
            ego_yaw = math.radians(_as_float(getattr(telemetry, "yaw_deg", 0.0)))

            dx = _as_float(point_x) - ego_x
            dy = _as_float(point_y) - ego_y
            cos_yaw = math.cos(ego_yaw)
            sin_yaw = math.sin(ego_yaw)

            forward = cos_yaw * dx + sin_yaw * dy
            lateral = -sin_yaw * dx + cos_yaw * dy
            return [float(forward), float(lateral)]

        target_point_previous = [0.0, 0.0]
        target_point = [0.0, 0.0]
        target_point_next = [0.0, 0.0]

        previous_world = None
        target_world = None
        next_world = None

        if goal is not None and getattr(goal, "is_valid", lambda: False)():
            target_world = [
                _as_float(getattr(goal, "target_x_m", 0.0)),
                _as_float(getattr(goal, "target_y_m", 0.0)),
            ]

            previous_world = [
                _as_float(getattr(goal, "previous_x_m", getattr(telemetry, "x_m", target_world[0]))),
                _as_float(getattr(goal, "previous_y_m", getattr(telemetry, "y_m", target_world[1]))),
            ]

            next_world = [
                _as_float(getattr(goal, "next_x_m", target_world[0])),
                _as_float(getattr(goal, "next_y_m", target_world[1])),
            ]

            target_point_previous = _world_to_ego(previous_world[0], previous_world[1])
            target_point = _world_to_ego(target_world[0], target_world[1])
            target_point_next = _world_to_ego(next_world[0], next_world[1])

            if os.environ.get("MITSU_DEBUG_NAV_EGO", "0").lower() in {"1", "true", "yes", "on"}:
                now = time.monotonic()
                last = float(getattr(self, "_last_nav_ego_debug_ts", 0.0) or 0.0)
                if now - last >= 1.0:
                    self._last_nav_ego_debug_ts = now
                    print(
                        "REAL AGENT NAV EGO: "
                        f"pose=({getattr(telemetry, 'x_m', 0.0):.2f},"
                        f"{getattr(telemetry, 'y_m', 0.0):.2f},"
                        f"{getattr(telemetry, 'yaw_deg', 0.0):.1f}) "
                        f"prev={target_point_previous} "
                        f"target={target_point} "
                        f"next={target_point_next}"
                    )

        return {
            "telemetry": telemetry,
            "goal": goal,
            "speed_kmh": float(getattr(telemetry, "speed_kmh", 0.0) or 0.0),
            "speed_mps": float(getattr(telemetry, "speed_kmh", 0.0) or 0.0) / 3.6,
            "pose_valid": bool(getattr(telemetry, "pose_valid", False)),
            "pose_source": str(getattr(telemetry, "pose_source", "none") or "none"),
            "pose_mode": str(getattr(self, "real_pose_mode", "external") or "external"),

            # Ego-local model inputs.
            "target_point_previous": target_point_previous,
            "target_point": target_point,
            "target_point_next": target_point_next,

            # Explicit aliases used by lead_real_model_0011_adapter.
            "target_point_previous_ego": target_point_previous,
            "target_point_ego": target_point,
            "target_point_next_ego": target_point_next,

            # Raw route/world points for debugging only.
            "target_point_previous_world": previous_world,
            "target_point_world": target_world,
            "target_point_next_world": next_world,

            **payload,
        }

    @Slot(dict)
    def handle_real_agent_prediction(self, payload):
        if str(getattr(self, "runtime_mode", "") or "") == "real" and not self._real_direct_model_inference_enabled():
            return
        if self.vehicle_control is None or self.real_agent_bridge is None or AgentPrediction is None:
            return
        try:
            self._log_real_agent_prediction(dict(payload or {}))
            prediction = AgentPrediction.from_dict(dict(payload or {}))
            goal = self.vehicle_control.get_current_goal() if hasattr(self.vehicle_control, "get_current_goal") else None
            intent = self.real_agent_bridge.build_intent(prediction, goal)
            self._schedule_async(self.vehicle_control.submit_external_agent_intent(intent), "real_agent_intent")
        except Exception as exc:
            print(f"REAL AGENT: prediction conversion failed: {exc}")

    def _log_real_agent_prediction(self, payload):
        now = time.monotonic()
        period = float(os.environ.get("MITSU_REAL_AGENT_LOG_PERIOD_SEC", "1.0") or 1.0)
        last = float(getattr(self, "_last_real_agent_prediction_log_at", 0.0) or 0.0)
        if now - last < period:
            return
        self._last_real_agent_prediction_log_at = now

        context = self._build_real_agent_context()
        goal = context.get("goal")

        def _fmt_pair(value):
            try:
                return f"({float(value[0]):.2f},{float(value[1]):.2f})"
            except Exception:
                return str(value)

        target_speed = payload.get("pred_target_speed_mps", payload.get("target_speed_mps", "n/a"))
        try:
            target_speed_text = f"{float(target_speed):.2f}m/s"
        except Exception:
            target_speed_text = str(target_speed)

        goal_text = ""
        if goal is not None:
            goal_text = (
                f" goal=({_fmt_pair([getattr(goal, 'target_x_m', 0.0), getattr(goal, 'target_y_m', 0.0)])})"
                f" option={getattr(goal, 'command', '-')}"
            )

        print(
            "REAL AGENT PRED: "
            f"frame={payload.get('frame_id', '-')} "
            f"steer={float(payload.get('steer', 0.0)):.3f} "
            f"thr={float(payload.get('throttle', 0.0)):.3f} "
            f"brk={float(payload.get('brake', 0.0)):.3f} "
            f"target_speed={target_speed_text} "
            f"speed={float(context.get('speed_mps', 0.0)):.2f}m/s "
            f"prev={_fmt_pair(context.get('target_point_previous'))} "
            f"target={_fmt_pair(context.get('target_point'))} "
            f"next={_fmt_pair(context.get('target_point_next'))} "
            f"world_target={_fmt_pair(context.get('target_point_world'))}"
            f"{goal_text}"
        )

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
            "real_dead_reckoning_timer",
        ):
            timer = getattr(self, timer_name, None)
            try:
                if timer is not None:
                    timer.stop()
            except Exception:
                pass

        try:
            self._stop_real_camera_stack()
            self._stop_real_pose_provider()
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

    def _is_carla_mode(self):
        return self.runtime_mode == "carla"

    def _is_real_mode(self):
        return self.runtime_mode == "real"

    def _ignore_non_carla_call(self, source):
        # Real/mock mode must never mutate CARLA route queue, watchdog or agent state.
        # Keep this log sparse: it is a developer guard, not a user-facing error.
        print(f"MODE GUARD: {source} ignored because active mode is {self.runtime_mode}")

    def _ignore_non_real_call(self, source):
        print(f"MODE GUARD: {source} ignored because active mode is {self.runtime_mode}")

    def _set_runtime_mode_from_device(self, device_text):
        descriptor = self._descriptor_from_text(device_text)
        next_mode = "carla" if descriptor is None or descriptor.kind == DeviceKind.VIRTUAL_DEMO else "real"

        if next_mode == self.runtime_mode:
            if hasattr(self.view, "set_ui_mode"):
                self.view.set_ui_mode(next_mode)
            return

        previous_mode = self.runtime_mode
        self.runtime_mode = next_mode

        if previous_mode == "carla" and next_mode == "real":
            # Do not carry a prepared CARLA route/agent state into real/mock mode.
            # If a CARLA scenario is actually running, stop only that CARLA scenario.
            try:
                if self.agent_thread is not None or self.ai_agent_loading or self.ai_agent_running:
                    self.abort_active_scenario("РџРµСЂРµРєР»СЋС‡РµРЅРёРµ РІ СЂРµР¶РёРј СЂРµР°Р»СЊРЅРѕРіРѕ Р°РІС‚Рѕ: CARLA-СЃС†РµРЅР°СЂРёР№ РѕСЃС‚Р°РЅРѕРІР»РµРЅ", keep_plan=False)
                else:
                    self.pending_route_queue = []
                    self.current_route_index = -1
                    self.queue_mode = None
                    self.route_plan_prepared = False
                    self.queue_stop_requested = False
                    self._clear_route_watchdog()
            except Exception as exc:
                print(f"MODE GUARD: РѕС€РёР±РєР° РѕС‡РёСЃС‚РєРё CARLA-СЃРѕСЃС‚РѕСЏРЅРёСЏ: {exc}")

        if previous_mode == "real" and next_mode == "carla":
            # Do not carry mission/AI authority state into CARLA mode.
            self._stop_real_camera_stack()
            self.real_mission_prepared = False
            self.real_mission = None
            try:
                if self.vehicle_control is not None:
                    self.vehicle_control.set_ai_preview_enabled(False)
            except Exception:
                pass

        if hasattr(self.view, "set_ui_mode"):
            self.view.set_ui_mode(next_mode)

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
        self.real_pose_mode = str(
            self.real_mission.get("pose_mode")
            or os.environ.get("MITSU_REAL_POSE_MODE", "external")
            or "external"
        ).strip().lower()
        if self.real_pose_mode == "dead_reckoning_ab":
            if self._configure_real_dead_reckoning_pose(self.real_mission) and self.is_connected:
                self._start_real_dead_reckoning_pose()
        else:
            self._stop_real_dead_reckoning_pose()
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
                        print(f"REAL CONTROL: road routing failed, fallback to direct Aв†’B: {route_exc}")
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
            self.view.set_real_readiness(route=True, pose=self.real_pose_mode == "dead_reckoning_ab", cameras=False, ai=self.ai_control_requested, vehicle=self.is_connected)

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
        current_state = getattr(getattr(self.vehicle_control, "state_machine", None), "state", None)
        state_text = getattr(current_state, "value", str(current_state)) if current_state is not None else "unknown"
        print(f"REAL CONTROL: toggle requested={bool(is_active)} state={state_text} control_active={bool(self.control_active)}")
        if is_active:
            manual_enabled = bool(
                hasattr(self.view, "is_manual_control_enabled")
                and self.view.is_manual_control_enabled()
            )
            try:
                self.vehicle_control.set_manual_mode_enabled(manual_enabled)
            except Exception:
                pass
            if manual_enabled:
                self.ai_control_requested = False
                try:
                    self.vehicle_control.set_ai_preview_enabled(False)
                except Exception:
                    pass
            ok = await self.vehicle_control.activate_control()
            return {
                "ok": bool(ok),
                "requested": True,
                "active": bool(ok),
                "message": ("Manual control active" if manual_enabled else "AI control active") if ok else "Activation blocked",
            }
        if not bool(self.control_active) and state_text not in {"AI_ACTIVE", "MANUAL_ACTIVE"}:
            print(f"REAL CONTROL: OFF toggle ignored because control is not active; state={state_text}")
            return {
                "ok": True,
                "requested": False,
                "active": False,
                "message": "Control is not active",
            }
        await self.vehicle_control.deactivate_control(reason="user_requested", park=True)
        still_manual = getattr(getattr(self.vehicle_control, "state_machine", None), "state", None)
        manual_active = still_manual is not None and getattr(still_manual, "value", str(still_manual)) == "MANUAL_ACTIVE"
        return {
            "ok": True,
            "requested": False,
            "active": manual_active,
            "message": "Manual control active" if manual_active else "Control disabled",
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
        # Hard split:
        #   VIRTUAL_DEMO_MODE  -> existing CARLA/LeadAgent route flow only.
        #   TEST_*/REAL COM    -> vehicle_control service only.
        # The branches below intentionally do not share route queues, watchdogs,
        # agent threads or CAN command paths.
        self._set_runtime_mode_from_device(port_name)
        if self._is_carla_mode():
            self.is_virtual = True
            self.is_connected = True
            self.control_active = False
            self.ai_agent_loading = False
            self.ai_agent_running = False
            self.physics_timer.start(PHYSICS_UPDATE_RATE_MS)
            self.view.set_connection_status(True, "Virtual Mode Active")
            return

        if self._is_real_mode():
            self.is_virtual = False
            self.is_connected = False
            self.control_active = False
            self.physics_timer.stop()
            self.raw_telemetry_timer.stop()
            self._clear_route_watchdog()
            self._schedule_async(self._connect_vehicle_control_device(port_name), "connect_vehicle_control_device")
            return

        # Fallback legacy serial path, kept only for compatibility.
        self.is_virtual = False
        self.is_connected = False
        self._schedule_async(self.serial.connect_serial(port_name), "serial_connect")

    @Slot(bool, str)
    def handle_serial_connection_status(self, is_connected, message):
        # Legacy serial status is authoritative only outside real-control mode.
        # In real/mock mode VehicleControlService publishes connection/readiness.
        if self._is_real_mode():
            return
        self.is_connected = bool(is_connected)

    def handle_disconnect(self):
        if self._is_real_mode():
            self._schedule_async(self._disconnect_vehicle_control_device(), "disconnect_vehicle_control_device")
        elif self._is_carla_mode():
            self.abort_active_scenario("Disconnected", keep_plan=True)

        self.is_virtual = False
        self.is_connected = False
        self.control_active = False
        self.physics_timer.stop()
        self.raw_telemetry_timer.stop()
        self.serial.close()
        self.view.set_connection_status(False, "Disconnected")

    def _set_control_button_checked_later(self, checked):
        btn = getattr(self.view, "btn_control", None)
        if btn is not None and btn.isChecked() != checked:
            QTimer.singleShot(0, lambda: btn.setChecked(checked))

    def _reject_control_activation(self, message):
        self.control_active = False
        print(f"AI CONTROL: Р·Р°РїСѓСЃРє РѕС‚РєР»РѕРЅРµРЅ: {message}")
        self.view.statusBar().showMessage(message)
        if hasattr(self.view, "set_route_runtime_state"):
            state = "prepared" if self.route_plan_prepared else "selected"
            self.view.set_route_runtime_state(state, message)
        elif hasattr(self.view, "set_route_error_status"):
            self.view.set_route_error_status(message)
        self._set_control_button_checked_later(False)

    def _validate_control_start_requirements(self):
        if not self.is_connected:
            return False, "РЎРЅР°С‡Р°Р»Р° РЅР°Р¶РјРёС‚Рµ Connect"
        if not self.ai_control_requested:
            return False, "Р’РєР»СЋС‡РёС‚Рµ AI Control"
        if not self.pending_route_queue:
            return False, "Р’С‹Р±РµСЂРёС‚Рµ РјР°СЂС€СЂСѓС‚ Рё РїРѕРґРіРѕС‚РѕРІСЊС‚Рµ Р·Р°РїСѓСЃРє"
        if not self.route_plan_prepared:
            count = len(self.pending_route_queue)
            if count > 1:
                return False, "РќР°Р¶РјРёС‚Рµ В«РџРѕРґРіРѕС‚РѕРІРёС‚СЊ РѕС‡РµСЂРµРґСЊВ» РїРµСЂРµРґ Activate Control"
            return False, "РќР°Р¶РјРёС‚Рµ В«РџРѕРґРіРѕС‚РѕРІРёС‚СЊ РјР°СЂС€СЂСѓС‚В» РїРµСЂРµРґ Activate Control"
        if self.agent_thread is not None or self.ai_agent_loading:
            return False, "РўРµРєСѓС‰РёР№ СЃС†РµРЅР°СЂРёР№ РµС‰Рµ Р·Р°РїСѓСЃРєР°РµС‚СЃСЏ РёР»Рё РІС‹РїРѕР»РЅСЏРµС‚СЃСЏ"
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
            print(f"AI CONTROL: РѕС€РёР±РєР° РѕС‚РїСЂР°РІРєРё cruise state: {exc}")

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

            print("AI CONTROL: Activate Control РІРєР»СЋС‡РµРЅ; Р·Р°РїСѓСЃРє РїРѕРґРіРѕС‚РѕРІР»РµРЅРЅРѕРіРѕ СЃС†РµРЅР°СЂРёСЏ")
            self.start_pending_route()
            return

        # РџРѕРІС‚РѕСЂРЅРѕРµ РЅР°Р¶Р°С‚РёРµ Activate Control РІС‹РєР»СЋС‡Р°РµС‚ Р°РєС‚РёРІРЅС‹Р№ СЃС†РµРЅР°СЂРёР№. РџР»Р°РЅ РјР°СЂС€СЂСѓС‚РѕРІ
        # РѕСЃС‚Р°РµС‚СЃСЏ РїРѕРґРіРѕС‚РѕРІР»РµРЅРЅС‹Рј, С‡С‚РѕР±С‹ РјРѕР¶РЅРѕ Р±С‹Р»Рѕ РїРѕРІС‚РѕСЂРЅРѕ СЃС‚Р°СЂС‚РѕРІР°С‚СЊ СЃ РїРµСЂРІРѕРіРѕ РјР°СЂС€СЂСѓС‚Р°.
        self.control_active = False
        self._send_cruise_state(False)
        self.abort_active_scenario("Control РІС‹РєР»СЋС‡РµРЅ: СЃС†РµРЅР°СЂРёР№ РѕСЃС‚Р°РЅРѕРІР»РµРЅ", keep_plan=True)

    def handle_ai_toggle(self, is_active):
        self.ai_control_requested = bool(is_active)
        if self.runtime_mode == "real":
            if bool(is_active):
                self._reset_real_direct_model_sample()
                self._start_real_camera_stack()
                if self.real_agent_analyzer is not None:
                    self.real_agent_analyzer.set_ai_enabled(False)
                if self.vehicle_control is not None:
                    self.vehicle_control.set_ai_preview_enabled(True)
                self.view.statusBar().showMessage("РџСЂРµРґРїСЂРѕСЃРјРѕС‚СЂ РР РІРєР»СЋС‡РµРЅ: Р°РіРµРЅС‚ РїРѕР»СѓС‡Р°РµС‚ РґРІРµ СЂРµР°Р»СЊРЅС‹Рµ РєР°РјРµСЂС‹")
                return

            # Turning AI Preview off while AI has authority must first disengage
            # control. Do not let the service remain in AI_ACTIVE with preview off.
            if self.control_active:
                self.view.statusBar().showMessage("РџСЂРµРґРїСЂРѕСЃРјРѕС‚СЂ РР Р±СѓРґРµС‚ РІС‹РєР»СЋС‡РµРЅ РїРѕСЃР»Рµ Р±РµР·РѕРїР°СЃРЅРѕРіРѕ РѕС‚РєР»СЋС‡РµРЅРёСЏ СѓРїСЂР°РІР»РµРЅРёСЏ")
                self._schedule_async(self._disable_real_ai_preview_after_disengage(), "real_ai_preview_disable")
                return

            if self.real_agent_analyzer is not None:
                self.real_agent_analyzer.set_ai_enabled(False)
            self._stop_real_direct_model_worker()
            self._reset_real_direct_model_sample()
            if self.vehicle_control is not None:
                self.vehicle_control.set_ai_preview_enabled(False)
            self.view.statusBar().showMessage("РџСЂРµРґРїСЂРѕСЃРјРѕС‚СЂ РР РІС‹РєР»СЋС‡РµРЅ")
            return
        if is_active:
            if self.route_plan_prepared:
                message = "AI Control РІРєР»СЋС‡РµРЅ. РќР°Р¶РјРёС‚Рµ Activate Control РґР»СЏ Р·Р°РїСѓСЃРєР°"
                state = "prepared"
            else:
                message = "AI Control РІРєР»СЋС‡РµРЅ. РџРѕРґРіРѕС‚РѕРІСЊС‚Рµ РјР°СЂС€СЂСѓС‚/РѕС‡РµСЂРµРґСЊ"
                state = "armed"
            self.view.statusBar().showMessage(message)
            if hasattr(self.view, "set_route_runtime_state"):
                self.view.set_route_runtime_state(state, message)
        else:
            self.abort_active_scenario("AI Control РІС‹РєР»СЋС‡РµРЅ", keep_plan=True)
            self.view.statusBar().showMessage("AI Control РІС‹РєР»СЋС‡РµРЅ")
            if hasattr(self.view, "set_route_runtime_state"):
                self.view.set_route_runtime_state("selected", "AI Control РІС‹РєР»СЋС‡РµРЅ")

    def handle_manual_toggle(self, is_active):
        self.manual_control_requested = bool(is_active)
        if self.runtime_mode != "real":
            return
        if self.vehicle_control is not None:
            self.vehicle_control.set_manual_mode_enabled(bool(is_active))
        if bool(is_active):
            self.ai_control_requested = False
            if self.real_agent_analyzer is not None:
                self.real_agent_analyzer.set_ai_enabled(False)
            if self.vehicle_control is not None:
                self.vehicle_control.set_ai_preview_enabled(False)
            if hasattr(self.view, "set_ai_checkbox"):
                self.view.set_ai_checkbox(False)
            self.view.statusBar().showMessage("Manual control enabled: mission/pose/cameras/AI Preview are not required")
        else:
            self.view.statusBar().showMessage("Manual control disabled")

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
                print(f"AI WATCHDOG: РѕС€РёР±РєР° РѕСЃС‚Р°РЅРѕРІРєРё CARLA monitor: {exc}")

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
                f"AI WATCHDOG: РЅР°Р№РґРµРЅ ego actor id={actor_id}, "
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
            print(f"AI WATCHDOG: РјР°СЂС€СЂСѓС‚ РЅР°С‡Р°Р» РґРІРёР¶РµРЅРёРµ/РїСЂРѕРіСЂРµСЃСЃ: {reason}")
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
        if not self._is_carla_mode():
            return
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
                reason = f"CARLA ego vehicle РЅРµ РЅР°Р№РґРµРЅ {timeout_s:.0f} СЃРµРєСѓРЅРґ ({detail})"
                self._skip_current_route_due_to_timeout(reason)
                return

            if self._route_total_distance_m < self.route_startup_min_distance_m:
                reason = (
                    f"Р·Р° {timeout_s:.0f} СЃРµРєСѓРЅРґ ego actor РїСЂРѕРµС…Р°Р» С‚РѕР»СЊРєРѕ "
                    f"{self._route_total_distance_m:.1f} Рј < {self.route_startup_min_distance_m:.1f} Рј; "
                    "СЃС†РµРЅР°СЂРёР№ СЃС‡РёС‚Р°РµС‚СЃСЏ Р·Р°РІРёСЃС€РёРј"
                )
                self._skip_current_route_due_to_timeout(reason)
                return

        # After the route has clearly started, also catch long hard-stalls. This
        # uses CARLA displacement deltas only, not agent commands.
        if self._route_last_physical_progress_monotonic is not None:
            stalled_ms = (now - self._route_last_physical_progress_monotonic) * 1000.0
            if stalled_ms >= self.route_startup_timeout_ms:
                reason = (
                    f"РЅРµС‚ РЅРѕРІРѕРіРѕ CARLA-СЃРјРµС‰РµРЅРёСЏ {timeout_s:.0f} СЃРµРєСѓРЅРґ "
                    f"РїРѕСЃР»Рµ РїРѕСЃР»РµРґРЅРµРіРѕ РїСЂРѕРіСЂРµСЃСЃР° ({self._route_progress_reason}); "
                    f"dist_from_start={self._route_total_distance_m:.1f} Рј"
                )
                self._skip_current_route_due_to_timeout(reason)

    def _skip_current_route_due_to_timeout(self, reason):
        if self._route_skip_after_stop:
            return

        total = len(self.pending_route_queue)
        idx = self.current_route_index
        if idx < 0:
            return

        message = f"РўР°Р№РјР°СѓС‚ Р·Р°РїСѓСЃРєР° РјР°СЂС€СЂСѓС‚Р° {idx + 1}/{total}: {reason}"
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
                f"AI WATCHDOG: Р¶РґРµРј Р·Р°РІРµСЂС€РµРЅРёСЏ LeadAgentThread РґРѕ "
                f"{self.route_stop_grace_ms / 1000:.1f}s; Р·Р°С‚РµРј Р±СѓРґРµС‚ РїСЂРёРЅСѓРґРёС‚РµР»СЊРЅС‹Р№ РїРµСЂРµС…РѕРґ"
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
                        "AI WATCHDOG: LeadAgentThread РЅРµ Р·Р°РІРµСЂС€РёР»СЃСЏ РїРѕСЃР»Рµ stop(); "
                        "РїСЂРёРЅСѓРґРёС‚РµР»СЊРЅРѕ Р·Р°РІРµСЂС€Р°РµРј QThread РїРµСЂРµРґ РїРµСЂРµС…РѕРґРѕРј Рє СЃР»РµРґСѓСЋС‰РµРјСѓ РјР°СЂС€СЂСѓС‚Сѓ"
                    )
                    thread.terminate()
                    thread.wait(5000)
            except RuntimeError:
                pass
            except Exception as exc:
                print(f"AI WATCHDOG: РѕС€РёР±РєР° РїСЂРёРЅСѓРґРёС‚РµР»СЊРЅРѕРіРѕ Р·Р°РІРµСЂС€РµРЅРёСЏ LeadAgentThread: {exc}")
            self.agent_thread = None

        self._continue_after_route_skip()

    def _continue_after_route_skip(self):
        total = len(self.pending_route_queue)
        skipped_index = self.current_route_index
        reason = self._route_skip_reason or "С‚Р°Р№РјР°СѓС‚ Р·Р°РїСѓСЃРєР°"
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
            msg = f"РњР°СЂС€СЂСѓС‚ {skipped_index + 1}/{total} РїСЂРѕРїСѓС‰РµРЅ: {reason}. Р—Р°РїСѓСЃРє СЃР»РµРґСѓСЋС‰РµРіРѕ"
            print(f"AI WATCHDOG: {msg}")
            self.view.statusBar().showMessage(msg)
            if hasattr(self.view, "set_route_queue_position"):
                self.view.set_route_queue_position(self.current_route_index + 1, total)
            QTimer.singleShot(self.route_transition_delay_ms, self._start_current_route)
        else:
            self._finish_route_queue(
                f"РњР°СЂС€СЂСѓС‚ РѕСЃС‚Р°РЅРѕРІР»РµРЅ РїРѕ С‚Р°Р№РјР°СѓС‚Сѓ: {reason}. РЎР»РµРґСѓСЋС‰РµРіРѕ РјР°СЂС€СЂСѓС‚Р° РЅРµС‚"
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

    def _parse_route_metadata(self, route_path):
        scenario_type = "unknown"
        route_id = "route"
        if route_path:
            route_id = os.path.splitext(os.path.basename(route_path))[0] or route_id
        try:
            if route_path and os.path.exists(route_path):
                tree = ET.parse(route_path)
                root = tree.getroot()
                scenario_node = root.find(".//scenario")
                if scenario_node is not None:
                    scenario_type = scenario_node.get("type") or scenario_type
                route_node = root.find(".//route")
                if route_node is not None:
                    route_id = route_node.get("id") or route_id
        except Exception as exc:
            print(f"AI ROUTES: РЅРµ СѓРґР°Р»РѕСЃСЊ СЂР°Р·РѕР±СЂР°С‚СЊ РјРµС‚Р°РґР°РЅРЅС‹Рµ РјР°СЂС€СЂСѓС‚Р°: {exc}")
        return scenario_type, route_id

    def _build_route_stdout_log_path(self, project_root, route_path):
        scenario_type, route_id = self._parse_route_metadata(route_path)
        scenario_tag = str(scenario_type or "unknown").replace(" ", "").strip() or "unknown"
        route_tag = str(route_id or "route").replace(" ", "").strip() or "route"
        host = (self.carla_watchdog_host or "127.0.0.1").strip()
        port = int(self.carla_watchdog_port or 2000)
        tm_port = int(os.environ.get("MITSU_CARLA_TM_PORT", "8000"))
        log_tag = f"{scenario_tag}_{route_tag}_Rep0_attempt0_{host}_{port}_{tm_port}"
        stdout_root = os.path.join(project_root, "data", "carla_leaderboard2_dual_cameras", "stdout")
        os.makedirs(stdout_root, exist_ok=True)
        return os.path.join(stdout_root, f"{log_tag}.log")

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
        if not self._is_carla_mode():
            self._ignore_non_carla_call("handle_route_queue_updated")
            return
        # РР·РјРµРЅРµРЅРёРµ РІС‹Р±РѕСЂР° РЅРµ Р·Р°РїСѓСЃРєР°РµС‚ Рё РЅРµ РіРѕС‚РѕРІРёС‚ СЃС†РµРЅР°СЂРёР№. РћРЅРѕ С‚РѕР»СЊРєРѕ РґРµР»Р°РµС‚
        # РїСЂРµРґС‹РґСѓС‰РёР№ РїР»Р°РЅ СѓСЃС‚Р°СЂРµРІС€РёРј: РїРѕР»СЊР·РѕРІР°С‚РµР»СЊ РґРѕР»Р¶РµРЅ СЏРІРЅРѕ РЅР°Р¶Р°С‚СЊ РєРЅРѕРїРєСѓ РїРѕРґРіРѕС‚РѕРІРєРё.
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
        if not self._is_carla_mode():
            self._ignore_non_carla_call("handle_route_launch_requested")
            return
        # РљРЅРѕРїРєР° РјР°СЂС€СЂСѓС‚Р°/РѕС‡РµСЂРµРґРё С‚РѕР»СЊРєРѕ РїРѕРґРіРѕС‚Р°РІР»РёРІР°РµС‚ РїР»Р°РЅ. CARLA/LeadAgentThread
        # СЃС‚Р°СЂС‚СѓСЋС‚ СЃС‚СЂРѕРіРѕ РїРѕ СЃРѕР±С‹С‚РёСЋ Activate Control ON.
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
            message = "РњР°СЂС€СЂСѓС‚ РЅРµ РІС‹Р±СЂР°РЅ"
            print(f"AI ROUTES: {message}")
            self.view.statusBar().showMessage(message)
            if hasattr(self.view, "set_route_error_status"):
                self.view.set_route_error_status(message)
            return

        label = f"РѕС‡РµСЂРµРґСЊ РёР· {len(queue)} РјР°СЂС€СЂСѓС‚РѕРІ" if len(queue) > 1 else "РјР°СЂС€СЂСѓС‚"
        message = f"РџРѕРґРіРѕС‚РѕРІР»РµРЅРѕ: {label}. РќР°Р¶РјРёС‚Рµ Activate Control РґР»СЏ Р·Р°РїСѓСЃРєР°"
        print(f"AI ROUTES: {message}")
        self.view.statusBar().showMessage(message)
        if hasattr(self.view, "set_route_runtime_state"):
            self.view.set_route_runtime_state("prepared", message)
        elif hasattr(self.view, "set_route_loading_status"):
            self.view.set_route_loading_status(message)

    def start_pending_route(self):
        if not self._is_carla_mode():
            self._ignore_non_carla_call("start_pending_route")
            return
        if self.agent_thread is not None or self.ai_agent_loading:
            self.view.statusBar().showMessage("AI Status: С‚РµРєСѓС‰РёР№ СЃС†РµРЅР°СЂРёР№ РµС‰Рµ РІС‹РїРѕР»РЅСЏРµС‚СЃСЏ")
            return

        ok, message = self._validate_control_start_requirements()
        if not ok:
            self._reject_control_activation(message)
            return

        # РљР°Р¶РґС‹Р№ РЅРѕРІС‹Р№ Р·Р°РїСѓСЃРє РѕС‡РµСЂРµРґРё РЅР°С‡РёРЅР°РµС‚СЃСЏ СЃ РїРµСЂРІРѕРіРѕ РїРѕРґРіРѕС‚РѕРІР»РµРЅРЅРѕРіРѕ РјР°СЂС€СЂСѓС‚Р°.
        self.current_route_index = 0
        self.queue_stop_requested = False
        self._agent_stop_requested = False
        self._start_current_route()

    # РЎС‚Р°СЂРѕРµ РёРјСЏ РѕСЃС‚Р°РІР»РµРЅРѕ РґР»СЏ СЃРѕРІРјРµСЃС‚РёРјРѕСЃС‚Рё СЃ РїСЂРµР¶РЅРёРј flow.
    def start_ai_agent(self):
        self.start_pending_route()

    def _start_current_route(self):
        if not self._is_carla_mode():
            self._ignore_non_carla_call("_start_current_route")
            return
        if self.queue_stop_requested:
            return
        if self.agent_thread is not None or self.ai_agent_loading:
            return
        if not (0 <= self.current_route_index < len(self.pending_route_queue)):
            self._finish_route_queue("РћС‡РµСЂРµРґСЊ Р·Р°РІРµСЂС€РµРЅР°")
            return

        route_record = self.pending_route_queue[self.current_route_index]
        selected_route_path = self._route_path_from_record(route_record)
        if not selected_route_path:
            self.handle_agent_error("РЈ РІС‹Р±СЂР°РЅРЅРѕРіРѕ РјР°СЂС€СЂСѓС‚Р° РЅРµС‚ XML path")
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
            "checkpoint_path": os.environ.get(
                "MITSU_LEAD_CHECKPOINT",
                os.path.join(project_root, "outputs", "model_0011"),
            ),
            "telemetry_file": telemetry_file,
            "expert_mode": False,
            "host": self.carla_watchdog_host or "127.0.0.1",
            "port": self.carla_watchdog_port or 2000,
            "traffic_manager_port": int(os.environ.get("MITSU_CARLA_TM_PORT", "8000")),
            "stdout_log_path": self._build_route_stdout_log_path(project_root, selected_route_path),
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
                    message=f"Р—Р°РіСЂСѓР·РєР° СЃС†РµРЅР°СЂРёСЏ {idx + 1} / {total}",
                )
            elif hasattr(self.view, "set_route_loading_status"):
                self.view.set_route_loading_status(f"Р—Р°РіСЂСѓР·РєР° СЃС†РµРЅР°СЂРёСЏ {idx + 1} / {total}")

            self.view.statusBar().showMessage(f"AI Status: loading route {idx + 1}/{total}")
            print(f"AI ROUTES: СЃС‚Р°СЂС‚ РјР°СЂС€СЂСѓС‚Р° {idx + 1}/{total}: {selected_route_path}")
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
            print(f"AI CONTROL: РѕС€РёР±РєР° СЃР±СЂРѕСЃР° Р»РѕРєР°Р»СЊРЅРѕРіРѕ СЃРѕСЃС‚РѕСЏРЅРёСЏ: {exc}")

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
            print(f"AI ROUTES: РѕС€РёР±РєР° stop(): {exc}")

        # РќРµ РѕР±РЅСѓР»СЏРµРј self.agent_thread Р·РґРµСЃСЊ. РРЅР°С‡Рµ Qt РјРѕР¶РµС‚ СѓРЅРёС‡С‚РѕР¶РёС‚СЊ QThread,
        # РїРѕРєР° РѕРЅ РµС‰Рµ Р·Р°РІРµСЂС€Р°РµС‚ CARLA/ScenarioRunner: СЌС‚Рѕ Рё РґР°РІР°Р»Рѕ
        # 'QThread: Destroyed while thread is still running'.
        try:
            if hasattr(thread, "isRunning") and thread.isRunning():
                print("AI ROUTES: РѕР¶РёРґР°РµРј С€С‚Р°С‚РЅРѕРіРѕ Р·Р°РІРµСЂС€РµРЅРёСЏ LeadAgentThread")
                return False
        except RuntimeError:
            pass

        self.agent_thread = None
        return True

    def stop_ai_agent(self):
        self.abort_active_scenario("Lead Agent: stopping", keep_plan=True)

    def abort_active_scenario(self, message="РЎС†РµРЅР°СЂРёР№ РѕСЃС‚Р°РЅРѕРІР»РµРЅ", keep_plan=True):
        # Р‘РµР·РѕРїР°СЃРЅР°СЏ РѕСЃС‚Р°РЅРѕРІРєР°: РїСЂРѕСЃРёРј LeadAgentThread Р·Р°РІРµСЂС€РёС‚СЊСЃСЏ, РЅРѕ РЅРµ СѓРЅРёС‡С‚РѕР¶Р°РµРј
        # РѕР±СЉРµРєС‚ QThread РґРѕ СЃРёРіРЅР°Р»Р° finished.
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
            suffix = " РћС‡РµСЂРµРґСЊ РѕСЃС‚Р°РµС‚СЃСЏ РїРѕРґРіРѕС‚РѕРІР»РµРЅРЅРѕР№." if keep_plan and self.route_plan_prepared else ""
            self.view.set_route_stopped_status(f"{message}.{suffix}")

    # РЎС‚Р°СЂРѕРµ РёРјСЏ РѕСЃС‚Р°РІР»РµРЅРѕ РґР»СЏ СЃРѕРІРјРµСЃС‚РёРјРѕСЃС‚Рё.
    def stop_route_queue(self, message="РћС‡РµСЂРµРґСЊ РѕСЃС‚Р°РЅРѕРІР»РµРЅР°"):
        self.abort_active_scenario(message, keep_plan=True)

    @Slot()
    def handle_agent_finished(self):
        if not self._is_carla_mode():
            self.raw_telemetry_timer.stop()
            self.raw_telemetry_reader = None
            self.agent_thread = None
            self.ai_agent_loading = False
            self.ai_agent_running = False
            return
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
                        "РЎС†РµРЅР°СЂРёР№ РїСЂРµСЂРІР°РЅ. CARLA РѕС‡РёС‰РµРЅР°, РјРѕР¶РЅРѕ СЃРЅРѕРІР° РЅР°Р¶Р°С‚СЊ Activate Control",
                    )
                else:
                    self.view.set_route_runtime_state("stopped", "РЎС†РµРЅР°СЂРёР№ РїСЂРµСЂРІР°РЅ")
            return

        total = len(self.pending_route_queue)
        finished_index = self.current_route_index
        if total <= 0 or finished_index < 0:
            return

        print(f"AI ROUTES: РјР°СЂС€СЂСѓС‚ {finished_index + 1}/{total} Р·Р°РІРµСЂС€РµРЅ")

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
            self._finish_route_queue("РћС‡РµСЂРµРґСЊ Р·Р°РІРµСЂС€РµРЅР°" if self.queue_mode == "queue" else "РњР°СЂС€СЂСѓС‚ Р·Р°РІРµСЂС€РµРЅ")

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
        if not self._is_carla_mode():
            return
        if not self.raw_telemetry_reader: 
            return
            
        data = self.raw_telemetry_reader.get_latest_data()
        
        if data is not None:
            self._update_route_watchdog_from_telemetry(data)
            if not self.ai_agent_running:
                self.ai_agent_loading = False
                self.ai_agent_running = True
                if hasattr(self.view, "set_route_running_status"):
                    self.view.set_route_running_status("РЎС†РµРЅР°СЂРёР№ РІС‹РїРѕР»РЅСЏРµС‚СЃСЏ: С‚РµР»РµРјРµС‚СЂРёСЏ РїРѕР»СѓС‡РµРЅР°")
                self.view.statusBar().showMessage("AI Status: scenario running")
            # РР·РІР»РµРєР°РµРј Р·РЅР°С‡РµРЅРёСЏ
            steer = data.get("steer", 0.0)
            throttle = data.get("throttle", 0.0)
            brake = data.get("brake", 0.0)
            
            # РџСЂРµРѕР±СЂР°Р·СѓРµРј РґР»СЏ РјРѕРґРµР»Рё
            target_angle = int(steer * 630)
            target_accel = int(throttle * 100)
            target_brake = int(brake * 100)
            
            # --- Р’Р«Р’РћР” Р›РћР“РћР’ Р’ РљРћРќРЎРћР›Р¬ (РєР°Рє РЅР° СЃРєСЂРёРЅС€РѕС‚Рµ) ---
            # РСЃРїРѕР»СЊР·СѓРµРј f-СЃС‚СЂРѕРєРё РґР»СЏ РІС‹СЂР°РІРЅРёРІР°РЅРёСЏ
            log_line = (
                f"[AI TELEMETRY] "
                f"Steer: {steer:>6.2f} | "
                f"Thr: {throttle:>5.2f} | "
                f"Brk: {brake:>5.2f} | "
                f"Target Angle: {target_angle:>4}В°"
            )
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] {log_line}")  

            # РћР±РЅРѕРІР»СЏРµРј СЃРѕСЃС‚РѕСЏРЅРёРµ РјРѕРґРµР»Рё
            # РћР±РЅРѕРІР»СЏРµРј СЃРѕСЃС‚РѕСЏРЅРёРµ РјРѕРґРµР»Рё
            self.vehicle.target_angle = target_angle
            self.vehicle.target_accel = target_accel
            self.vehicle.target_brake = target_brake

            # РђРіРµРЅС‚ РЅРµ РїРµСЂРµРєР»СЋС‡Р°РµС‚ РїРµСЂРµРґР°С‡Рё. Р”Р»СЏ AI-СЃС†РµРЅР°СЂРёСЏ РґРµСЂР¶РёРј Р»РѕРєР°Р»СЊРЅСѓСЋ РјРѕРґРµР»СЊ РІ D.
            if hasattr(self.vehicle, "force_drive_gear"):
                self.vehicle.force_drive_gear()
            else:
                self.vehicle.target_gear = 4
                self.vehicle.gear = 4

            # Р•СЃР»Рё trace_log.jsonl СЃРѕРґРµСЂР¶РёС‚ СЂРµР°Р»СЊРЅСѓСЋ СЃРєРѕСЂРѕСЃС‚СЊ CARLA, РёСЃРїРѕР»СЊР·СѓРµРј РµРµ.
            # Р•СЃР»Рё СЃРєРѕСЂРѕСЃС‚Рё РЅРµС‚, РѕР±РЅРѕРІР»СЏРµРј fallback-С„РёР·РёРєСѓ.
            if hasattr(self.vehicle, "apply_telemetry") and self.vehicle.apply_telemetry(data):
                pass
            else:
                self.vehicle.update_physics(dt=0.05)

            # Р•СЃР»Рё СѓРїСЂР°РІР»РµРЅРёРµ Р°РєС‚РёРІРЅРѕ, РѕС‚РїСЂР°РІР»СЏРµРј РІ CAN
            if self.control_active:
                self.handle_manual_input(target_angle, target_accel, target_brake)

    @Slot(str)
    def handle_agent_log(self, message):
        if not self._is_carla_mode():
            return
        text = str(message)
        self.view.statusBar().showMessage(f"AI: {text[:40]}")
        self._update_route_loading_from_text(text)

    @Slot(str)
    def handle_agent_status(self, status):
        if not self._is_carla_mode():
            return
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
        if not self._is_carla_mode():
            return
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
            state = getattr(getattr(self.vehicle_control, "state_machine", None), "state", None)
            state_text = getattr(state, "value", str(state or ""))
            if self.control_active and state_text != "MANUAL_ACTIVE":
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
            state = getattr(getattr(self.vehicle_control, "state_machine", None), "state", None)
            state_text = getattr(state, "value", str(state or ""))
            if self.control_active and state_text != "MANUAL_ACTIVE":
                now = time.monotonic()
                if now - self._last_real_gear_ignore_log_at > 1.0:
                    self._last_real_gear_ignore_log_at = now
                    print("REAL CONTROL: СЂСѓС‡РЅР°СЏ РїРµСЂРµРґР°С‡Р° Р·Р°Р±Р»РѕРєРёСЂРѕРІР°РЅР°: СЃРЅР°С‡Р°Р»Р° РѕС‚РєР»СЋС‡РёС‚Рµ Activate Control")
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
        if not self._is_carla_mode():
            return
        if self.is_virtual:
            self.vehicle.update_physics()
            if self.telemetry.enabled:
                self.telemetry.log(self.vehicle.speed, self.vehicle.angle, 
                                   self.vehicle.accel, self.vehicle.brake, self.vehicle.gear)

    def handle_can_packet(self, pkt):
        if self._is_real_mode():
            # RealSerialVehicleAdapter owns real telemetry parsing. Legacy CAN parsing
            # is for CARLA/virtual/old serial flow only.
            return
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

# --- MITSU_LIFECYCLE_CAMERA_CLEAN_PATCH_BEGIN ---
# Runtime lifecycle patch:
# - telemetry != scenario running;
# - no phantom throttle/brake during loading;
# - watchdog is armed only after hero actor + first telemetry are both available;
# - RUNNING is shown only after the CARLA ego actor physically moves.

def _mitsu_zero_visual_controls(self):
    try:
        self.vehicle.target_accel = 0
        self.vehicle.target_brake = 0
        self.vehicle.accel = 0
        self.vehicle.brake = 0
    except Exception:
        pass

def _mitsu_set_route_loading_text(self, text):
    self._route_lifecycle_phase = "loading"
    if hasattr(self.view, "set_route_loading_status"):
        self.view.set_route_loading_status(text)
    elif hasattr(self.view, "set_route_runtime_state"):
        self.view.set_route_runtime_state("loading", text)

def _mitsu_set_route_waiting_motion_status(self, message=None):
    self._route_lifecycle_phase = "waiting_motion"
    text = message or "РЎС†РµРЅР°СЂРёР№ Р·Р°РіСЂСѓР¶РµРЅ: РѕР¶РёРґР°РЅРёРµ С„РёР·РёС‡РµСЃРєРѕРіРѕ РґРІРёР¶РµРЅРёСЏ ego"
    if hasattr(self.view, "set_route_loading_status"):
        self.view.set_route_loading_status(text)
    elif hasattr(self.view, "set_route_runtime_state"):
        self.view.set_route_runtime_state("loading", text)

def _mitsu_arm_motion_watchdog_if_ready(self, reason=""):
    if getattr(self, "_route_started_monotonic", None) is None:
        return
    if getattr(self, "_route_motion_watchdog_armed_at", None) is not None:
        return
    if not getattr(self, "_route_carla_vehicle_seen", False):
        return
    if getattr(self, "_route_last_telemetry_monotonic", None) is None:
        return

    self._route_motion_watchdog_armed_at = time.monotonic()
    self._set_route_waiting_motion_status("РЎС†РµРЅР°СЂРёР№ Р·Р°РіСЂСѓР¶РµРЅ: РѕР¶РёРґР°РЅРёРµ РЅР°С‡Р°Р»Р° РґРІРёР¶РµРЅРёСЏ ego")
    print(f"AI WATCHDOG: motion watchdog armed ({reason})")

def _mitsu_reset_route_watchdog(self):
    self._stop_carla_motion_monitor()

    self._route_started_monotonic = time.monotonic()
    self._route_motion_watchdog_armed_at = None
    self._route_lifecycle_phase = "loading"

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

    self._zero_visual_controls()
    self._set_route_loading_text("Р—Р°РіСЂСѓР·РєР° РјРёСЂР° Рё СЃС†РµРЅР°СЂРёСЏ")

    if hasattr(self, "_route_force_continue_timer"):
        self._route_force_continue_timer.stop()

    self._start_carla_motion_monitor()

def _mitsu_clear_route_watchdog(self):
    self._stop_carla_motion_monitor()

    self._route_started_monotonic = None
    self._route_motion_watchdog_armed_at = None
    self._route_lifecycle_phase = "idle"

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

def _mitsu_mark_route_progress(self, reason):
    if not getattr(self, "_route_progress_seen", False):
        print(f"AI WATCHDOG: РјР°СЂС€СЂСѓС‚ РЅР°С‡Р°Р» С„РёР·РёС‡РµСЃРєРѕРµ РґРІРёР¶РµРЅРёРµ: {reason}")
        self._route_progress_seen = True

        self.ai_agent_loading = False
        self.ai_agent_running = True
        self._route_lifecycle_phase = "running"

        if hasattr(self.view, "set_route_running_status"):
            self.view.set_route_running_status("РЎС†РµРЅР°СЂРёР№ РІС‹РїРѕР»РЅСЏРµС‚СЃСЏ: ego РЅР°С‡Р°Р» РґРІРёР¶РµРЅРёРµ")
        elif hasattr(self.view, "set_route_runtime_state"):
            self.view.set_route_runtime_state("running", "РЎС†РµРЅР°СЂРёР№ РІС‹РїРѕР»РЅСЏРµС‚СЃСЏ: ego РЅР°С‡Р°Р» РґРІРёР¶РµРЅРёРµ")

        try:
            self.view.statusBar().showMessage("AI Status: scenario running; ego is moving")
        except Exception:
            pass

    self._route_progress_reason = str(reason or "physical progress")
    self._route_last_physical_progress_monotonic = time.monotonic()

def _mitsu_handle_carla_motion_update(self, info):
    if getattr(self, "_route_started_monotonic", None) is None or getattr(self, "_route_skip_after_stop", False):
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

    actor_id = info.get("actor_id")
    location = (
        self._safe_float(info.get("x")),
        self._safe_float(info.get("y")),
        self._safe_float(info.get("z", 0.0)) or 0.0,
    )
    if location[0] is None or location[1] is None:
        location = None

    if actor_id != getattr(self, "_route_carla_actor_id", None):
        self._route_carla_actor_id = actor_id
        self._route_carla_start_location = location
        self._route_last_progress_location = location
        self._route_total_distance_m = 0.0
        print(
            f"AI WATCHDOG: РЅР°Р№РґРµРЅ ego actor id={actor_id}, "
            f"type={info.get('type_id', '-')}, role={info.get('role', '-')}, "
            f"start={location}"
        )

    self._route_carla_last_location = location

    if getattr(self, "_route_carla_start_location", None) is not None and location is not None:
        self._route_total_distance_m = self._distance_2d(self._route_carla_start_location, location)

    speed_kmh = self._safe_float(info.get("speed_kmh"))
    if speed_kmh is not None:
        self._route_carla_last_speed_kmh = speed_kmh

    if not getattr(self, "_route_progress_seen", False):
        self._zero_visual_controls()
        self._set_route_waiting_motion_status("РЎС†РµРЅР°СЂРёР№ Р·Р°РіСЂСѓР¶РµРЅ: РѕР¶РёРґР°РЅРёРµ РґРІРёР¶РµРЅРёСЏ ego")

    self._arm_motion_watchdog_if_ready("hero actor visible")

    if getattr(self, "_route_total_distance_m", 0.0) >= getattr(self, "route_startup_min_distance_m", 5.0):
        self._mark_route_progress(
            f"CARLA actor moved {self._route_total_distance_m:.1f} m from start"
        )

    if getattr(self, "_route_progress_seen", False) and speed_kmh is not None:
        try:
            if hasattr(self.vehicle, "apply_external_speed_kmh"):
                self.vehicle.apply_external_speed_kmh(speed_kmh, smooth=True)
            else:
                self.vehicle.speed = speed_kmh
        except Exception:
            pass

    if self._route_debug_last_log_monotonic is None or now - self._route_debug_last_log_monotonic >= 15.0:
        self._route_debug_last_log_monotonic = now
        elapsed = self._route_elapsed_seconds()
        armed = "yes" if getattr(self, "_route_motion_watchdog_armed_at", None) is not None else "no"
        try:
            idx_text = f"{self.current_route_index + 1}/{len(self.pending_route_queue)}"
        except Exception:
            idx_text = "-/-"
        print(
            f"AI WATCHDOG: route {idx_text} "
            f"phase={getattr(self, '_route_lifecycle_phase', '-')}, armed={armed}, "
            f"elapsed={elapsed:.0f}s, speed={getattr(self, '_route_carla_last_speed_kmh', 0.0):.1f} km/h, "
            f"dist={getattr(self, '_route_total_distance_m', 0.0):.1f} m"
        )

def _mitsu_check_route_startup_watchdog(self):
    if not self._is_carla_mode():
        return
    if getattr(self, "_route_started_monotonic", None) is None:
        return
    if getattr(self, "_route_skip_after_stop", False) or getattr(self, "queue_stop_requested", False):
        return
    if not (self.control_active and self.route_plan_prepared and self.current_route_index >= 0):
        return
    if self.agent_thread is None and not self.ai_agent_loading and not self.ai_agent_running:
        return

    now = time.monotonic()
    timeout_s = self.route_startup_timeout_ms / 1000.0

    # Loading phase: do NOT kill slow CARLA/Leaderboard startup.
    if not getattr(self, "_route_carla_vehicle_seen", False) or getattr(self, "_route_last_telemetry_monotonic", None) is None:
        return

    self._arm_motion_watchdog_if_ready("watchdog tick")

    # Loaded + telemetry + hero actor, but no physical movement yet.
    if not getattr(self, "_route_progress_seen", False):
        armed_at = getattr(self, "_route_motion_watchdog_armed_at", None)
        if armed_at is None:
            return
        waiting_ms = (now - armed_at) * 1000.0
        if waiting_ms >= self.route_startup_timeout_ms:
            reason = (
                f"РїРѕСЃР»Рµ Р·Р°РіСЂСѓР·РєРё СЃС†РµРЅР°СЂРёСЏ ego actor РЅРµ РЅР°С‡Р°Р» РґРІРёР¶РµРЅРёРµ Р·Р° {timeout_s:.0f} СЃРµРєСѓРЅРґ; "
                f"dist_from_start={getattr(self, '_route_total_distance_m', 0.0):.1f} Рј "
                f"< {getattr(self, 'route_startup_min_distance_m', 5.0):.1f} Рј"
            )
            self._skip_current_route_due_to_timeout(reason)
        return

    # Already running: catch hard stalls only after confirmed physical movement.
    last_progress = getattr(self, "_route_last_physical_progress_monotonic", None)
    if last_progress is not None:
        stalled_ms = (now - last_progress) * 1000.0
        if stalled_ms >= self.route_startup_timeout_ms:
            reason = (
                f"РЅРµС‚ РЅРѕРІРѕРіРѕ С„РёР·РёС‡РµСЃРєРѕРіРѕ РґРІРёР¶РµРЅРёСЏ ego {timeout_s:.0f} СЃРµРєСѓРЅРґ "
                f"РїРѕСЃР»Рµ РїРѕСЃР»РµРґРЅРµРіРѕ РїСЂРѕРіСЂРµСЃСЃР° ({getattr(self, '_route_progress_reason', '')}); "
                f"dist_from_start={getattr(self, '_route_total_distance_m', 0.0):.1f} Рј"
            )
            self._skip_current_route_due_to_timeout(reason)

def _mitsu_poll_ai_telemetry(self):
    if not self._is_carla_mode():
        return
    if not self.raw_telemetry_reader:
        return

    data = self.raw_telemetry_reader.get_latest_data()
    if data is None:
        return

    self._update_route_watchdog_from_telemetry(data)
    self._arm_motion_watchdog_if_ready("first telemetry")

    steer = data.get("steer", 0.0)
    throttle = data.get("throttle", 0.0)
    brake = data.get("brake", 0.0)

    target_angle = int(steer * 630)
    target_accel = int(throttle * 100)
    target_brake = int(brake * 100)

    log_line = (
        f"[AI TELEMETRY] "
        f"Steer: {steer:>6.2f} | "
        f"Thr: {throttle:>5.2f} | "
        f"Brk: {brake:>5.2f} | "
        f"Target Angle: {target_angle:>4}В°"
    )
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {log_line}")

    # Telemetry means the agent is alive, not that the route is physically running.
    if not getattr(self, "_route_progress_seen", False):
        self.ai_agent_loading = True
        self.ai_agent_running = False
        self._zero_visual_controls()

        if getattr(self, "_route_carla_vehicle_seen", False):
            self._set_route_waiting_motion_status("РЎС†РµРЅР°СЂРёР№ Р·Р°РіСЂСѓР¶РµРЅ: РѕР¶РёРґР°РЅРёРµ РґРІРёР¶РµРЅРёСЏ ego")
        else:
            self._set_route_loading_text("Р—Р°РіСЂСѓР·РєР° РјРёСЂР° Рё СЃС†РµРЅР°СЂРёСЏ")
        return

    # RUNNING phase: only after CARLA physical displacement has been confirmed.
    self.vehicle.target_angle = target_angle
    self.vehicle.target_accel = target_accel
    self.vehicle.target_brake = target_brake

    if hasattr(self.vehicle, "force_drive_gear"):
        self.vehicle.force_drive_gear()
    else:
        self.vehicle.target_gear = 4
        self.vehicle.gear = 4

    if hasattr(self.vehicle, "apply_telemetry") and self.vehicle.apply_telemetry(data):
        pass
    else:
        self.vehicle.update_physics(dt=0.05)

    if self.control_active:
        self.handle_manual_input(target_angle, target_accel, target_brake)

# Patch methods before AppController is instantiated.
AppController._zero_visual_controls = _mitsu_zero_visual_controls
AppController._set_route_loading_text = _mitsu_set_route_loading_text
AppController._set_route_waiting_motion_status = _mitsu_set_route_waiting_motion_status
AppController._arm_motion_watchdog_if_ready = _mitsu_arm_motion_watchdog_if_ready
AppController._reset_route_watchdog = _mitsu_reset_route_watchdog
AppController._clear_route_watchdog = _mitsu_clear_route_watchdog
AppController._mark_route_progress = _mitsu_mark_route_progress
AppController.handle_carla_motion_update = _mitsu_handle_carla_motion_update
AppController.check_route_startup_watchdog = _mitsu_check_route_startup_watchdog
AppController.poll_ai_telemetry = _mitsu_poll_ai_telemetry

_mitsu_original_appcontroller_init = AppController.__init__

def _mitsu_appcontroller_init(self, *args, **kwargs):
    _mitsu_original_appcontroller_init(self, *args, **kwargs)

    # Ensure fields exist even if the base file did not define them.
    self._route_motion_watchdog_armed_at = None
    self._route_lifecycle_phase = "idle"
    if not hasattr(self, "route_startup_min_distance_m"):
        self.route_startup_min_distance_m = 5.0

AppController.__init__ = _mitsu_appcontroller_init
# --- MITSU_LIFECYCLE_CAMERA_CLEAN_PATCH_END ---

# --- MITSU_ROUTE_COMPLETION_STATUS_V2_BEGIN ---
# Non-blocking completion notifications for routes/queues.
# No modal QMessageBox: queue execution should never be blocked by UI.

_mitsu_original_finish_route_queue = AppController._finish_route_queue
_mitsu_original_handle_agent_finished = AppController.handle_agent_finished

def _mitsu_is_success_finish_message(message):
    low = str(message or "").lower()
    bad = ("С‚Р°Р№РјР°СѓС‚", "РѕС€РёР±", "error", "failed", "РѕСЃС‚Р°РЅРѕРІ", "РїСЂРµСЂРІР°РЅ", "stop", "abort")
    return not any(token in low for token in bad)

def _mitsu_success_message_for_finish(self, message, total_before):
    text = str(message or "")
    if not _mitsu_is_success_finish_message(text):
        return text

    if getattr(self, "queue_mode", None) == "queue" or total_before > 1 or "РѕС‡РµСЂРµРґ" in text.lower():
        done = max(1, total_before)
        return f"РћС‡РµСЂРµРґСЊ Р·Р°РІРµСЂС€РµРЅР° СѓСЃРїРµС€РЅРѕ: {done}/{done} РјР°СЂС€СЂСѓС‚РѕРІ"

    return "РњР°СЂС€СЂСѓС‚ СѓСЃРїРµС€РЅРѕ Р·Р°РІРµСЂС€С‘РЅ"

def _mitsu_finish_route_queue(self, message):
    total_before = len(getattr(self, "pending_route_queue", []) or [])
    success = _mitsu_is_success_finish_message(message)
    final_message = _mitsu_success_message_for_finish(self, message, total_before)

    result = _mitsu_original_finish_route_queue(self, final_message)

    if success:
        print(f"AI ROUTES: SUCCESS: {final_message}")
        try:
            self.view.statusBar().showMessage(final_message, 7000)
        except TypeError:
            self.view.statusBar().showMessage(final_message)
        except Exception:
            pass

    return result

def _mitsu_handle_agent_finished(self):
    total = len(getattr(self, "pending_route_queue", []) or [])
    idx = getattr(self, "current_route_index", -1)
    skip = getattr(self, "_route_skip_after_stop", False)
    stop = getattr(self, "_agent_stop_requested", False) or getattr(self, "queue_stop_requested", False)

    if not skip and not stop and total > 0 and idx >= 0:
        print(f"AI ROUTES: РјР°СЂС€СЂСѓС‚ {idx + 1}/{total} Р·Р°РІРµСЂС€С‘РЅ С€С‚Р°С‚РЅРѕ")
        try:
            self.view.statusBar().showMessage(f"РњР°СЂС€СЂСѓС‚ {idx + 1}/{total} Р·Р°РІРµСЂС€С‘РЅ", 5000)
        except TypeError:
            self.view.statusBar().showMessage(f"РњР°СЂС€СЂСѓС‚ {idx + 1}/{total} Р·Р°РІРµСЂС€С‘РЅ")
        except Exception:
            pass

    return _mitsu_original_handle_agent_finished(self)

AppController._finish_route_queue = _mitsu_finish_route_queue
AppController.handle_agent_finished = _mitsu_handle_agent_finished
# --- MITSU_ROUTE_COMPLETION_STATUS_V2_END ---

# --- MITSU_ROUTE_COMPLETION_STATUS_V4_BEGIN ---
# Non-blocking route/queue completion status.
# No modal QMessageBox: queue execution must not be blocked by UI.

_mitsu_v4_original_finish_route_queue = AppController._finish_route_queue
_mitsu_v4_original_handle_agent_finished = AppController.handle_agent_finished

def _mitsu_v4_is_success_finish_message(message):
    low = str(message or "").lower()
    bad = ("С‚Р°Р№РјР°СѓС‚", "РѕС€РёР±", "error", "failed", "РѕСЃС‚Р°РЅРѕРІ", "РїСЂРµСЂРІР°РЅ", "stop", "abort")
    return not any(token in low for token in bad)

def _mitsu_v4_success_message_for_finish(self, message, total_before):
    text = str(message or "")
    if not _mitsu_v4_is_success_finish_message(text):
        return text

    if getattr(self, "queue_mode", None) == "queue" or total_before > 1 or "РѕС‡РµСЂРµРґ" in text.lower():
        done = max(1, total_before)
        return f"РћС‡РµСЂРµРґСЊ Р·Р°РІРµСЂС€РµРЅР° СѓСЃРїРµС€РЅРѕ: {done}/{done} РјР°СЂС€СЂСѓС‚РѕРІ"

    return "РњР°СЂС€СЂСѓС‚ СѓСЃРїРµС€РЅРѕ Р·Р°РІРµСЂС€С‘РЅ"

def _mitsu_v4_finish_route_queue(self, message):
    total_before = len(getattr(self, "pending_route_queue", []) or [])
    success = _mitsu_v4_is_success_finish_message(message)
    final_message = _mitsu_v4_success_message_for_finish(self, message, total_before)

    result = _mitsu_v4_original_finish_route_queue(self, final_message)

    if success:
        print(f"AI ROUTES: SUCCESS: {final_message}")
        try:
            self.view.statusBar().showMessage(final_message, 7000)
        except TypeError:
            self.view.statusBar().showMessage(final_message)
        except Exception:
            pass

    return result

def _mitsu_v4_handle_agent_finished(self):
    total = len(getattr(self, "pending_route_queue", []) or [])
    idx = getattr(self, "current_route_index", -1)
    skip = getattr(self, "_route_skip_after_stop", False)
    stop = getattr(self, "_agent_stop_requested", False) or getattr(self, "queue_stop_requested", False)

    if not skip and not stop and total > 0 and idx >= 0:
        print(f"AI ROUTES: РјР°СЂС€СЂСѓС‚ {idx + 1}/{total} Р·Р°РІРµСЂС€С‘РЅ С€С‚Р°С‚РЅРѕ")
        try:
            self.view.statusBar().showMessage(f"РњР°СЂС€СЂСѓС‚ {idx + 1}/{total} Р·Р°РІРµСЂС€С‘РЅ", 5000)
        except TypeError:
            self.view.statusBar().showMessage(f"РњР°СЂС€СЂСѓС‚ {idx + 1}/{total} Р·Р°РІРµСЂС€С‘РЅ")
        except Exception:
            pass

    return _mitsu_v4_original_handle_agent_finished(self)

AppController._finish_route_queue = _mitsu_v4_finish_route_queue
AppController.handle_agent_finished = _mitsu_v4_handle_agent_finished
# --- MITSU_ROUTE_COMPLETION_STATUS_V4_END ---

# --- MITSU_REAL_TAKEOVER_SINGLE_SOURCE_V18_BEGIN ---
# Single source of truth for real/mock authority state.
#
# This block replaces the previous V12/V17 takeover UI patches. It fixes the
# core race:
#   AI_ACTIVE + clicked checked button => Qt emits False.
#   Service correctly transfers to MANUAL_ACTIVE.
#   UI must keep control button checked and set authority=manual_takeover.
#
# CAN/service behavior is not bypassed. The service remains source of truth for
# gear, brake, Drive/Park and command submission.

def _mitsu_v18_state(controller):
    vc = getattr(controller, "vehicle_control", None)
    state = getattr(getattr(vc, "state_machine", None), "state", None)
    return str(getattr(state, "value", state) or "")


def _mitsu_v18_is_test_cameraless(controller):
    vc = getattr(controller, "vehicle_control", None)
    descriptor = getattr(vc, "descriptor", None)
    text = ""
    if descriptor is not None:
        text = " ".join([
            str(getattr(descriptor, "label", "") or ""),
            str(getattr(getattr(descriptor, "kind", None), "name", getattr(descriptor, "kind", "")) or ""),
        ]).upper()
    return any(token in text for token in ("MOCK", "REPLAY", "LOOPBACK"))


def _mitsu_v18_seed_test_cameras(controller):
    if not _mitsu_v18_is_test_cameraless(controller):
        return False

    view = getattr(controller, "view", None)
    helper = getattr(view, "_mitsu_v18_enable_test_camera_frames", None) if view is not None else None
    if callable(helper):
        try:
            helper(force=True)
        except Exception:
            pass

    try:
        controller.real_camera_status = {"wide_90": True, "narrow_50": True}
    except Exception:
        pass

    vc = getattr(controller, "vehicle_control", None)
    if vc is not None:
        try:
            vc.set_camera_status(True)
        except Exception:
            try:
                vc.cameras_ok = True
                if hasattr(vc, "_refresh_readiness_state"):
                    vc._refresh_readiness_state()
            except Exception:
                pass

    if not bool(getattr(controller, "_mitsu_v18_test_camera_logged", False)):
        controller._mitsu_v18_test_camera_logged = True
        print("TEST CAMERA: physical cameras bypassed for test/mock contour")

    return True


def _mitsu_v18_set_button(controller, checked, text=None):
    view = getattr(controller, "view", None)
    helper = getattr(view, "set_control_button_silent", None) if view is not None else None
    if callable(helper):
        try:
            helper(bool(checked), text=text)
            return
        except Exception:
            pass

    btn = getattr(view, "btn_control", None) if view is not None else None
    if btn is None:
        return

    try:
        btn.blockSignals(True)
        btn.setChecked(bool(checked))
        if text is not None:
            btn.setText(str(text))
    finally:
        try:
            btn.blockSignals(False)
        except Exception:
            pass


def _mitsu_v18_set_authority(controller, authority, message=""):
    view = getattr(controller, "view", None)
    if view is None:
        return

    helper = getattr(view, "set_real_authority_mode", None)
    if callable(helper):
        try:
            helper(authority, message)
            return
        except Exception as exc:
            print(f"REAL UI: authority update failed: {exc}")


def _mitsu_v18_apply_state_to_ui(controller, previous_state=""):
    state = _mitsu_v18_state(controller)

    if state == "AI_ACTIVE":
        controller.control_active = True
        controller.manual_control_requested = False
        _mitsu_v18_set_button(controller, True, "РћС‚РєР»СЋС‡РёС‚СЊ РР / СЂСѓС‡РЅРѕРµ")
        _mitsu_v18_set_authority(controller, "ai", "РђРІС‚РѕРЅРѕРјРЅРѕРµ СѓРїСЂР°РІР»РµРЅРёРµ Р°РєС‚РёРІРЅРѕ")
        return

    if state == "MANUAL_ACTIVE":
        controller.control_active = True
        controller.ai_control_requested = False
        controller.manual_control_requested = True

        takeover = bool(getattr(controller, "_mitsu_v18_takeover_latched", False)) or previous_state == "AI_ACTIVE"
        _mitsu_v18_set_button(controller, True, "Р СѓС‡РЅРѕРµ СѓРїСЂР°РІР»РµРЅРёРµ Р°РєС‚РёРІРЅРѕ")
        _mitsu_v18_set_authority(
            controller,
            "manual_takeover" if takeover else "manual",
            "РџРµСЂРµС…РІР°С‚ РР: СЂСѓС‡РЅРѕРµ СѓРїСЂР°РІР»РµРЅРёРµ Р°РєС‚РёРІРЅРѕ" if takeover else "Р СѓС‡РЅРѕРµ СѓРїСЂР°РІР»РµРЅРёРµ Р°РєС‚РёРІРЅРѕ",
        )
        return

    if state in {"IDLE", "DISCONNECTED"}:
        if bool(getattr(controller, "control_active", False)):
            # If service says IDLE after a real full deactivation, release UI. If
            # a delayed IDLE leaks during takeover, it will be overwritten by the
            # next service state; this is still safer than using toggle-requested.
            controller.control_active = False
            controller.ai_control_requested = False
            controller.manual_control_requested = False
            controller._mitsu_v18_takeover_latched = False
            _mitsu_v18_set_button(controller, False, "РђРєС‚РёРІРёСЂРѕРІР°С‚СЊ СѓРїСЂР°РІР»РµРЅРёРµ")
            _mitsu_v18_set_authority(controller, "off", "РњРёСЃСЃРёСЏ РїРѕРґРіРѕС‚РѕРІР»РµРЅР°")


if hasattr(AppController, "_start_real_camera_stack") and not hasattr(AppController, "_mitsu_v18_original_start_real_camera_stack"):
    AppController._mitsu_v18_original_start_real_camera_stack = AppController._start_real_camera_stack

    def _mitsu_v18_start_real_camera_stack(self):
        if _mitsu_v18_seed_test_cameras(self):
            return
        return AppController._mitsu_v18_original_start_real_camera_stack(self)

    AppController._start_real_camera_stack = _mitsu_v18_start_real_camera_stack


if hasattr(AppController, "handle_real_camera_cell_status") and not hasattr(AppController, "_mitsu_v18_original_handle_real_camera_cell_status"):
    AppController._mitsu_v18_original_handle_real_camera_cell_status = AppController.handle_real_camera_cell_status

    def _mitsu_v18_handle_real_camera_cell_status(self, camera_name, ok, message):
        if _mitsu_v18_seed_test_cameras(self):
            return
        return AppController._mitsu_v18_original_handle_real_camera_cell_status(self, camera_name, ok, message)

    AppController.handle_real_camera_cell_status = _mitsu_v18_handle_real_camera_cell_status


if hasattr(AppController, "handle_real_agent_camera_status") and not hasattr(AppController, "_mitsu_v18_original_handle_real_agent_camera_status"):
    AppController._mitsu_v18_original_handle_real_agent_camera_status = AppController.handle_real_agent_camera_status

    def _mitsu_v18_handle_real_agent_camera_status(self, ok, message):
        if _mitsu_v18_seed_test_cameras(self):
            return
        return AppController._mitsu_v18_original_handle_real_agent_camera_status(self, ok, message)

    AppController.handle_real_agent_camera_status = _mitsu_v18_handle_real_agent_camera_status


if hasattr(AppController, "_handle_real_control_toggle") and not hasattr(AppController, "_mitsu_v18_original_handle_real_control_toggle"):
    AppController._mitsu_v18_original_handle_real_control_toggle = AppController._handle_real_control_toggle

    async def _mitsu_v18_handle_real_control_toggle(self, is_active):
        before_state = _mitsu_v18_state(self)
        takeover = self.runtime_mode == "real" and not bool(is_active) and before_state == "AI_ACTIVE"

        if self.runtime_mode == "real":
            _mitsu_v18_seed_test_cameras(self)

        if takeover:
            self._mitsu_v18_takeover_latched = True
            _mitsu_v18_set_authority(self, "manual_takeover", "РџРµСЂРµС…РІР°С‚ РР: СЂСѓС‡РЅРѕРµ СѓРїСЂР°РІР»РµРЅРёРµ Р°РєС‚РёРІРЅРѕ")
            _mitsu_v18_set_button(self, True, "РџРµСЂРµС…РѕРґ РІ СЂСѓС‡РЅРѕРµ")

        result = await AppController._mitsu_v18_original_handle_real_control_toggle(self, is_active)
        if not isinstance(result, dict):
            result = {}

        after_state = _mitsu_v18_state(self)

        if self.runtime_mode == "real":
            if takeover and after_state == "MANUAL_ACTIVE":
                # Normalize result so async completion cannot turn the button off
                # just because the original Qt toggle value was False.
                result["requested"] = True
                result["active"] = True
                result["authority"] = "manual_takeover"
                result["message"] = "РџРµСЂРµС…РІР°С‚ РР: СЂСѓС‡РЅРѕРµ СѓРїСЂР°РІР»РµРЅРёРµ Р°РєС‚РёРІРЅРѕ"
                self.control_active = True
                self.ai_control_requested = False
                self.manual_control_requested = True

                vc = getattr(self, "vehicle_control", None)
                if vc is not None:
                    try:
                        vc.set_ai_preview_enabled(False)
                    except Exception:
                        pass
                    try:
                        vc.set_manual_mode_enabled(True)
                    except Exception:
                        pass

                analyzer = getattr(self, "real_agent_analyzer", None)
                if analyzer is not None:
                    try:
                        analyzer.set_ai_enabled(False)
                    except Exception:
                        pass

            _mitsu_v18_apply_state_to_ui(self, previous_state=before_state)

        return result

    AppController._handle_real_control_toggle = _mitsu_v18_handle_real_control_toggle


if hasattr(AppController, "_handle_async_task_completed") and not hasattr(AppController, "_mitsu_v18_original_handle_async_task_completed"):
    AppController._mitsu_v18_original_handle_async_task_completed = AppController._handle_async_task_completed

    @Slot(str, object, str)
    def _mitsu_v18_handle_async_task_completed(self, label, result, error):
        if label != "real_control_toggle":
            return AppController._mitsu_v18_original_handle_async_task_completed(self, label, result, error)

        result = result or {}
        error = str(error or "")
        if error and error != "cancelled":
            print(f"ASYNC WORKER: {label} failed: {error}")

        # Service state wins over requested toggle value.
        state = _mitsu_v18_state(self)
        if state in {"AI_ACTIVE", "MANUAL_ACTIVE"} and not error:
            result = dict(result)
            result["active"] = True
            result["requested"] = True
            if state == "MANUAL_ACTIVE" and bool(getattr(self, "_mitsu_v18_takeover_latched", False)):
                result["authority"] = "manual_takeover"

        # Let the original handler process non-active failures and normal off.
        if not bool(result.get("active")):
            return AppController._mitsu_v18_original_handle_async_task_completed(self, label, result, error)

        self.control_active = True
        if state == "AI_ACTIVE":
            analyzer = getattr(self, "real_agent_analyzer", None)
            if analyzer is not None:
                try:
                    analyzer.set_ai_enabled(True)
                except Exception:
                    pass
            try:
                self._reset_real_direct_model_sample()
                self._ensure_real_direct_model_worker()
            except Exception as exc:
                print(f"REAL MODEL PREVIEW: activation start failed: {exc}")
        else:
            analyzer = getattr(self, "real_agent_analyzer", None)
            if analyzer is not None:
                try:
                    analyzer.set_ai_enabled(False)
                except Exception:
                    pass
            try:
                self._stop_real_direct_model_worker()
                self._reset_real_direct_model_sample()
            except Exception:
                pass
        _mitsu_v18_apply_state_to_ui(self)

        message = result.get("message")
        if message:
            self.view.statusBar().showMessage(str(message))
        return

    AppController._handle_async_task_completed = _mitsu_v18_handle_async_task_completed


if hasattr(AppController, "handle_vehicle_control_state") and not hasattr(AppController, "_mitsu_v18_original_handle_vehicle_control_state"):
    AppController._mitsu_v18_original_handle_vehicle_control_state = AppController.handle_vehicle_control_state

    @Slot(str)
    def _mitsu_v18_handle_vehicle_control_state(self, state):
        AppController._mitsu_v18_original_handle_vehicle_control_state(self, state)

        if self.runtime_mode != "real":
            return

        # Do not repaint route-preview for transient states.
        if str(state or "") in {"ARMING", "DISENGAGING", "READY", "MANUAL_READY"}:
            return

        _mitsu_v18_apply_state_to_ui(self)

    AppController.handle_vehicle_control_state = _mitsu_v18_handle_vehicle_control_state

# --- MITSU_REAL_TAKEOVER_SINGLE_SOURCE_V18_END ---

if __name__ == "__main__":
    # Use the plain Qt event loop for the GUI. Async serial/real-control work
    # runs in AppController.async_runtime on a dedicated standard asyncio loop.
    # This avoids qasync QTimer assertion failures on Windows while preserving
    # the existing CARLA/QThread simulation flow.
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--carla-host",
        default="127.0.0.1",
        help="CARLA host/IP for watchdog (default: 127.0.0.1)",
    )
    args, qt_args = parser.parse_known_args()

    app = QApplication([sys.argv[0], *qt_args])
    main_window = MainWindow()
    controller = AppController(main_window)
    host = str(getattr(args, "carla_host", "") or "").strip()
    if host:
        controller.carla_watchdog_host = host
    auto_device = os.environ.get("MITSU_AUTO_CONNECT_DEVICE", "").strip()
    if auto_device:
        def _auto_connect_real_device():
            controller._set_runtime_mode_from_device(auto_device)
            controller._schedule_async(
                controller._connect_vehicle_control_device(auto_device),
                "connect_vehicle_control_device",
            )
            if os.environ.get("MITSU_AUTO_AI_PREVIEW", "").strip().lower() in {"1", "true", "yes", "on"}:
                controller.handle_ai_toggle(True)

        QTimer.singleShot(500, _auto_connect_real_device)
    app.aboutToQuit.connect(controller.shutdown)
    main_window.show()
    sys.exit(app.exec())
