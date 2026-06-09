from __future__ import annotations

import asyncio
import concurrent.futures
import os
import time
from typing import Optional

try:
    from PySide6.QtCore import QObject, Signal
except Exception:  # pragma: no cover
    class _Signal:
        def __init__(self, *args, **kwargs):
            self._slots = []
        def connect(self, slot):
            self._slots.append(slot)
        def emit(self, *args, **kwargs):
            for slot in list(self._slots):
                slot(*args, **kwargs)
    class QObject:
        def __init__(self, *args, **kwargs):
            pass
    def Signal(*args, **kwargs):
        return _Signal()

from .models import (
    DeviceDescriptor,
    DeviceKind,
    Gear,
    DriveState,
    VehicleCommand,
    VehicleTelemetry,
    ReadinessStatus,
    ControlIntent,
    Mission,
    Pose2D,
)
from .state_machine import DriveStateMachine
from .arbiter import ControlArbiter
from .adapters import VehicleAdapterFactory, VehicleAdapterError
from .navigation import NavigatorService
from .pid import WaypointPIDController


class VehicleControlService(QObject):
    state_changed = Signal(str)
    telemetry_changed = Signal(object)
    nav_goal_changed = Signal(object)
    event_logged = Signal(str)
    activation_blocked = Signal(str)

    def __init__(
        self,
        adapter_factory: VehicleAdapterFactory,
        arbiter: Optional[ControlArbiter] = None,
        navigator: Optional[NavigatorService] = None,
        waypoint_controller: Optional[WaypointPIDController] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.adapter_factory = adapter_factory
        self.arbiter = arbiter or ControlArbiter()
        self.navigator = navigator or NavigatorService()
        self.waypoint_controller = waypoint_controller or WaypointPIDController()
        self.state_machine = DriveStateMachine()
        self.adapter = None
        self.descriptor: Optional[DeviceDescriptor] = None
        self.ai_preview_enabled = False
        self.manual_mode_enabled = False
        self.mission: Optional[Mission] = None
        self.pose_ok = True
        self.cameras_ok = True
        self.vehicle_ok = False
        self.autonomy_loop_enabled = True
        self.autonomy_tick_hz = 20.0
        self._command_seq = 0
        self._last_intent: Optional[ControlIntent] = None
        self._telemetry_task = None
        self._autonomy_task = None
        self._connected = False
        self.arming_delay_sec = 0.55
        self.park_speed_threshold_kmh = 0.25
        self.disengage_timeout_sec = 6.0
        self._mode_lock = asyncio.Lock()
        self._last_autonomy_log_at = 0.0
        self.nav_log_interval_sec = 2.0
        self.external_agent_enabled = False
        self._last_external_intent: Optional[ControlIntent] = None
        self._last_goal = None
        self.external_intent_max_age_ms = 150.0
        self.ai_authority_confirmed = False
        self._manual_last_update_monotonic: Optional[float] = None
        self._manual_accel_pct = 0.0
        self._manual_brake_pct = 0.0

    def _schedule_coro_threadsafe(self, coro, label="vehicle control task"):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError as exc:
                try:
                    coro.close()
                except Exception:
                    pass
                self._log(f"Async schedule failed for {label}: {exc}")
                return None
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception as exc:
            try:
                coro.close()
            except Exception:
                pass
            self._log(f"Async schedule failed for {label}: {exc}")
            return None

        def _done(done_future):
            try:
                done_future.result()
            except (asyncio.CancelledError, concurrent.futures.CancelledError):
                pass
            except Exception as exc:
                self._log(f"{label} failed: {exc}")
        future.add_done_callback(_done)
        return future

    def _emit_state(self) -> None:
        self.state_changed.emit(self.state_machine.state.value)

    def _log(self, message: str) -> None:
        self.event_logged.emit(str(message))

    def get_telemetry(self) -> VehicleTelemetry:
        if self.adapter is None:
            return VehicleTelemetry()
        return self.adapter.get_telemetry()

    def set_mission(self, mission: Optional[Mission]) -> None:
        self.mission = mission
        self.navigator.reset()
        self.waypoint_controller.reset()
        self._seed_pose_from_mission_for_test_modes(mission)
        self._refresh_readiness_state()
        if mission is None:
            self._log("Mission cleared")
        else:
            self._log(f"Mission validated: {mission.name}; waypoints={len(mission.waypoints)}")

    def _seed_pose_from_mission_for_test_modes(self, mission: Optional[Mission]) -> None:
        """Make TEST_MOCK/REPLAY usable without external localization.

        Real serial mode must not fake localization. For mock/HIL UI tests, the
        navigator should start at the first waypoint so A/B route tests can run
        immediately after Validate.
        """
        if mission is None or not mission.waypoints or self.adapter is None:
            return
        if self.descriptor is None or self.descriptor.kind not in {DeviceKind.MOCK_VEHICLE, DeviceKind.REPLAY_LOG, DeviceKind.SERIAL_LOOPBACK}:
            return
        try:
            telemetry = self.adapter.get_telemetry()
            first = mission.waypoints[0]
            telemetry.x_m = float(first.x_m)
            telemetry.y_m = float(first.y_m)
            telemetry.yaw_deg = float(first.yaw_deg)
            telemetry.pose_valid = True
            telemetry.pose_source = "mission_start_mock"
            telemetry.last_rx_monotonic = time.monotonic()
        except Exception as exc:
            self._log(f"Mock pose seed skipped: {exc}")

    def set_mission_selected(self, selected: bool) -> None:
        self.set_mission(Mission.default_test_mission() if selected else None)

    def set_ai_preview_enabled(self, enabled: bool) -> None:
        self.ai_preview_enabled = bool(enabled)
        self.external_agent_enabled = bool(enabled)
        if not enabled:
            self._last_external_intent = None
        self.state_machine.on_ai_preview(self.ai_preview_enabled, self._build_readiness())
        self._emit_state()
        self._log("AI Preview enabled" if enabled else "AI Preview disabled")

    def set_external_agent_enabled(self, enabled: bool) -> None:
        self.external_agent_enabled = bool(enabled)
        if not enabled:
            self._last_external_intent = None

    def set_manual_mode_enabled(self, enabled: bool) -> None:
        self.manual_mode_enabled = bool(enabled)
        # Manual mode is an explicit local authority mode. It does not require a
        # mission, pose, camera stream or AI Preview; those are autonomous-control
        # preconditions. It still requires a connected vehicle controller with a
        # live heartbeat and normal safety checks before Drive can be armed.
        self._refresh_readiness_state()
        if not enabled:
            self._reset_manual_pedal_slew()
        self._log("Manual mode enabled" if enabled else "Manual mode disabled")

    def get_current_goal(self):
        return self._last_goal

    def update_navigation_preview(self):
        if self.mission is None:
            self._last_goal = None
            return None
        telemetry = self.get_telemetry()
        pose = telemetry.pose()
        goal = self.navigator.update(self.mission, pose, telemetry.speed_kmh)
        self._last_goal = goal
        self.nav_goal_changed.emit(goal)
        return goal

    @staticmethod
    def _slew_value(current: float, target: float, dt: float, rise_rate: float, fall_rate: float) -> float:
        current = float(current)
        target = float(target)
        dt = max(0.0, float(dt))
        rate = float(rise_rate if target > current else fall_rate)
        if rate <= 0.0 or dt <= 0.0:
            return current
        step = rate * dt
        if target > current:
            return min(target, current + step)
        return max(target, current - step)

    def _manual_rate(self, name: str, default: float) -> float:
        cfg = getattr(self.adapter, "safety_config", None)
        return float(getattr(cfg, name, default))

    def _gear_confirm_timeout_sec(self) -> float:
        cfg = getattr(self.adapter, "safety_config", None)
        configured = os.environ.get("MITSU_GEAR_CONFIRM_TIMEOUT_SEC", "").strip()
        if configured:
            return max(0.5, float(configured))
        return max(2.5, float(getattr(cfg, "gear_confirm_timeout_sec", self.arming_delay_sec)))

    async def _request_gear_until_confirmed(self, gear: Gear, reason: str = "gear_request", brake_pct: int = 35) -> bool:
        if self.adapter is None:
            return False
        deadline = time.monotonic() + self._gear_confirm_timeout_sec()
        while time.monotonic() <= deadline:
            await self._send_raw_command(
                VehicleCommand(
                    active=False,
                    gear_request=gear,
                    steering_raw=0,
                    accel_pct=0,
                    brake_pct=max(0, min(100, int(brake_pct))),
                    cruise_enabled=False,
                    send_cruise_frame=True,
                    reason=reason,
                )
            )
            await asyncio.sleep(0.10)
            telemetry = self.get_telemetry()
            if telemetry.gear == gear:
                return True
        return self.get_telemetry().gear == gear

    async def _send_ai_authority_hold(self, reason: str, brake_pct: int = 0) -> None:
        await self._send_raw_command(
            VehicleCommand(
                active=True,
                gear_request=None,
                accel_pct=0,
                brake_pct=max(0, min(100, int(brake_pct))),
                cruise_enabled=True,
                send_cruise_frame=True,
                reason=reason,
            )
        )

    async def _arm_ai_authority(self) -> None:
        self.ai_authority_confirmed = False
        brake_pct = max(0, min(100, int(os.environ.get("MITSU_AI_AUTHORITY_BRAKE_PCT", "35") or 35)))
        repeats = max(1, int(float(os.environ.get("MITSU_AI_AUTHORITY_ENABLE_SEC", "1.0") or 1.0) / 0.05))
        for _ in range(repeats):
            await self._send_ai_authority_hold("ai_control_enable", brake_pct=brake_pct)
            await asyncio.sleep(0.05)
        self._log(f"AI authority enable sent: packets={repeats} brake={brake_pct}%")

    async def _send_brake_hold(self, reason: str, brake_pct: int = 35, duration_sec: float = 0.0) -> None:
        deadline = time.monotonic() + max(0.0, float(duration_sec))
        first = True
        while first or time.monotonic() < deadline:
            first = False
            await self._send_raw_command(
                VehicleCommand(
                    active=False,
                    gear_request=None,
                    steering_raw=0,
                    accel_pct=0,
                    brake_pct=max(0, min(100, int(brake_pct))),
                    cruise_enabled=False,
                    send_cruise_frame=True,
                    reason=reason,
                )
            )
            await asyncio.sleep(0.05)

    async def _request_park_continuously(self, brake_pct: int = 35) -> bool:
        duration = max(0.5, float(os.environ.get("MITSU_PARK_REQUEST_SEC", "2.0") or 2.0))
        deadline = time.monotonic() + duration
        confirmed = False
        while time.monotonic() <= deadline:
            await self._send_raw_command(
                VehicleCommand(
                    active=False,
                    gear_request=Gear.P,
                    steering_raw=0,
                    accel_pct=0,
                    brake_pct=max(0, min(100, int(brake_pct))),
                    cruise_enabled=False,
                    send_cruise_frame=True,
                    reason="park_request",
                )
            )
            telemetry = self.get_telemetry()
            confirmed = confirmed or telemetry.gear == Gear.P
            await asyncio.sleep(0.10)
        return confirmed or self.get_telemetry().gear == Gear.P

    async def _send_pedal_release(self, reason: str, repeats: int = 3, interval_sec: float = 0.03) -> None:
        repeats = max(1, int(repeats))
        for _ in range(repeats):
            await self._send_raw_command(
                VehicleCommand(
                    active=False,
                    gear_request=None,
                    steering_raw=0,
                    accel_pct=0,
                    brake_pct=0,
                    cruise_enabled=False,
                    send_cruise_frame=True,
                    reason=reason,
                )
            )
            await asyncio.sleep(max(0.0, float(interval_sec)))
        self._reset_manual_pedal_slew()

    def _reset_manual_pedal_slew(self) -> None:
        self._manual_last_update_monotonic = None
        self._manual_accel_pct = 0.0
        self._manual_brake_pct = 0.0

    def set_camera_status(self, ok: bool) -> None:
        self.cameras_ok = bool(ok)
        self._refresh_readiness_state()

    def set_pose_status(self, ok: bool) -> None:
        self.pose_ok = bool(ok)
        self._refresh_readiness_state()

    def submit_pose(self, pose: Pose2D) -> None:
        telemetry = self.get_telemetry()
        telemetry.x_m = float(pose.x_m)
        telemetry.y_m = float(pose.y_m)
        telemetry.yaw_deg = float(pose.yaw_deg)
        telemetry.pose_valid = bool(pose.valid)
        telemetry.pose_source = str(pose.source or "external")
        telemetry.last_rx_monotonic = time.monotonic()
        self._refresh_readiness_state()

    def _build_readiness(self) -> ReadinessStatus:
        telemetry = self.get_telemetry()
        pose_valid = bool(getattr(telemetry, "pose_valid", False))
        pose_fresh = telemetry.age_ms() <= 750.0
        manual = bool(self.manual_mode_enabled)
        return ReadinessStatus(
            connected=self._connected and telemetry.connected,
            mission_ok=manual or (self.mission is not None and bool(self.mission.waypoints)),
            pose_ok=manual or (self.pose_ok and pose_valid and pose_fresh),
            cameras_ok=manual or self.cameras_ok,
            ai_preview_ok=manual or self.ai_preview_enabled,
            vehicle_ok=self.vehicle_ok and telemetry.heartbeat_ok,
            speed_zero=telemetry.speed_kmh <= self.park_speed_threshold_kmh,
            gear_known=telemetry.gear is not None,
            fault=telemetry.fault,
        )

    def _refresh_readiness_state(self) -> None:
        self.state_machine.refresh_ready_state(self._build_readiness())
        self._emit_state()

    async def connect_device(self, device_text: str) -> None:
        descriptor = DeviceDescriptor.from_combo_text(device_text)
        if descriptor.kind == DeviceKind.VIRTUAL_DEMO:
            raise VehicleAdapterError("VIRTUAL_DEMO_MODE is handled by the CARLA controller")
        self.descriptor = descriptor
        self.adapter = self.adapter_factory.create(descriptor)
        await self.adapter.connect(descriptor)
        telemetry = self.adapter.get_telemetry()
        self._connected = True
        self.vehicle_ok = telemetry.connected and telemetry.heartbeat_ok
        # Real serial mode must prove that both physical camera streams are fresh.
        # Test modes can run without physical cameras.
        if descriptor.kind in {DeviceKind.MOCK_VEHICLE, DeviceKind.REPLAY_LOG, DeviceKind.SERIAL_LOOPBACK}:
            self.cameras_ok = True
        else:
            self.cameras_ok = False
        self.state_machine.on_connected()
        self._emit_state()
        self.telemetry_changed.emit(telemetry)
        self._log(f"Vehicle connected: {descriptor.label}")
        self._start_telemetry_poll()

    async def disconnect(self) -> None:
        await self.deactivate_control(reason="disconnect")
        if self.adapter is not None:
            await self.adapter.disconnect()
        self._connected = False
        self.vehicle_ok = False
        self.ai_preview_enabled = False
        self.manual_mode_enabled = False
        self.state_machine.on_disconnected()
        self._emit_state()
        self._log("Vehicle disconnected")
        self._stop_autonomy_loop()
        self._stop_telemetry_poll()

    def _start_telemetry_poll(self) -> None:
        self._stop_telemetry_poll()
        try:
            self._telemetry_task = self._schedule_coro_threadsafe(self._telemetry_loop(), "telemetry_loop")
        except Exception:
            self._telemetry_task = None

    def _stop_telemetry_poll(self) -> None:
        task = self._telemetry_task
        self._telemetry_task = None
        if task is not None and not task.done():
            task.cancel()

    async def _telemetry_loop(self) -> None:
        while self._connected and self.adapter is not None:
            telemetry = self.adapter.get_telemetry()
            self.vehicle_ok = telemetry.connected and telemetry.heartbeat_ok
            self.telemetry_changed.emit(telemetry)
            self._refresh_readiness_state()
            await asyncio.sleep(0.05)

    async def activate_control(self) -> bool:
        async with self._mode_lock:
            readiness = self._build_readiness()
            decision = self.state_machine.can_activate(readiness)
            if not decision.allowed:
                self.activation_blocked.emit(decision.reason)
                self._log(f"Activation blocked: {decision.reason}")
                self._log(
                    "Readiness: "
                    f"connected={readiness.connected} mission={readiness.mission_ok} "
                    f"pose={readiness.pose_ok} cameras={readiness.cameras_ok} "
                    f"ai={readiness.ai_preview_ok} vehicle={readiness.vehicle_ok} "
                    f"speed_zero={readiness.speed_zero} gear_known={readiness.gear_known}"
                )
                return False

            self.state_machine.on_activate_requested()
            self._emit_state()
            try:
                self._log("ARMING: brake hold")
                await self._send_raw_command(VehicleCommand(active=False, gear_request=None, accel_pct=0, brake_pct=35, cruise_enabled=False, reason="brake_hold"))
                await asyncio.sleep(0.1)

                manual_active = bool(self.manual_mode_enabled and not self.ai_preview_enabled)
                if manual_active:
                    telemetry = self.get_telemetry()
                    gear_text = telemetry.gear.name if telemetry.gear else "unknown"
                    self._log(f"ARMING: manual mode in current gear {gear_text}")
                else:
                    self._log("ARMING: enabling AI authority")
                    await self._arm_ai_authority()
                    self._log("ARMING: requesting Drive")
                    if not await self._request_gear_until_confirmed(Gear.D, reason="drive_request", brake_pct=35):
                        telemetry = self.get_telemetry()
                        reason = f"Gear D not confirmed, actual={telemetry.gear.name if telemetry.gear else 'unknown'}"
                        self.state_machine.on_fault(reason)
                        self._emit_state()
                        self.activation_blocked.emit(reason)
                        self._log(f"Activation failed: {reason}")
                        return False
                    self.ai_authority_confirmed = True
            except Exception as exc:
                reason = f"arming failed: {exc}"
                self.ai_authority_confirmed = False
                self.state_machine.on_fault(reason)
                self._emit_state()
                self.activation_blocked.emit(reason)
                self._log(f"Activation failed: {reason}")
                return False

            self.state_machine.on_gear_drive_confirmed(manual=manual_active)
            self._emit_state()
            if manual_active:
                self.ai_authority_confirmed = False
                self._reset_manual_pedal_slew()
                await self._send_raw_command(
                    VehicleCommand(
                        active=True,
                        gear_request=None,
                        accel_pct=0,
                        brake_pct=0,
                        cruise_enabled=True,
                        send_cruise_frame=True,
                        reason="manual_control_enable",
                    )
                )
                self._log("MANUAL_ACTIVE")
            else:
                self._log("AI_ACTIVE")
                self._start_autonomy_loop()
            return True

    async def transfer_ai_to_manual(self, reason: str = "driver_takeover") -> bool:
        """Transfer authority from AI to manual control without requesting Park.

        This method is used when the operator presses Activate Control again
        while AI_ACTIVE. The vehicle remains in Drive, AI commands are stopped,
        a short zero-throttle/brake command is sent, and manual steering/throttle/
        brake commands can be accepted immediately afterwards. Park is handled
        only by full deactivation after the vehicle has stopped.
        """
        async with self._mode_lock:
            if self.adapter is None:
                return False
            if self.state_machine.state == DriveState.DISCONNECTED:
                return False
            if self.state_machine.state == DriveState.MANUAL_ACTIVE:
                self.manual_mode_enabled = True
                return True
            if self.state_machine.state != DriveState.AI_ACTIVE:
                self._log(f"Manual takeover ignored: state={self.state_machine.state.value}")
                return False

            self._stop_autonomy_loop()
            self.manual_mode_enabled = True

            # Drop AI authority and remove both pedals immediately. This is not
            # a Park request and must not wait for full stop.
            self.ai_authority_confirmed = False
            await self._send_pedal_release(reason=f"{reason}_pedal_release")

            telemetry = self.get_telemetry()
            if telemetry.gear != Gear.D and self.adapter is not None:
                self._log(f"Manual takeover: Drive not confirmed, requesting D; actual={telemetry.gear.name if telemetry.gear else 'unknown'}")
                await self.adapter.request_gear(Gear.D)
                await asyncio.sleep(min(0.15, float(self.arming_delay_sec)))

            self.state_machine.on_manual_takeover()
            # If state machine implementation is stricter, force the resulting
            # authority state only after a valid AI_ACTIVE precondition above.
            if self.state_machine.state != DriveState.MANUAL_ACTIVE:
                self.state_machine.state = DriveState.MANUAL_ACTIVE
            self._emit_state()
            self.telemetry_changed.emit(self.get_telemetry())
            self._log(f"MANUAL_ACTIVE: {reason}")
            return True

    async def deactivate_control(self, reason: str = "user_requested", park: bool = True) -> None:
        async with self._mode_lock:
            if self.adapter is None:
                return
            if self.state_machine.state == DriveState.DISCONNECTED:
                return

            # Repeated Activate/Deactivate while AI has authority is a driver takeover,
            # not an immediate Park request. Park is allowed only after explicit full
            # disengage and after the vehicle has stopped.
            if not park and self.state_machine.state == DriveState.AI_ACTIVE:
                self._stop_autonomy_loop()
                self.ai_authority_confirmed = False
                await self._send_pedal_release(reason=f"{reason}_pedal_release")
                self.manual_mode_enabled = True
                self.state_machine.on_manual_takeover()
                self._emit_state()
                self._log(f"MANUAL_ACTIVE: {reason}")
                return

            self._stop_autonomy_loop()
            self.ai_authority_confirmed = False
            self.state_machine.on_deactivate_requested(park=park)
            self._emit_state()
            self._log(f"DISENGAGING: {reason}")

            deadline = time.monotonic() + float(self.disengage_timeout_sec)
            hold_sec = float(os.environ.get("MITSU_DISENGAGE_BRAKE_HOLD_SEC", "2.0") or 2.0)
            self._log(f"DISENGAGING: brake hold before Park ({hold_sec:.1f}s)")
            await self._send_brake_hold(reason="disengage_brake_hold", brake_pct=35, duration_sec=hold_sec)
            while time.monotonic() < deadline:
                telemetry = self.get_telemetry()
                if telemetry.speed_kmh <= self.park_speed_threshold_kmh:
                    break
                await self._send_raw_command(VehicleCommand.safe_stop(reason="stop_before_park"))
                await asyncio.sleep(0.05)

            telemetry = self.get_telemetry()
            if telemetry.speed_kmh <= self.park_speed_threshold_kmh:
                try:
                    self._log("DISENGAGING: requesting Park continuously")
                    park_confirmed = await self._request_park_continuously(brake_pct=35)
                    telemetry = self.get_telemetry()
                    if park_confirmed:
                        self._log("Gear P confirmed after stop")
                    else:
                        actual = telemetry.gear.name if telemetry.gear else "unknown"
                        self._log(f"Gear P requested, waiting feedback: actual={actual}")
                except Exception as exc:
                    self._log(f"Park request skipped: {exc}")
            else:
                self._log(f"Park skipped: vehicle still moving ({telemetry.speed_kmh:.2f} km/h)")

            await self._send_pedal_release(reason="disengage_pedal_release")
            self.state_machine.on_manual_ready()
            self._emit_state()

    def _start_autonomy_loop(self) -> None:
        self._stop_autonomy_loop()
        if not self.autonomy_loop_enabled:
            return
        try:
            self._autonomy_task = asyncio.create_task(self._autonomy_loop())
        except Exception as exc:
            self._log(f"Autonomy loop failed to start: {exc}")
            self._autonomy_task = None

    def _stop_autonomy_loop(self) -> None:
        task = self._autonomy_task
        self._autonomy_task = None
        if task is not None and not task.done():
            task.cancel()

    async def _autonomy_loop(self) -> None:
        period = 1.0 / max(1.0, float(self.autonomy_tick_hz))
        while self.state_machine.state == DriveState.AI_ACTIVE and self.adapter is not None:
            telemetry = self.get_telemetry()
            if not telemetry.connected or not telemetry.heartbeat_ok:
                await self._send_ai_authority_hold("telemetry_lost", brake_pct=35)
                self._log("Autonomy safe-stop: telemetry/heartbeat lost")
                await asyncio.sleep(period)
                continue
            if self.mission is None or not self.mission.waypoints:
                await self._send_ai_authority_hold("no_mission", brake_pct=35)
                await asyncio.sleep(period)
                continue
            pose = telemetry.pose()
            goal = self.navigator.update(self.mission, pose, telemetry.speed_kmh)
            self._last_goal = goal
            self.nav_goal_changed.emit(goal)
            if not goal.is_valid() or goal.maneuver == "pose_lost":
                await self._send_ai_authority_hold("pose_lost", brake_pct=35)
                if time.monotonic() - self._last_autonomy_log_at > 1.0:
                    self._last_autonomy_log_at = time.monotonic()
                    self._log("Autonomy safe-stop: pose lost/stale")
                await asyncio.sleep(period)
                continue

            source = "route_pid"
            intent = None
            if self.external_agent_enabled and self._last_external_intent is not None:
                candidate = self._last_external_intent
                if not candidate.is_expired() and float(candidate.prediction_age_ms or 0.0) <= self.external_intent_max_age_ms:
                    intent = candidate
                    source = "camera_model"

            if intent is None:
                # Fallback path for dry-run and initial real-car testing. It keeps
                # the car following the validated route even when the neural camera
                # adapter is not configured yet.
                intent = self.waypoint_controller.build_intent(goal, telemetry)

            await self.submit_ai_intent(intent)
            if time.monotonic() - self._last_autonomy_log_at > float(self.nav_log_interval_sec):
                self._last_autonomy_log_at = time.monotonic()
                self._log(
                    f"NAV[{source}]: wp={goal.waypoint_index} {goal.maneuver} "
                    f"target=({goal.target_x_m:.1f},{goal.target_y_m:.1f}) "
                    f"dist={goal.distance_to_target_m:.1f}m desired={goal.desired_speed_kmh:.1f}km/h "
                    f"actual={telemetry.speed_kmh:.2f}km/h cmd_thr={intent.throttle_norm:.2f} cmd_brk={intent.brake_norm:.2f}"
                )
            await asyncio.sleep(period)

    async def submit_external_agent_intent(self, intent: ControlIntent) -> None:
        self._last_external_intent = intent
        if self.state_machine.state == DriveState.AI_ACTIVE:
            if time.monotonic() - self._last_autonomy_log_at > 0.25:
                self._last_autonomy_log_at = time.monotonic()
                self._log(
                    "MODEL->CAN: "
                    f"frame={int(getattr(intent, 'frame_id', 0) or 0)} "
                    f"steer={float(getattr(intent, 'steer_norm', 0.0) or 0.0):.3f} "
                    f"thr={float(getattr(intent, 'throttle_norm', 0.0) or 0.0):.3f} "
                    f"brk={float(getattr(intent, 'brake_norm', 0.0) or 0.0):.3f}"
                )
            await self.submit_ai_intent(intent)

    async def submit_ai_intent(self, intent: ControlIntent) -> None:
        self._last_intent = intent
        if self.state_machine.state != DriveState.AI_ACTIVE:
            return
        if not self.ai_authority_confirmed:
            return
        telemetry = self.get_telemetry()
        command = self.arbiter.build_command(intent, telemetry=telemetry, gear=Gear.D, active=True)
        await self._send_raw_command(command)

    async def request_manual_gear(self, gear_value: int) -> None:
        """Manual/service-mode gear request with the same safety interlocks as AI disengage.

        This is intentionally not used while AI has authority. In real vehicle
        mode, keyboard gear shortcuts are allowed only after Activate Control has
        been disabled and the drive state returned to CONNECTED_MANUAL.
        """
        if self.adapter is None:
            return
        if self.state_machine.state == DriveState.AI_ACTIVE:
            self._log("Manual gear request ignored: AI control is active")
            return
        if self.state_machine.state in (DriveState.ARMING, DriveState.DISENGAGING):
            self._log(f"Manual gear request ignored: state={self.state_machine.state.value}")
            return

        gear = Gear.from_value(gear_value, Gear.P)
        telemetry = self.get_telemetry()

        if gear == Gear.P and telemetry.speed_kmh > self.park_speed_threshold_kmh:
            await self._send_raw_command(VehicleCommand.safe_stop(reason="manual_park_requested"))
            self._log(f"Manual P delayed: vehicle still moving ({telemetry.speed_kmh:.2f} km/h)")
            return

        if gear in (Gear.R, Gear.D, Gear.E, Gear.B) and telemetry.speed_kmh > 0.5:
            self._log(f"Manual {gear.name} blocked while moving ({telemetry.speed_kmh:.2f} km/h)")
            return

        try:
            await self.adapter.request_gear(gear)
            self.telemetry_changed.emit(self.adapter.get_telemetry())
            self._log(f"Manual gear {gear.name} confirmed")
        except Exception as exc:
            self._log(f"Manual gear {gear.name} failed: {exc}")

    async def submit_manual_command(self, angle_deg: int, accel_pct: int, brake_pct: int) -> None:
        if self.adapter is None:
            return
        if self.state_machine.state == DriveState.AI_ACTIVE:
            # Manual keys should not mix with AI authority. Driver takeover is
            # performed by Deactivate Control first, then manual commands are accepted.
            return
        if self.state_machine.state in (DriveState.DISCONNECTED, DriveState.ARMING, DriveState.DISENGAGING, DriveState.FAULT):
            return

        telemetry = self.get_telemetry()
        steering_raw = max(-100, min(100, int(float(angle_deg) / 630.0 * 100.0)))
        target_accel = max(0, min(100, int(accel_pct)))
        target_brake = max(0, min(100, int(brake_pct)))

        # Manual bench mode must match the proven GTK controller: pedal values
        # are operator percentages, not scaled again by AI/PID limits. They are
        # only slew-limited before CAN TX so a held key cannot step the actuator.
        now = time.monotonic()
        if self._manual_last_update_monotonic is None:
            dt = 0.05
        else:
            dt = min(0.25, max(0.0, now - self._manual_last_update_monotonic))
        self._manual_last_update_monotonic = now
        accel = self._slew_value(
            self._manual_accel_pct,
            target_accel,
            dt,
            self._manual_rate("manual_accel_rise_pct_per_sec", 35.0),
            self._manual_rate("manual_accel_fall_pct_per_sec", 80.0),
        )
        brake = self._slew_value(
            self._manual_brake_pct,
            target_brake,
            dt,
            self._manual_rate("manual_brake_rise_pct_per_sec", 120.0),
            self._manual_rate("manual_brake_fall_pct_per_sec", 160.0),
        )

        # Unknown gear feedback is still unsafe for throttle.
        if telemetry.gear is None:
            accel = 0.0
        if target_brake > 0 or brake > 0.0:
            accel = 0.0
        self._manual_accel_pct = accel
        self._manual_brake_pct = brake

        self._command_seq += 1
        command = VehicleCommand(
            seq=self._command_seq,
            active=True,
            gear_request=None,
            steering_raw=steering_raw,
            accel_pct=int(round(accel)),
            brake_pct=int(round(brake)),
            cruise_enabled=False,
            valid_for_ms=120,
            reason="manual",
        )
        await self._send_raw_command(command)

    async def _send_raw_command(self, command: VehicleCommand) -> None:
        if self.adapter is None:
            return
        await self.adapter.send_command(command)
        self.telemetry_changed.emit(self.adapter.get_telemetry())
