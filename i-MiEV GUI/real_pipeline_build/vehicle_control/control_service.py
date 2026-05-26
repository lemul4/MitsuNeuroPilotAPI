from __future__ import annotations

import asyncio
import concurrent.futures
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

    def get_current_goal(self):
        return self._last_goal

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
        return ReadinessStatus(
            connected=self._connected and telemetry.connected,
            mission_ok=self.mission is not None and bool(self.mission.waypoints),
            pose_ok=self.pose_ok and pose_valid and pose_fresh,
            cameras_ok=self.cameras_ok,
            ai_preview_ok=self.ai_preview_enabled,
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
                return False

            self.state_machine.on_activate_requested()
            self._emit_state()
            self._log("ARMING: brake hold")
            await self._send_raw_command(VehicleCommand(active=False, gear_request=None, accel_pct=0, brake_pct=35, cruise_enabled=False, reason="brake_hold"))
            await asyncio.sleep(0.1)

            self._log("ARMING: requesting Drive")
            if self.adapter is not None:
                await self.adapter.request_gear(Gear.D)
            await asyncio.sleep(self.arming_delay_sec)

            telemetry = self.get_telemetry()
            if telemetry.gear != Gear.D:
                reason = f"Gear D not confirmed, actual={telemetry.gear.name if telemetry.gear else 'unknown'}"
                self.state_machine.on_fault(reason)
                self._emit_state()
                self.activation_blocked.emit(reason)
                self._log(f"Activation failed: {reason}")
                return False

            self.state_machine.on_gear_drive_confirmed()
            self._emit_state()
            self._log("AI_ACTIVE")
            self._start_autonomy_loop()
            return True

    async def deactivate_control(self, reason: str = "user_requested") -> None:
        async with self._mode_lock:
            if self.adapter is None:
                return
            if self.state_machine.state == DriveState.DISCONNECTED:
                return
            self._stop_autonomy_loop()
            self.state_machine.on_deactivate_requested()
            self._emit_state()
            self._log(f"DISENGAGING: {reason}")

            deadline = time.monotonic() + float(self.disengage_timeout_sec)
            await self._send_raw_command(VehicleCommand.safe_stop(reason="deactivate"))
            while time.monotonic() < deadline:
                telemetry = self.get_telemetry()
                if telemetry.speed_kmh <= self.park_speed_threshold_kmh:
                    break
                await self._send_raw_command(VehicleCommand.safe_stop(reason="stop_before_park"))
                await asyncio.sleep(0.05)

            telemetry = self.get_telemetry()
            if telemetry.speed_kmh <= self.park_speed_threshold_kmh:
                try:
                    await self.adapter.request_gear(Gear.P)
                    telemetry = self.get_telemetry()
                    if telemetry.gear == Gear.P:
                        self._log("Gear P confirmed after stop")
                    else:
                        actual = telemetry.gear.name if telemetry.gear else "unknown"
                        self._log(f"Gear P requested, waiting feedback: actual={actual}")
                except Exception as exc:
                    self._log(f"Park request skipped: {exc}")
            else:
                self._log(f"Park skipped: vehicle still moving ({telemetry.speed_kmh:.2f} km/h)")

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
                await self._send_raw_command(VehicleCommand.safe_stop(reason="telemetry_lost"))
                self._log("Autonomy safe-stop: telemetry/heartbeat lost")
                await asyncio.sleep(period)
                continue
            if self.mission is None or not self.mission.waypoints:
                await self._send_raw_command(VehicleCommand.safe_stop(reason="no_mission"))
                await asyncio.sleep(period)
                continue
            pose = telemetry.pose()
            goal = self.navigator.update(self.mission, pose, telemetry.speed_kmh)
            self._last_goal = goal
            self.nav_goal_changed.emit(goal)
            if not goal.is_valid() or goal.maneuver == "pose_lost":
                await self._send_raw_command(VehicleCommand.safe_stop(reason="pose_lost"))
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
            await self.submit_ai_intent(intent)

    async def submit_ai_intent(self, intent: ControlIntent) -> None:
        self._last_intent = intent
        if self.state_machine.state != DriveState.AI_ACTIVE:
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
        steer_norm = max(-1.0, min(1.0, float(angle_deg) / 630.0))
        throttle = max(0.0, min(1.0, float(accel_pct) / 100.0))
        brake = max(0.0, min(1.0, float(brake_pct) / 100.0))

        # Do not apply throttle in Park/Neutral. User must manually request D/R
        # first, which is visible and auditable in the event log.
        if telemetry.gear not in (Gear.D, Gear.R, Gear.E, Gear.B):
            throttle = 0.0

        intent = ControlIntent(
            seq=self._command_seq,
            steer_norm=steer_norm,
            throttle_norm=throttle,
            brake_norm=brake,
            target_angle_deg=float(angle_deg),
            confidence=1.0,
            speed_cap_kmh=self.mission.speed_cap_kmh if self.mission else 3.0,
            valid_for_ms=120,
        )
        self._command_seq += 1
        command = self.arbiter.build_command(intent, telemetry=telemetry, gear=telemetry.gear, active=True)
        command = VehicleCommand(
            seq=command.seq,
            timestamp_monotonic=command.timestamp_monotonic,
            active=True,
            gear_request=None,
            steering_raw=command.steering_raw,
            accel_pct=command.accel_pct,
            brake_pct=command.brake_pct,
            cruise_enabled=False,
            valid_for_ms=command.valid_for_ms,
            reason="manual",
        )
        await self._send_raw_command(command)

    async def _send_raw_command(self, command: VehicleCommand) -> None:
        if self.adapter is None:
            return
        await self.adapter.send_command(command)
        self.telemetry_changed.emit(self.adapter.get_telemetry())
