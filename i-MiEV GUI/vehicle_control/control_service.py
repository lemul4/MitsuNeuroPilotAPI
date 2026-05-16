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
)
from .state_machine import DriveStateMachine
from .arbiter import ControlArbiter
from .adapters import VehicleAdapterFactory, VehicleAdapterError


class VehicleControlService(QObject):
    state_changed = Signal(str)
    telemetry_changed = Signal(object)
    event_logged = Signal(str)
    activation_blocked = Signal(str)

    def __init__(self, adapter_factory: VehicleAdapterFactory, arbiter: Optional[ControlArbiter] = None, parent=None):
        super().__init__(parent)
        self.adapter_factory = adapter_factory
        self.arbiter = arbiter or ControlArbiter()
        self.state_machine = DriveStateMachine()
        self.adapter = None
        self.descriptor: Optional[DeviceDescriptor] = None
        self.ai_preview_enabled = False
        self.mission: Optional[Mission] = None
        self.pose_ok = True
        self.cameras_ok = True
        self.vehicle_ok = False
        self._command_seq = 0
        self._last_intent: Optional[ControlIntent] = None
        self._telemetry_task = None
        self._connected = False
        self.arming_delay_sec = 0.55
        self.park_speed_threshold_kmh = 0.5

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
        self._refresh_readiness_state()
        if mission is None:
            self._log("Mission cleared")
        else:
            self._log(f"Mission validated: {mission.name}")

    def set_mission_selected(self, selected: bool) -> None:
        self.set_mission(Mission.default_test_mission() if selected else None)

    def set_ai_preview_enabled(self, enabled: bool) -> None:
        self.ai_preview_enabled = bool(enabled)
        self.state_machine.on_ai_preview(self.ai_preview_enabled, self._build_readiness())
        self._emit_state()
        self._log("AI Preview enabled" if enabled else "AI Preview disabled")

    def set_camera_status(self, ok: bool) -> None:
        self.cameras_ok = bool(ok)
        self._refresh_readiness_state()

    def set_pose_status(self, ok: bool) -> None:
        self.pose_ok = bool(ok)
        self._refresh_readiness_state()

    def _build_readiness(self) -> ReadinessStatus:
        telemetry = self.get_telemetry()
        return ReadinessStatus(
            connected=self._connected and telemetry.connected,
            mission_ok=self.mission is not None,
            pose_ok=self.pose_ok,
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
        readiness = self._build_readiness()
        decision = self.state_machine.can_activate(readiness)
        if not decision.allowed:
            self.activation_blocked.emit(decision.reason)
            self._log(f"Activation blocked: {decision.reason}")
            return False

        self.state_machine.on_activate_requested()
        self._emit_state()
        self._log("ARMING: brake hold")

        # Keep throttle zero and hold brake before gear D.
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
        return True

    async def deactivate_control(self, reason: str = "user_requested") -> None:
        if self.adapter is None:
            return
        if self.state_machine.state == DriveState.DISCONNECTED:
            return

        self.state_machine.on_deactivate_requested()
        self._emit_state()
        self._log(f"DISENGAGING: {reason}")

        # AI authority is removed immediately. Park is requested only after the
        # vehicle is practically stopped.
        await self._send_raw_command(VehicleCommand.safe_stop(reason="deactivate"))
        for _ in range(30):
            telemetry = self.get_telemetry()
            if telemetry.speed_kmh <= self.park_speed_threshold_kmh:
                break
            await self._send_raw_command(VehicleCommand.safe_stop(reason="stop_before_park"))
            await asyncio.sleep(0.05)

        telemetry = self.get_telemetry()
        if telemetry.speed_kmh <= self.park_speed_threshold_kmh:
            try:
                await self.adapter.request_gear(Gear.P)
                self._log("Gear P requested after stop")
            except Exception as exc:
                self._log(f"Park request failed: {exc}")
        else:
            self._log("Park skipped: vehicle still moving")

        self.state_machine.on_manual_ready()
        self._emit_state()

    async def submit_ai_intent(self, intent: ControlIntent) -> None:
        self._last_intent = intent
        if self.state_machine.state != DriveState.AI_ACTIVE:
            return
        telemetry = self.get_telemetry()
        command = self.arbiter.build_command(intent, telemetry=telemetry, gear=Gear.D, active=True)
        await self._send_raw_command(command)

    async def submit_manual_command(self, angle_deg: int, accel_pct: int, brake_pct: int) -> None:
        # Used for mock/bench testing through the same command path.
        steer_norm = max(-1.0, min(1.0, float(angle_deg) / 630.0))
        intent = ControlIntent(
            seq=self._command_seq,
            steer_norm=steer_norm,
            throttle_norm=max(0.0, min(1.0, float(accel_pct) / 100.0)),
            brake_norm=max(0.0, min(1.0, float(brake_pct) / 100.0)),
            target_angle_deg=float(angle_deg),
            confidence=1.0,
            speed_cap_kmh=self.mission.speed_cap_kmh if self.mission else 3.0,
            valid_for_ms=120,
        )
        self._command_seq += 1
        await self.submit_ai_intent(intent)

    async def _send_raw_command(self, command: VehicleCommand) -> None:
        if self.adapter is None:
            return
        await self.adapter.send_command(command)
        self.telemetry_changed.emit(self.adapter.get_telemetry())
