from __future__ import annotations

import asyncio
import math
import time
from typing import Dict, Optional

from .models import DeviceDescriptor, DeviceKind, VehicleCommand, VehicleTelemetry, Gear
from .vehicle_gateway import VehicleGateway
from .mcu_protocol import McuTelemetryParser
from .safety_config import RealVehicleSafetyConfig


class VehicleAdapterError(RuntimeError):
    pass


class BaseVehicleAdapter:
    async def connect(self, descriptor: DeviceDescriptor) -> None:
        raise NotImplementedError

    async def disconnect(self) -> None:
        raise NotImplementedError

    async def send_command(self, command: VehicleCommand) -> None:
        raise NotImplementedError

    async def request_gear(self, gear: Gear) -> None:
        command = VehicleCommand(active=False, gear_request=gear, accel_pct=0, brake_pct=25)
        await self.send_command(command)

    def get_telemetry(self) -> VehicleTelemetry:
        raise NotImplementedError


class RealSerialVehicleAdapter(BaseVehicleAdapter):
    def __init__(self, serial_manager, gateway: VehicleGateway, safety_config: RealVehicleSafetyConfig | None = None, telemetry_parser: McuTelemetryParser | None = None):
        self.serial = serial_manager
        self.gateway = gateway
        self.safety_config = safety_config or RealVehicleSafetyConfig.load()
        self.telemetry_parser = telemetry_parser or McuTelemetryParser.load()
        self.telemetry = VehicleTelemetry()
        self.descriptor: Optional[DeviceDescriptor] = None
        self._connected_event = asyncio.Event()
        self._rx_packet_count = 0
        self._mapped_packet_count = 0
        self._last_packet_can_id: Optional[int] = None
        try:
            self.serial.data_received.connect(self.handle_can_packet)
            self.serial.connection_status.connect(self._handle_connection_status)
        except Exception:
            pass

    async def connect(self, descriptor: DeviceDescriptor) -> None:
        if not descriptor.port:
            raise VehicleAdapterError("Serial port is not specified")
        self.descriptor = descriptor
        self._connected_event.clear()
        await self.serial.connect_serial(descriptor.port)
        # SerialManager emits the status signal when the port opens. Do not block
        # forever if Qt signals are not delivered in tests.
        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            if not getattr(self.serial, "running", False):
                raise VehicleAdapterError(f"Cannot connect to {descriptor.port}")
        self.telemetry.connected = True
        self.telemetry.heartbeat_ok = False
        self.telemetry.pose_source = f"{descriptor.port}:no_packets"

    async def disconnect(self) -> None:
        self.serial.close()
        self.telemetry.connected = False
        self.telemetry.heartbeat_ok = False

    async def send_command(self, command: VehicleCommand) -> None:
        command_to_send = command
        if not self.safety_config.actuation_allowed:
            # Host-side dry-run guard. The GUI/control stack can exercise the full
            # pipeline, but no active acceleration or gear request is emitted unless
            # MITSU_REAL_ENABLE_ACTUATION=1 and dry_run=false in safety config.
            if command.active or command.gear_request is not None or getattr(command, "send_cruise_frame", False):
                print(
                    "REAL CONTROL: host dry-run blocked physical command "
                    f"reason={command.reason}; "
                    f"actuation_allowed={self.safety_config.actuation_allowed}; "
                    f"dry_run={self.safety_config.dry_run}"
                )
            command_to_send = VehicleCommand.safe_stop(seq=command.seq, brake_pct=max(25, int(command.brake_pct)), reason="host_dry_run_guard")
        self.gateway.write_vehicle_command_now(command_to_send)
        self.telemetry.requested_gear = command.gear_request or self.telemetry.requested_gear
        self.telemetry.target_angle_deg = float(command.steering_raw) / 100.0 * 630.0
        self.telemetry.accel_pct = float(command.accel_pct)
        self.telemetry.brake_pct = float(command.brake_pct)

    def get_telemetry(self) -> VehicleTelemetry:
        return self.telemetry

    def _handle_connection_status(self, connected: bool, message: str) -> None:
        self.telemetry.connected = bool(connected)
        if connected:
            try:
                self._connected_event.set()
            except Exception:
                pass
        else:
            self.telemetry.heartbeat_ok = False

    def handle_can_packet(self, pkt) -> None:
        try:
            can_id = int(pkt.CAN_ID)
            data = pkt.CAN_DATA.DATA
            self._rx_packet_count += 1
            self._last_packet_can_id = can_id
            if self.telemetry_parser.apply_packet(can_id, data, self.telemetry):
                self._mapped_packet_count += 1
            self.telemetry.last_rx_monotonic = time.monotonic()
            self.telemetry.heartbeat_ok = True
        except Exception as exc:
            self.telemetry.fault = str(exc)

    def get_diagnostics(self) -> Dict[str, object]:
        return {
            "connected": bool(self.telemetry.connected),
            "heartbeat_ok": bool(self.telemetry.heartbeat_ok),
            "rx_packet_count": int(self._rx_packet_count),
            "mapped_packet_count": int(self._mapped_packet_count),
            "last_packet_can_id": self._last_packet_can_id,
        }


class MockVehicleAdapter(BaseVehicleAdapter):
    def __init__(self, label: str = "TEST_MOCK_VEHICLE"):
        self.label = label
        self.telemetry = VehicleTelemetry()
        self.last_command: Optional[VehicleCommand] = None
        self.gear_shift_delay_sec = 0.65
        self.inject_heartbeat_loss = False
        self.inject_gear_timeout = False
        self.inject_serial_ack_loss = False

    async def connect(self, descriptor: DeviceDescriptor) -> None:
        await asyncio.sleep(0.05)
        self.telemetry = VehicleTelemetry(
            connected=True,
            heartbeat_ok=True,
            gear=Gear.P,
            requested_gear=Gear.P,
            speed_kmh=0.0,
            x_m=0.0,
            y_m=0.0,
            yaw_deg=0.0,
            pose_valid=True,
            pose_source="mock_odometry",
            last_rx_monotonic=time.monotonic(),
        )

    async def disconnect(self) -> None:
        self.telemetry.connected = False
        self.telemetry.heartbeat_ok = False
        self.telemetry.speed_kmh = 0.0

    async def request_gear(self, gear: Gear) -> None:
        gear = Gear.from_value(gear)
        self.telemetry.requested_gear = gear
        if self.telemetry.gear == gear:
            self.telemetry.last_rx_monotonic = time.monotonic()
            return
        await asyncio.sleep(self.gear_shift_delay_sec)
        if self.inject_gear_timeout:
            return
        if gear == Gear.P and self.telemetry.speed_kmh > 0.3:
            raise VehicleAdapterError("Mock refuses Park while moving")
        self.telemetry.gear = gear
        self.telemetry.last_rx_monotonic = time.monotonic()

    async def send_command(self, command: VehicleCommand) -> None:
        self.last_command = command
        self.telemetry.heartbeat_ok = not self.inject_heartbeat_loss
        self.telemetry.requested_gear = command.gear_request or self.telemetry.requested_gear
        self.telemetry.target_angle_deg = float(command.steering_raw) / 100.0 * 630.0
        self.telemetry.angle_deg += (self.telemetry.target_angle_deg - self.telemetry.angle_deg) * 0.35
        self.telemetry.accel_pct = float(command.accel_pct)
        self.telemetry.brake_pct = float(command.brake_pct)

        if command.gear_request is not None:
            await self.request_gear(command.gear_request)

        dt = 0.05
        if self.telemetry.gear == Gear.D and command.active:
            accel_term = float(command.accel_pct) * 0.018
        else:
            accel_term = 0.0
        brake_term = float(command.brake_pct) * 0.035
        drag = self.telemetry.speed_kmh * 0.015
        self.telemetry.speed_kmh = max(0.0, self.telemetry.speed_kmh + accel_term - brake_term - drag * dt)

        # Lightweight bicycle-like mock odometry. This is for UI/HIL testing only.
        steering_fraction = max(-1.0, min(1.0, float(command.steering_raw) / 100.0))
        yaw_rate_deg_s = steering_fraction * max(10.0, self.telemetry.speed_kmh * 8.0)
        self.telemetry.yaw_deg = (self.telemetry.yaw_deg + yaw_rate_deg_s * dt + 180.0) % 360.0 - 180.0
        distance_m = (self.telemetry.speed_kmh / 3.6) * dt
        yaw_rad = math.radians(self.telemetry.yaw_deg)
        self.telemetry.x_m += math.cos(yaw_rad) * distance_m
        self.telemetry.y_m += math.sin(yaw_rad) * distance_m
        self.telemetry.pose_valid = True
        self.telemetry.pose_source = "mock_odometry"
        self.telemetry.last_rx_monotonic = time.monotonic()

    def get_telemetry(self) -> VehicleTelemetry:
        if self.inject_heartbeat_loss:
            self.telemetry.heartbeat_ok = False
        elif self.telemetry.connected:
            self.telemetry.heartbeat_ok = True
            self.telemetry.pose_valid = True
            self.telemetry.pose_source = self.telemetry.pose_source or "mock_odometry"
            self.telemetry.last_rx_monotonic = time.monotonic()
        return self.telemetry


class VehicleAdapterFactory:
    def __init__(self, real_adapter: Optional[RealSerialVehicleAdapter], mock_adapter: MockVehicleAdapter, loopback_adapter: Optional[MockVehicleAdapter] = None):
        self.real_adapter = real_adapter
        self.mock_adapter = mock_adapter
        self.loopback_adapter = loopback_adapter or MockVehicleAdapter(label="TEST_SERIAL_LOOPBACK")

    def create(self, descriptor: DeviceDescriptor) -> BaseVehicleAdapter:
        if descriptor.kind == DeviceKind.MOCK_VEHICLE:
            return self.mock_adapter
        if descriptor.kind == DeviceKind.SERIAL_LOOPBACK:
            return self.loopback_adapter
        if descriptor.kind == DeviceKind.REPLAY_LOG:
            return self.mock_adapter
        if descriptor.kind == DeviceKind.REAL_SERIAL:
            if self.real_adapter is None:
                raise VehicleAdapterError("Real serial adapter is not configured")
            return self.real_adapter
        raise VehicleAdapterError(f"Unsupported vehicle adapter kind: {descriptor.kind}")
