from __future__ import annotations

import asyncio
import time
from typing import Dict, Optional

from .models import DeviceDescriptor, DeviceKind, VehicleCommand, VehicleTelemetry, Gear
from .vehicle_gateway import VehicleGateway


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
    def __init__(self, serial_manager, gateway: VehicleGateway):
        self.serial = serial_manager
        self.gateway = gateway
        self.telemetry = VehicleTelemetry()
        self.descriptor: Optional[DeviceDescriptor] = None
        self._connected_event = asyncio.Event()
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
        self.telemetry.heartbeat_ok = True
        self.telemetry.last_rx_monotonic = time.monotonic()

    async def disconnect(self) -> None:
        self.serial.close()
        self.telemetry.connected = False
        self.telemetry.heartbeat_ok = False

    async def send_command(self, command: VehicleCommand) -> None:
        self.gateway.write_vehicle_command_now(command)
        self.telemetry.requested_gear = command.gear_request or self.telemetry.requested_gear
        self.telemetry.target_angle_deg = float(command.steering_raw) / 100.0 * 630.0
        self.telemetry.accel_pct = float(command.accel_pct)
        self.telemetry.brake_pct = float(command.brake_pct)

    def get_telemetry(self) -> VehicleTelemetry:
        return self.telemetry

    def _handle_connection_status(self, connected: bool, message: str) -> None:
        self.telemetry.connected = bool(connected)
        if connected:
            self.telemetry.heartbeat_ok = True
            self.telemetry.last_rx_monotonic = time.monotonic()
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
            if can_id == 0x0001:
                val = int.from_bytes(data[0:2], "little", signed=True)
                self.telemetry.angle_deg = int(val * 630 / 0x500)
            elif can_id == 0x0003:
                self.telemetry.speed_kmh = float(data[0])
            elif can_id == 0x0017:
                self.telemetry.brake_pct = float(data[0])
            elif can_id == 0x0018:
                self.telemetry.accel_pct = float(data[0])
            elif can_id == 0x0004:
                self.telemetry.gear = Gear.from_value(data[0])
            self.telemetry.last_rx_monotonic = time.monotonic()
            self.telemetry.heartbeat_ok = True
        except Exception as exc:
            self.telemetry.fault = str(exc)


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
        self.telemetry.last_rx_monotonic = time.monotonic()

    def get_telemetry(self) -> VehicleTelemetry:
        if self.inject_heartbeat_loss:
            self.telemetry.heartbeat_ok = False
        return self.telemetry


class VehicleAdapterFactory:
    def __init__(self, real_adapter: RealSerialVehicleAdapter, mock_adapter: MockVehicleAdapter, loopback_adapter: Optional[MockVehicleAdapter] = None):
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
            return self.real_adapter
        raise VehicleAdapterError(f"Unsupported vehicle adapter kind: {descriptor.kind}")
