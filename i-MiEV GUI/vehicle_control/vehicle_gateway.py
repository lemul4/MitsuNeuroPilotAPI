from __future__ import annotations

from typing import List, Optional

from .models import VehicleCommand, Gear

try:
    from hardware.can_commands import (
        CANCommandFactory,
        GEAR_TEMPLATE,
        ACCEL_TEMPLATE,
        BRAKE_TEMPLATE,
        ANGLE_TEMPLATE,
        CRUISE_TEMPLATE,
    )
except Exception:  # pragma: no cover - import fallback for static checks
    CANCommandFactory = None
    GEAR_TEMPLATE = ACCEL_TEMPLATE = BRAKE_TEMPLATE = ANGLE_TEMPLATE = CRUISE_TEMPLATE = ""


class VehicleGateway:
    """Converts VehicleCommand into the existing Serial_Data CAN packet objects.

    This class is the only layer that should know about CAN templates. UI code
    and AI code should only work with VehicleCommand / ControlIntent.
    """

    def __init__(self, serial_manager=None, can_factory=None):
        if can_factory is None:
            if CANCommandFactory is None:
                raise RuntimeError("CANCommandFactory is unavailable")
            can_factory = CANCommandFactory()
        self.serial = serial_manager
        self.can = can_factory
        self.cmd_gear = self.can.create_base_packet(GEAR_TEMPLATE)
        self.cmd_accel = self.can.create_base_packet(ACCEL_TEMPLATE)
        self.cmd_brake = self.can.create_base_packet(BRAKE_TEMPLATE)
        self.cmd_angle = self.can.create_base_packet(ANGLE_TEMPLATE)
        self.cmd_cruise = self.can.create_base_packet(CRUISE_TEMPLATE)

    @staticmethod
    def _signed_raw_to_byte(value: int) -> int:
        value = max(-100, min(100, int(value)))
        return value.to_bytes(1, "little", signed=True)[0]

    def encode_vehicle_command(self, command: VehicleCommand) -> List[object]:
        packets = []

        # Brake first, then gear, then angle, then accel, then cruise. This order
        # makes stale or partially delivered frames fail safer.
        packets.append(self.can.prepare_packet(self.cmd_brake, max(0, min(100, int(command.brake_pct)))))

        if command.gear_request is not None:
            gear = Gear.from_value(command.gear_request)
            packets.append(self.can.prepare_packet(self.cmd_gear, int(gear.value)))

        packets.append(self.can.prepare_packet(self.cmd_angle, self._signed_raw_to_byte(command.steering_raw)))

        accel = 0 if int(command.brake_pct) > 0 else int(command.accel_pct)
        packets.append(self.can.prepare_packet(self.cmd_accel, max(0, min(100, accel))))
        packets.append(self.can.prepare_packet(self.cmd_cruise, 1 if command.cruise_enabled and command.active else 0))
        return packets

    def write_vehicle_command_now(self, command: VehicleCommand) -> None:
        packets = self.encode_vehicle_command(command)
        if self.serial is None:
            return
        write_now = getattr(self.serial, "write_packet_immediate", None)
        if callable(write_now):
            for packet in packets:
                write_now(packet)
            return
        for packet in packets:
            self.serial.send_command(packet)

    def send_vehicle_command_latest(self, command: VehicleCommand) -> None:
        packets = self.encode_vehicle_command(command)
        if self.serial is None:
            return
        send_set = getattr(self.serial, "send_control_packet_set_latest", None)
        if callable(send_set):
            send_set(packets)
            return
        self.write_vehicle_command_now(command)
