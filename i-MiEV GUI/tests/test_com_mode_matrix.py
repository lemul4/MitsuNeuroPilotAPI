import asyncio
import time
import unittest

from vehicle_control.adapters import MockVehicleAdapter, RealSerialVehicleAdapter, VehicleAdapterFactory
from vehicle_control.arbiter import ControlArbiter
from vehicle_control.control_service import VehicleControlService
from vehicle_control.mcu_protocol import McuTelemetryParser, TelemetryFieldSpec
from vehicle_control.models import (
    ControlIntent,
    DeviceDescriptor,
    DeviceKind,
    DriveState,
    Gear,
    Mission,
    Pose2D,
    VehicleCommand,
    VehicleTelemetry,
)
from vehicle_control.safety_config import RealVehicleSafetyConfig
from vehicle_control.vehicle_gateway import VehicleGateway


class _Signal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def emit(self, *args):
        for slot in list(self.slots):
            slot(*args)


class _FakeSerialManager:
    def __init__(self):
        self.data_received = _Signal()
        self.connection_status = _Signal()
        self.running = False
        self.port = None
        self.closed = False

    async def connect_serial(self, port):
        self.running = True
        self.port = port
        self.connection_status.emit(True, f"connected {port}")

    def close(self):
        self.running = False
        self.closed = True
        self.connection_status.emit(False, "closed")


class _CommandSink:
    def __init__(self):
        self.commands = []

    def write_vehicle_command_now(self, command):
        self.commands.append(command)


class _Packet:
    class _CanData:
        def __init__(self, data):
            self.DATA = data

    def __init__(self, can_id, data):
        self.CAN_ID = can_id
        self.CAN_DATA = self._CanData(data)


class _FakeCanFactory:
    def create_base_packet(self, template):
        return template

    def prepare_packet(self, base, value):
        return {"base": base, "value": value}


def _parser():
    return McuTelemetryParser(
        [
            TelemetryFieldSpec(0x44, "speed_kmh", 0, 1, False, 1.0),
            TelemetryFieldSpec(0x45, "gear", 0, 1, False, enum="gear_1_6"),
            TelemetryFieldSpec(0x46, "x_m", 0, 2, True, 0.01),
            TelemetryFieldSpec(0x46, "y_m", 2, 2, True, 0.01),
            TelemetryFieldSpec(0x47, "yaw_deg", 0, 2, True, 0.01),
            TelemetryFieldSpec(0x48, "pose_valid", 0, 1, False, 1.0, valid_value=1),
        ]
    )


async def _wait_until(predicate, timeout=1.0, interval=0.01):
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return bool(predicate())


class ComModeAdapterMatrixTests(unittest.IsolatedAsyncioTestCase):
    async def test_open_com_port_is_not_mcu_heartbeat_until_packet_arrives(self):
        serial = _FakeSerialManager()
        sink = _CommandSink()
        adapter = RealSerialVehicleAdapter(
            serial,
            sink,
            safety_config=RealVehicleSafetyConfig(),
            telemetry_parser=_parser(),
        )

        await adapter.connect(DeviceDescriptor("serial:COM4", "COM4", DeviceKind.REAL_SERIAL, "COM4"))
        self.assertTrue(adapter.get_telemetry().connected)
        self.assertFalse(adapter.get_telemetry().heartbeat_ok)
        self.assertEqual(adapter.get_diagnostics()["rx_packet_count"], 0)

        serial.data_received.emit(_Packet(0x44, [36, 0, 0, 0]))
        self.assertTrue(adapter.get_telemetry().heartbeat_ok)
        self.assertEqual(adapter.get_telemetry().speed_kmh, 36.0)
        self.assertEqual(adapter.get_diagnostics()["rx_packet_count"], 1)
        self.assertEqual(adapter.get_diagnostics()["mapped_packet_count"], 1)

        serial.connection_status.emit(False, "closed")
        self.assertFalse(adapter.get_telemetry().connected)
        self.assertFalse(adapter.get_telemetry().heartbeat_ok)

    async def test_unmapped_or_short_mcu_packet_does_not_update_mapped_telemetry(self):
        telemetry = VehicleTelemetry()
        parser = _parser()
        self.assertFalse(parser.apply_packet(0x44, [], telemetry))
        self.assertEqual(telemetry.speed_kmh, 0.0)
        self.assertFalse(parser.apply_packet(0x999, [1, 2, 3, 4], telemetry))

        serial = _FakeSerialManager()
        adapter = RealSerialVehicleAdapter(
            serial,
            _CommandSink(),
            safety_config=RealVehicleSafetyConfig(),
            telemetry_parser=parser,
        )
        await adapter.connect(DeviceDescriptor("serial:COM4", "COM4", DeviceKind.REAL_SERIAL, "COM4"))
        serial.data_received.emit(_Packet(0x999, [1, 2, 3, 4]))

        diagnostics = adapter.get_diagnostics()
        self.assertEqual(diagnostics["rx_packet_count"], 1)
        self.assertEqual(diagnostics["mapped_packet_count"], 0)
        self.assertEqual(adapter.get_telemetry().speed_kmh, 0.0)

    async def test_real_com_readiness_blocks_without_mcu_then_allows_after_packet_pose_and_camera(self):
        serial = _FakeSerialManager()
        sink = _CommandSink()
        adapter = RealSerialVehicleAdapter(
            serial,
            sink,
            safety_config=RealVehicleSafetyConfig(),
            telemetry_parser=_parser(),
        )
        service = VehicleControlService(
            VehicleAdapterFactory(real_adapter=adapter, mock_adapter=None),
            ControlArbiter(),
        )
        service.arming_delay_sec = 0.0
        service.autonomy_loop_enabled = False

        try:
            await service.connect_device("COM4")
            service.set_mission(Mission.default_test_mission())
            service.submit_pose(Pose2D(0.0, 0.0, 0.0, valid=True, source="test_pose"))
            service.set_camera_status(True)
            service.set_ai_preview_enabled(True)

            self.assertFalse(service._build_readiness().vehicle_ok)
            self.assertFalse(await service.activate_control())

            serial.data_received.emit(_Packet(0x44, [0, 0, 0, 0]))
            serial.data_received.emit(_Packet(0x45, [Gear.D.value, 0, 0, 0]))
            await asyncio.sleep(0.06)

            self.assertTrue(service._build_readiness().all_ok)
            self.assertTrue(await service.activate_control())
            self.assertEqual(service.get_telemetry().gear, Gear.D)
            self.assertTrue(sink.commands)
            self.assertEqual(sink.commands[-1].reason, "host_dry_run_guard")
        finally:
            await service.disconnect()

    async def test_real_com_readiness_blocks_when_camera_stream_is_not_ready(self):
        serial = _FakeSerialManager()
        adapter = RealSerialVehicleAdapter(
            serial,
            _CommandSink(),
            safety_config=RealVehicleSafetyConfig(),
            telemetry_parser=_parser(),
        )
        service = VehicleControlService(
            VehicleAdapterFactory(real_adapter=adapter, mock_adapter=None),
            ControlArbiter(),
        )
        try:
            await service.connect_device("COM4")
            service.set_mission(Mission.default_test_mission())
            service.submit_pose(Pose2D(0.0, 0.0, 0.0, valid=True, source="test_pose"))
            service.set_ai_preview_enabled(True)
            serial.data_received.emit(_Packet(0x44, [0, 0, 0, 0]))
            serial.data_received.emit(_Packet(0x45, [Gear.P.value, 0, 0, 0]))
            await asyncio.sleep(0.06)

            readiness = service._build_readiness()
            self.assertFalse(readiness.cameras_ok)
            self.assertIn("Camera stream is not ready", readiness.blocked_reasons())
            self.assertFalse(await service.activate_control())
        finally:
            await service.disconnect()

    async def test_real_com_manual_activation_does_not_require_ai_mission_pose_or_cameras(self):
        serial = _FakeSerialManager()
        sink = _CommandSink()
        adapter = RealSerialVehicleAdapter(
            serial,
            sink,
            safety_config=RealVehicleSafetyConfig(),
            telemetry_parser=_parser(),
        )
        service = VehicleControlService(
            VehicleAdapterFactory(real_adapter=adapter, mock_adapter=MockVehicleAdapter()),
            ControlArbiter(),
        )
        try:
            await service.connect_device("COM4")
            service.set_manual_mode_enabled(True)
            serial.data_received.emit(_Packet(0x44, [0, 0, 0, 0]))
            serial.data_received.emit(_Packet(0x45, [Gear.P.value, 0, 0, 0]))
            await asyncio.sleep(0.06)

            readiness = service._build_readiness()
            self.assertTrue(readiness.mission_ok)
            self.assertTrue(readiness.pose_ok)
            self.assertTrue(readiness.cameras_ok)
            self.assertTrue(readiness.ai_preview_ok)
            self.assertTrue(readiness.vehicle_ok)

            self.assertTrue(await service.activate_control())
            self.assertEqual(service.state_machine.state, DriveState.MANUAL_ACTIVE)
            self.assertEqual(service.get_telemetry().gear, Gear.P)
            self.assertTrue(sink.commands)
        finally:
            await service.disconnect()

    async def test_real_com_manual_active_allows_pedal_and_steering_commands_in_park(self):
        serial = _FakeSerialManager()
        sink = _CommandSink()
        adapter = RealSerialVehicleAdapter(
            serial,
            sink,
            safety_config=RealVehicleSafetyConfig(
                allow_real_actuation=True,
                dry_run=False,
                manual_accel_rise_pct_per_sec=1000.0,
                manual_accel_fall_pct_per_sec=1000.0,
                manual_brake_rise_pct_per_sec=1000.0,
                manual_brake_fall_pct_per_sec=1000.0,
            ),
            telemetry_parser=_parser(),
        )
        service = VehicleControlService(
            VehicleAdapterFactory(real_adapter=adapter, mock_adapter=MockVehicleAdapter()),
            ControlArbiter(),
        )
        try:
            await service.connect_device("COM4")
            service.set_manual_mode_enabled(True)
            serial.data_received.emit(_Packet(0x44, [0, 0, 0, 0]))
            serial.data_received.emit(_Packet(0x45, [Gear.P.value, 0, 0, 0]))
            await asyncio.sleep(0.06)

            self.assertTrue(await service.activate_control())
            self.assertEqual(service.state_machine.state, DriveState.MANUAL_ACTIVE)
            self.assertEqual(service.get_telemetry().gear, Gear.P)

            await service.submit_manual_command(angle_deg=120, accel_pct=7, brake_pct=0)

            self.assertTrue(sink.commands)
            self.assertTrue(any(command.reason == "manual_control_enable" for command in sink.commands))
            enable_command = next(command for command in sink.commands if command.reason == "manual_control_enable")
            self.assertTrue(enable_command.send_cruise_frame)
            self.assertTrue(enable_command.cruise_enabled)
            command = sink.commands[-1]
            self.assertEqual(command.reason, "manual")
            self.assertTrue(command.active)
            self.assertFalse(command.send_cruise_frame)
            self.assertGreater(command.steering_raw, 0)
            self.assertEqual(command.accel_pct, 7)
            self.assertEqual(command.brake_pct, 0)
            self.assertIsNone(command.gear_request)

            await asyncio.sleep(0.02)
            await service.submit_manual_command(angle_deg=0, accel_pct=0, brake_pct=11)
            command = sink.commands[-1]
            self.assertEqual(command.reason, "manual")
            self.assertFalse(command.send_cruise_frame)
            self.assertEqual(command.accel_pct, 0)
            self.assertEqual(command.brake_pct, 11)
            self.assertIsNone(command.gear_request)
        finally:
            await service.disconnect()

    async def test_real_com_manual_pedals_are_slew_limited(self):
        serial = _FakeSerialManager()
        sink = _CommandSink()
        adapter = RealSerialVehicleAdapter(
            serial,
            sink,
            safety_config=RealVehicleSafetyConfig(
                allow_real_actuation=True,
                dry_run=False,
                manual_accel_rise_pct_per_sec=20.0,
                manual_accel_fall_pct_per_sec=40.0,
                manual_brake_rise_pct_per_sec=40.0,
                manual_brake_fall_pct_per_sec=80.0,
            ),
            telemetry_parser=_parser(),
        )
        service = VehicleControlService(
            VehicleAdapterFactory(real_adapter=adapter, mock_adapter=MockVehicleAdapter()),
            ControlArbiter(),
        )
        try:
            await service.connect_device("COM4")
            service.set_manual_mode_enabled(True)
            serial.data_received.emit(_Packet(0x44, [0, 0, 0, 0]))
            serial.data_received.emit(_Packet(0x45, [Gear.P.value, 0, 0, 0]))
            await asyncio.sleep(0.06)

            self.assertTrue(await service.activate_control())

            await service.submit_manual_command(angle_deg=0, accel_pct=100, brake_pct=0)
            first = sink.commands[-1]
            self.assertEqual(first.reason, "manual")
            self.assertGreater(first.accel_pct, 0)
            self.assertLess(first.accel_pct, 100)

            await asyncio.sleep(0.10)
            await service.submit_manual_command(angle_deg=0, accel_pct=100, brake_pct=0)
            second = sink.commands[-1]
            self.assertGreater(second.accel_pct, first.accel_pct)
            self.assertLess(second.accel_pct, 100)

            await asyncio.sleep(0.10)
            await service.submit_manual_command(angle_deg=0, accel_pct=0, brake_pct=100)
            third = sink.commands[-1]
            self.assertEqual(third.accel_pct, 0)
            self.assertGreater(third.brake_pct, 0)
            self.assertLess(third.brake_pct, 100)
        finally:
            await service.disconnect()

    async def test_real_com_heartbeat_loss_in_ai_loop_results_in_dry_run_safe_stop(self):
        serial = _FakeSerialManager()
        sink = _CommandSink()
        adapter = RealSerialVehicleAdapter(
            serial,
            sink,
            safety_config=RealVehicleSafetyConfig(),
            telemetry_parser=_parser(),
        )
        service = VehicleControlService(
            VehicleAdapterFactory(real_adapter=adapter, mock_adapter=None),
            ControlArbiter(),
        )
        service.arming_delay_sec = 0.0
        service.autonomy_tick_hz = 40.0
        service.nav_log_interval_sec = 999.0

        try:
            await service.connect_device("COM4")
            service.set_mission(Mission.default_test_mission())
            service.submit_pose(Pose2D(0.0, 0.0, 0.0, valid=True, source="test_pose"))
            service.set_camera_status(True)
            service.set_ai_preview_enabled(True)
            serial.data_received.emit(_Packet(0x44, [0, 0, 0, 0]))
            serial.data_received.emit(_Packet(0x45, [Gear.D.value, 0, 0, 0]))
            await asyncio.sleep(0.06)

            self.assertTrue(await service.activate_control())
            adapter.telemetry.heartbeat_ok = False

            self.assertTrue(
                await _wait_until(
                    lambda: sink.commands
                    and sink.commands[-1].reason == "host_dry_run_guard"
                    and sink.commands[-1].brake_pct >= 35,
                    timeout=1.0,
                )
            )
        finally:
            await service.disconnect()

    async def test_dry_run_guard_replaces_active_real_command_with_safe_stop(self):
        sink = _CommandSink()
        adapter = RealSerialVehicleAdapter(
            _FakeSerialManager(),
            sink,
            safety_config=RealVehicleSafetyConfig(allow_real_actuation=False, dry_run=True),
            telemetry_parser=_parser(),
        )
        command = VehicleCommand(
            seq=10,
            active=True,
            gear_request=Gear.D,
            steering_raw=50,
            accel_pct=30,
            brake_pct=0,
            cruise_enabled=True,
            reason="unit_active",
        )

        await adapter.send_command(command)

        self.assertEqual(len(sink.commands), 1)
        sent = sink.commands[0]
        self.assertEqual(sent.reason, "host_dry_run_guard")
        self.assertFalse(sent.active)
        self.assertIsNone(sent.gear_request)
        self.assertEqual(sent.accel_pct, 0)
        self.assertGreaterEqual(sent.brake_pct, 25)
        self.assertEqual(adapter.get_telemetry().requested_gear, Gear.D)

    async def test_actuation_allowed_passes_real_command_to_gateway(self):
        sink = _CommandSink()
        adapter = RealSerialVehicleAdapter(
            _FakeSerialManager(),
            sink,
            safety_config=RealVehicleSafetyConfig(allow_real_actuation=True, dry_run=False),
            telemetry_parser=_parser(),
        )
        command = VehicleCommand(
            seq=11,
            active=True,
            gear_request=Gear.D,
            steering_raw=-30,
            accel_pct=8,
            brake_pct=0,
            cruise_enabled=True,
            reason="unit_active_allowed",
        )

        await adapter.send_command(command)

        self.assertEqual(sink.commands, [command])
        self.assertEqual(adapter.get_telemetry().target_angle_deg, -189.0)
        self.assertEqual(adapter.get_telemetry().accel_pct, 8.0)


class ComModeGatewayMatrixTests(unittest.TestCase):
    def test_device_descriptor_and_factory_classify_real_and_test_com_modes(self):
        real = DeviceDescriptor.from_combo_text("USB Serial Port - COM4")
        self.assertEqual(real.kind, DeviceKind.REAL_SERIAL)
        self.assertEqual(real.port, "COM4")

        self.assertEqual(DeviceDescriptor.from_combo_text("TEST_MOCK_VEHICLE").kind, DeviceKind.MOCK_VEHICLE)
        self.assertEqual(DeviceDescriptor.from_combo_text("TEST_REPLAY_LOG").kind, DeviceKind.REPLAY_LOG)
        self.assertEqual(DeviceDescriptor.from_combo_text("TEST_SERIAL_LOOPBACK").kind, DeviceKind.SERIAL_LOOPBACK)

    def test_gateway_encodes_brake_first_and_suppresses_accel_while_braking(self):
        gateway = VehicleGateway(can_factory=_FakeCanFactory())
        packets = gateway.encode_vehicle_command(
            VehicleCommand(
                active=True,
                gear_request=Gear.D,
                steering_raw=-1,
                accel_pct=80,
                brake_pct=12,
                cruise_enabled=True,
            )
        )

        values = [packet["value"] for packet in packets]
        self.assertEqual(values, [12, Gear.D.value, 255, 0])

        cruise_packets = gateway.encode_vehicle_command(
            VehicleCommand(active=True, cruise_enabled=True, send_cruise_frame=True)
        )
        self.assertEqual([packet["value"] for packet in cruise_packets], [0, 0, 0, 1])

    def test_gateway_encodes_safe_stop_without_gear_or_accel(self):
        gateway = VehicleGateway(can_factory=_FakeCanFactory())
        packets = gateway.encode_vehicle_command(VehicleCommand.safe_stop(brake_pct=40))

        values = [packet["value"] for packet in packets]
        self.assertEqual(values, [40, 0, 0])

    def test_arbiter_replaces_expired_intent_with_safe_stop(self):
        command = ControlArbiter().build_command(
            ControlIntent(
                timestamp_monotonic=time.monotonic() - 1.0,
                confidence=1.0,
                prediction_age_ms=1000.0,
                valid_for_ms=10,
            )
        )

        self.assertFalse(command.active)
        self.assertEqual(command.reason, "stale_prediction")
        self.assertGreaterEqual(command.brake_pct, 35)

    def test_gateway_uses_latest_batch_sender_when_available(self):
        class _BatchSerial:
            def __init__(self):
                self.batches = []

            def send_control_packet_set_latest(self, packets):
                self.batches.append(packets)

        serial = _BatchSerial()
        gateway = VehicleGateway(serial_manager=serial, can_factory=_FakeCanFactory())
        gateway.send_vehicle_command_latest(VehicleCommand(active=True, accel_pct=5))

        self.assertEqual(len(serial.batches), 1)
        self.assertEqual(serial.batches[0][-1]["value"], 5)


if __name__ == "__main__":
    unittest.main()
