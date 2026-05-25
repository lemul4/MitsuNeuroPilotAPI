import asyncio
import unittest

from vehicle_control.models import DeviceDescriptor, Gear, ControlIntent, Mission
from vehicle_control.adapters import MockVehicleAdapter, VehicleAdapterFactory
from vehicle_control.vehicle_gateway import VehicleGateway
from vehicle_control.control_service import VehicleControlService
from vehicle_control.arbiter import ControlArbiter


class DummyGateway:
    def __init__(self):
        self.commands = []
    def write_vehicle_command_now(self, command):
        self.commands.append(command)


class VehicleControlTests(unittest.IsolatedAsyncioTestCase):
    async def test_mock_activation_requires_mission_and_ai_preview(self):
        mock = MockVehicleAdapter()
        factory = VehicleAdapterFactory(real_adapter=None, mock_adapter=mock)
        service = VehicleControlService(factory, ControlArbiter())
        await service.connect_device("TEST_MOCK_VEHICLE")
        ok = await service.activate_control()
        self.assertFalse(ok)

        service.set_mission(Mission.default_test_mission())
        service.set_ai_preview_enabled(True)
        ok = await service.activate_control()
        self.assertTrue(ok)
        self.assertEqual(service.get_telemetry().gear, Gear.D)

    async def test_deactivate_parks_after_stop(self):
        mock = MockVehicleAdapter()
        factory = VehicleAdapterFactory(real_adapter=None, mock_adapter=mock)
        service = VehicleControlService(factory, ControlArbiter())
        await service.connect_device("TEST_MOCK_VEHICLE")
        service.set_mission(Mission.default_test_mission())
        service.set_ai_preview_enabled(True)
        self.assertTrue(await service.activate_control())
        await service.deactivate_control()
        self.assertEqual(service.get_telemetry().gear, Gear.P)


class VehicleControlDisengageTests(unittest.IsolatedAsyncioTestCase):
    async def test_deactivate_from_moving_mock_does_not_raise_and_parks(self):
        mock = MockVehicleAdapter()
        factory = VehicleAdapterFactory(real_adapter=None, mock_adapter=mock)
        service = VehicleControlService(factory, ControlArbiter())
        await service.connect_device("TEST_MOCK_VEHICLE")
        service.set_mission(Mission.default_test_mission())
        service.set_ai_preview_enabled(True)
        self.assertTrue(await service.activate_control())
        mock.telemetry.speed_kmh = 4.0
        await service.deactivate_control(reason="test_moving_stop")
        self.assertEqual(service.get_telemetry().gear, Gear.P)
        self.assertLessEqual(service.get_telemetry().speed_kmh, service.park_speed_threshold_kmh)


if __name__ == "__main__":
    unittest.main()

class ManualControlTests(unittest.IsolatedAsyncioTestCase):
    async def test_manual_gear_and_throttle_after_deactivate(self):
        mock = MockVehicleAdapter()
        factory = VehicleAdapterFactory(real_adapter=None, mock_adapter=mock)
        service = VehicleControlService(factory, ControlArbiter())
        await service.connect_device("TEST_MOCK_VEHICLE")
        service.set_mission(Mission.default_test_mission())
        service.set_ai_preview_enabled(True)
        self.assertTrue(await service.activate_control())
        await service.deactivate_control(reason="manual_takeover")
        self.assertEqual(service.get_telemetry().gear, Gear.P)
        await service.request_manual_gear(Gear.D.value)
        self.assertEqual(service.get_telemetry().gear, Gear.D)
        before = service.get_telemetry().speed_kmh
        for _ in range(10):
            await service.submit_manual_command(angle_deg=0, accel_pct=25, brake_pct=0)
        self.assertGreater(service.get_telemetry().speed_kmh, before)

class ManualTakeoverTests(unittest.IsolatedAsyncioTestCase):
    async def test_ai_deactivate_transfers_to_manual_without_parking(self):
        mock = MockVehicleAdapter()
        factory = VehicleAdapterFactory(real_adapter=None, mock_adapter=mock)
        service = VehicleControlService(factory, ControlArbiter())
        await service.connect_device("TEST_MOCK_VEHICLE")
        service.set_mission(Mission.default_test_mission())
        service.set_ai_preview_enabled(True)
        self.assertTrue(await service.activate_control())
        self.assertEqual(service.get_telemetry().gear, Gear.D)
        ok = await service.transfer_ai_to_manual(reason="unit_test_takeover")
        self.assertTrue(ok)
        self.assertEqual(service.get_telemetry().gear, Gear.D)
        before = service.get_telemetry().speed_kmh
        for _ in range(10):
            await service.submit_manual_command(angle_deg=0, accel_pct=20, brake_pct=0)
        self.assertGreaterEqual(service.get_telemetry().speed_kmh, before)
