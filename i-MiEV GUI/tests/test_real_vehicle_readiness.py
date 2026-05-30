import tempfile
import unittest
from pathlib import Path

from vehicle_control.mcu_protocol import McuTelemetryParser, TelemetryFieldSpec
from vehicle_control.models import VehicleTelemetry, Gear, DeviceDescriptor, Mission, DriveState
from vehicle_control.safety_config import RealVehicleSafetyConfig
from vehicle_control.arbiter import ControlArbiter
from vehicle_control.adapters import MockVehicleAdapter, VehicleAdapterFactory
from vehicle_control.control_service import VehicleControlService


class McuTelemetryParserTests(unittest.TestCase):
    def test_configurable_parser_updates_feedback_and_pose(self):
        parser = McuTelemetryParser([
            TelemetryFieldSpec(0x44, "speed_kmh", 0, 2, False, 0.1),
            TelemetryFieldSpec(0x45, "gear", 0, 1, False, enum="gear_1_6"),
            TelemetryFieldSpec(0x46, "x_m", 0, 2, True, 0.01),
            TelemetryFieldSpec(0x46, "y_m", 2, 2, True, 0.01),
        ])
        t = VehicleTelemetry()
        parser.apply_packet(0x44, [123, 0], t)
        parser.apply_packet(0x45, [4], t)
        parser.apply_packet(0x46, [100, 0, 200, 0], t)
        self.assertAlmostEqual(t.speed_kmh, 12.3)
        self.assertEqual(t.gear, Gear.D)
        self.assertAlmostEqual(t.x_m, 1.0)
        self.assertAlmostEqual(t.y_m, 2.0)
        self.assertTrue(t.pose_valid)


class SafetyConfigTests(unittest.TestCase):
    def test_default_safety_config_blocks_actuation_and_limits_arbiter(self):
        cfg = RealVehicleSafetyConfig()
        self.assertFalse(cfg.actuation_allowed)
        arbiter = ControlArbiter.from_safety_config(cfg)
        self.assertLessEqual(arbiter.max_accel_pct, 10)
        self.assertLessEqual(arbiter.max_steering_raw, 60)


class ManualModeActivationTests(unittest.IsolatedAsyncioTestCase):
    async def test_manual_mode_can_activate_without_ai_preview(self):
        mock = MockVehicleAdapter()
        service = VehicleControlService(VehicleAdapterFactory(real_adapter=None, mock_adapter=mock), ControlArbiter())
        await service.connect_device("TEST_MOCK_VEHICLE")
        service.set_mission(Mission.default_test_mission())
        service.set_manual_mode_enabled(True)
        ok = await service.activate_control()
        self.assertTrue(ok)
        self.assertEqual(service.state_machine.state, DriveState.MANUAL_ACTIVE)
        self.assertEqual(service.get_telemetry().gear, Gear.D)


if __name__ == "__main__":
    unittest.main()
