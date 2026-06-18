import asyncio
import unittest

import main
from vehicle_control.models import DeviceDescriptor, Gear, DriveState, ControlIntent, Mission, Pose2D, VehicleTelemetry, Waypoint
from vehicle_control.adapters import MockVehicleAdapter, VehicleAdapterFactory
from vehicle_control.vehicle_gateway import VehicleGateway
from vehicle_control.control_service import VehicleControlService
from vehicle_control.arbiter import ControlArbiter
from vehicle_control.navigation import NavigatorService


class DummyGateway:
    def __init__(self):
        self.commands = []
    def write_vehicle_command_now(self, command):
        self.commands.append(command)
    def send_vehicle_command_latest(self, command):
        self.commands.append(command)


class _StatusBar:
    def __init__(self):
        self.messages = []

    def showMessage(self, message, *_args):
        self.messages.append(str(message))


class _View:
    def __init__(self):
        self._status = _StatusBar()

    def statusBar(self):
        return self._status


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

    async def test_ui_pid_update_changes_real_steering_pid(self):
        mock = MockVehicleAdapter()
        service = VehicleControlService(VehicleAdapterFactory(real_adapter=None, mock_adapter=mock), ControlArbiter())
        controller = main.AppController.__new__(main.AppController)
        controller.view = _View()
        controller.vehicle_control = service
        pid = service.waypoint_controller.steering_pid
        pid._integral = 10.0
        pid._last_error = 5.0

        controller.handle_pid_update(0.015, 0.02, 0.003)

        self.assertEqual(pid.kp, 0.015)
        self.assertEqual(pid.ki, 0.02)
        self.assertEqual(pid.kd, 0.003)
        self.assertEqual(pid._integral, 0.0)
        self.assertIsNone(pid._last_error)
        self.assertAlmostEqual(service.arbiter.steering_output_gain, 0.5)
        self.assertAlmostEqual(service.arbiter.max_steering_rate_raw_per_sec, 70.0)
        command = service.arbiter.build_command(ControlIntent(steer_norm=0.5, confidence=1.0, prediction_age_ms=0.0))
        self.assertEqual(command.steering_raw, 25)
        self.assertIn("Steering PID updated", controller.view.statusBar().messages[-1])

    async def test_arbiter_does_not_brake_on_speed_cap(self):
        arbiter = ControlArbiter(max_accel_pct=25, max_brake_pct=80)
        intent = ControlIntent(throttle_norm=1.0, brake_norm=0.0, speed_cap_kmh=1.0, confidence=1.0, prediction_age_ms=0.0)
        telemetry = VehicleTelemetry(speed_kmh=20.0)

        command = arbiter.build_command(intent, telemetry=telemetry, active=True)

        self.assertEqual(command.accel_pct, 25)
        self.assertEqual(command.brake_pct, 0)

    async def test_physical_steering_gain_and_minimum_raw_are_applied(self):
        arbiter = ControlArbiter(
            max_steering_raw=100,
            steering_output_gain=3.0,
            min_effective_steering_raw=15,
            steering_deadband_norm=0.015,
        )

        model_command = arbiter.build_command(ControlIntent(steer_norm=-0.061, confidence=1.0, prediction_age_ms=0.0))
        self.assertEqual(model_command.steering_raw, -18)

        arbiter = ControlArbiter(
            max_steering_raw=100,
            steering_output_gain=3.0,
            min_effective_steering_raw=15,
            steering_deadband_norm=0.015,
        )
        small_command = arbiter.build_command(ControlIntent(steer_norm=0.02, confidence=1.0, prediction_age_ms=0.0))
        self.assertEqual(small_command.steering_raw, 15)

        arbiter = ControlArbiter(
            max_steering_raw=100,
            steering_output_gain=3.0,
            min_effective_steering_raw=15,
            steering_deadband_norm=0.015,
        )
        deadband_command = arbiter.build_command(ControlIntent(steer_norm=0.01, confidence=1.0, prediction_age_ms=0.0))
        self.assertEqual(deadband_command.steering_raw, 0)

    async def test_physical_steering_minimum_is_rate_limited(self):
        arbiter = ControlArbiter(
            max_steering_raw=100,
            max_steering_rate_raw_per_sec=1.0,
            steering_output_gain=3.0,
            min_effective_steering_raw=64,
            steering_deadband_norm=0.015,
        )
        arbiter.build_command(ControlIntent(steer_norm=0.0, confidence=1.0, prediction_age_ms=0.0))

        command = arbiter.build_command(ControlIntent(steer_norm=-0.059, confidence=1.0, prediction_age_ms=0.0))

        self.assertEqual(command.steering_raw, -1)
        self.assertTrue(arbiter.last_steering_rate_limited)
        self.assertEqual(arbiter.last_target_steering_raw, -64)

    async def test_safety_config_loads_steering_rate_limit(self):
        from vehicle_control.safety_config import RealVehicleSafetyConfig

        cfg = RealVehicleSafetyConfig.from_dict(
            {
                "max_steering_raw": 84,
                "max_steering_rate_raw_per_sec": 180.0,
                "steering_output_gain": 3.0,
            }
        )
        arbiter = ControlArbiter.from_safety_config(cfg)

        self.assertEqual(arbiter.max_steering_raw, 84)
        self.assertAlmostEqual(arbiter.max_steering_rate_raw_per_sec, 180.0)

    async def test_enabled_real_vehicle_profile_uses_full_steering_authority(self):
        import json
        from pathlib import Path
        from vehicle_control.safety_config import RealVehicleSafetyConfig

        cfg_path = Path(__file__).resolve().parents[1] / "config" / "real_vehicle_safety.json"
        cfg = RealVehicleSafetyConfig.from_dict(json.loads(cfg_path.read_text(encoding="utf-8")))
        arbiter = ControlArbiter.from_safety_config(cfg)

        self.assertEqual(arbiter.max_steering_raw, 100)
        self.assertAlmostEqual(arbiter.max_steering_rate_raw_per_sec, 300.0)
        self.assertEqual(arbiter.min_effective_steering_raw, 72)

    async def test_navigation_uses_waypoint_speed_without_mission_cap(self):
        mission = Mission(
            mission_id="speed_unlimited",
            name="speed_unlimited",
            speed_cap_kmh=1.0,
            waypoints=(
                Waypoint(0.0, 0.0, 0.0, 20.0, "straight"),
                Waypoint(10.0, 0.0, 0.0, 20.0, "straight"),
            ),
        )
        navigator = NavigatorService(lookahead_m=1.0)

        goal = navigator.update(mission, Pose2D(0.0, 0.0, 0.0, valid=True), speed_kmh=0.0)

        self.assertEqual(goal.desired_speed_kmh, 20.0)

    async def test_manual_takeover_clears_model_intent_buffer(self):
        mock = MockVehicleAdapter()
        service = VehicleControlService(VehicleAdapterFactory(real_adapter=None, mock_adapter=mock), ControlArbiter())
        service._last_external_intent = ControlIntent(steer_norm=0.5)
        service._last_intent = ControlIntent(steer_norm=0.5)

        service.set_manual_mode_enabled(True)

        self.assertIsNone(service._last_external_intent)
        self.assertIsNone(service._last_intent)

    async def test_external_model_intent_is_ignored_outside_ai_authority(self):
        mock = MockVehicleAdapter()
        service = VehicleControlService(VehicleAdapterFactory(real_adapter=None, mock_adapter=mock), ControlArbiter())
        service.state_machine.state = DriveState.MANUAL_ACTIVE
        service.ai_authority_confirmed = False

        await service.submit_external_agent_intent(ControlIntent(steer_norm=0.5, confidence=1.0, prediction_age_ms=0.0))

        self.assertIsNone(service._last_external_intent)

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
