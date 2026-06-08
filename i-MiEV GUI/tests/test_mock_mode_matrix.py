import asyncio
import time
import unittest

from vehicle_control.adapters import MockVehicleAdapter, VehicleAdapterFactory
from vehicle_control.arbiter import ControlArbiter
from vehicle_control.control_service import VehicleControlService
from vehicle_control.models import ControlIntent, DriveState, Gear, Mission


async def _wait_until(predicate, timeout=1.0, interval=0.01):
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return bool(predicate())


class _SpyWaypointController:
    def __init__(self):
        self.calls = 0
        self.last_goal = None

    def reset(self):
        self.calls = 0
        self.last_goal = None

    def build_intent(self, goal, telemetry):
        self.calls += 1
        self.last_goal = goal
        return ControlIntent(
            steer_norm=0.12,
            throttle_norm=0.20,
            brake_norm=0.0,
            confidence=1.0,
            desired_speed_kmh=float(goal.desired_speed_kmh),
            speed_cap_kmh=float(goal.speed_cap_kmh or 3.0),
            nav_maneuver=str(goal.maneuver),
            valid_for_ms=120,
        )


def _make_service(mock=None, waypoint_controller=None):
    mock = mock or MockVehicleAdapter()
    mock.gear_shift_delay_sec = 0.0
    factory = VehicleAdapterFactory(
        real_adapter=None,
        mock_adapter=mock,
        loopback_adapter=MockVehicleAdapter(label="TEST_SERIAL_LOOPBACK"),
    )
    service = VehicleControlService(
        factory,
        ControlArbiter(),
        waypoint_controller=waypoint_controller,
    )
    service.arming_delay_sec = 0.0
    service.disengage_timeout_sec = 1.0
    service.autonomy_tick_hz = 40.0
    service.nav_log_interval_sec = 999.0
    return service, mock


class MockModeReadinessMatrixTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_test_modes_seed_pose_and_do_not_require_physical_cameras(self):
        for device_text in ("TEST_MOCK_VEHICLE", "TEST_REPLAY_LOG", "TEST_SERIAL_LOOPBACK"):
            with self.subTest(device_text=device_text):
                service, _mock = _make_service()
                try:
                    await service.connect_device(device_text)
                    service.set_mission(Mission.default_test_mission())

                    readiness = service._build_readiness()
                    self.assertTrue(readiness.connected)
                    self.assertTrue(readiness.mission_ok)
                    self.assertTrue(readiness.pose_ok)
                    self.assertTrue(readiness.cameras_ok)
                    self.assertTrue(readiness.vehicle_ok)
                    self.assertFalse(readiness.ai_preview_ok)

                    service.set_ai_preview_enabled(True)
                    readiness = service._build_readiness()
                    self.assertTrue(readiness.all_ok)
                    self.assertEqual(service.state_machine.state, DriveState.READY_TO_ARM)
                    self.assertEqual(service.get_telemetry().pose_source, "mission_start_mock")
                finally:
                    await service.disconnect()

    async def test_ai_preview_without_activate_does_not_send_vehicle_command(self):
        service, mock = _make_service()
        try:
            await service.connect_device("TEST_MOCK_VEHICLE")
            service.set_mission(Mission.default_test_mission())

            service.set_ai_preview_enabled(True)

            self.assertEqual(service.state_machine.state, DriveState.READY_TO_ARM)
            self.assertIsNone(mock.last_command)
            self.assertTrue(service.external_agent_enabled)
        finally:
            await service.disconnect()

    async def test_disabling_ai_preview_clears_external_agent_influence(self):
        service, _mock = _make_service()
        try:
            await service.connect_device("TEST_MOCK_VEHICLE")
            service.set_mission(Mission.default_test_mission())
            service.set_ai_preview_enabled(True)
            service._last_external_intent = ControlIntent(throttle_norm=1.0, confidence=1.0)

            service.set_ai_preview_enabled(False)

            self.assertFalse(service.ai_preview_enabled)
            self.assertFalse(service.external_agent_enabled)
            self.assertIsNone(service._last_external_intent)
            self.assertEqual(service.state_machine.state, DriveState.CONNECTED_MANUAL)
        finally:
            await service.disconnect()

    async def test_activation_is_blocked_by_missing_readiness_conditions(self):
        service, _mock = _make_service()
        try:
            await service.connect_device("TEST_MOCK_VEHICLE")

            self.assertFalse(await service.activate_control())
            self.assertIn("No valid mission", service._build_readiness().blocked_reasons())

            service.set_mission(Mission.default_test_mission())
            self.assertFalse(await service.activate_control())
            self.assertIn("AI Preview is not enabled", service._build_readiness().blocked_reasons())

            service.set_ai_preview_enabled(True)
            service.set_pose_status(False)
            self.assertFalse(await service.activate_control())
            self.assertIn("Pose is not valid", service._build_readiness().blocked_reasons())
        finally:
            await service.disconnect()

    async def test_activation_fails_when_drive_feedback_is_not_confirmed(self):
        mock = MockVehicleAdapter()
        mock.gear_shift_delay_sec = 0.0
        mock.inject_gear_timeout = True
        service, _mock = _make_service(mock=mock)
        try:
            await service.connect_device("TEST_MOCK_VEHICLE")
            service.set_mission(Mission.default_test_mission())
            service.set_ai_preview_enabled(True)

            self.assertFalse(await service.activate_control())
            self.assertEqual(service.state_machine.state, DriveState.FAULT)
            self.assertIn("Gear D not confirmed", service.state_machine.last_fault)
        finally:
            await service.disconnect()


class MockModeControlMatrixTests(unittest.IsolatedAsyncioTestCase):
    async def test_stale_external_intent_falls_back_to_route_pid_in_ai_loop(self):
        spy = _SpyWaypointController()
        service, mock = _make_service(waypoint_controller=spy)
        try:
            await service.connect_device("TEST_MOCK_VEHICLE")
            service.set_mission(Mission.default_test_mission())
            service.set_ai_preview_enabled(True)
            service.set_external_agent_enabled(True)
            service._last_external_intent = ControlIntent(
                timestamp_monotonic=time.monotonic() - 1.0,
                throttle_norm=1.0,
                confidence=1.0,
                prediction_age_ms=1000.0,
                valid_for_ms=10,
            )

            self.assertTrue(await service.activate_control())
            self.assertTrue(await _wait_until(lambda: spy.calls > 0))
            self.assertIsNotNone(spy.last_goal)
            self.assertIsNotNone(mock.last_command)
            self.assertEqual(mock.last_command.reason, "ai_intent")
            self.assertGreater(mock.last_command.accel_pct, 0)
            self.assertLess(mock.last_command.accel_pct, 25)
        finally:
            await service.disconnect()

    async def test_mock_heartbeat_loss_in_ai_loop_sends_safe_stop(self):
        service, mock = _make_service()
        try:
            await service.connect_device("TEST_MOCK_VEHICLE")
            service.set_mission(Mission.default_test_mission())
            service.set_ai_preview_enabled(True)
            self.assertTrue(await service.activate_control())

            mock.inject_heartbeat_loss = True
            self.assertTrue(
                await _wait_until(
                    lambda: mock.last_command is not None
                    and mock.last_command.reason == "telemetry_lost"
                    and mock.last_command.brake_pct >= 35,
                    timeout=1.0,
                )
            )
            self.assertFalse(service.get_telemetry().heartbeat_ok)
        finally:
            mock.inject_heartbeat_loss = False
            await service.disconnect()

    async def test_manual_commands_are_blocked_under_ai_and_limited_by_gear_and_speed(self):
        service, mock = _make_service()
        service.autonomy_loop_enabled = False
        try:
            await service.connect_device("TEST_MOCK_VEHICLE")
            service.set_mission(Mission.default_test_mission())
            service.set_ai_preview_enabled(True)
            self.assertTrue(await service.activate_control())
            self.assertEqual(service.state_machine.state, DriveState.AI_ACTIVE)

            last_before = mock.last_command
            await service.submit_manual_command(angle_deg=0, accel_pct=60, brake_pct=0)
            self.assertIs(mock.last_command, last_before)

            self.assertTrue(await service.transfer_ai_to_manual(reason="unit_test_takeover"))
            self.assertEqual(service.state_machine.state, DriveState.MANUAL_ACTIVE)
            await service.submit_manual_command(angle_deg=0, accel_pct=20, brake_pct=0)
            self.assertEqual(mock.last_command.reason, "manual")
            self.assertGreater(mock.last_command.accel_pct, 0)

            mock.telemetry.gear = Gear.P
            mock.telemetry.speed_kmh = 0.0
            await service.submit_manual_command(angle_deg=0, accel_pct=80, brake_pct=0)
            self.assertEqual(mock.last_command.reason, "manual")
            self.assertGreater(mock.last_command.accel_pct, 0)
            self.assertLessEqual(mock.last_command.accel_pct, 80)

            mock.telemetry.gear = Gear.D
            mock.telemetry.speed_kmh = 2.0
            await service.request_manual_gear(Gear.P.value)
            self.assertEqual(mock.last_command.reason, "manual_park_requested")
            self.assertEqual(mock.telemetry.gear, Gear.D)
        finally:
            await service.disconnect()


if __name__ == "__main__":
    unittest.main()
