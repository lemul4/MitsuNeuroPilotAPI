import unittest

from vehicle_control.adapters import MockVehicleAdapter, VehicleAdapterFactory
from vehicle_control.arbiter import ControlArbiter
from vehicle_control.control_service import VehicleControlService
from vehicle_control.models import Mission, Pose2D
from vehicle_control.navigation import ABRouteRequest, CoordinateRoutePlanner
from vehicle_control.pose_modes import (
    DeadReckoningPoseState,
    advance_dead_reckoning_pose,
    mission_targets_from_dict,
    parse_xyyaw,
)


class DeadReckoningPoseModeTests(unittest.TestCase):
    def test_pose_advances_towards_b_by_speed_and_time(self):
        state = DeadReckoningPoseState(0.0, 0.0, 0.0)

        updated = advance_dead_reckoning_pose(
            state,
            targets=((10.0, 0.0),),
            speed_mps=2.0,
            dt_sec=1.5,
        )

        self.assertAlmostEqual(updated.x_m, 3.0, places=3)
        self.assertAlmostEqual(updated.y_m, 0.0, places=3)
        self.assertAlmostEqual(updated.yaw_deg, 0.0, places=3)
        self.assertFalse(updated.finished)

    def test_pose_follows_hints_then_goal(self):
        mission = {
            "start": {"x_m": 0.0, "y_m": 0.0, "yaw_deg": 0.0},
            "hints": [{"x_m": 2.0, "y_m": 0.0, "command": "straight"}],
            "goal": {"x_m": 2.0, "y_m": 2.0, "yaw_deg": 90.0},
        }
        seed = parse_xyyaw(mission["start"])
        state = DeadReckoningPoseState(seed.x_m, seed.y_m, seed.yaw_deg)

        updated = advance_dead_reckoning_pose(
            state,
            targets=mission_targets_from_dict(mission),
            speed_mps=2.0,
            dt_sec=2.0,
            arrive_radius_m=0.01,
        )

        self.assertAlmostEqual(updated.x_m, 2.0, places=3)
        self.assertAlmostEqual(updated.y_m, 2.0, places=3)
        self.assertAlmostEqual(updated.yaw_deg, 90.0, places=3)
        self.assertTrue(updated.finished)


class NavigationPreviewPoseModeTests(unittest.TestCase):
    def test_navigation_preview_goal_is_available_before_ai_activation(self):
        mock = MockVehicleAdapter()
        service = VehicleControlService(VehicleAdapterFactory(real_adapter=None, mock_adapter=mock), ControlArbiter())
        service.adapter = mock
        mission = CoordinateRoutePlanner().build_from_ab(
            ABRouteRequest(
                mission_id="ab_no_gps",
                name="A to B no GPS",
                start_x_m=0.0,
                start_y_m=0.0,
                goal_x_m=10.0,
                goal_y_m=0.0,
                speed_cap_kmh=3.0,
                spacing_m=2.0,
            )
        )
        service.set_mission(mission)
        service.submit_pose(Pose2D(0.0, 0.0, 0.0, True, "dead_reckoning_ab_no_gps"))

        goal = service.update_navigation_preview()

        self.assertIsNotNone(goal)
        self.assertTrue(goal.is_valid())
        self.assertEqual(service.get_current_goal(), goal)
        self.assertEqual(goal.mission_id, "ab_no_gps")
        self.assertGreater(goal.distance_to_goal_m, 0.0)
        self.assertEqual(service.state_machine.state.value, "DISCONNECTED")


if __name__ == "__main__":
    unittest.main()
