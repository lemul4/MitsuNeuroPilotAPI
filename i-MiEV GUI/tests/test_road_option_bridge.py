import unittest

from vehicle_control.models import NavCommand, Pose2D, Waypoint, Mission
from vehicle_control.navigation import ABRouteRequest, CoordinateRoutePlanner, NavigatorService
from vehicle_control.road_option import RoadOption, nav_command_to_road_option, road_option_to_one_hot, goal_command_payload
from vehicle_control.ai_bridge import RealAgentBridge


class RoadOptionBridgeTests(unittest.TestCase):
    def test_nav_command_maps_to_carla_road_option_values(self):
        self.assertEqual(nav_command_to_road_option("turn_left"), RoadOption.LEFT)
        self.assertEqual(nav_command_to_road_option("turn_right"), RoadOption.RIGHT)
        self.assertEqual(nav_command_to_road_option("straight"), RoadOption.STRAIGHT)
        self.assertEqual(nav_command_to_road_option("lane_follow"), RoadOption.LANEFOLLOW)
        self.assertEqual(nav_command_to_road_option("change_lane_left"), RoadOption.CHANGELANELEFT)
        self.assertEqual(nav_command_to_road_option("change_lane_right"), RoadOption.CHANGELANERIGHT)

    def test_one_hot_matches_carla_dataset_order(self):
        self.assertEqual(road_option_to_one_hot(RoadOption.LEFT), (1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual(road_option_to_one_hot(RoadOption.RIGHT), (0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual(road_option_to_one_hot(RoadOption.STRAIGHT), (0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        self.assertEqual(road_option_to_one_hot(RoadOption.LANEFOLLOW), (0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        self.assertEqual(road_option_to_one_hot(RoadOption.CHANGELANELEFT), (0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        self.assertEqual(road_option_to_one_hot(RoadOption.CHANGELANERIGHT), (0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
        self.assertEqual(road_option_to_one_hot(RoadOption.VOID), road_option_to_one_hot(RoadOption.LANEFOLLOW))

    def test_navigator_goal_contains_current_and_next_road_option(self):
        mission = Mission(
            mission_id="cmd",
            name="cmd",
            speed_cap_kmh=2.0,
            waypoints=(
                Waypoint(0, 0, 0, 1.0, "start", "start"),
                Waypoint(4, 0, 0, 1.0, "lane_follow", "straight"),
                Waypoint(6, 2, 45, 1.0, "turn_left", "turn_left"),
                Waypoint(7, 4, 90, 0.0, "stop", "stop"),
            ),
        )
        goal = NavigatorService(lookahead_m=3.0, target_waypoint_lookahead=0).update(mission, Pose2D(0, 0, 0, valid=True, source="test"), 0)
        self.assertEqual(goal.road_option, int(RoadOption.STRAIGHT))
        self.assertEqual(goal.road_option_name, "STRAIGHT")
        self.assertEqual(goal.next_road_option, int(RoadOption.LEFT))

    def test_agent_bridge_builds_model_payload(self):
        mission = CoordinateRoutePlanner().build_from_ab(
            ABRouteRequest(mission_id="test", name="A-B", start_x_m=0, start_y_m=0, goal_x_m=10, goal_y_m=0, spacing_m=2)
        )
        goal = NavigatorService(lookahead_m=3.0, target_waypoint_lookahead=0).update(mission, Pose2D(0, 0, 0, valid=True, source="test"), 0)
        payload = RealAgentBridge.build_model_command_payload(goal)
        self.assertEqual(payload["road_option"], int(RoadOption.STRAIGHT))
        self.assertEqual(len(payload["command_one_hot"]), 6)
        self.assertAlmostEqual(sum(payload["command_one_hot"]), 1.0)
        self.assertEqual(payload, goal_command_payload(goal))


if __name__ == "__main__":
    unittest.main()
