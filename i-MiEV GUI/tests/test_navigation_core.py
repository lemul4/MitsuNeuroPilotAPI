import unittest

from vehicle_control.navigation import ABRouteRequest, RouteHint, CoordinateRoutePlanner, NavigatorService
from vehicle_control.models import Mission, Pose2D, NavCommand, VehicleTelemetry, Waypoint
from vehicle_control.pid import WaypointPIDController


class NavigationCoreTests(unittest.TestCase):
    def test_ab_route_densifies_and_marks_turns(self):
        req = ABRouteRequest(
            mission_id="test",
            name="A-B",
            start_x_m=0,
            start_y_m=0,
            goal_x_m=10,
            goal_y_m=5,
            speed_cap_kmh=3,
            spacing_m=2,
            hints=(RouteHint(6, 0, "turn_left", 1.2),),
        )
        mission = CoordinateRoutePlanner().build_from_ab(req)
        self.assertGreaterEqual(len(mission.waypoints), 6)
        commands = {wp.command for wp in mission.waypoints}
        self.assertIn(NavCommand.TURN_LEFT.value, commands)
        self.assertEqual(mission.waypoints[-1].command, NavCommand.STOP.value)

    def test_navigator_outputs_local_goal(self):
        mission = CoordinateRoutePlanner().build_from_ab(ABRouteRequest(
            mission_id="test", name="A-B", start_x_m=0, start_y_m=0, goal_x_m=10, goal_y_m=0, spacing_m=2
        ))
        navigator = NavigatorService(lookahead_m=3.0, target_waypoint_lookahead=0)
        goal = navigator.update(mission, Pose2D(0, 0, 0, valid=True, source="test"), speed_kmh=0)
        self.assertTrue(goal.is_valid())
        self.assertGreater(goal.target_x_m, 0)
        self.assertLess(abs(goal.heading_error_deg), 5)
        self.assertGreater(goal.desired_speed_kmh, 0.0)

    def test_navigation_goal_uses_previous_target_next_waypoint_triplet(self):
        mission = Mission(
            mission_id="triplet",
            name="triplet",
            speed_cap_kmh=3.0,
            waypoints=(
                Waypoint(0.0, 0.0, 0.0, 3.0, "start"),
                Waypoint(2.0, 0.0, 0.0, 3.0, "straight"),
                Waypoint(4.0, 0.0, 0.0, 3.0, "straight"),
                Waypoint(6.0, 0.0, 0.0, 0.0, "stop"),
            ),
        )
        goal = NavigatorService(lookahead_m=3.0, target_waypoint_lookahead=0).update(
            mission,
            Pose2D(0.0, 0.0, 0.0, valid=True, source="test"),
            speed_kmh=0.0,
        )

        self.assertEqual(goal.previous_waypoint_index, 0)
        self.assertEqual(goal.waypoint_index, 1)
        self.assertEqual(goal.next_waypoint_index, 2)
        self.assertEqual(goal.control_waypoint_index, 2)
        self.assertEqual((goal.previous_x_m, goal.previous_y_m), (0.0, 0.0))
        self.assertEqual((goal.target_x_m, goal.target_y_m), (2.0, 0.0))
        self.assertEqual((goal.next_x_m, goal.next_y_m), (4.0, 0.0))

    def test_default_navigation_uses_current_triplet_and_eighth_control_waypoint(self):
        waypoints = tuple(Waypoint(float(idx), 0.0, 0.0, 3.0, "straight") for idx in range(12))
        mission = Mission(
            mission_id="eighth",
            name="eighth",
            speed_cap_kmh=3.0,
            waypoints=waypoints,
        )

        goal = NavigatorService().update(
            mission,
            Pose2D(0.0, 0.0, 0.0, valid=True, source="test"),
            speed_kmh=0.0,
        )

        self.assertEqual(goal.previous_waypoint_index, 0)
        self.assertEqual(goal.waypoint_index, 1)
        self.assertEqual(goal.next_waypoint_index, 2)
        self.assertEqual(goal.control_waypoint_index, 8)
        self.assertEqual((goal.previous_x_m, goal.previous_y_m), (0.0, 0.0))
        self.assertEqual((goal.target_x_m, goal.target_y_m), (1.0, 0.0))
        self.assertEqual((goal.next_x_m, goal.next_y_m), (2.0, 0.0))

    def test_heading_error_uses_current_target_not_distant_control_waypoint(self):
        waypoints = tuple(
            [Waypoint(0.0, 0.0, 0.0, 3.0, "start")]
            + [Waypoint(float(idx), 0.0, 0.0, 3.0, "straight") for idx in range(1, 8)]
            + [Waypoint(8.0, 8.0, 45.0, 3.0, "straight")]
        )
        mission = Mission(
            mission_id="heading_target",
            name="heading_target",
            speed_cap_kmh=3.0,
            waypoints=waypoints,
        )

        goal = NavigatorService(target_waypoint_lookahead=8).update(
            mission,
            Pose2D(0.0, 0.0, 0.0, valid=True, source="test"),
            speed_kmh=0.0,
        )

        self.assertEqual(goal.waypoint_index, 1)
        self.assertEqual(goal.control_waypoint_index, 8)
        self.assertAlmostEqual(goal.heading_error_deg, 0.0, places=6)

    def test_navigation_target_triplet_moves_as_conveyor(self):
        waypoints = tuple(Waypoint(float(idx) * 2.0, 0.0, 0.0, 3.0, "straight") for idx in range(8))
        mission = Mission(
            mission_id="conveyor",
            name="conveyor",
            speed_cap_kmh=3.0,
            waypoints=waypoints,
        )
        navigator = NavigatorService(target_waypoint_lookahead=0, waypoint_reached_m=0.6)

        first = navigator.update(mission, Pose2D(0.0, 0.0, 0.0, valid=True, source="test"), 0.0)
        second = navigator.update(mission, Pose2D(2.0, 0.0, 0.0, valid=True, source="test"), 0.0)
        third = navigator.update(mission, Pose2D(4.0, 0.0, 0.0, valid=True, source="test"), 0.0)

        self.assertEqual(
            (first.previous_waypoint_index, first.waypoint_index, first.next_waypoint_index),
            (0, 1, 2),
        )
        self.assertEqual(
            (second.previous_waypoint_index, second.waypoint_index, second.next_waypoint_index),
            (1, 2, 3),
        )
        self.assertEqual(
            (third.previous_waypoint_index, third.waypoint_index, third.next_waypoint_index),
            (2, 3, 4),
        )

    def test_route_target_spacing_is_separate_from_model_waypoint_spacing(self):
        mission = CoordinateRoutePlanner().build_from_ab(ABRouteRequest.from_dict({
            "mission_id": "spacing",
            "name": "spacing",
            "start": {"x_m": 0.0, "y_m": 0.0},
            "goal": {"x_m": 10.0, "y_m": 0.0},
            "spacing_m": 3.0,
            "route_target_spacing_m": 1.0,
            "model_waypoints_spacing_m": 3.0,
        }))

        self.assertAlmostEqual(float(mission.metadata["spacing_m"]), 1.0)
        self.assertGreaterEqual(len(mission.waypoints), 10)

    def test_navigator_advances_by_projection_when_waypoint_radius_is_missed(self):
        mission = CoordinateRoutePlanner().build_from_ab(ABRouteRequest(
            mission_id="test",
            name="A-B",
            start_x_m=0,
            start_y_m=0,
            goal_x_m=12,
            goal_y_m=0,
            spacing_m=2,
        ))
        navigator = NavigatorService(lookahead_m=3.0, target_waypoint_lookahead=0, waypoint_reached_m=0.8)

        goal = navigator.update(
            mission,
            Pose2D(5.0, 1.2, 0.0, valid=True, source="gps_offset"),
            speed_kmh=1.0,
        )

        self.assertTrue(goal.is_valid())
        self.assertGreater(goal.waypoint_index, 0)
        self.assertGreater(goal.target_x_m, 5.0)

    def test_pid_intent_moves_forward_on_straight_goal(self):
        mission = CoordinateRoutePlanner().build_from_ab(ABRouteRequest(
            mission_id="test", name="A-B", start_x_m=0, start_y_m=0, goal_x_m=10, goal_y_m=0, speed_cap_kmh=3
        ))
        goal = NavigatorService(lookahead_m=3.0, target_waypoint_lookahead=0).update(mission, Pose2D(0, 0, 0, valid=True, source="test"), 0)
        intent = WaypointPIDController().build_intent(goal, VehicleTelemetry(speed_kmh=0.0))
        self.assertGreater(intent.throttle_norm, 0.0)
        self.assertLess(abs(intent.steer_norm), 0.4)

    def test_pid_steering_sign_matches_real_vehicle_left_negative(self):
        left_goal = NavigatorService(lookahead_m=3.0, target_waypoint_lookahead=0).update(
            Mission(
                mission_id="left",
                name="left",
                speed_cap_kmh=3.0,
                waypoints=(
                    Waypoint(0.0, 0.0, 0.0, 3.0, "start"),
                    Waypoint(0.0, 5.0, 90.0, 3.0, "straight"),
                ),
            ),
            Pose2D(0.0, 0.0, 0.0, valid=True, source="test"),
            0.0,
        )
        right_goal = NavigatorService(lookahead_m=3.0, target_waypoint_lookahead=0).update(
            Mission(
                mission_id="right",
                name="right",
                speed_cap_kmh=3.0,
                waypoints=(
                    Waypoint(0.0, 0.0, 0.0, 3.0, "start"),
                    Waypoint(0.0, -5.0, -90.0, 3.0, "straight"),
                ),
            ),
            Pose2D(0.0, 0.0, 0.0, valid=True, source="test"),
            0.0,
        )

        controller = WaypointPIDController()
        left_intent = controller.build_intent(left_goal, VehicleTelemetry(speed_kmh=0.0))
        controller.reset()
        right_intent = controller.build_intent(right_goal, VehicleTelemetry(speed_kmh=0.0))

        self.assertLess(left_intent.steer_norm, 0.0)
        self.assertGreater(right_intent.steer_norm, 0.0)


if __name__ == "__main__":
    unittest.main()

class GeoMapConversionTests(unittest.TestCase):
    def test_geo_ab_converts_to_local_meters(self):
        from vehicle_control.geo import GeoPoint, geo_points_to_local_ab
        local = geo_points_to_local_ab(GeoPoint(55.0, 37.0), GeoPoint(55.0001, 37.0002))
        self.assertAlmostEqual(local["start"]["x_m"], 0.0, places=5)
        self.assertAlmostEqual(local["start"]["y_m"], 0.0, places=5)
        self.assertGreater(abs(local["goal"]["x_m"]), 1.0)
        self.assertGreater(abs(local["goal"]["y_m"]), 1.0)

class RoadRouteProviderTests(unittest.TestCase):
    def test_osrm_provider_converts_geometry_to_mission_without_network(self):
        from vehicle_control.geo import GeoPoint
        from vehicle_control.road_routing import OsrmRoadRouteProvider, RoadRouteRequest

        class FakeProvider(OsrmRoadRouteProvider):
            def _fetch_osrm_geojson(self, request):
                return [(55.0, 37.0), (55.00005, 37.00005), (55.0001, 37.0002)]

        request = RoadRouteRequest(
            mission_id="road",
            name="road",
            start=GeoPoint(55.0, 37.0),
            goal=GeoPoint(55.0001, 37.0002),
            speed_cap_kmh=2.0,
            spacing_m=2.0,
        )
        mission = FakeProvider().build_mission(request)
        self.assertGreater(len(mission.waypoints), 2)
        self.assertEqual(mission.metadata.get("routing_provider"), "osrm")
        self.assertEqual(mission.waypoints[-1].command, NavCommand.STOP.value)

    def test_osrm_provider_does_not_infer_turn_commands_from_geometry(self):
        from vehicle_control.geo import GeoPoint
        from vehicle_control.road_routing import OsrmRoadRouteProvider, RoadRouteRequest

        class FakeProvider(OsrmRoadRouteProvider):
            def _fetch_osrm_geojson(self, request):
                return [(55.0, 37.0), (55.00006, 37.0), (55.00008, 36.99994)]

        request = RoadRouteRequest(
            mission_id="road_turn_shape",
            name="road_turn_shape",
            start=GeoPoint(55.0, 37.0),
            goal=GeoPoint(55.00008, 36.99994),
            spacing_m=1.0,
        )
        mission = FakeProvider().build_mission(request)
        commands = {wp.command for wp in mission.waypoints[:-1]}

        self.assertNotIn(NavCommand.TURN_RIGHT.value, commands)
        self.assertNotIn(NavCommand.TURN_LEFT.value, commands)

    def test_osrm_waypoints_follow_road_geometry_not_direct_chord(self):
        from vehicle_control.geo import GeoPoint
        from vehicle_control.road_routing import OsrmRoadRouteProvider, RoadRouteRequest

        class FakeProvider(OsrmRoadRouteProvider):
            def _fetch_osrm_geojson(self, request):
                # Local shape: first north, then east. A direct A->B route would
                # stay near the diagonal; the road route must keep the bend.
                return [(55.0, 37.0), (55.00010, 37.0), (55.00010, 37.00020)]

        request = RoadRouteRequest(
            mission_id="road_bend",
            name="road_bend",
            start=GeoPoint(55.0, 37.0),
            goal=GeoPoint(55.00010, 37.00020),
            spacing_m=1.0,
        )
        mission = FakeProvider().build_mission(request)
        waypoints = list(mission.waypoints)

        self.assertGreater(len(waypoints), 10)
        self.assertTrue(any(abs(wp.x_m) < 0.5 and wp.y_m > 5.0 for wp in waypoints))
        self.assertTrue(any(wp.x_m > 5.0 and wp.y_m > 5.0 for wp in waypoints))

        navigator = NavigatorService(lookahead_m=3.0, target_waypoint_lookahead=0)
        goal = navigator.update(mission, Pose2D(0.0, 5.0, 90.0, valid=True, source="test"), speed_kmh=0.0)

        self.assertTrue(goal.is_valid())
        self.assertAlmostEqual(goal.target_x_m, 0.0, delta=0.75)
        self.assertGreater(goal.target_y_m, 5.0)

class RoadRouteLaneOffsetTests(unittest.TestCase):
    def test_road_route_default_keeps_centerline_for_proving_ground(self):
        from vehicle_control.geo import GeoPoint
        from vehicle_control.road_routing import OsrmRoadRouteProvider, RoadRouteRequest

        class FakeProvider(OsrmRoadRouteProvider):
            def _fetch_osrm_geojson(self, request):
                return [(55.0, 37.0), (55.0001, 37.0)]

        request = RoadRouteRequest(
            mission_id="road_default",
            name="road_default",
            start=GeoPoint(55.0, 37.0),
            goal=GeoPoint(55.0001, 37.0),
        )
        mission = FakeProvider().build_mission(request)

        self.assertEqual(mission.metadata.get("trajectory_geometry"), "centerline")
        self.assertAlmostEqual(float(mission.metadata.get("lane_offset_m")), 0.0, places=3)
        self.assertAlmostEqual(mission.waypoints[0].x_m, 0.0, places=3)

    def test_right_side_offset_moves_control_trajectory_off_centerline(self):
        from vehicle_control.geo import GeoPoint
        from vehicle_control.road_routing import OsrmRoadRouteProvider, RoadRouteRequest

        class FakeProvider(OsrmRoadRouteProvider):
            def _fetch_osrm_geojson(self, request):
                # Northbound route in local coordinates. Right-hand side should
                # offset east (positive x) by approximately lane_offset_m.
                return [(55.0, 37.0), (55.0001, 37.0), (55.0002, 37.0)]

        request = RoadRouteRequest(
            mission_id="road_lane",
            name="road_lane",
            start=GeoPoint(55.0, 37.0),
            goal=GeoPoint(55.0002, 37.0),
            speed_cap_kmh=2.0,
            spacing_m=2.0,
            lane_policy="right_side",
            lane_offset_m=1.7,
            traffic_side="right",
        )
        mission = FakeProvider().build_mission(request)
        self.assertEqual(mission.metadata.get("trajectory_geometry"), "right_side_offset_approximation")
        self.assertGreater(mission.waypoints[0].x_m, 1.0)
        self.assertAlmostEqual(float(mission.metadata.get("lane_offset_m")), 1.7, places=2)

    def test_centerline_policy_keeps_centerline(self):
        from vehicle_control.geo import GeoPoint
        from vehicle_control.road_routing import OsrmRoadRouteProvider, RoadRouteRequest

        class FakeProvider(OsrmRoadRouteProvider):
            def _fetch_osrm_geojson(self, request):
                return [(55.0, 37.0), (55.0001, 37.0)]

        request = RoadRouteRequest(
            mission_id="center",
            name="center",
            start=GeoPoint(55.0, 37.0),
            goal=GeoPoint(55.0001, 37.0),
            lane_policy="centerline",
            lane_offset_m=1.7,
        )
        mission = FakeProvider().build_mission(request)
        self.assertEqual(mission.metadata.get("trajectory_geometry"), "centerline")
        self.assertAlmostEqual(mission.waypoints[0].x_m, 0.0, places=3)
