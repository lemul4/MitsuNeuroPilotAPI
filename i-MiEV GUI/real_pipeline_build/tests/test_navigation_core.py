import unittest

from vehicle_control.navigation import ABRouteRequest, RouteHint, CoordinateRoutePlanner, NavigatorService
from vehicle_control.models import Pose2D, NavCommand, VehicleTelemetry
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
        navigator = NavigatorService(lookahead_m=3.0)
        goal = navigator.update(mission, Pose2D(0, 0, 0, valid=True, source="test"), speed_kmh=0)
        self.assertTrue(goal.is_valid())
        self.assertGreater(goal.target_x_m, 0)
        self.assertLess(abs(goal.heading_error_deg), 5)
        self.assertGreater(goal.desired_speed_kmh, 0.0)

    def test_pid_intent_moves_forward_on_straight_goal(self):
        mission = CoordinateRoutePlanner().build_from_ab(ABRouteRequest(
            mission_id="test", name="A-B", start_x_m=0, start_y_m=0, goal_x_m=10, goal_y_m=0, speed_cap_kmh=3
        ))
        goal = NavigatorService(lookahead_m=3.0).update(mission, Pose2D(0, 0, 0, valid=True, source="test"), 0)
        intent = WaypointPIDController().build_intent(goal, VehicleTelemetry(speed_kmh=0.0))
        self.assertGreater(intent.throttle_norm, 0.0)
        self.assertLess(abs(intent.steer_norm), 0.4)


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
