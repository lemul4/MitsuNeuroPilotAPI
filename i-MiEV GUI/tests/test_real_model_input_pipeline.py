import time
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import main
from real_agent_adapters.lead_real_model_0011_adapter import LeadModel0011RealPortAdapter
from hardware.gstreamer_udp_camera import GStreamerUdpH265ReceiverThread
from vehicle_control.adapters import MockVehicleAdapter, VehicleAdapterFactory
from vehicle_control.arbiter import ControlArbiter
from vehicle_control.control_service import VehicleControlService
from vehicle_control.models import Mission, Pose2D, Waypoint


class _PredictionSink:
    def __init__(self):
        self.calls = []

    def predict(self, frames, context):
        self.calls.append((frames, context))
        return {
            "steer": 0.1,
            "throttle": 0.2,
            "brake": 0.0,
            "confidence": 1.0,
            "timestamp_monotonic": time.monotonic(),
        }


class RealModelInputPipelineTests(unittest.TestCase):
    def _controller_with_service(self):
        mock = MockVehicleAdapter()
        service = VehicleControlService(VehicleAdapterFactory(real_adapter=None, mock_adapter=mock), ControlArbiter())
        service.adapter = mock
        service.navigator.target_waypoint_lookahead = 0
        mission = Mission(
            mission_id="real_model_input",
            name="Real model input",
            speed_cap_kmh=3.0,
            waypoints=(
                Waypoint(0.0, 0.0, 0.0, 3.0, "start", "start"),
                Waypoint(5.0, 0.0, 0.0, 3.0, "lane_follow", "straight"),
                Waypoint(10.0, 0.0, 0.0, 0.0, "stop", "stop"),
            ),
        )
        service.set_mission(mission)
        service.submit_pose(Pose2D(0.0, 0.0, 0.0, True, "dead_reckoning_ab_no_gps"))
        service.update_navigation_preview()

        controller = main.AppController.__new__(main.AppController)
        controller.vehicle_control = service
        controller.real_agent_bridge = main.RealAgentBridge()
        controller.real_pose_mode = "dead_reckoning_ab"
        controller.ai_control_requested = True
        controller.control_active = True
        controller.real_direct_model_frame_parts = {}
        controller.real_direct_model_latest_sample = None
        controller.real_direct_model_sample_seq = 0
        controller.real_direct_model_consumed_sample_seq = 0
        controller.real_direct_model_part_seq = 0
        controller.real_direct_model_sample_generation = 0
        controller.real_direct_model_condition = __import__("threading").Condition()
        controller.real_direct_model_worker_thread = None
        controller.real_direct_model_worker_stop = False
        controller.real_direct_model_auto_worker = False
        controller.real_direct_model_frame_seq = 0
        controller.real_direct_model_adapter = _PredictionSink()
        controller.handle_real_agent_prediction = lambda payload: setattr(controller, "_last_prediction_payload", payload)
        return controller

    def test_gstreamer_receiver_exposes_model_frame_signal(self):
        self.assertTrue(hasattr(GStreamerUdpH265ReceiverThread, "model_frame_received"))

    def test_context_contains_pose_and_ego_waypoints_for_model(self):
        controller = self._controller_with_service()

        context = controller._build_real_agent_context()

        self.assertEqual(context["pose_source"], "dead_reckoning_ab_no_gps")
        self.assertEqual(context["pose_mode"], "dead_reckoning_ab")
        self.assertEqual(context["target_point_previous_ego"], [0.0, 0.0])
        self.assertGreater(context["target_point_ego"][0], 0.0)
        self.assertEqual(context["target_point_ego"][1], 0.0)
        self.assertIn("command_one_hot", context)
        self.assertIn("next_command_one_hot", context)

    def test_context_world_targets_are_previous_current_next_waypoints(self):
        controller = self._controller_with_service()
        mission = Mission(
            mission_id="target_triplet",
            name="Target triplet",
            speed_cap_kmh=3.0,
            waypoints=(
                Waypoint(0.0, 0.0, 0.0, 3.0, "start", "start"),
                Waypoint(2.0, 0.0, 0.0, 3.0, "lane_follow", "straight"),
                Waypoint(4.0, 0.0, 0.0, 3.0, "lane_follow", "straight"),
                Waypoint(6.0, 0.0, 0.0, 0.0, "stop", "stop"),
            ),
        )
        controller.vehicle_control.set_mission(mission)
        controller.vehicle_control.submit_pose(Pose2D(0.0, 0.0, 0.0, True, "dead_reckoning_ab_no_gps"))
        controller.vehicle_control.update_navigation_preview()

        context = controller._build_real_agent_context()

        self.assertEqual(context["target_point_previous_world"], [0.0, 0.0])
        self.assertEqual(context["target_point_world"], [2.0, 0.0])
        self.assertEqual(context["target_point_next_world"], [4.0, 0.0])
        self.assertEqual(context["target_point_previous_ego"], [0.0, 0.0])
        self.assertEqual(context["target_point_ego"], [2.0, 0.0])
        self.assertEqual(context["target_point_next_ego"], [4.0, 0.0])

    def test_context_default_route_target_triplet_uses_current_route_points(self):
        controller = self._controller_with_service()
        controller.vehicle_control.navigator.target_waypoint_lookahead = 8
        mission = Mission(
            mission_id="target_triplet_eighth",
            name="Target triplet eighth",
            speed_cap_kmh=3.0,
            waypoints=tuple(
                Waypoint(float(idx), 0.0, 0.0, 3.0, "lane_follow", "straight")
                for idx in range(12)
            ),
        )
        controller.vehicle_control.set_mission(mission)
        controller.vehicle_control.submit_pose(Pose2D(0.0, 0.0, 0.0, True, "dead_reckoning_ab_no_gps"))
        controller.vehicle_control.update_navigation_preview()

        context = controller._build_real_agent_context()

        self.assertEqual(context["target_point_previous_waypoint_index"], 0)
        self.assertEqual(context["target_point_waypoint_index"], 1)
        self.assertEqual(context["target_point_next_waypoint_index"], 2)
        self.assertEqual(context["target_point_previous_world"], [0.0, 0.0])
        self.assertEqual(context["target_point_world"], [1.0, 0.0])
        self.assertEqual(context["target_point_next_world"], [2.0, 0.0])

    def test_nmea_route_heading_keeps_straight_route_in_front_of_model(self):
        controller = self._controller_with_service()
        mission = Mission(
            mission_id="nmea_straight_route",
            name="NMEA straight route",
            speed_cap_kmh=3.0,
            waypoints=(
                Waypoint(0.0, 0.0, 90.0, 3.0, "start", "start"),
                Waypoint(0.0, 5.0, 90.0, 3.0, "lane_follow", "straight"),
                Waypoint(0.0, 10.0, 90.0, 0.0, "stop", "stop"),
            ),
        )
        controller.vehicle_control.set_mission(mission)
        controller.vehicle_control.submit_pose(Pose2D(0.0, 0.0, 0.0, True, "nmea0183_gnss"))
        controller.vehicle_control.update_navigation_preview()

        context = controller._build_real_agent_context()

        self.assertGreater(context["target_point_ego"][0], 0.0)
        self.assertAlmostEqual(context["target_point_ego"][1], 0.0, places=6)
        self.assertAlmostEqual(context["model_ego_yaw_deg"], 90.0, places=6)

    def test_direct_model_preview_without_active_control_does_not_predict(self):
        controller = self._controller_with_service()
        controller.control_active = False
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        controller.handle_real_direct_model_frame("wide_90", frame)
        controller.handle_real_direct_model_frame("narrow_50", frame)

        self.assertIsNotNone(controller.real_direct_model_latest_sample)
        self.assertFalse(controller._consume_real_direct_model_sample_once_for_test())
        self.assertEqual(len(controller.real_direct_model_adapter.calls), 0)

    def test_direct_model_prediction_receives_two_camera_frames_and_context(self):
        controller = self._controller_with_service()
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        controller.handle_real_direct_model_frame("wide_90", frame)
        self.assertEqual(len(controller.real_direct_model_adapter.calls), 0)

        controller.handle_real_direct_model_frame("narrow_50", frame)
        controller._consume_real_direct_model_sample_once_for_test()

        self.assertEqual(len(controller.real_direct_model_adapter.calls), 1)
        frames, context = controller.real_direct_model_adapter.calls[0]
        self.assertIs(frames["wide_90"], frame)
        self.assertIs(frames["narrow_50"], frame)
        self.assertEqual(context["pose_source"], "dead_reckoning_ab_no_gps")
        self.assertTrue(hasattr(controller, "_last_prediction_payload"))
        self.assertEqual(controller._last_prediction_payload["input_sample_seq"], 1)
        self.assertIn("inference_started_at_monotonic", controller._last_prediction_payload)
        self.assertIn("inference_finished_at_monotonic", controller._last_prediction_payload)
        self.assertIn("inference_duration_ms", controller._last_prediction_payload)

    def test_prediction_to_control_latency_is_logged(self):
        controller = self._controller_with_service()
        controller.runtime_mode = "carla"
        controller._last_real_model_latency_log_at = 0.0
        payload = {
            "steer": 0.1,
            "throttle": 0.0,
            "brake": 0.0,
            "confidence": 1.0,
            "timestamp_monotonic": time.monotonic(),
            "frame_id": 42,
            "input_sample_seq": 7,
            "input_sample_created_at_monotonic": time.monotonic() - 0.2,
            "inference_started_at_monotonic": time.monotonic() - 0.1,
            "inference_finished_at_monotonic": time.monotonic() - 0.02,
            "inference_duration_ms": 80.0,
        }
        controller._schedule_latest_real_agent_intent = lambda intent: setattr(controller, "_last_intent", intent)

        with patch("main.print") as mocked_print:
            main.AppController.handle_real_agent_prediction(controller, payload)

        printed = " ".join(str(call.args[0]) for call in mocked_print.call_args_list if call.args)
        self.assertIn("REAL MODEL LATENCY", printed)
        self.assertTrue(hasattr(controller, "_last_intent"))

    def test_direct_model_prediction_does_not_repeat_without_new_sample(self):
        controller = self._controller_with_service()
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        controller.handle_real_direct_model_frame("wide_90", frame, time.monotonic() - 2.0)
        controller.handle_real_direct_model_frame("narrow_50", frame, time.monotonic() - 2.0)

        self.assertTrue(controller._consume_real_direct_model_sample_once_for_test())
        self.assertFalse(controller._consume_real_direct_model_sample_once_for_test())
        self.assertEqual(len(controller.real_direct_model_adapter.calls), 1)

    def test_direct_model_latest_sample_overwrites_before_consumption(self):
        controller = self._controller_with_service()
        wide_old = np.zeros((32, 32, 3), dtype=np.uint8)
        narrow_old = np.zeros((32, 32, 3), dtype=np.uint8)
        wide_new = np.ones((32, 32, 3), dtype=np.uint8)
        narrow_new = np.full((32, 32, 3), 2, dtype=np.uint8)

        controller.handle_real_direct_model_frame("wide_90", wide_old, time.monotonic())
        controller.handle_real_direct_model_frame("narrow_50", narrow_old, time.monotonic())
        controller.handle_real_direct_model_frame("wide_90", wide_new, time.monotonic())
        controller.handle_real_direct_model_frame("narrow_50", narrow_new, time.monotonic())
        controller._consume_real_direct_model_sample_once_for_test()

        self.assertEqual(len(controller.real_direct_model_adapter.calls), 1)
        frames, _context = controller.real_direct_model_adapter.calls[0]
        self.assertIs(frames["wide_90"], wide_new)
        self.assertIs(frames["narrow_50"], narrow_new)
        self.assertEqual(controller._last_prediction_payload["input_sample_seq"], 3)

    def test_direct_model_sample_reset_forces_wait_for_new_camera_pair(self):
        controller = self._controller_with_service()
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        controller.handle_real_direct_model_frame("wide_90", frame, time.monotonic())
        controller._reset_real_direct_model_sample()
        controller.handle_real_direct_model_frame("narrow_50", frame, time.monotonic())

        self.assertFalse(controller._consume_real_direct_model_sample_once_for_test())

    def test_direct_model_drops_prediction_from_reset_generation(self):
        controller = self._controller_with_service()
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        controller.handle_real_direct_model_frame("wide_90", frame, time.monotonic())
        sample = controller._publish_real_direct_model_sample("narrow_50", frame, time.monotonic())
        controller._reset_real_direct_model_sample()

        self.assertIsNone(controller._predict_real_direct_model_sample(sample))
        self.assertFalse(hasattr(controller, "_last_prediction_payload"))

    def test_visualization_metadata_uses_sample_creation_time(self):
        image = np.zeros((80, 320, 3), dtype=np.uint8)
        label = LeadModel0011RealPortAdapter._format_sample_time(123.456789)
        annotated = LeadModel0011RealPortAdapter._annotate_visualization(image, label)

        self.assertEqual(label, "sample monotonic: 123.456789s")
        self.assertGreater(int(annotated.sum()), 0)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "000001.json"
            LeadModel0011RealPortAdapter._write_json(
                str(path),
                {"input_sample_created_at_monotonic": 123.456789},
            )
            self.assertIn("123.456789", path.read_text(encoding="utf-8"))

    def test_real_steering_uses_second_model_waypoint_by_default(self):
        adapter = LeadModel0011RealPortAdapter.__new__(LeadModel0011RealPortAdapter)
        pred = SimpleNamespace(
            pred_future_waypoints=np.array(
                [
                    [1.0, 0.1],
                    [2.0, 0.2],
                    [3.0, 3.0],
                ],
                dtype=np.float32,
            )
        )

        with patch.dict("os.environ", {}, clear=False):
            _steer, metadata = adapter._real_steer_from_model_waypoints(pred)

        self.assertEqual(metadata["real_steer_waypoint_index"], 2)
        self.assertAlmostEqual(metadata["real_steer_aim_x_m"], 2.0)
        self.assertAlmostEqual(metadata["real_steer_aim_y_m"], 0.2)


if __name__ == "__main__":
    unittest.main()

