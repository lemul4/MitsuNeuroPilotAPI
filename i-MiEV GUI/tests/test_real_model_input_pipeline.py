import time
import unittest

import numpy as np

import main
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
        controller.real_direct_model_frames = {}
        controller.real_direct_model_last_predict_at = 0.0
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

    def test_direct_model_prediction_receives_two_camera_frames_and_context(self):
        controller = self._controller_with_service()
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        controller.handle_real_direct_model_frame("wide_90", frame)
        self.assertEqual(len(controller.real_direct_model_adapter.calls), 0)

        controller.handle_real_direct_model_frame("narrow_50", frame)

        self.assertEqual(len(controller.real_direct_model_adapter.calls), 1)
        frames, context = controller.real_direct_model_adapter.calls[0]
        self.assertIs(frames["wide_90"], frame)
        self.assertIs(frames["narrow_50"], frame)
        self.assertEqual(context["pose_source"], "dead_reckoning_ab_no_gps")
        self.assertTrue(hasattr(controller, "_last_prediction_payload"))
        self.assertIn("input_frame_age_ms", controller._last_prediction_payload)

    def test_direct_model_prediction_rejects_stale_timestamped_frames(self):
        controller = self._controller_with_service()
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        stale_ts = time.monotonic() - 2.0

        controller.handle_real_direct_model_frame("wide_90", frame, stale_ts)
        controller.handle_real_direct_model_frame("narrow_50", frame, stale_ts)

        self.assertEqual(len(controller.real_direct_model_adapter.calls), 0)
        self.assertFalse(hasattr(controller, "_last_prediction_payload"))

    def test_direct_model_cache_reset_forces_wait_for_new_camera_pair(self):
        controller = self._controller_with_service()
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        controller.handle_real_direct_model_frame("wide_90", frame, time.monotonic())
        controller._reset_real_direct_model_frame_cache()
        controller.handle_real_direct_model_frame("narrow_50", frame, time.monotonic())

        self.assertEqual(len(controller.real_direct_model_adapter.calls), 0)


if __name__ == "__main__":
    unittest.main()

