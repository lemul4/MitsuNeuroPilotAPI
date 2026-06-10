import time
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()

