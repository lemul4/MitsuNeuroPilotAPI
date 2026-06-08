import os
import unittest
from pathlib import Path

from hardware.gstreamer_udp_camera import UdpH265CameraSpec, build_udp_h265_pipeline


class GStreamerUdpCameraPipelineTests(unittest.TestCase):
    def test_pipeline_uses_low_latency_udp_h265_settings(self):
        pipeline = build_udp_h265_pipeline(
            UdpH265CameraSpec(
                name="wide_90",
                port=5601,
                decoder="avdec_h265",
                payload=96,
                jitter_latency_ms=0,
            )
        )

        self.assertIn("udpsrc port=5601", pipeline)
        self.assertIn("encoding-name=(string)H265", pipeline)
        self.assertIn("payload=(int)96", pipeline)
        self.assertIn("rtpjitterbuffer latency=0 drop-on-latency=true", pipeline)
        self.assertIn("queue max-size-buffers=1", pipeline)
        self.assertIn("leaky=downstream", pipeline)
        self.assertIn("avdec_h265", pipeline)
        self.assertIn("appsink name=appsink", pipeline)
        self.assertIn("max-buffers=1 drop=true sync=false", pipeline)

    def test_appcontroller_loads_udp_h265_default_specs(self):
        import main

        old_config = os.environ.pop("MITSU_REAL_CAMERAS_CONFIG", None)
        old_backend = os.environ.pop("MITSU_REAL_CAMERA_BACKEND", None)
        try:
            os.environ["MITSU_REAL_CAMERAS_CONFIG"] = str(
                Path(__file__).resolve().parents[1] / "config" / "real_cameras.udp_h265.example.json"
            )
            controller = main.AppController.__new__(main.AppController)

            self.assertEqual(controller._real_camera_backend(), "gstreamer_udp")
            specs = controller._load_udp_h265_camera_specs()
            self.assertGreaterEqual(len(specs), 4)
            self.assertEqual(specs[0].name, "wide_90")
            self.assertEqual(specs[0].port, 5601)
            self.assertEqual(specs[1].name, "narrow_50")
            self.assertEqual(specs[1].port, 5602)
        finally:
            if old_config is not None:
                os.environ["MITSU_REAL_CAMERAS_CONFIG"] = old_config
            if old_backend is not None:
                os.environ["MITSU_REAL_CAMERA_BACKEND"] = old_backend


if __name__ == "__main__":
    unittest.main()
