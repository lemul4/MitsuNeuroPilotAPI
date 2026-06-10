import os
import unittest
from pathlib import Path

from hardware.gstreamer_udp_camera import UdpH265CameraSpec, build_udp_h265_pipeline
from hardware.opencv_udp_camera import OpenCvUdpH265CameraSpec, build_h265_rtp_sdp, read_latest_raw_frame


class _BufferedPipe:
    def __init__(self, chunks):
        self.data = b"".join(chunks)
        self.offset = 0

    def read(self, size):
        end = min(len(self.data), self.offset + int(size))
        chunk = self.data[self.offset:end]
        self.offset = end
        return chunk


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

    def test_opencv_udp_sdp_describes_h265_rtp_stream(self):
        sdp = build_h265_rtp_sdp(OpenCvUdpH265CameraSpec("wide_90", 5601, payload=96))

        self.assertIn("m=video 5601 RTP/AVP 96", sdp)
        self.assertIn("a=rtpmap:96 H265/90000", sdp)
        self.assertIn("c=IN IP4 127.0.0.1", sdp)

    def test_opencv_raw_pipe_reader_returns_newest_complete_frame(self):
        frames = [bytes([idx]) * 4 for idx in range(1, 4)]
        pipe = _BufferedPipe(frames)
        old_available = __import__("hardware.opencv_udp_camera", fromlist=["_pipe_bytes_available"])._pipe_bytes_available
        try:
            import hardware.opencv_udp_camera as opencv_udp_camera

            opencv_udp_camera._pipe_bytes_available = lambda stream: len(stream.data) - stream.offset
            self.assertEqual(read_latest_raw_frame(pipe, 4), frames[-1])
        finally:
            import hardware.opencv_udp_camera as opencv_udp_camera

            opencv_udp_camera._pipe_bytes_available = old_available

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

    def test_appcontroller_loads_opencv_udp_h265_specs(self):
        import main

        old_config = os.environ.pop("MITSU_REAL_CAMERAS_CONFIG", None)
        old_backend = os.environ.pop("MITSU_REAL_CAMERA_BACKEND", None)
        try:
            os.environ["MITSU_REAL_CAMERAS_CONFIG"] = str(
                Path(__file__).resolve().parents[1] / "config" / "real_cameras.opencv_udp.example.json"
            )
            controller = main.AppController.__new__(main.AppController)

            self.assertEqual(controller._real_camera_backend(), "opencv_udp")
            specs = controller._load_opencv_udp_h265_camera_specs()
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
