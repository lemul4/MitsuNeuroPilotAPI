import json
import os
import tempfile
import unittest
from pathlib import Path

from lead.common.carla_vehicle_sensor_profile import (
    apply_camera_sensor_profile,
    build_camera_sensor_specs,
    load_sensor_profile,
)


class CarlaVehicleSensorProfileTests(unittest.TestCase):
    def _write_profile(self, tmp: Path) -> Path:
        profile = {
            "keep_non_camera_sensors": True,
            "cameras": [
                {
                    "id": "rgb_1",
                    "role": "wide_90",
                    "x": 0.9,
                    "y": -0.0675,
                    "z": 1.55,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "width": 960,
                    "height": 540,
                    "fov": 90.0,
                },
                {
                    "id": "rgb_3",
                    "role": "narrow_50",
                    "x": 0.9,
                    "y": 0.0675,
                    "z": 1.55,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "width": 960,
                    "height": 540,
                    "fov": 50.03,
                },
            ],
        }
        path = tmp / "profile.json"
        path.write_text(json.dumps(profile), encoding="utf-8")
        return path

    def test_profile_builds_two_rgb_specs(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_profile(Path(td))
            specs = build_camera_sensor_specs(load_sensor_profile(path))
            self.assertEqual([s["id"] for s in specs], ["rgb_1", "rgb_3"])
            self.assertEqual(specs[0]["type"], "sensor.camera.rgb")
            self.assertEqual(specs[0]["fov"], 90.0)
            self.assertEqual(specs[1]["fov"], 50.03)

    def test_profile_replaces_only_rgb_and_keeps_non_camera_sensors(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_profile(Path(td))
            sensors = [
                {"type": "sensor.camera.rgb", "id": "rgb_1"},
                {"type": "sensor.camera.rgb", "id": "rgb_2"},
                {"type": "sensor.lidar.ray_cast", "id": "lidar"},
                {"type": "sensor.other.gnss", "id": "gps"},
            ]
            result = apply_camera_sensor_profile(sensors, profile_path=path)
            self.assertEqual([s["id"] for s in result], ["rgb_1", "rgb_3", "lidar", "gps"])


if __name__ == "__main__":
    unittest.main()
