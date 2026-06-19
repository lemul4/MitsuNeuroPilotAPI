import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "real_model_mock_benchmark.py"
SPEC = importlib.util.spec_from_file_location("real_model_mock_benchmark", SCRIPT_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


class RealModelMockBenchmarkTests(unittest.TestCase):
    def test_mock_samples_change_and_keep_real_input_format(self):
        mocker = benchmark.RealVehicleInputMocker(width=64, height=36, seed=123)

        first = mocker.sample(1)
        second = mocker.sample(2)

        self.assertEqual(first.frames["wide_90"].shape, (36, 64, 3))
        self.assertEqual(first.frames["wide_90"].dtype, np.uint8)
        self.assertEqual(first.frames["narrow_50"].shape, (36, 64, 3))
        self.assertIn("telemetry", first.context)
        self.assertIn("goal", first.context)
        self.assertIn("speed_mps", first.context)
        self.assertIn("gps_lat", first.context)
        self.assertIn("gps_lon", first.context)
        self.assertIn("target_point_previous_ego", first.context)
        self.assertIn("target_point_ego", first.context)
        self.assertIn("target_point_next_ego", first.context)
        self.assertEqual(len(first.context["command_one_hot"]), 6)
        self.assertEqual(len(first.context["next_command_one_hot"]), 6)
        self.assertIn("steering_raw", first.vehicle_command)
        self.assertIn("accel_pct", first.vehicle_command)
        self.assertIn("brake_pct", first.vehicle_command)

        self.assertFalse(np.array_equal(first.frames["wide_90"], second.frames["wide_90"]))
        self.assertNotEqual(first.context["speed_kmh"], second.context["speed_kmh"])
        self.assertNotEqual(first.context["gps_lon"], second.context["gps_lon"])
        self.assertNotEqual(first.context["target_point_ego"], second.context["target_point_ego"])

    def test_mock_benchmark_runs_100_samples_and_reports_timing(self):
        result = benchmark.run_benchmark(
            samples=100,
            mock_model=True,
            width=64,
            height=36,
            seed=456,
        )

        self.assertEqual(result["mode"], "mock_model")
        self.assertEqual(result["samples"], 100)
        self.assertGreaterEqual(result["startup_ms"], 0.0)
        self.assertGreaterEqual(result["compile_ms"], 0.0)
        self.assertEqual(result["inference"]["count"], 100)
        self.assertIn("mean_ms", result["inference"])
        self.assertIn("median_ms", result["inference"])
        self.assertIn("p95_ms", result["inference"])
        self.assertGreaterEqual(result["inference"]["p95_ms"], result["inference"]["median_ms"])
        self.assertNotEqual(result["first_sample"]["speed_kmh"], result["last_sample"]["speed_kmh"])
        self.assertNotEqual(result["first_sample"]["wide_90_mean"], result["last_sample"]["wide_90_mean"])
        self.assertEqual(result["last_prediction"]["frame_id"], 100)


if __name__ == "__main__":
    unittest.main()
