from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np


GUI_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = GUI_ROOT.parent
if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


COMMANDS = (
    "lane_follow",
    "straight",
    "turn_left",
    "turn_right",
    "change_lane_left",
    "change_lane_right",
)

COMMAND_ONE_HOT = {
    "turn_left": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "turn_right": (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    "straight": (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    "lane_follow": (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    "change_lane_left": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    "change_lane_right": (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
}


@dataclass(frozen=True)
class MockRealVehicleSample:
    seq: int
    frames: dict[str, np.ndarray]
    context: dict[str, Any]
    vehicle_command: dict[str, Any]


@dataclass(frozen=True)
class TimingSummary:
    count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def summarize_ms(values: Iterable[float]) -> TimingSummary:
    items = [float(v) for v in values]
    if not items:
        return TimingSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return TimingSummary(
        count=len(items),
        mean_ms=statistics.fmean(items),
        median_ms=statistics.median(items),
        p95_ms=percentile(items, 0.95),
        min_ms=min(items),
        max_ms=max(items),
    )


def world_to_ego(
    point_x: float,
    point_y: float,
    ego_x: float,
    ego_y: float,
    ego_yaw_deg: float,
) -> list[float]:
    dx = float(point_x) - float(ego_x)
    dy = float(point_y) - float(ego_y)
    yaw = math.radians(float(ego_yaw_deg))
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return [
        float(cos_yaw * dx + sin_yaw * dy),
        float(-sin_yaw * dx + cos_yaw * dy),
    ]


class RealVehicleInputMocker:
    """Generate changing real-car input payloads for the i-MiEV real adapter."""

    def __init__(
        self,
        *,
        width: int = 384,
        height: int = 384,
        seed: int = 20260619,
        start_lat: float = 55.9949,
        start_lon: float = 92.7934,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.rng = np.random.default_rng(seed)
        self.start_lat = float(start_lat)
        self.start_lon = float(start_lon)

    def _frame(self, seq: int, fov_deg: int) -> np.ndarray:
        h = self.height
        w = self.width
        x = np.linspace(0, 255, w, dtype=np.float32)[None, :]
        y = np.linspace(0, 255, h, dtype=np.float32)[:, None]
        phase = float(seq * (3 if fov_deg > 60 else 5))
        blue = (x + phase) % 255.0
        green = (y + phase * 0.7) % 255.0
        red = ((x * 0.25 + y * 0.75) + fov_deg + phase * 0.3) % 255.0
        img = np.dstack(
            [
                np.broadcast_to(blue, (h, w)),
                np.broadcast_to(green, (h, w)),
                red,
            ]
        )

        road_center = int(w * (0.5 + 0.12 * math.sin(seq / 13.0)))
        lane_half_width = max(24, w // (10 if fov_deg > 60 else 7))
        img[:, max(0, road_center - lane_half_width) : max(0, road_center - lane_half_width + 4), :] = (40, 210, 210)
        img[:, min(w, road_center + lane_half_width) : min(w, road_center + lane_half_width + 4), :] = (40, 210, 210)

        noise = self.rng.normal(0.0, 4.0, size=(h, w, 3))
        return np.clip(img + noise, 0, 255).astype(np.uint8)

    def sample(self, seq: int) -> MockRealVehicleSample:
        dt = 0.1
        route_x = seq * 0.35
        route_y = 1.4 * math.sin(seq / 18.0)
        yaw_deg = math.degrees(math.atan2(
            1.4 / 18.0 * math.cos(seq / 18.0),
            1.0,
        ))
        speed_kmh = 8.0 + 4.0 * math.sin(seq / 11.0) + 0.4 * self.rng.normal()
        speed_kmh = max(0.0, speed_kmh)
        speed_mps = speed_kmh / 3.6

        lat = self.start_lat + route_y / 111_320.0
        lon = self.start_lon + route_x / (111_320.0 * math.cos(math.radians(self.start_lat)))

        previous_world = (route_x + 3.0, route_y)
        target_world = (route_x + 6.0, route_y + 0.4 * math.sin(seq / 9.0))
        next_world = (route_x + 9.0, route_y + 0.6 * math.sin((seq + 3) / 9.0))

        command_name = COMMANDS[(seq // 17) % len(COMMANDS)]
        next_command_name = COMMANDS[((seq // 17) + 1) % len(COMMANDS)]
        command_one_hot = COMMAND_ONE_HOT[command_name]
        next_command_one_hot = COMMAND_ONE_HOT[next_command_name]

        telemetry = SimpleNamespace(
            connected=True,
            heartbeat_ok=True,
            speed_kmh=float(speed_kmh),
            speed_source="mock_mcu_can",
            angle_deg=float(4.0 * math.sin(seq / 8.0)),
            target_angle_deg=float(5.0 * math.sin((seq + 1) / 8.0)),
            accel_pct=float(18.0 + 6.0 * math.sin(seq / 10.0)),
            brake_pct=float(max(0.0, 3.0 * math.sin(seq / 15.0))),
            x_m=float(route_x),
            y_m=float(route_y),
            yaw_deg=float(yaw_deg),
            pose_valid=True,
            pose_source="mock_nmea0183_gnss",
            gps_lat=float(lat),
            gps_lon=float(lon),
            last_rx_monotonic=time.monotonic(),
            last_pose_monotonic=time.monotonic(),
        )
        goal = SimpleNamespace(
            target_x_m=float(target_world[0]),
            target_y_m=float(target_world[1]),
            previous_x_m=float(previous_world[0]),
            previous_y_m=float(previous_world[1]),
            next_x_m=float(next_world[0]),
            next_y_m=float(next_world[1]),
            previous_waypoint_index=int(seq + 1),
            waypoint_index=int(seq + 2),
            next_waypoint_index=int(seq + 3),
            control_waypoint_index=int(seq + 2),
            road_option_name=command_name,
            next_road_option_name=next_command_name,
            desired_speed_kmh=float(min(12.0, speed_kmh + 1.5)),
            speed_cap_kmh=12.0,
            distance_to_target_m=6.0,
            distance_to_goal_m=max(0.0, 120.0 - seq * 0.35),
            maneuver=command_name,
            is_valid=lambda: True,
        )

        target_previous_ego = world_to_ego(*previous_world, route_x, route_y, yaw_deg)
        target_ego = world_to_ego(*target_world, route_x, route_y, yaw_deg)
        target_next_ego = world_to_ego(*next_world, route_x, route_y, yaw_deg)
        captured_at = time.monotonic()

        context = {
            "telemetry": telemetry,
            "goal": goal,
            "speed_kmh": float(speed_kmh),
            "speed_mps": float(speed_mps),
            "pose_valid": True,
            "pose_source": telemetry.pose_source,
            "pose_mode": "mock_real_car_nmea",
            "gps_lat": float(lat),
            "gps_lon": float(lon),
            "model_ego_yaw_deg": float(yaw_deg),
            "target_point_previous": target_previous_ego,
            "target_point": target_ego,
            "target_point_next": target_next_ego,
            "target_point_previous_ego": target_previous_ego,
            "target_point_ego": target_ego,
            "target_point_next_ego": target_next_ego,
            "target_point_previous_world": [float(previous_world[0]), float(previous_world[1])],
            "target_point_world": [float(target_world[0]), float(target_world[1])],
            "target_point_next_world": [float(next_world[0]), float(next_world[1])],
            "target_point_previous_waypoint_index": int(seq + 1),
            "target_point_waypoint_index": int(seq + 2),
            "target_point_next_waypoint_index": int(seq + 3),
            "target_point_control_waypoint_index": int(seq + 2),
            "command_name": command_name,
            "next_command_name": next_command_name,
            "command_one_hot": command_one_hot,
            "next_command_one_hot": next_command_one_hot,
            "town": "RealWorld",
            "route_target_spacing_m": 3.0,
            "model_waypoints_spacing_m": 3.0,
            "input_sample_seq": int(seq),
            "input_sample_generation": 1,
            "input_sample_created_at_monotonic": captured_at,
            "input_frame_timestamps_monotonic": {
                "wide_90": captured_at - 0.012,
                "narrow_50": captured_at - 0.006,
            },
            "input_frame_part_sequences": {
                "wide_90": int(seq * 2 - 1),
                "narrow_50": int(seq * 2),
            },
        }
        vehicle_command = {
            "seq": int(seq),
            "active": True,
            "gear_request": "D",
            "steering_raw": int(round(telemetry.target_angle_deg / 630.0 * 100.0)),
            "accel_pct": int(round(telemetry.accel_pct)),
            "brake_pct": int(round(telemetry.brake_pct)),
            "valid_for_ms": 100,
            "reason": "mock_real_vehicle_model_benchmark",
        }
        return MockRealVehicleSample(
            seq=int(seq),
            frames={
                "wide_90": self._frame(seq, 90),
                "narrow_50": self._frame(seq, 50),
            },
            context=context,
            vehicle_command=vehicle_command,
        )


class TimedMockRealModel:
    """Small deterministic stand-in with the same predict(frames, context) API."""

    def __init__(self, startup_delay_ms: float = 2.0, compile_delay_ms: float = 1.0) -> None:
        if startup_delay_ms > 0:
            time.sleep(startup_delay_ms / 1000.0)
        self.compile_delay_ms = float(compile_delay_ms)
        self.compiled = False
        self.frame_id = 0

    def compile(self, sample: MockRealVehicleSample | None = None) -> None:
        if self.compiled:
            return
        if self.compile_delay_ms > 0:
            time.sleep(self.compile_delay_ms / 1000.0)
        self.compiled = True

    def predict(self, frames: dict[str, np.ndarray], context: dict[str, Any]) -> dict[str, Any]:
        if not self.compiled:
            self.compile()
        self.frame_id += 1
        speed_mps = float(context.get("speed_mps", 0.0) or 0.0)
        target = np.asarray(context.get("target_point_ego", context.get("target_point", [0.0, 0.0])), dtype=np.float32)
        lateral = float(target.reshape(-1)[1]) if target.size >= 2 else 0.0
        wide_mean = float(np.asarray(frames["wide_90"], dtype=np.float32).mean())
        narrow_mean = float(np.asarray(frames["narrow_50"], dtype=np.float32).mean())
        visual_bias = (wide_mean - narrow_mean) / 255.0
        steer = max(-1.0, min(1.0, lateral / 6.0 + visual_bias * 0.05))
        throttle = max(0.0, min(1.0, 0.28 - speed_mps * 0.025))
        brake = max(0.0, min(1.0, (speed_mps - 4.0) * 0.05))
        now = time.monotonic()
        return {
            "steer": float(steer),
            "throttle": float(throttle),
            "brake": float(brake),
            "confidence": 1.0,
            "timestamp_monotonic": now,
            "frame_id": self.frame_id,
            "model_forward_ms": 0.0,
            "source": "timed_mock_real_model",
        }


def create_real_model(checkpoint_dir: str | None) -> Any:
    from real_agent_adapters.lead_real_model_0011_adapter import LeadModel0011RealPortAdapter

    os.environ.setdefault("MITSU_REAL_VISUALIZATION", "0")
    return LeadModel0011RealPortAdapter(checkpoint_dir=checkpoint_dir)


def sync_model_device(model: Any) -> None:
    device = getattr(model, "device", None)
    if device is None:
        return
    try:
        import torch

        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(device)
    except Exception:
        return


def run_benchmark(
    *,
    samples: int = 100,
    mock_model: bool = True,
    width: int = 1280,
    height: int = 720,
    checkpoint_dir: str | None = None,
    seed: int = 20260619,
) -> dict[str, Any]:
    mocker = RealVehicleInputMocker(width=width, height=height, seed=seed)

    startup_start = time.perf_counter()
    model = TimedMockRealModel() if mock_model else create_real_model(checkpoint_dir)
    sync_model_device(model)
    startup_ms = (time.perf_counter() - startup_start) * 1000.0

    warmup_sample = mocker.sample(0)
    compile_start = time.perf_counter()
    if hasattr(model, "compile"):
        model.compile(warmup_sample)
    else:
        model.predict(warmup_sample.frames, warmup_sample.context)
    sync_model_device(model)
    compile_ms = (time.perf_counter() - compile_start) * 1000.0

    inference_ms: list[float] = []
    forward_ms: list[float] = []
    predictions: list[dict[str, Any]] = []
    first_sample_digest: dict[str, Any] | None = None
    last_sample_digest: dict[str, Any] | None = None

    for seq in range(1, samples + 1):
        sample = mocker.sample(seq)
        if seq == 1:
            first_sample_digest = digest_sample(sample)

        started = time.perf_counter()
        prediction = model.predict(sample.frames, sample.context)
        sync_model_device(model)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        inference_ms.append(elapsed_ms)
        if "model_forward_ms" in prediction:
            try:
                forward_ms.append(float(prediction["model_forward_ms"]))
            except Exception:
                pass
        predictions.append(prediction)
        last_sample_digest = digest_sample(sample)

    if hasattr(model, "close"):
        model.close()

    return {
        "mode": "mock_model" if mock_model else "real_model_0011",
        "samples": int(samples),
        "image_shape_bgr": [int(height), int(width), 3],
        "startup_ms": float(startup_ms),
        "compile_ms": float(compile_ms),
        "inference": asdict(summarize_ms(inference_ms)),
        "model_forward": asdict(summarize_ms(forward_ms)),
        "first_sample": first_sample_digest,
        "last_sample": last_sample_digest,
        "first_prediction": compact_prediction(predictions[0]) if predictions else None,
        "last_prediction": compact_prediction(predictions[-1]) if predictions else None,
    }


def compact_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    keys = ("frame_id", "steer", "throttle", "brake", "confidence", "source", "device")
    return {key: prediction[key] for key in keys if key in prediction}


def digest_sample(sample: MockRealVehicleSample) -> dict[str, Any]:
    context = sample.context
    telemetry = context["telemetry"]
    return {
        "seq": int(sample.seq),
        "wide_90_mean": float(sample.frames["wide_90"].mean()),
        "narrow_50_mean": float(sample.frames["narrow_50"].mean()),
        "speed_kmh": float(context["speed_kmh"]),
        "gps_lat": float(context["gps_lat"]),
        "gps_lon": float(context["gps_lon"]),
        "pose": {
            "x_m": float(telemetry.x_m),
            "y_m": float(telemetry.y_m),
            "yaw_deg": float(telemetry.yaw_deg),
            "source": str(telemetry.pose_source),
        },
        "target_point_ego": [float(x) for x in context["target_point_ego"]],
        "target_point_world": [float(x) for x in context["target_point_world"]],
        "command_name": str(context["command_name"]),
        "next_command_name": str(context["next_command_name"]),
        "command_one_hot": [float(x) for x in context["command_one_hot"]],
        "vehicle_command": dict(sample.vehicle_command),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mock real i-MiEV camera/GPS/controller inputs and benchmark model "
            "startup, compile/warmup, and per-frame inference timing."
        )
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument(
        "--real-model",
        action="store_true",
        help="Load real LeadModel0011RealPortAdapter instead of the lightweight mock model.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path for the JSON timing report.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_benchmark(
        samples=max(1, int(args.samples)),
        mock_model=not bool(args.real_model),
        width=max(16, int(args.width)),
        height=max(16, int(args.height)),
        checkpoint_dir=args.checkpoint_dir,
        seed=int(args.seed),
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(payload)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
