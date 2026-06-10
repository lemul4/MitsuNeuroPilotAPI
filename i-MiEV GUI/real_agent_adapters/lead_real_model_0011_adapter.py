from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover
    cv2 = None
    np = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


# --- MITSU_REAL_CPU_CARLA_SHIM_BEGIN ---
def _mitsu_install_carla_shims() -> None:
    """Install minimal CARLA symbols for real-COM CPU inference.

    Real COM-port inference does not use CARLA actors. Some local environments
    import a partial `carla` module. LEAD imports type annotations and visual
    constants such as carla.Color / carla.Transform / carla.Location. These
    lightweight classes are sufficient for import-time use and simple local
    coordinate containers.
    """
    try:
        import carla  # type: ignore
    except Exception:
        class _CarlaShim:
            pass

        carla = _CarlaShim()  # type: ignore
        sys.modules["carla"] = carla  # type: ignore

    if not hasattr(carla, "Color"):
        class Color:
            def __init__(self, r=0, g=0, b=0, a=255):
                self.r = int(r)
                self.g = int(g)
                self.b = int(b)
                self.a = int(a)

            def __iter__(self):
                return iter((self.r, self.g, self.b, self.a))

            def __repr__(self):
                return f"Color(r={self.r}, g={self.g}, b={self.b}, a={self.a})"

        carla.Color = Color  # type: ignore[attr-defined]

    if not hasattr(carla, "Location"):
        class Location:
            def __init__(self, x=0.0, y=0.0, z=0.0):
                self.x = float(x)
                self.y = float(y)
                self.z = float(z)

            def distance(self, other):
                dx = self.x - getattr(other, "x", 0.0)
                dy = self.y - getattr(other, "y", 0.0)
                dz = self.z - getattr(other, "z", 0.0)
                return (dx * dx + dy * dy + dz * dz) ** 0.5

            def __repr__(self):
                return f"Location(x={self.x}, y={self.y}, z={self.z})"

        carla.Location = Location  # type: ignore[attr-defined]

    if not hasattr(carla, "Rotation"):
        class Rotation:
            def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
                self.pitch = float(pitch)
                self.yaw = float(yaw)
                self.roll = float(roll)

            def __repr__(self):
                return f"Rotation(pitch={self.pitch}, yaw={self.yaw}, roll={self.roll})"

        carla.Rotation = Rotation  # type: ignore[attr-defined]

    if not hasattr(carla, "Vector3D"):
        class Vector3D:
            def __init__(self, x=0.0, y=0.0, z=0.0):
                self.x = float(x)
                self.y = float(y)
                self.z = float(z)

            def __repr__(self):
                return f"Vector3D(x={self.x}, y={self.y}, z={self.z})"

        carla.Vector3D = Vector3D  # type: ignore[attr-defined]

    if not hasattr(carla, "Transform"):
        class Transform:
            def __init__(self, location=None, rotation=None):
                self.location = location if location is not None else carla.Location()
                self.rotation = rotation if rotation is not None else carla.Rotation()

            def transform(self, location):
                return location

            def get_forward_vector(self):
                import math
                yaw = math.radians(getattr(self.rotation, "yaw", 0.0))
                return carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)

            def __repr__(self):
                return f"Transform(location={self.location}, rotation={self.rotation})"

        carla.Transform = Transform  # type: ignore[attr-defined]

    if not hasattr(carla, "BoundingBox"):
        class BoundingBox:
            def __init__(self, location=None, extent=None):
                self.location = location if location is not None else carla.Location()
                self.extent = extent if extent is not None else carla.Vector3D()

        carla.BoundingBox = BoundingBox  # type: ignore[attr-defined]

    if not hasattr(carla, "VehicleControl"):
        class VehicleControl:
            def __init__(
                self,
                throttle=0.0,
                steer=0.0,
                brake=0.0,
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
                gear=0,
            ):
                self.throttle = float(throttle)
                self.steer = float(steer)
                self.brake = float(brake)
                self.hand_brake = bool(hand_brake)
                self.reverse = bool(reverse)
                self.manual_gear_shift = bool(manual_gear_shift)
                self.gear = int(gear)

            def __repr__(self):
                return (
                    "VehicleControl("
                    f"throttle={self.throttle}, steer={self.steer}, brake={self.brake}, "
                    f"hand_brake={self.hand_brake}, reverse={self.reverse}, "
                    f"manual_gear_shift={self.manual_gear_shift}, gear={self.gear})"
                )

        carla.VehicleControl = VehicleControl  # type: ignore[attr-defined]


_mitsu_install_carla_shims()
# --- MITSU_REAL_CPU_CARLA_SHIM_END ---



def _project_root() -> Path:
    # <repo>/i-MiEV GUI/real_agent_adapters/lead_real_model_0011_adapter.py
    return Path(__file__).resolve().parents[2]


def _ensure_project_import_paths() -> Path:
    root = _project_root()
    candidates = [
        root,
        root / "3rd_party" / "CARLA_0915" / "PythonAPI",
        root / "3rd_party" / "CARLA_0915" / "PythonAPI" / "carla",
        root / "3rd_party" / "leaderboard_autopilot",
        root / "3rd_party" / "scenario_runner_autopilot",
    ]
    for path in candidates:
        value = str(path)
        if path.exists() and value not in sys.path:
            sys.path.insert(0, value)
    return root


def _select_device() -> "torch.device":
    if torch is None:
        raise RuntimeError("torch is required for model_0011 real-port inference")

    requested = os.environ.get("MITSU_REAL_MODEL_DEVICE", "auto").strip().lower()
    if requested in {"", "auto"}:
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"

    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"MITSU_REAL_MODEL_DEVICE={requested!r}, but torch.cuda.is_available() is False. "
            "Use MITSU_REAL_MODEL_DEVICE=cpu for CPU inference, or install CUDA-enabled PyTorch."
        )

    return torch.device(requested)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _read_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _world_to_ego_local(
    point_x: float,
    point_y: float,
    ego_x: float,
    ego_y: float,
    ego_yaw_deg: float,
) -> list[float]:
    """Convert route/world local coordinates into ego-relative coordinates.

    The model expects target points in the vehicle coordinate frame:
    ego vehicle == [0, 0], x forward, y lateral.
    """
    dx = float(point_x) - float(ego_x)
    dy = float(point_y) - float(ego_y)
    yaw = math.radians(float(ego_yaw_deg))
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    forward = cos_yaw * dx + sin_yaw * dy
    lateral = -sin_yaw * dx + cos_yaw * dy
    return [float(forward), float(lateral)]


@dataclass
class RealModel0011InputBuilder:
    image_height: int = int(os.environ.get("MITSU_REAL_MODEL_IMAGE_HEIGHT", "384"))
    image_width: int = int(os.environ.get("MITSU_REAL_MODEL_IMAGE_WIDTH", "384"))
    town: str = os.environ.get("MITSU_REAL_TOWN", "RealWorld")

    # Straight/lane-follow default. RealAgentBridge normally overrides this.
    default_command_one_hot: tuple[float, ...] = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

    def configure_from_training(self, training_config: Any) -> None:
        self.image_height = int(getattr(training_config, "final_image_height", self.image_height))
        self.image_width = int(getattr(training_config, "final_image_width", self.image_width))

    def _resize_camera(self, frame: Any):
        if cv2 is None or np is None:
            raise RuntimeError("OpenCV/numpy are required for model_0011 real camera input")

        if frame is None:
            return np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

        img = frame
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim != 3:
            raise RuntimeError(f"Invalid camera frame ndim={getattr(img, 'ndim', None)}")
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        img = cv2.resize(
            img,
            (self.image_width, self.image_height),
            interpolation=cv2.INTER_LINEAR,
        )

        # camera_zmq uses cv2.imdecode, so incoming frames are BGR.
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _camera_to_chw(self, frame: Any):
        img = self._resize_camera(frame)
        return np.transpose(img, (2, 0, 1)).astype(np.float32)

    def _command(self, context: Dict[str, Any], key: str) -> "np.ndarray":
        value = context.get(key)
        if value is None:
            value = self.default_command_one_hot
        return np.asarray(value, dtype=np.float32).reshape(6)

    def _target_points(self, context: Dict[str, Any]):
        telemetry = context.get("telemetry")
        goal = context.get("goal")

        def _as_pair(value, default):
            try:
                return [float(value[0]), float(value[1])]
            except Exception:
                return list(default)

        # Preferred path: main.py already converted navigator prev/target/next
        # into ego-local points. Keep all three points in the same frame:
        # ego vehicle == [0, 0], x forward, y lateral.
        if context.get("target_point_ego") is not None:
            previous = _as_pair(context.get("target_point_previous_ego"), [0.0, 0.0])
            target = _as_pair(context.get("target_point_ego"), [0.0, 0.0])
            next_point = _as_pair(context.get("target_point_next_ego"), target)

        elif goal is not None:
            ego_x = _as_float(_read_attr_or_key(telemetry, "x_m", 0.0))
            ego_y = _as_float(_read_attr_or_key(telemetry, "y_m", 0.0))
            ego_yaw = _as_float(_read_attr_or_key(telemetry, "yaw_deg", 0.0))

            target_x = _as_float(_read_attr_or_key(goal, "target_x_m", 0.0))
            target_y = _as_float(_read_attr_or_key(goal, "target_y_m", 0.0))

            previous_x = _as_float(_read_attr_or_key(goal, "previous_x_m", ego_x))
            previous_y = _as_float(_read_attr_or_key(goal, "previous_y_m", ego_y))
            next_x = _as_float(_read_attr_or_key(goal, "next_x_m", target_x))
            next_y = _as_float(_read_attr_or_key(goal, "next_y_m", target_y))

            previous = _world_to_ego_local(previous_x, previous_y, ego_x, ego_y, ego_yaw)
            target = _world_to_ego_local(target_x, target_y, ego_x, ego_y, ego_yaw)
            next_point = _world_to_ego_local(next_x, next_y, ego_x, ego_y, ego_yaw)

        else:
            previous = _as_pair(context.get("target_point_previous"), [0.0, 0.0])
            target = _as_pair(context.get("target_point"), [0.0, 0.0])
            next_point = _as_pair(context.get("target_point_next"), target)

        previous = np.asarray(previous, dtype=np.float32).reshape(2)
        target = np.asarray(target, dtype=np.float32).reshape(2)
        next_point = np.asarray(next_point, dtype=np.float32).reshape(2)
        return previous, target, next_point

    def build_numpy(self, frames: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if np is None:
            raise RuntimeError("numpy is required for model_0011 real camera input")

        context = dict(context or {})
        previous, target, next_point = self._target_points(context)

        speed_mps = context.get("speed_mps")
        if speed_mps is None:
            speed_mps = _as_float(context.get("speed_kmh", 0.0)) / 3.6

        return {
            "rgb_left": self._camera_to_chw(frames.get("wide_90")),
            "rgb_right": self._camera_to_chw(frames.get("narrow_50")),
            "target_point_previous": previous,
            "target_point": target,
            "target_point_next": next_point,
            "speed": np.asarray([_as_float(speed_mps, 0.0)], dtype=np.float32),
            "command": self._command(context, "command_one_hot"),
            "next_command": self._command(context, "next_command_one_hot"),
            "town": np.asarray([str(context.get("town", self.town))]),
        }

    def build_torch(
        self,
        frames: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        device: "torch.device",
    ) -> Dict[str, Any]:
        if torch is None:
            raise RuntimeError("torch is required for model_0011 inference")

        data = self.build_numpy(frames, context)

        return {
            "rgb_left": torch.tensor(data["rgb_left"], dtype=torch.float32, device=device).unsqueeze(0),
            "rgb_right": torch.tensor(data["rgb_right"], dtype=torch.float32, device=device).unsqueeze(0),
            "target_point_previous": torch.tensor(data["target_point_previous"], dtype=torch.float32, device=device).view(1, 2),
            "target_point": torch.tensor(data["target_point"], dtype=torch.float32, device=device).view(1, 2),
            "target_point_next": torch.tensor(data["target_point_next"], dtype=torch.float32, device=device).view(1, 2),
            "speed": torch.tensor(data["speed"], dtype=torch.float32, device=device).view(1),
            "command": torch.tensor(data["command"], dtype=torch.float32, device=device).view(1, 6),
            "next_command": torch.tensor(data["next_command"], dtype=torch.float32, device=device).view(1, 6),
            "town": data["town"],
        }


class LeadModel0011RealPortAdapter:
    """Run outputs/model_0011 in real COM-port mode.

    This adapter returns normalized steer/throttle/brake predictions. It does
    not send CAN commands directly; VehicleControlService/RealAgentBridge remain
    responsible for safety gating and command arbitration.
    """

    @staticmethod
    def _timing_enabled() -> bool:
        return os.environ.get("MITSU_REAL_MODEL_TIMING", "1").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @classmethod
    def _log_timing(cls, stage: str, start_s: float, **fields: Any) -> None:
        if not cls._timing_enabled():
            return
        elapsed_ms = (time.perf_counter() - start_s) * 1000.0
        suffix = ""
        if fields:
            suffix = " " + " ".join(f"{key}={value}" for key, value in fields.items())
        print(f"REAL MODEL TIMING: {stage} {elapsed_ms:.1f} ms{suffix}", flush=True)

    def __init__(
        self,
        builder: Optional[RealModel0011InputBuilder] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        total_start = time.perf_counter()
        if torch is None:
            raise RuntimeError("torch is required for model_0011 inference")

        step_start = time.perf_counter()
        root = _ensure_project_import_paths()
        _mitsu_install_carla_shims()
        self._log_timing("project_paths_and_carla_shims", step_start)

        step_start = time.perf_counter()
        from lead.expert.config_expert import ExpertConfig
        _mitsu_install_carla_shims()

        from lead.inference.closed_loop_inference import ClosedLoopInference
        from lead.inference.config_closed_loop import ClosedLoopConfig
        from lead.training.config_training import TrainingConfig
        try:
            from lead.visualization.visualizer import Visualizer
        except Exception as exc:
            Visualizer = None
            print(f"REAL MODEL VIS: visualizer import failed: {exc}", flush=True)
        self._log_timing("lead_imports", step_start)

        self.root = root
        step_start = time.perf_counter()
        self.device = _select_device()
        self._log_timing("select_device", step_start, device=self.device)
        self.builder = builder or RealModel0011InputBuilder()

        step_start = time.perf_counter()
        checkpoint = Path(
            checkpoint_dir
            or os.environ.get("MITSU_REAL_MODEL_CHECKPOINT", "")
            or (root / "outputs" / "model_0011")
        )
        if not checkpoint.is_absolute():
            checkpoint = root / checkpoint
        checkpoint = checkpoint.resolve()

        if not checkpoint.is_dir():
            raise RuntimeError(f"model_0011 checkpoint directory not found: {checkpoint}")

        config_path = checkpoint / "config_dual_front_camera.json"
        if not config_path.exists():
            json_files = sorted(checkpoint.glob("*.json"))
            if not json_files:
                raise RuntimeError(f"No *.json config found in checkpoint directory: {checkpoint}")
            config_path = json_files[0]

        model_files = sorted(checkpoint.glob("model*.pth"))
        if not model_files:
            raise RuntimeError(f"No model*.pth weights found in checkpoint directory: {checkpoint}")
        self._log_timing(
            "resolve_checkpoint",
            step_start,
            checkpoint=checkpoint,
            config=config_path.name,
            model_files=len(model_files),
        )

        step_start = time.perf_counter()
        os.environ.setdefault("SAVE_PATH", str(root / "outputs" / "real_model_inference"))
        os.environ.setdefault("BENCHMARK_ROUTE_ID", "real_serial")
        os.environ.setdefault("ROUTE_NUMBER", "real_serial")
        os.environ.setdefault("IS_BENCH2DRIVE", "0")
        os.environ.setdefault("LEAD_PROJECT_ROOT", str(root))
        self._log_timing("runtime_env_defaults", step_start)

        step_start = time.perf_counter()
        with open(config_path, "r", encoding="utf-8") as file:
            training_payload = json.load(file)
        self._log_timing("read_training_config_json", step_start, config=config_path)

        # Real-COM preview favors compatibility over throughput. Some Windows
        # Torch builds cannot convert or run parts of this model in bfloat16.
        step_start = time.perf_counter()
        training_payload["compile"] = False
        training_payload["jit_compile"] = False
        training_payload["jit_compile_warmup_steps"] = 0
        training_payload["use_mixed_precision_training"] = False
        training_payload["channel_last"] = False
        training_payload["additional_metrics_dtype"] = "float32"
        training_payload["visualize_lidar_bev"] = False

        if self.device.type == "cpu":
            training_payload["inference_device"] = "cpu"
        else:
            training_payload["inference_device"] = str(self.device)
        self._log_timing(
            "apply_real_runtime_overrides",
            step_start,
            inference_device=training_payload["inference_device"],
        )

        step_start = time.perf_counter()
        self.training_config = TrainingConfig(training_payload)
        self.builder.configure_from_training(self.training_config)
        self.visualizer_cls = Visualizer
        self._log_timing(
            "build_training_config",
            step_start,
            image=f"{self.builder.image_width}x{self.builder.image_height}",
        )

        step_start = time.perf_counter()
        self.closed_loop_config = ClosedLoopConfig()
        self.expert_config = ExpertConfig()
        self._log_timing("build_control_configs", step_start)

        step_start = time.perf_counter()
        self.inference = ClosedLoopInference(
            config_training=self.training_config,
            config_closed_loop=self.closed_loop_config,
            config_expert=self.expert_config,
            model_path=str(checkpoint),
            device=self.device,
            prefix="model",
        )
        self._log_timing("closed_loop_inference", step_start, nets=len(self.inference.nets))

        if not self.inference.nets:
            raise RuntimeError(f"ClosedLoopInference loaded no model*.pth files from {checkpoint}")
        step_start = time.perf_counter()
        self._force_float32_modules()
        self._log_timing("force_float32_modules", step_start)

        self.checkpoint = str(checkpoint)
        self.config_path = str(config_path)
        self._frame_id = 0
        self._visualization_executor = ThreadPoolExecutor(
            max_workers=self._visualization_save_workers(),
            thread_name_prefix="real-visual-save",
        )
        self._visualization_futures: list[Future] = []
        self._log_timing("real_adapter_total", total_start, checkpoint=self.checkpoint)

    def _force_float32_modules(self) -> None:
        nets = getattr(self.inference, "nets", None)
        if not nets:
            return
        for net in nets:
            try:
                net.float()
            except Exception:
                pass

    @staticmethod
    def _to_float(value: Any) -> float:
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        if np is not None:
            value = np.asarray(value).reshape(-1)[0]
        return float(value)

    def _visualization_enabled(self) -> bool:
        if getattr(self, "visualizer_cls", None) is None:
            return False
        if bool(getattr(self.training_config, "disable_visual_artifacts", False)):
            return False
        value = os.environ.get("MITSU_REAL_VISUALIZATION", "").strip().lower()
        if value:
            return value in {"1", "true", "yes", "on"}
        return bool(getattr(self.closed_loop_config, "produce_debug_image", True))

    def _visualization_frequency(self) -> int:
        value = os.environ.get("MITSU_REAL_VISUALIZATION_FREQUENCY", "").strip()
        if value:
            return max(1, int(value))
        return max(1, int(getattr(self.training_config, "real_visualization_frequency", 5)))

    def _visualization_scale(self) -> float:
        return float(getattr(self.training_config, "visual_artifact_resolution_scale", 1.0))

    def _visualization_save_workers(self) -> int:
        return max(1, int(getattr(self.training_config, "visual_artifact_save_workers", 2)))

    def _visualization_dir(self) -> Path:
        save_path = os.environ.get("SAVE_PATH", "").strip()
        if save_path:
            root = Path(save_path)
        else:
            root = self.root / "outputs" / "real_model_inference"
        path = root / "real_debug_images"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _resize_visualization(self, image: "np.ndarray") -> "np.ndarray":
        scale = self._visualization_scale()
        if scale == 1.0:
            return image
        height, width = image.shape[:2]
        target = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        if target == (width, height):
            return image
        return cv2.resize(image, target, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _format_sample_time(value: Any) -> str:
        try:
            sample_ts = float(value)
        except Exception:
            sample_ts = 0.0
        if sample_ts <= 0.0:
            return "sample: n/a"
        return f"sample monotonic: {sample_ts:.6f}s"

    @staticmethod
    def _annotate_visualization(image: "np.ndarray", text: str) -> "np.ndarray":
        if cv2 is None or np is None or not text:
            return image
        annotated = image.copy()
        if annotated.ndim != 3 or annotated.shape[0] <= 0 or annotated.shape[1] <= 0:
            return annotated
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        margin = 10
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        box_w = min(annotated.shape[1], text_w + margin * 2)
        box_h = text_h + baseline + margin * 2
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)
        cv2.putText(annotated, text, (margin, margin + text_h), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return annotated

    @staticmethod
    def _write_visualization(path: str, image: "np.ndarray") -> None:
        ok = cv2.imwrite(
            path,
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_PNG_COMPRESSION), 0],
        )
        if not ok:
            raise OSError(f"Failed to write visualization: {path}")

    @staticmethod
    def _write_json(path: str, payload: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)

    def _prune_visualization_saves(self) -> None:
        pending = []
        for future in self._visualization_futures:
            if future.done():
                try:
                    future.result()
                except Exception as exc:
                    print(f"REAL MODEL VIS: async save failed: {exc}", flush=True)
            else:
                pending.append(future)
        self._visualization_futures = pending

    def _submit_visualization_save(self, path: Path, image: "np.ndarray", metadata: Optional[Dict[str, Any]] = None) -> None:
        self._prune_visualization_saves()
        max_pending = self._visualization_save_workers() * 8
        if len(self._visualization_futures) >= max_pending:
            done, pending = wait(
                self._visualization_futures,
                return_when=FIRST_COMPLETED,
            )
            for future in done:
                try:
                    future.result()
                except Exception as exc:
                    print(f"REAL MODEL VIS: async save failed: {exc}", flush=True)
            self._visualization_futures = list(pending)
        self._visualization_futures.append(
            self._visualization_executor.submit(
                self._write_visualization,
                str(path),
                image.copy(),
            )
        )
        if metadata is not None:
            self._visualization_futures.append(
                self._visualization_executor.submit(
                    self._write_json,
                    str(path.with_suffix(".json")),
                    dict(metadata),
                )
            )

    def _visualization_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        visual_data = dict(data)
        left = visual_data.get("rgb_left")
        right = visual_data.get("rgb_right")
        if left is not None and right is not None:
            visual_data["rgb"] = torch.cat([left, right], dim=3)
        elif left is not None:
            visual_data["rgb"] = left
        elif right is not None:
            visual_data["rgb"] = right
        return visual_data

    def _maybe_save_visualization(self, data: Dict[str, Any], prediction: Any) -> None:
        if cv2 is None or np is None or not self._visualization_enabled():
            return
        if self._frame_id % self._visualization_frequency() != 0:
            return
        try:
            image = self.visualizer_cls(
                config=self.training_config,
                data=self._visualization_data(data),
                prediction=prediction,
                config_test_time=self.closed_loop_config,
                test_time=True,
            ).visualize_inference_prediction()
            image = self._resize_visualization(np.asarray(image, dtype=np.uint8))
            out_path = self._visualization_dir() / f"{self._frame_id:06}.png"
            sample_created_at = data.get("input_sample_created_at_monotonic")
            label = self._format_sample_time(sample_created_at)
            image = self._annotate_visualization(image, label)
            metadata: Dict[str, Any] = {
                "frame_id": int(self._frame_id),
                "input_sample_created_at_monotonic": sample_created_at,
                "input_sample_time_label": label,
                "visualization_path": str(out_path),
            }
            for key in (
                "input_sample_seq",
                "input_sample_generation",
                "input_frame_timestamps_monotonic",
                "input_frame_part_sequences",
            ):
                if key in data:
                    metadata[key] = data.get(key)
            self._submit_visualization_save(out_path, image, metadata)
        except Exception as exc:
            print(f"REAL MODEL VIS: skipped frame {self._frame_id}: {exc}", flush=True)

    def close(self) -> None:
        futures = list(getattr(self, "_visualization_futures", []) or [])
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"REAL MODEL VIS: async save failed: {exc}", flush=True)
        self._visualization_futures = []
        executor = getattr(self, "_visualization_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def predict(self, frames: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = dict(context or {})
        data = self.builder.build_torch(frames, context, self.device)

        with torch.inference_mode():
            pred = self.inference.forward(data)

        self._frame_id += 1
        self._maybe_save_visualization(data, pred)

        steer = max(-1.0, min(1.0, self._to_float(pred.steer)))
        throttle = max(0.0, min(1.0, self._to_float(pred.throttle)))
        brake = max(0.0, min(1.0, self._to_float(pred.brake)))

        metadata: Dict[str, Any] = {
            "source": "model_0011_closed_loop",
            "checkpoint": self.checkpoint,
            "device": str(self.device),
        }

        if getattr(pred, "pred_target_speed_scalar", None) is not None:
            metadata["pred_target_speed_mps"] = self._to_float(pred.pred_target_speed_scalar)
        if getattr(pred, "route_steer", None) is not None:
            metadata["route_steer"] = float(pred.route_steer)
        if getattr(pred, "target_speed_throttle", None) is not None:
            metadata["target_speed_throttle"] = float(pred.target_speed_throttle)
        if getattr(pred, "target_speed_brake", None) is not None:
            metadata["target_speed_brake"] = float(pred.target_speed_brake)

        return {
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
            "confidence": 1.0,
            "timestamp_monotonic": time.monotonic(),
            "frame_id": self._frame_id,
            **metadata,
        }


def create_agent() -> LeadModel0011RealPortAdapter:
    return LeadModel0011RealPortAdapter()
