from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

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


def _project_root() -> Path:
    # <repo>/i-MiEV GUI/real_agent_adapters/lead_real_adapter.py
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


def _parse_shape(value: str, default: Sequence[int]) -> tuple[int, ...]:
    try:
        parts = [
            int(x.strip())
            for x in str(value).replace("x", ",").split(",")
            if x.strip()
        ]
        return tuple(parts) if parts else tuple(default)
    except Exception:
        return tuple(default)


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
    """Convert world/local-route coordinates into ego-relative coordinates.

    Ego vehicle is (0, 0). x is forward, y is lateral. This matches the model
    report requirement that target points are coordinates relative to the car.
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
class RealDualCameraInputBuilder:
    """Build model_0011 dual-front-camera inputs from two real camera streams."""

    image_size: int = int(os.environ.get("MITSU_REAL_MODEL_IMAGE_SIZE", "384"))
    device: str = os.environ.get("MITSU_REAL_MODEL_DEVICE", "cuda:0")
    town: str = os.environ.get("MITSU_REAL_TOWN", "RealWorld")
    default_command_one_hot: tuple[float, ...] = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

    def _resize_camera(self, frame):
        if cv2 is None or np is None:
            raise RuntimeError("OpenCV/numpy are required for real camera model input")

        if frame is None:
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        img = frame
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        img = cv2.resize(
            img,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        # cv2.imdecode gives BGR; model expects RGB tensors.
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _camera_to_chw(self, frame):
        img = self._resize_camera(frame)
        return np.transpose(img, (2, 0, 1)).astype(np.float32)

    def _command(self, context: Dict[str, Any], key: str) -> np.ndarray:
        value = context.get(key)
        if value is None:
            value = self.default_command_one_hot
        return np.asarray(value, dtype=np.float32).reshape(6)

    def _target_points(self, context: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        telemetry = context.get("telemetry")
        goal = context.get("goal")

        ego_x = _as_float(_read_attr_or_key(telemetry, "x_m", 0.0))
        ego_y = _as_float(_read_attr_or_key(telemetry, "y_m", 0.0))
        ego_yaw = _as_float(_read_attr_or_key(telemetry, "yaw_deg", 0.0))

        if goal is not None:
            target = _world_to_ego_local(
                _as_float(_read_attr_or_key(goal, "target_x_m", 0.0)),
                _as_float(_read_attr_or_key(goal, "target_y_m", 0.0)),
                ego_x,
                ego_y,
                ego_yaw,
            )
        else:
            # Fall back to context-provided target_point. It is expected to be
            # ego-relative if no goal object is supplied.
            target = context.get("target_point", [0.0, 0.0])

        # The car itself is the origin. If there is no reliable previous/next
        # route sample from NavigatorService, use origin and current target.
        previous = context.get("target_point_previous", [0.0, 0.0])
        if goal is not None:
            previous = [0.0, 0.0]

        next_point = context.get("target_point_next", target)
        if goal is not None:
            next_point = target

        previous = np.asarray(previous, dtype=np.float32).reshape(2)
        target = np.asarray(target, dtype=np.float32).reshape(2)
        next_point = np.asarray(next_point, dtype=np.float32).reshape(2)
        return previous, target, next_point

    def build_numpy(self, frames: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if np is None:
            raise RuntimeError("numpy is required for real camera model input")

        context = dict(context or {})

        rgb_left = self._camera_to_chw(frames.get("wide_90"))
        rgb_right = self._camera_to_chw(frames.get("narrow_50"))

        previous, target, next_point = self._target_points(context)

        speed_mps = context.get("speed_mps")
        if speed_mps is None:
            speed_mps = _as_float(context.get("speed_kmh", 0.0)) / 3.6

        return {
            "rgb_left": rgb_left,
            "rgb_right": rgb_right,
            "target_point_previous": previous,
            "target_point": target,
            "target_point_next": next_point,
            "speed": np.asarray([_as_float(speed_mps, 0.0)], dtype=np.float32),
            "command": self._command(context, "command_one_hot"),
            "next_command": self._command(context, "next_command_one_hot"),
            "town": np.asarray([str(context.get("town", self.town))]),
        }

    def build_torch(self, frames: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if torch is None:
            raise RuntimeError("torch is required for model_0011 inference")

        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "PyTorch CUDA is not available. model_0011 real-port inference "
                "requires an NVIDIA/CUDA runtime."
            )

        device = torch.device(self.device)
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


class LeadClosedLoopRealModel:
    """Loads outputs/model_0011 and returns normalized real-car control predictions."""

    def __init__(self, builder: RealDualCameraInputBuilder, checkpoint_dir: Optional[str] = None):
        if torch is None:
            raise RuntimeError("torch is required for model_0011 inference")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "PyTorch CUDA is not available. model_0011 real-port inference "
                "requires CUDA-enabled PyTorch."
            )

        root = _ensure_project_import_paths()

        from lead.expert.config_expert import ExpertConfig
        from lead.inference.closed_loop_inference import ClosedLoopInference
        from lead.inference.config_closed_loop import ClosedLoopConfig
        from lead.training.config_training import TrainingConfig

        self.builder = builder
        self.device = torch.device(builder.device)

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

        # Defaults used by ClosedLoopConfig lazy properties.
        os.environ.setdefault("SAVE_PATH", str(root / "outputs" / "real_model_inference"))
        os.environ.setdefault("BENCHMARK_ROUTE_ID", "real_serial")
        os.environ.setdefault("ROUTE_NUMBER", "real_serial")
        os.environ.setdefault("IS_BENCH2DRIVE", "0")

        with open(config_path, "r", encoding="utf-8") as file:
            training_payload = json.load(file)

        self.training_config = TrainingConfig(training_payload)
        self.closed_loop_config = ClosedLoopConfig()
        self.expert_config = ExpertConfig()

        self.inference = ClosedLoopInference(
            config_training=self.training_config,
            config_closed_loop=self.closed_loop_config,
            config_expert=self.expert_config,
            model_path=str(checkpoint),
            device=self.device,
            prefix="model",
        )

        if not self.inference.nets:
            raise RuntimeError(f"ClosedLoopInference loaded no model*.pth files from {checkpoint}")

        self.checkpoint = str(checkpoint)
        self.config_path = str(config_path)

    @staticmethod
    def _tensor_to_float(value: Any) -> float:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if np is not None:
            value = np.asarray(value).reshape(-1)[0]
        return float(value)

    def predict(self, input_tensors: Dict[str, Any]) -> Dict[str, Any]:
        with torch.inference_mode():
            pred = self.inference.forward(input_tensors)

        steer = max(-1.0, min(1.0, self._tensor_to_float(pred.steer)))
        throttle = max(0.0, min(1.0, self._tensor_to_float(pred.throttle)))
        brake = max(0.0, min(1.0, self._tensor_to_float(pred.brake)))

        metadata: Dict[str, Any] = {
            "source": "model_0011_closed_loop",
            "checkpoint": self.checkpoint,
        }

        if getattr(pred, "pred_target_speed_scalar", None) is not None:
            metadata["pred_target_speed_mps"] = self._tensor_to_float(pred.pred_target_speed_scalar)
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
            "metadata": metadata,
        }


class RealDualCameraAgentAdapter:
    """Default real-COM adapter.

    It always builds model_0011-compatible inputs and, by default, loads and runs
    outputs/model_0011. Set MITSU_REAL_MODEL_FACTORY only if you want to replace
    the model object for tests.
    """

    def __init__(self, model: Any = None, builder: Optional[RealDualCameraInputBuilder] = None):
        _ensure_project_import_paths()
        self.builder = builder or RealDualCameraInputBuilder()
        self.model = model if model is not None else self._load_model()
        self._frame_id = 0

    def _load_model(self):
        spec = os.environ.get("MITSU_REAL_MODEL_FACTORY", "").strip()
        if spec:
            module_name, func_name = spec.split(":", 1)
            module = __import__(module_name, fromlist=[func_name])
            factory = getattr(module, func_name)
            return factory(self.builder)
        return LeadClosedLoopRealModel(self.builder)

    @staticmethod
    def _prediction_to_dict(prediction: Any) -> Optional[Dict[str, Any]]:
        if prediction is None:
            return None
        if isinstance(prediction, dict):
            return dict(prediction)

        keys = ["steer", "throttle", "brake", "target_angle_deg", "confidence"]
        out: Dict[str, Any] = {}
        for key in keys:
            if hasattr(prediction, key):
                out[key] = float(getattr(prediction, key))
        return out or None

    def predict(self, frames: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        context = dict(context or {})
        inputs = self.builder.build_torch(frames, context)

        if hasattr(self.model, "predict"):
            try:
                pred = self.model.predict(inputs)
            except TypeError:
                pred = self.model.predict(frames, context)
        elif hasattr(self.model, "forward"):
            try:
                pred = self.model.forward(data=inputs)
            except TypeError:
                pred = self.model.forward(inputs)
        elif callable(self.model):
            pred = self.model(inputs)
        else:
            raise RuntimeError("Model object must implement predict(), forward() or __call__")

        result = self._prediction_to_dict(pred)
        if result:
            self._frame_id += 1
            result.setdefault("frame_id", self._frame_id)
            result.setdefault("timestamp_monotonic", time.monotonic())
            result.setdefault("confidence", 1.0)
        return result


def create_model(builder: Optional[RealDualCameraInputBuilder] = None):
    return LeadClosedLoopRealModel(builder or RealDualCameraInputBuilder())


def create_agent():
    return RealDualCameraAgentAdapter()
