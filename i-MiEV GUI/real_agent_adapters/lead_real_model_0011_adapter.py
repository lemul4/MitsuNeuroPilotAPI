from __future__ import annotations

import json
import math
import os
import sys
import time
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

    def __init__(
        self,
        builder: Optional[RealModel0011InputBuilder] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        if torch is None:
            raise RuntimeError("torch is required for model_0011 inference")

        root = _ensure_project_import_paths()

        _mitsu_install_carla_shims()

        from lead.expert.config_expert import ExpertConfig
        _mitsu_install_carla_shims()

        from lead.inference.closed_loop_inference import ClosedLoopInference
        from lead.inference.config_closed_loop import ClosedLoopConfig
        from lead.training.config_training import TrainingConfig

        self.root = root
        self.device = _select_device()
        self.builder = builder or RealModel0011InputBuilder()

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

        os.environ.setdefault("SAVE_PATH", str(root / "outputs" / "real_model_inference"))
        os.environ.setdefault("BENCHMARK_ROUTE_ID", "real_serial")
        os.environ.setdefault("ROUTE_NUMBER", "real_serial")
        os.environ.setdefault("IS_BENCH2DRIVE", "0")
        os.environ.setdefault("LEAD_PROJECT_ROOT", str(root))

        with open(config_path, "r", encoding="utf-8") as file:
            training_payload = json.load(file)

        # CPU-safe inference overrides. They do not change the weights.
        if self.device.type == "cpu":
            training_payload["inference_device"] = "cpu"
            training_payload["compile"] = False
            training_payload["jit_compile"] = False
            training_payload["jit_compile_warmup_steps"] = 0
            training_payload["use_mixed_precision_training"] = False
            training_payload["channel_last"] = False
        else:
            training_payload["inference_device"] = str(self.device)

        self.training_config = TrainingConfig(training_payload)
        self.builder.configure_from_training(self.training_config)

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
        self._frame_id = 0

    @staticmethod
    def _to_float(value: Any) -> float:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if np is not None:
            value = np.asarray(value).reshape(-1)[0]
        return float(value)

    def predict(self, frames: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = dict(context or {})
        data = self.builder.build_torch(frames, context, self.device)

        with torch.inference_mode():
            pred = self.inference.forward(data)

        self._frame_id += 1

        steer = max(-1.0, min(1.0, float(pred.steer)))
        throttle = max(0.0, min(1.0, float(pred.throttle)))
        brake = max(0.0, min(1.0, float(pred.brake)))

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
