from __future__ import annotations

import importlib
import os
import time
from dataclasses import dataclass
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


def _parse_shape(value: str, default: Sequence[int]) -> tuple[int, ...]:
    try:
        parts = [int(x.strip()) for x in str(value).replace("x", ",").split(",") if x.strip()]
        return tuple(parts) if parts else tuple(default)
    except Exception:
        return tuple(default)


@dataclass
class RealDualCameraInputBuilder:
    """Builds SensorAgent-like input tensors from two real cameras.

    The repository SensorAgent feeds ClosedLoopInference with keys: rgb,
    rasterized_lidar, target_point_previous, target_point, target_point_next,
    speed, command, next_command and town. This builder creates the same key set
    from real camera frames and the real NavigatorService context.
    """

    rgb_height: int = int(os.environ.get("MITSU_REAL_RGB_HEIGHT", "256"))
    rgb_width_per_camera: int = int(os.environ.get("MITSU_REAL_RGB_WIDTH_PER_CAMERA", "512"))
    jpeg_quality: int = int(os.environ.get("MITSU_REAL_MODEL_JPEG_QUALITY", "95"))
    simulate_jpeg: bool = os.environ.get("MITSU_REAL_MODEL_SIMULATE_JPEG", "1").lower() in {"1", "true", "yes", "on"}
    device: str = os.environ.get("MITSU_REAL_MODEL_DEVICE", "cuda:0" if torch is not None else "cpu")
    lidar_shape: tuple[int, ...] = _parse_shape(os.environ.get("MITSU_REAL_LIDAR_SHAPE", "1,1,256,256"), (1, 1, 256, 256))
    town: str = os.environ.get("MITSU_REAL_TOWN", "RealWorld")

    def _resize_camera(self, frame):
        if cv2 is None or np is None:
            raise RuntimeError("OpenCV/numpy are required for real camera model input")
        if frame is None:
            return np.zeros((self.rgb_height, self.rgb_width_per_camera, 3), dtype=np.uint8)
        img = frame
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.resize(img, (self.rgb_width_per_camera, self.rgb_height), interpolation=cv2.INTER_LINEAR)

    def build_numpy(self, frames: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if np is None or cv2 is None:
            raise RuntimeError("OpenCV/numpy are required for real camera model input")
        context = dict(context or {})
        wide = self._resize_camera(frames.get("wide_90"))
        narrow = self._resize_camera(frames.get("narrow_50"))
        rgb_bgr = np.concatenate([wide, narrow], axis=1)
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        if self.simulate_jpeg:
            ok, encoded = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)])
            if ok:
                rgb = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        rgb_chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32)

        def vec2(name: str):
            value = context.get(name, [0.0, 0.0])
            return np.asarray(value, dtype=np.float32).reshape(2)

        command = np.asarray(context.get("command_one_hot", [0, 0, 0, 1, 0, 0]), dtype=np.float32).reshape(6)
        next_command = np.asarray(context.get("next_command_one_hot", [0, 0, 0, 1, 0, 0]), dtype=np.float32).reshape(6)
        speed = float(context.get("speed_mps", 0.0) or 0.0)
        return {
            "rgb": rgb_chw,
            "rasterized_lidar": np.zeros(self.lidar_shape, dtype=np.float32),
            "target_point_previous": vec2("target_point_previous"),
            "target_point": vec2("target_point"),
            "target_point_next": vec2("target_point_next"),
            "speed": np.asarray([speed], dtype=np.float32),
            "command": command,
            "next_command": next_command,
            "town": np.asarray([str(context.get("town", self.town))]),
        }

    def build_torch(self, frames: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if torch is None:
            raise RuntimeError("torch is required for ClosedLoopInference")
        data = self.build_numpy(frames, context)
        device = torch.device(self.device if torch.cuda.is_available() or not str(self.device).startswith("cuda") else "cpu")
        return {
            "rgb": torch.tensor(data["rgb"], dtype=torch.float32, device=device)[None],
            "rasterized_lidar": torch.tensor(data["rasterized_lidar"], dtype=torch.float32, device=device),
            "target_point_previous": torch.tensor(data["target_point_previous"], dtype=torch.float32, device=device).view(1, 2),
            "target_point": torch.tensor(data["target_point"], dtype=torch.float32, device=device).view(1, 2),
            "target_point_next": torch.tensor(data["target_point_next"], dtype=torch.float32, device=device).view(1, 2),
            "speed": torch.tensor(data["speed"], dtype=torch.float32, device=device).view(1),
            "command": torch.tensor(data["command"], dtype=torch.float32, device=device).view(1, 6),
            "next_command": torch.tensor(data["next_command"], dtype=torch.float32, device=device).view(1, 6),
            "town": data["town"],
        }


class RealDualCameraAgentAdapter:
    """Adapter used by MITSU_REAL_AGENT_FACTORY.

    By default it only validates and builds model inputs. To run an actual model,
    set MITSU_REAL_MODEL_FACTORY=module:function. The factory may return an object
    with predict(input_tensors) or forward(data=input_tensors). The returned object
    or dict must contain steer/throttle/brake.
    """

    def __init__(self, model: Any = None, builder: Optional[RealDualCameraInputBuilder] = None):
        self.builder = builder or RealDualCameraInputBuilder()
        self.model = model if model is not None else self._load_model_factory()
        self._frame_id = 0

    def _load_model_factory(self):
        spec = os.environ.get("MITSU_REAL_MODEL_FACTORY", "").strip()
        if not spec:
            return None
        module_name, func_name = spec.split(":", 1)
        module = importlib.import_module(module_name)
        factory = getattr(module, func_name)
        return factory(self.builder)

    @staticmethod
    def _prediction_to_dict(prediction: Any) -> Optional[Dict[str, Any]]:
        if prediction is None:
            return None
        if isinstance(prediction, dict):
            return dict(prediction)
        keys = ["steer", "throttle", "brake", "target_angle_deg", "confidence"]
        out = {}
        for key in keys:
            if hasattr(prediction, key):
                out[key] = float(getattr(prediction, key))
        return out or None

    def predict(self, frames: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        context = dict(context or {})
        # Always build inputs; this catches camera/order/shape mistakes during mock tests.
        inputs = self.builder.build_torch(frames, context) if torch is not None else self.builder.build_numpy(frames, context)
        if self.model is None:
            return None
        if hasattr(self.model, "predict"):
            pred = self.model.predict(inputs)
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


def create_agent():
    return RealDualCameraAgentAdapter()
