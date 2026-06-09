from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from importlib.util import find_spec

import jaxtyping as jt
import numpy as np
import torch
import torch.nn.functional as F
from beartype import beartype

from lead.common.constants import TransfuserBoundingBoxIndex
from lead.data_loader import carla_dataset_utils
from lead.inference import inference_utils
from lead.inference.config_open_loop import OpenLoopConfig
from lead.tfv6.center_net_decoder import PredictedBoundingBox
from lead.tfv6.planning_decoder import decode_two_hot
from lead.tfv6.tfv6 import Prediction, TFv6
from lead.training.config_training import TrainingConfig

np.set_printoptions(suppress=True)

LOG = logging.getLogger(__name__)


class OpenLoopInference:
    @beartype
    def __init__(
        self,
        config_training: TrainingConfig,
        config_open_loop: OpenLoopConfig,
        model_path: str,
        device: torch.device,
        prefix: str = "model",
    ):
        """
        Open-Loop-Inference constructor.

        Args:
            config_training: Training config object belong to model.
            config_open_loop: Open loop config object.
            model_path: Path to the trained model weights.
            device: Device to run inference on.
            prefix: Prefix of the model weights files to load.
        """
        self.config_training = config_training
        self.config_open_loop = config_open_loop
        self.device = device
        self._startup_timing_enabled = bool(
            getattr(self.config_training, "model_startup_timing", False)
        )
        startup_total_start = time.perf_counter()

        # Loading models
        self.nets: list[TFv6] = []
        model_files = [
            file
            for file in sorted(os.listdir(model_path))
            if file.startswith(prefix) and file.endswith(".pth")
        ]
        self._log_startup_timing(
            "scan_model_files",
            startup_total_start,
            files=len(model_files),
            model_path=model_path,
            prefix=prefix,
        )
        for file in model_files:
            model_file_path = os.path.join(model_path, file)
            model_start = time.perf_counter()
            LOG.info(f"Loading model weight from {model_file_path}")

            step_start = time.perf_counter()
            net = TFv6(self.device, self.config_training)
            self._log_startup_timing(file, step_start, stage="build_model")

            if self.config_training.sync_batchnorm:
                step_start = time.perf_counter()
                net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
                self._log_startup_timing(file, step_start, stage="sync_batchnorm")

            step_start = time.perf_counter()
            state_dict = torch.load(
                model_file_path,
                map_location=self.device,
                weights_only=True,
            )
            self._log_startup_timing(file, step_start, stage="load_weights_from_disk")

            step_start = time.perf_counter()
            net.load_state_dict(state_dict, strict=config_open_loop.strict_weight_load)
            self._log_startup_timing(file, step_start, stage="load_state_dict")

            step_start = time.perf_counter()
            net.to(device=self.device).eval()
            self._sync_device()
            self._log_startup_timing(
                file,
                step_start,
                stage="move_to_device_and_eval",
                device=str(self.device),
            )

            if self.config_training.channel_last:
                step_start = time.perf_counter()
                net.to(memory_format=torch.channels_last)
                self._sync_device()
                self._log_startup_timing(file, step_start, stage="channels_last")

            step_start = time.perf_counter()
            self._prepare_jit_compile(net)
            self._log_startup_timing(file, step_start, stage="prepare_jit_compile")

            self.nets.append(net)
            self._log_startup_timing(file, model_start, stage="model_total")

        step_start = time.perf_counter()
        self._warmup_jit_compile()
        self._log_startup_timing("warmup_jit_compile", step_start)
        self._log_startup_timing("open_loop_inference_total", startup_total_start)
        self.step = 4  # Constant so produced images start with 5, not really important

    def _sync_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _log_startup_timing(
        self,
        name: str,
        start_s: float,
        **fields: object,
    ) -> None:
        if not getattr(self, "_startup_timing_enabled", False):
            return
        self._sync_device()
        elapsed_ms = (time.perf_counter() - start_s) * 1000.0
        suffix = ""
        if fields:
            suffix = " " + " ".join(f"{key}={value}" for key, value in fields.items())
        LOG.info("[ModelStartupTiming] %s elapsed=%.3f ms%s", name, elapsed_ms, suffix)

    def _jit_compile_mode(self) -> str:
        configured_mode = getattr(self.config_training, "jit_compile_mode", None)
        if configured_mode is None:
            configured_mode = getattr(
                self.config_training,
                "compile_mode",
                "reduce-overhead",
            )
        return str(configured_mode).lower()

    def _prepare_jit_compile(self, net: TFv6) -> None:
        if not bool(getattr(self.config_training, "jit_compile", False)):
            return
        if self.device.type != "cuda":
            LOG.warning("Skipping torch.compile because inference device is not CUDA.")
            return
        if find_spec("triton") is None:
            LOG.warning(
                "Skipping torch.compile because Triton is not installed in the active Python environment."
            )
            return
        jit_compile_mode = self._jit_compile_mode()
        net.prepare_compile(
            fullgraph=False,
            dynamic=False,
            backend="inductor",
            mode=jit_compile_mode,
        )
        LOG.info(
            "Using torch.compile on TFv6 forward core for inference, mode=%s",
            jit_compile_mode,
        )

    def _dummy_forward_data(self) -> dict[str, torch.Tensor | np.ndarray]:
        batch_size = 1
        data: dict[str, torch.Tensor | np.ndarray] = {
            "target_point_previous": torch.zeros(
                (batch_size, 2), device=self.device, dtype=torch.float32
            ),
            "target_point": torch.zeros(
                (batch_size, 2), device=self.device, dtype=torch.float32
            ),
            "target_point_next": torch.zeros(
                (batch_size, 2), device=self.device, dtype=torch.float32
            ),
            "speed": torch.zeros((batch_size,), device=self.device, dtype=torch.float32),
            "command": torch.zeros(
                (batch_size, self.config_training.discrete_command_dim),
                device=self.device,
                dtype=torch.float32,
            ),
            "next_command": torch.zeros(
                (batch_size, self.config_training.discrete_command_dim),
                device=self.device,
                dtype=torch.float32,
            ),
            "town": np.array(["Town01"]),
        }
        data["command"][:, 3] = 1.0
        data["next_command"][:, 3] = 1.0

        if bool(getattr(self.config_training, "dual_front_camera_mode", False)):
            for key in (
                str(getattr(self.config_training, "left_camera_key", "rgb_left")),
                str(getattr(self.config_training, "right_camera_key", "rgb_right")),
            ):
                data[key] = torch.zeros(
                    (
                        batch_size,
                        3,
                        self.config_training.final_image_height,
                        self.config_training.final_image_width,
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )
        else:
            data["rgb"] = torch.zeros(
                (
                    batch_size,
                    3,
                    self.config_training.final_image_height,
                    self.config_training.final_image_width,
                ),
                device=self.device,
                dtype=torch.float32,
            )
            if not self.config_training.LTF:
                data["rasterized_lidar"] = torch.zeros(
                    (
                        batch_size,
                        1,
                        self.config_training.lidar_height_pixel,
                        self.config_training.lidar_width_pixel,
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )

        if self.config_training.use_radars:
            data["radar"] = torch.zeros(
                (
                    batch_size,
                    self.config_training.num_radar_sensors
                    * self.config_training.num_radar_points_per_sensor,
                    5,
                ),
                device=self.device,
                dtype=torch.float32,
            )
        return data

    def _warmup_jit_compile(self) -> None:
        warmup_steps = int(getattr(self.config_training, "jit_compile_warmup_steps", 0))
        if (
            warmup_steps <= 0
            or not bool(getattr(self.config_training, "jit_compile", False))
            or len(self.nets) == 0
        ):
            return

        data = self._dummy_forward_data()
        use_autocast = self._uses_autocast()
        LOG.info("Running %d torch.compile warmup forward(s).", warmup_steps)
        with torch.inference_mode(), torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.config_training.torch_float_type,
            enabled=use_autocast,
        ):
            for step_index in range(warmup_steps):
                for net_index, net in enumerate(self.nets):
                    step_start = time.perf_counter()
                    try:
                        net(data)
                    except Exception as exc:
                        for fallback_net in self.nets:
                            fallback_net.disable_compile()
                        self.config_training.jit_compile = False
                        LOG.warning(
                            "Disabling torch.compile after warmup failed; falling back to eager inference. Error: %s",
                            exc,
                        )
                        return
                    self._sync_device()
                    self._log_startup_timing(
                        "jit_compile_warmup_forward",
                        step_start,
                        step=step_index + 1,
                        net=net_index,
                    )
        self._sync_device()

    def _uses_autocast(self) -> bool:
        return (
            self.device.type == "cuda"
            and bool(
                getattr(self.config_training, "use_mixed_precision_training", False)
            )
            and self.config_training.torch_float_type != torch.float32
        )

    @beartype
    def ensemble_planning_decoder(
        self, predictions: list[Prediction]
    ) -> tuple[
        jt.Float[torch.Tensor, "1 num_waypoints 2"] | None,
        jt.Float[torch.Tensor, "1 num_checkpoints 2"] | None,
        jt.Float[torch.Tensor, " 1 1"] | None,
        jt.Float[torch.Tensor, "1 num_speed_classes"] | None,
        jt.Float[torch.Tensor, "1 num_waypoints"] | None,
    ]:
        """Ensemble the outputs of the planning decoder from multiple models.

        Args:
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            pred_routes: The aggregated route.
            pred_future_waypoints: The aggregated future waypoints.
            pred_target_speed_scalar: The aggregated target speed.
            pred_target_speed_distribution: The aggregated target speed distribution.
        """
        pred_routes = pred_future_waypoints = pred_target_speed_scalar = (
            pred_target_speed_distribution
        ) = pred_future_headings = None

        if self.config_training.use_planning_decoder:
            if self.config_training.predict_target_speed:
                pred_target_speed_logits = torch.stack(
                    [pred.pred_target_speed_distribution[0] for pred in predictions]
                ).mean(dim=0, keepdim=True)  # Average target speed logits.

                pred_target_speed_distribution = F.softmax(
                    pred_target_speed_logits, dim=-1
                )  # softmax probabilities.
                pred_target_speed_scalar = decode_two_hot(
                    pred_target_speed_distribution,
                    self.config_training.target_speed_classes,
                    self.device,
                ).reshape(1, 1)  # Decode to scalar.
                if (
                    pred_target_speed_distribution[0, 0]
                    > self.config_open_loop.brake_threshold
                ):  # Brake if we are confident enough.
                    pred_target_speed_scalar = torch.Tensor([0.0]).reshape(1, -1)
                if (
                    self.config_open_loop.lower_target_speed
                ):  # Optionally lower the target speed.
                    pred_target_speed_scalar *= (
                        self.config_open_loop.lower_target_speed_factor
                    )

            if self.config_training.predict_temporal_spatial_waypoints:
                pred_future_waypoints = torch.stack(
                    [pred.pred_future_waypoints[0] for pred in predictions]
                ).mean(dim=0, keepdim=True)  # Average waypoints.

            if self.config_training.predict_spatial_path:
                pred_routes = torch.stack(
                    [pred.pred_route[0] for pred in predictions]
                ).mean(dim=0, keepdim=True)  # Average route.

            if (
                self.config_training.use_navsim_data
                and predictions[0].pred_headings is not None
            ):
                pred_future_headings = torch.stack(
                    [pred.pred_headings[0] for pred in predictions]
                ).mean(dim=0, keepdim=True)  # Average headings.

        return (
            pred_routes,
            pred_future_waypoints,
            pred_target_speed_scalar,
            pred_target_speed_distribution,
            pred_future_headings,
        )

    @beartype
    def ensemble_bounding_boxes(
        self, predictions: list[Prediction]
    ) -> tuple[list[PredictedBoundingBox], list[PredictedBoundingBox]]:
        """
        Args:
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            List of aggregated bounding boxes in vehicle system.
            List of aggregated bounding boxes in image system.
        """
        pred_bounding_boxes_vehicle_system, pred_bounding_boxes_image_system = [], []
        if self.config_training.detect_boxes:
            for prediction in predictions:
                pred_bb = prediction.pred_bounding_box.pred_bounding_box_vehicle_system.squeeze().reshape(
                    -1, 9
                )
                if len(pred_bb) > 0:
                    pred_bounding_boxes_vehicle_system.append(pred_bb)

        if len(pred_bounding_boxes_vehicle_system) > 0:
            pred_bounding_boxes_vehicle_system = (
                inference_utils.non_maximum_suppression(
                    pred_bounding_boxes_vehicle_system,
                    float(self.config_training.iou_threshold_nms),
                )
            )

            pred_bounding_boxes_image_system = (
                carla_dataset_utils.bb_vehicle_to_image_system(
                    pred_bounding_boxes_vehicle_system,
                    self.config_training.pixels_per_meter,
                    self.config_training.min_x_meter,
                    self.config_training.min_y_meter,
                )
            )

            pred_bounding_boxes_vehicle_system = [
                PredictedBoundingBox(
                    x=float(bb[TransfuserBoundingBoxIndex.X]),
                    y=float(bb[TransfuserBoundingBoxIndex.Y]),
                    w=float(bb[TransfuserBoundingBoxIndex.W]),
                    h=float(bb[TransfuserBoundingBoxIndex.H]),
                    yaw=float(bb[TransfuserBoundingBoxIndex.YAW]),
                    velocity=float(bb[TransfuserBoundingBoxIndex.VELOCITY]),
                    brake=float(bb[TransfuserBoundingBoxIndex.BRAKE]),
                    clazz=int(bb[TransfuserBoundingBoxIndex.CLASS]),
                    score=float(bb[TransfuserBoundingBoxIndex.SCORE]),
                )
                for bb in pred_bounding_boxes_vehicle_system
            ]

            pred_bounding_boxes_image_system = [
                PredictedBoundingBox(
                    x=float(bb[TransfuserBoundingBoxIndex.X]),
                    y=float(bb[TransfuserBoundingBoxIndex.Y]),
                    w=float(bb[TransfuserBoundingBoxIndex.W]),
                    h=float(bb[TransfuserBoundingBoxIndex.H]),
                    yaw=float(bb[TransfuserBoundingBoxIndex.YAW]),
                    velocity=float(bb[TransfuserBoundingBoxIndex.VELOCITY]),
                    brake=float(bb[TransfuserBoundingBoxIndex.BRAKE]),
                    clazz=int(bb[TransfuserBoundingBoxIndex.CLASS]),
                    score=float(bb[TransfuserBoundingBoxIndex.SCORE]),
                )
                for bb in pred_bounding_boxes_image_system
            ]

        return pred_bounding_boxes_vehicle_system, pred_bounding_boxes_image_system

    @beartype
    def ensemble_bev_semantic(
        self, predictions: list[Prediction]
    ) -> jt.Float[torch.Tensor, "B num_classes bev_height bev_width"] | None:
        """
        Args:
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            pred_bev_semantic: Tensor containing the aggregated BEV semantic map
        """
        if self.config_training.use_bev_semantic:
            pred_bev_semantic = []
            for prediction in predictions:
                pred_bev_semantic.append(prediction.pred_bev_semantic)
            stacked = torch.stack(
                pred_bev_semantic, dim=0
            )  # (num_models, num_batches, num_classes, H, W)
            ch0 = (
                stacked[:, :, 0].min(dim=0).values.unsqueeze(1)
            )  # (num_batches, 1, H, W)
            others = (
                stacked[:, :, 1:].max(dim=0).values
            )  # (num_batches, num_classes-1, H, W)
            return torch.cat([ch0, others], dim=1)  # (num_batches, num_classes, H, W)
        return None

    @beartype
    def ensemble_depth(
        self, predictions: list[Prediction]
    ) -> jt.Float[torch.Tensor, "B img_height img_width"] | None:
        """
        Args:
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            pred_depth: Tensor containing the aggregated depth map
        """
        if self.config_training.use_depth:
            pred_depth = []
            for prediction in predictions:
                pred_depth.append(prediction.pred_depth)
            stacked = torch.stack(pred_depth, dim=0)  # (num_models, num_batches, H, W)
            return stacked.mean(dim=0)  # (num_batches, H, W)
        return None

    @beartype
    def ensemble_semantic_segmentation(
        self, predictions: list[Prediction]
    ) -> jt.Float[torch.Tensor, "B num_classes img_height img_width"] | None:
        """
        Args:
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            pred_semantic: Tensor containing the aggregated semantic segmentation map
        """
        if self.config_training.use_semantic:
            pred_semantic = []
            for prediction in predictions:
                pred_semantic.append(prediction.pred_semantic)
            stacked = torch.stack(
                pred_semantic, dim=0
            )  # (num_models, num_batches, num_classes, H, W)
            ch0 = (
                stacked[:, :, 0].min(dim=0).values.unsqueeze(1)
            )  # (num_batches, 1, H, W)
            others = (
                stacked[:, :, 1:].max(dim=0).values
            )  # (num_batches, num_classes-1, H, W)
            return torch.cat([ch0, others], dim=1)  # (num_batches, num_classes, H, W)
        return None

    @beartype
    def ensemble(self, _, predictions: list[Prediction]) -> OpenLoopPrediction:
        """
        Args:
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            EnsemblePrediction object containing the aggregated predictions
        """
        # Bounding boxes
        pred_bounding_boxes_vehicle_system, pred_bounding_boxes_image_system = (
            None,
            None,
        )
        if self.config_training.carla_leaderboard_mode:
            pred_bounding_boxes_vehicle_system, pred_bounding_boxes_image_system = (
                self.ensemble_bounding_boxes(predictions)
            )

        # BEV semantic map
        pred_bev_semantic = None
        if self.config_training.carla_leaderboard_mode:
            pred_bev_semantic = self.ensemble_bev_semantic(predictions)

        # Semantic segmentation
        pred_semantic = None
        if self.config_training.carla_leaderboard_mode:
            pred_semantic = self.ensemble_semantic_segmentation(predictions)

        # Depth
        pred_depth = None
        if self.config_training.carla_leaderboard_mode:
            pred_depth = self.ensemble_depth(predictions)

        # Planning
        (
            pred_route,
            pred_future_waypoints,
            pred_target_speed_scalar,
            pred_target_speed_distribution,
            pred_future_headings,
        ) = self.ensemble_planning_decoder(predictions)

        return OpenLoopPrediction(
            pred_future_waypoints=pred_future_waypoints,
            pred_target_speed_scalar=pred_target_speed_scalar,
            pred_target_speed_distribution=pred_target_speed_distribution,
            pred_future_headings=pred_future_headings,
            pred_route=pred_route,
            pred_semantic=pred_semantic,
            pred_depth=pred_depth,
            pred_bev_semantic=pred_bev_semantic,
            pred_bounding_box_vehicle_system=pred_bounding_boxes_vehicle_system,
            pred_bounding_box_image_system=pred_bounding_boxes_image_system,
            pred_radar_predictions=None,
        )

    @beartype
    @torch.inference_mode()
    def forward(self, data: dict[str, torch.Tensor]) -> OpenLoopPrediction:
        """Run inference on the ensemble of models.
        Args:
            data: Dictionary containing the input data for the model

        Returns:
            EnsemblePrediction object containing the aggregated predictions
        """
        self.step += 1
        use_autocast = self._uses_autocast()
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.config_training.torch_float_type,
            enabled=use_autocast,
        ):
            self.predictions: list[Prediction] = [net(data) for net in self.nets]
        return self.ensemble(data, self.predictions)

    def __getitem__(self, index):
        return self.nets[index]


@jt.jaxtyped(typechecker=beartype)
@dataclass
class OpenLoopPrediction:
    """Raw output predictions from the open loop model."""

    pred_future_waypoints: jt.Float[torch.Tensor, "bs n_waypoints 2"] | None
    pred_future_headings: jt.Float[torch.Tensor, "bs n_waypoints"] | None
    pred_target_speed_scalar: jt.Float[torch.Tensor, "bs 1"] | None
    pred_target_speed_distribution: (
        jt.Float[torch.Tensor, "bs num_speed_classes"] | None
    )
    pred_route: jt.Float[torch.Tensor, "bs n_checkpoints 2"] | None
    pred_semantic: (
        jt.Float[torch.Tensor, "bs num_sem_classes img_height img_width"] | None
    )
    pred_depth: jt.Float[torch.Tensor, "bs img_height img_width"] | None
    pred_bev_semantic: (
        jt.Float[torch.Tensor, "bs num_bev_classes bev_height bev_width"] | None
    )
    pred_bounding_box_vehicle_system: list[PredictedBoundingBox] | None
    pred_bounding_box_image_system: list[PredictedBoundingBox] | None
    pred_radar_predictions: None
