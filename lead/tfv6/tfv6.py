from __future__ import annotations

import typing
from dataclasses import dataclass

import jaxtyping as jt
import torch
from beartype import beartype
from torch import nn

from lead.common.constants import SourceDataset
from lead.tfv6.bev_decoder import BEVDecoder
from lead.tfv6.center_net_decoder import (
    CenterNetBoundingBoxPrediction,
    CenterNetDecoder,
)
from lead.tfv6.perspective_decoder import PerspectiveDecoder
from lead.tfv6.planning_decoder import PlanningDecoder
from lead.tfv6.radar_detector import RadarDetector
from lead.tfv6.tfv5_planning_decoder import TFv5PlanningDecoder
from lead.tfv6.transfuser_backbone import TransfuserBackbone
from lead.training.config_training import TrainingConfig


class TFv6(nn.Module):
    @beartype
    def __init__(
        self,
        device: torch.device,
        config: TrainingConfig,
    ):
        super().__init__()
        self.device = device
        self.config = config
        self.log = {}
        self._compiled_forward_core = None
        self._compiled_forward_core_is_dual_static = False
        self.use_carla_data = bool(self.config.use_carla_data)
        self.use_navsim_data = bool(self.config.use_navsim_data)
        self.use_semantic = bool(self.config.use_semantic)
        self.use_depth = bool(self.config.use_depth)
        self.use_bev_semantic = bool(self.config.use_bev_semantic)
        self.detect_boxes = bool(self.config.detect_boxes)
        self.radar_detection = bool(self.config.radar_detection)
        self.use_radar_detection = bool(self.config.use_radar_detection)
        self.use_planning_decoder = bool(self.config.use_planning_decoder)

        self.backbone = TransfuserBackbone(self.device, self.config)

        if self.config.use_semantic and self.config.use_carla_data:
            self.semantic_decoder = PerspectiveDecoder(
                config=self.config,
                in_channels=self.backbone.num_image_features,
                out_channels=self.config.num_semantic_classes,
                perspective_upsample_factor=self.backbone.perspective_upsample_factor,
                modality="semantic",
                device=self.device,
                source_data=SourceDataset.CARLA,
            )

        if self.config.use_depth and self.config.use_carla_data:
            self.depth_decoder = PerspectiveDecoder(
                config=self.config,
                in_channels=self.backbone.num_image_features,
                out_channels=1,
                perspective_upsample_factor=self.backbone.perspective_upsample_factor,
                modality="depth",
                device=self.device,
                source_data=SourceDataset.CARLA,
            )

        if self.config.use_bev_semantic:
            if self.config.use_carla_data:
                self.bev_semantic_decoder = BEVDecoder(
                    self.config,
                    self.config.num_bev_semantic_classes,
                    self.device,
                    source_data=SourceDataset.CARLA,
                )

            if self.config.use_navsim_data:
                self.bev_semantic_decoder_navsim = BEVDecoder(
                    self.config,
                    self.config.navsim_num_bev_semantic_classes,
                    self.device,
                    source_data=SourceDataset.NAVSIM,
                )

        if self.config.detect_boxes:
            if self.config.use_carla_data:
                self.center_net_decoder = CenterNetDecoder(
                    self.config.num_bb_classes,
                    self.config,
                    self.device,
                    source_data=SourceDataset.CARLA,
                )
            if self.config.use_navsim_data:
                self.center_net_decoder_navsim = CenterNetDecoder(
                    self.config.navsim_num_bb_classes,
                    self.config,
                    self.device,
                    source_data=SourceDataset.NAVSIM,
                )

        if self.config.radar_detection and self.config.use_carla_data:
            self.radar_detector = RadarDetector(
                bev_input_dim=self.backbone.num_lidar_features,
                config=self.config,
                device=self.device,
            )

        if self.config.use_planning_decoder:
            if self.config.use_tfv5_planning_decoder:
                self.planning_decoder = TFv5PlanningDecoder(
                    input_bev_channels=self.backbone.num_lidar_features,
                    config=self.config,
                    device=self.device,
                ).to(self.device)
            else:
                self.planning_decoder = PlanningDecoder(
                    input_bev_channels=self.backbone.num_lidar_features,
                    config=self.config,
                    device=self.device,
                ).to(self.device)

    def prepare_compile(
        self,
        fullgraph: bool = True,
        dynamic: bool = False,
        backend: str = "inductor",
        mode: str = "max-autotune-no-cudagraphs",
    ) -> None:
        forward_core = self._forward_core
        self._compiled_forward_core_is_dual_static = False
        if self._can_use_dual_front_static_core():
            forward_core = self._forward_dual_front_static_core
            self._compiled_forward_core_is_dual_static = True
        self._compiled_forward_core = torch.compile(
            forward_core,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
        )

    def forward(self, data: dict[str, typing.Any]) -> Prediction:
        self.log = {}
        if self._can_use_dual_front_static_core() and not self._has_dynamic_camera_data(
            data
        ):
            left = data[self.backbone.left_camera_key].to(
                self.device, dtype=self.backbone.input_dtype, non_blocking=True
            )
            right = data[self.backbone.right_camera_key].to(
                self.device, dtype=self.backbone.input_dtype, non_blocking=True
            )
            forward_core = (
                self._compiled_forward_core
                if self._compiled_forward_core_is_dual_static
                else self._forward_dual_front_static_core
            )
            core_outputs = forward_core(left, right)
        else:
            forward_core = (
                self._compiled_forward_core
                if self._compiled_forward_core is not None
                and not self._compiled_forward_core_is_dual_static
                else self._forward_core
            )
            core_data = self._prepare_forward_data(data)
            core_outputs = forward_core(core_data)
        (
            pred_route,
            pred_future_waypoints,
            pred_target_speed_distribution,
            pred_target_speed_scalar,
            pred_headings,
            pred_semantic,
            pred_depth,
            pred_bounding_box_tensors,
            pred_bev_semantic,
            radar_features,
            radar_predictions,
            pred_bounding_box_navsim_tensors,
            pred_bev_semantic_navsim,
        ) = core_outputs
        pred_bounding_box = self._make_center_net_prediction(
            pred_bounding_box_tensors, self.config
        )
        pred_bounding_box_navsim = self._make_center_net_prediction(
            pred_bounding_box_navsim_tensors, self.config
        )

        return Prediction(
            # Planning prediction
            pred_future_waypoints=pred_future_waypoints,
            pred_target_speed_distribution=pred_target_speed_distribution,
            pred_target_speed_scalar=pred_target_speed_scalar,
            pred_route=pred_route,
            # CARLA perception prediction
            pred_semantic=pred_semantic,
            pred_depth=pred_depth,
            pred_bounding_box=pred_bounding_box,
            pred_bev_semantic=pred_bev_semantic,
            pred_radar_features=radar_features,
            pred_radar_predictions=radar_predictions,
            # NavSim perception prediction
            pred_bounding_box_navsim=pred_bounding_box_navsim,
            pred_bev_semantic_navsim=pred_bev_semantic_navsim,
            pred_headings=pred_headings,
        )

    def _can_use_dual_front_static_core(self) -> bool:
        return (
            self.backbone.backbone_sensor_mode == "dual_front_camera"
            and self.use_carla_data
            and not self.use_navsim_data
            and not self.radar_detection
            and not self.use_planning_decoder
        )

    @staticmethod
    def _has_dynamic_camera_data(data: dict[str, typing.Any]) -> bool:
        for camera_name in ("left", "right"):
            for key in (
                f"{camera_name}_camera_intrinsics",
                f"{camera_name}_intrinsics",
                f"rgb_{camera_name}_intrinsics",
                f"{camera_name}_camera_T_cam_ego",
                f"{camera_name}_T_cam_ego",
                f"rgb_{camera_name}_T_cam_ego",
            ):
                if key in data:
                    return True
        return False

    def _forward_dual_front_static_core(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> tuple:
        pred_semantic = pred_depth = pred_bounding_box = pred_bev_semantic = None
        bev_features, image_features = self.backbone._forward_dual_camera(
            left,
            right,
            data=None,
        )

        if self.use_semantic:
            pred_semantic = self.semantic_decoder(None, image_features, None)
        if self.use_depth:
            pred_depth = self.depth_decoder(None, image_features, None)

        bev_feature_grid = self.backbone.top_down(bev_features)
        if self.detect_boxes:
            pred_bounding_box = self.center_net_decoder.forward_tensors(
                bev_feature_grid
            )
        if self.use_bev_semantic:
            pred_bev_semantic = self.bev_semantic_decoder(bev_feature_grid, None)

        return (
            None,
            None,
            None,
            None,
            None,
            pred_semantic,
            pred_depth,
            pred_bounding_box,
            pred_bev_semantic,
            None,
            None,
            None,
            None,
        )

    def _forward_core(self, data: dict[str, typing.Any]) -> tuple:
        pred_route = pred_future_waypoints = pred_target_speed_distribution = (
            pred_target_speed_scalar
        ) = pred_headings = None
        pred_semantic = pred_depth = pred_bounding_box = pred_bev_semantic = None
        pred_bounding_box_navsim = pred_bev_semantic_navsim = None

        # Backbone
        bev_features, image_features = self.backbone(data)

        # Radar detection
        radar_features = radar_predictions = None
        if self.use_carla_data and self.radar_detection:
            radar_features, radar_predictions = self.radar_detector(bev_features, data)

        # Planning heads
        if self.use_planning_decoder:
            planner_radar_features = radar_features
            planner_radar_predictions = radar_predictions
            if not self.use_radar_detection or not self.use_carla_data:
                planner_radar_features = planner_radar_predictions = None
            (
                pred_route,
                pred_future_waypoints,
                pred_target_speed_distribution,
                pred_target_speed_scalar,
                pred_headings,
            ) = self.planning_decoder(
                bev_features,
                planner_radar_features,
                planner_radar_predictions,
                data,
                log=None,
            )

        # Semantic segmentation forward pass
        if self.use_carla_data and self.use_semantic:
            pred_semantic = self.semantic_decoder(data, image_features, None)

        # Depth estimation forward pass
        if self.use_carla_data and self.use_depth:
            pred_depth = self.depth_decoder(data, image_features, None)

        # Bounding box detection forward pass
        bev_feature_grid = self.backbone.top_down(bev_features)
        if self.detect_boxes:
            if self.use_carla_data:
                pred_bounding_box = self.center_net_decoder.forward_tensors(
                    bev_feature_grid
                )
            if self.use_navsim_data:
                pred_bounding_box_navsim = self.center_net_decoder_navsim.forward_tensors(
                    bev_feature_grid
                )

        # BEV semantic segmentation forward pass
        if self.use_bev_semantic:
            if self.use_carla_data:
                pred_bev_semantic = self.bev_semantic_decoder(bev_feature_grid, None)
            if self.use_navsim_data:
                pred_bev_semantic_navsim = self.bev_semantic_decoder_navsim(
                    bev_feature_grid, None
                )

        return (
            pred_route,
            pred_future_waypoints,
            pred_target_speed_distribution,
            pred_target_speed_scalar,
            pred_headings,
            pred_semantic,
            pred_depth,
            pred_bounding_box,
            pred_bev_semantic,
            radar_features,
            radar_predictions,
            pred_bounding_box_navsim,
            pred_bev_semantic_navsim,
        )

    def _prepare_forward_data(self, data: dict[str, typing.Any]) -> dict[str, typing.Any]:
        core_data: dict[str, typing.Any] = {}

        def add_tensor(
            key: str,
            dtype: torch.dtype | None = None,
            required: bool = False,
        ) -> None:
            if key not in data:
                if required:
                    raise KeyError(f"Missing required forward data key: {key!r}")
                return
            value = data[key]
            if torch.is_tensor(value):
                if dtype is None:
                    core_data[key] = value.to(self.device, non_blocking=True)
                else:
                    core_data[key] = value.to(
                        self.device,
                        dtype=dtype,
                        non_blocking=True,
                    )
            else:
                core_data[key] = value

        if self.backbone.backbone_sensor_mode == "dual_front_camera":
            add_tensor(self.backbone.left_camera_key, self.backbone.input_dtype, True)
            add_tensor(self.backbone.right_camera_key, self.backbone.input_dtype, True)
            for camera_name in ("left", "right"):
                for key in (
                    f"{camera_name}_camera_intrinsics",
                    f"{camera_name}_intrinsics",
                    f"rgb_{camera_name}_intrinsics",
                    f"{camera_name}_camera_T_cam_ego",
                    f"{camera_name}_T_cam_ego",
                    f"rgb_{camera_name}_T_cam_ego",
                ):
                    add_tensor(key, torch.float32)
        else:
            add_tensor("rgb", self.backbone.input_dtype, True)
            if not self.backbone.use_ltf:
                add_tensor("rasterized_lidar", self.backbone.input_dtype, True)

        if self.radar_detection:
            add_tensor("radar", self.radar_detector.input_dtype, True)
            add_tensor("speed", self.radar_detector.input_dtype, True)

        if self.use_planning_decoder:
            planning_dtype = getattr(self.planning_decoder, "input_dtype", None)
            for key in (
                "speed",
                "acceleration",
                "command",
                "target_point",
                "target_point_previous",
                "target_point_next",
            ):
                add_tensor(key, planning_dtype)

        return core_data

    def _make_center_net_prediction(
        self,
        tensors: tuple[torch.Tensor | None, ...] | None,
        config: TrainingConfig,
    ) -> CenterNetBoundingBoxPrediction | None:
        if tensors is None:
            return None
        return CenterNetBoundingBoxPrediction(*tensors, config=config)

    @beartype
    def compute_loss(
        self, predictions: Prediction, data: dict[str, typing.Any]
    ) -> tuple[dict, dict]:
        loss = {}
        # Semantic segmentation loss
        if self.config.use_semantic and self.config.use_carla_data:
            self.semantic_decoder.compute_loss(
                predictions.pred_semantic, data, loss, log=self.log
            )

        # Depth estimation loss
        if self.config.use_depth and self.config.use_carla_data:
            self.depth_decoder.compute_loss(
                predictions.pred_depth, data, loss, log=self.log
            )

        # BEV semantic segmentation loss
        if self.config.use_bev_semantic:
            if self.config.use_carla_data:
                self.bev_semantic_decoder.compute_loss(
                    predictions.pred_bev_semantic, data, loss, log=self.log
                )
            if self.config.use_navsim_data:
                self.bev_semantic_decoder_navsim.compute_loss(
                    predictions.pred_bev_semantic_navsim, data, loss, log=self.log
                )

        # Bounding box detection loss
        if self.config.detect_boxes:
            if self.config.use_carla_data:
                self.center_net_decoder.compute_loss(
                    data=data,
                    bounding_box_features=predictions.pred_bounding_box,
                    losses=loss,
                    log=self.log,
                )
            if self.config.use_navsim_data:
                self.center_net_decoder_navsim.compute_loss(
                    data=data,
                    bounding_box_features=predictions.pred_bounding_box_navsim,
                    losses=loss,
                    log=self.log,
                )

        # Radar detection loss
        if self.config.radar_detection and self.config.use_carla_data:
            self.radar_detector.compute_loss(
                pred=predictions.pred_radar_predictions,
                data=data,
                loss=loss,
                log=self.log,
            )

        # Planning loss
        if self.config.use_planning_decoder:
            self.planning_decoder.compute_loss(
                data=data, predictions=predictions, loss=loss, log=self.log
            )

        return loss, self.log


@jt.jaxtyped(typechecker=beartype)
@dataclass
class Prediction:
    """Raw output predictions from the model."""

    # Planning prediction
    pred_future_waypoints: jt.Float[torch.Tensor, "bs n_waypoints 2"] | None
    pred_target_speed_distribution: (
        jt.Float[torch.Tensor, "bs num_speed_classes"] | None
    )
    pred_target_speed_scalar: jt.Float[torch.Tensor, " bs"] | None
    pred_route: jt.Float[torch.Tensor, "bs n_checkpoints 2"] | None

    # CARLA perception prediction
    pred_semantic: (
        jt.Float[torch.Tensor, "bs num_semantic_classes img_height img_width"] | None
    )
    pred_bev_semantic: (
        jt.Float[torch.Tensor, "bs num_bev_classes bev_height bev_width"] | None
    )
    pred_depth: jt.Float[torch.Tensor, "bs img_height img_width"] | None
    pred_bounding_box: CenterNetBoundingBoxPrediction | None
    pred_radar_features: jt.Float[torch.Tensor, "B Q C"] | None
    pred_radar_predictions: jt.Float[torch.Tensor, "B Q 4"] | None

    # NavSim perception prediction
    pred_bounding_box_navsim: CenterNetBoundingBoxPrediction | None
    pred_bev_semantic_navsim: (
        jt.Float[torch.Tensor, "bs num_bev_classes_navsim bev_height bev_width"] | None
    )
    pred_headings: jt.Float[torch.Tensor, "bs n_waypoints"] | None
