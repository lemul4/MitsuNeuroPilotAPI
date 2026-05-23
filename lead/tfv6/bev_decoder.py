import torch
import torch.nn as nn
from beartype import beartype

import lead.common.common_utils as common_utils
from lead.common.constants import (
    SOURCE_DATASET_NAME_MAP,
    SourceDataset,
    TransfuserBEVSemanticClass,
)
from lead.training import metrics
from lead.training.config_training import TrainingConfig
from lead.tfv6.losses import (
    bev_cross_entropy_loss_scalar,
    bev_focal_loss_scalar,
    maybe_compile_focal_loss,
    maybe_compile_loss,
    sigmoid_focal_loss_one_hot_label_ignore,
)


class BEVDecoder(nn.Module):
    @beartype
    def __init__(
        self,
        config: TrainingConfig,
        num_classes: int,
        device: torch.device,
        source_data: int,
    ):
        """Dense BEV decoder for BEV semantic segmentation.

        Args:
            config: Training configuration object
            num_classes: Number of semantic classes to predict
            device: Device to run the model on
            source_data: Source dataset enum value for which this decoder is responsible
        """
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.device = device
        self.source_data = source_data
        self._bev_semantic_focal_loss = maybe_compile_focal_loss(
            sigmoid_focal_loss_one_hot_label_ignore,
            enabled=bool(self.config.compile),
            mode=str(self.config.compile_mode).lower(),
        )
        self._bev_semantic_focal_scalar_loss = maybe_compile_loss(
            bev_focal_loss_scalar,
            enabled=bool(self.config.compile),
            mode=str(self.config.compile_mode).lower(),
        )
        self._bev_semantic_ce_scalar_loss = maybe_compile_loss(
            bev_cross_entropy_loss_scalar,
            enabled=bool(self.config.compile),
            mode=str(self.config.compile_mode).lower(),
        )
        bev_class_weights = torch.ones(self.num_classes, dtype=torch.float32)
        walker_idx = int(TransfuserBEVSemanticClass.WALKER)
        biker_idx = int(TransfuserBEVSemanticClass.BIKER)
        traffic_light_indices = (
            int(TransfuserBEVSemanticClass.TRAFFIC_GREEN),
            int(TransfuserBEVSemanticClass.TRAFFIC_RED_NORMAL),
            int(TransfuserBEVSemanticClass.TRAFFIC_RED_NOT_NORMAL),
        )
        if walker_idx < self.num_classes:
            bev_class_weights[walker_idx] = float(
                self.config.bev_semantic_walker_loss_weight
            )
        if biker_idx < self.num_classes:
            bev_class_weights[biker_idx] = float(
                self.config.bev_semantic_biker_loss_weight
            )
        for traffic_light_idx in traffic_light_indices:
            if traffic_light_idx < self.num_classes:
                bev_class_weights[traffic_light_idx] = float(
                    self.config.bev_semantic_traffic_light_loss_weight
                )
        self.register_buffer(
            "bev_semantic_class_weights",
            bev_class_weights,
            persistent=False,
        )

        self.net = nn.Sequential(
            nn.Conv2d(
                self.config.bev_features_chanels,
                self.config.bev_features_chanels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.config.bev_features_chanels,
                num_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Upsample(
                size=(self.config.lidar_height_pixel, self.config.lidar_width_pixel),
                mode="bilinear",
                align_corners=False,
            ),
        )

        # Mask to mark occluded part of the BEV grid
        h, w = self.config.lidar_height_pixel, self.config.lidar_width_pixel
        valid_bev_pixels = torch.ones((h, w))
        center_y = h // 2
        center_x = -(self.config.min_x_meter * self.config.pixels_per_meter)
        for y in range(h):
            for x in range(w):
                valid_bev_pixels[y, x] = float(
                    common_utils.is_point_in_camera_frustum(
                        x=x,
                        y=y,
                        config=self.config,
                        center_x=center_x,
                        center_y=center_y,
                    )
                )

        # Register as parameter so that it will automatically be moved to the correct GPU with the rest of the network
        self.valid_bev_pixels = valid_bev_pixels.unsqueeze(0).to(
            dtype=self.config.torch_float_type, device=device
        )  # (1, H, W)

    @beartype
    def compute_loss(self, pred: torch.Tensor, data: dict, loss: dict, log: dict):
        """
        Compute BEV semantic segmentation loss.

        Args:
            pred: (B, C, H, W) BEV semantic prediction tensor
            data: dict containing the ground truth labels and masks
            loss: dict to store the computed loss
            log: dict to store computed metrics and logs
        Returns:
            None
        """
        if not self.config.use_bev_semantic:
            return

        # Mask for samples from the correct source dataset
        source_dataset = data["source_dataset"].to(
            pred.device, dtype=torch.long, non_blocking=True
        )  # (B,)
        source_mask = (source_dataset == self.source_data).float()  # (B,)
        if source_mask.sum() == 0:
            return  # No samples from this source dataset in the batch
        dataset_name = SOURCE_DATASET_NAME_MAP[self.source_data]
        prefix = f"{dataset_name}_"
        if self.source_data == SourceDataset.CARLA:
            prefix = ""

        label = data[f"{prefix}bev_semantic"].to(
            pred.device, dtype=torch.long, non_blocking=True
        )
        visible_label = self.mask_label(label).long()  # Mark invisible pixels to -1

        with torch.amp.autocast(device_type="cuda", enabled=False):
            # Compute per-sample loss
            pred = pred.float()
            loss_type = str(self.config.bev_semantic_loss_type).lower()
            if loss_type == "focal":
                loss_bev = self._bev_semantic_focal_scalar_loss(
                    pred,
                    label,
                    source_mask,
                    self.valid_bev_pixels,
                    self.bev_semantic_class_weights,
                    float(self.config.bev_semantic_focal_loss_alpha),
                    float(self.config.bev_semantic_focal_loss_gamma),
                )
            elif loss_type == "cross_entropy":
                loss_bev = self._bev_semantic_ce_scalar_loss(
                    pred,
                    label,
                    source_mask,
                    self.valid_bev_pixels,
                    self.bev_semantic_class_weights,
                )
            else:
                raise ValueError(
                    f"Unknown bev_semantic_loss_type: {self.config.bev_semantic_loss_type!r}"
                )

        if data.get("compute_additional_metrics", False):
            metric_prefix = f"{prefix}" if prefix else ""
            metric_name = f"metric/{metric_prefix}bev_semantic_miou"
            log[metric_name] = metrics.mean_iou_from_logits(
                logits=pred[source_mask.bool()],
                label=visible_label[source_mask.bool()],
                num_classes=self.num_classes,
                ignore_index=-1,
            )

        # Add dataset name prefix
        loss[f"{prefix}loss_bev_semantic"] = loss_bev

    def _bev_semantic_focal_loss_per_pixel(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        alpha = float(self.config.bev_semantic_focal_loss_alpha)
        gamma = float(self.config.bev_semantic_focal_loss_gamma)
        per_pixel = self._bev_semantic_focal_loss(pred, label, alpha, gamma, -1)
        return per_pixel * self.bev_semantic_class_weights.to(dtype=per_pixel.dtype)[
            label.clamp(min=0)
        ]

    def forward(self, bev_feature_grid: torch.Tensor, log: dict):
        """Forward pass for the BEV decoder.

        Args:
            bev_feature_grid: (B, D, H, W) BEV feature grid from the encoder
            log: dict to store computed metrics and logs

        Returns:
            (B, C, H, W) BEV feature grid after passing through the decoder
        """
        return self.net(bev_feature_grid)

    @beartype
    def mask_label(self, label: torch.Tensor):
        """
        Args:
            label: (B, H, W) BEV semantic label tensor
        Returns:
            label: (B, H, W) Masked BEV semantic label tensor
        """
        valid_bev_pixels = self.valid_bev_pixels
        visible_bev_semantic_label = (
            label * valid_bev_pixels
        )  # (B, H, W). Set invisible pixels to 0
        visible_bev_semantic_label = (
            valid_bev_pixels - 1
        ) + visible_bev_semantic_label  # Set invisible pixels to -1
        return visible_bev_semantic_label
