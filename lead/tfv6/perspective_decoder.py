import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype

from lead.common.constants import SOURCE_DATASET_NAME_MAP, SourceDataset
from lead.training import metrics
from lead.training.config_training import TrainingConfig
from lead.tfv6.losses import (
    maybe_compile_focal_loss,
    sigmoid_focal_loss_one_hot_label,
)


class PerspectiveDecoder(nn.Module):
    @beartype
    def __init__(
        self,
        config: TrainingConfig,
        in_channels: int,
        out_channels: int,
        perspective_upsample_factor: int,
        modality: str,
        device: torch.device,
        source_data: int,
    ):
        """Decodes a low resolution perspective grid to a full resolution output. E.g. semantic segmentation, depth

        Args:
            config: Training configuration object
            in_channels: Feature channels of the input feature grid
            out_channels: Feature channels of the output feature grid
            perspective_upsample_factor: Upsampling factor from input feature grid to output
            modality: "semantic" or "depth"
            device: torch device
            source_data: Source dataset identifier (e.g., SourceDataset.CARLA)
        """
        super().__init__()
        self.modality = modality
        self.config = config
        self.device = device
        self.source_data = source_data
        self.expected_h = int(self.config.final_image_height)
        self.expected_w = int(self.config.final_image_width)
        self.is_depth = self.modality == "depth"
        self._semantic_focal_loss = maybe_compile_focal_loss(
            sigmoid_focal_loss_one_hot_label,
            enabled=bool(self.config.compile),
            mode=str(self.config.compile_mode).lower(),
        )
        self.scale_factor_0 = (
            perspective_upsample_factor // self.config.deconv_scale_factor_0
        )
        self.scale_factor_1 = (
            perspective_upsample_factor // self.config.deconv_scale_factor_1
        )

        self.deconv1 = nn.Sequential(
            nn.Conv2d(in_channels, self.config.deconv_channel_num_0, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.config.deconv_channel_num_0,
                self.config.deconv_channel_num_1,
                3,
                1,
                1,
            ),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(
                self.config.deconv_channel_num_1,
                self.config.deconv_channel_num_2,
                3,
                1,
                1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.config.deconv_channel_num_2,
                self.config.deconv_channel_num_2,
                3,
                1,
                1,
            ),
            nn.ReLU(inplace=True),
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(
                self.config.deconv_channel_num_2,
                self.config.deconv_channel_num_2,
                3,
                1,
                1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.config.deconv_channel_num_2, out_channels, 3, 1, 1, bias=True
            ),
        )

    def compute_loss(
        self,
        prediction: torch.Tensor,
        data: dict[str, torch.Tensor],
        loss: dict,
        log: dict,
    ):
        """Compute loss and metrics for the given modality.

        Args:
            prediction: (B, C, H, W) Prediction tensor
            data: dict containing the ground truth labels and masks
            loss: dict to store the computed loss
            log: dict to store computed metrics and logs
        Returns:
            None
        """
        if self.config.use_semantic:
            # Mask for samples from the correct source dataset
            source_dataset = data["source_dataset"].to(
                prediction.device, dtype=torch.long, non_blocking=True
            )  # (B,)
            source_mask = (source_dataset == self.source_data).float()  # (B,)

            if source_mask.sum() == 0:
                return  # No samples from this source dataset in the batch

            label = data[self.modality].to(
                prediction.device, dtype=torch.long, non_blocking=True
            )
            if prediction.shape[-2:] != label.shape[-2:]:
                if prediction.ndim == 3:
                    prediction = F.interpolate(
                        prediction.float().unsqueeze(1),
                        size=label.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)
                else:
                    prediction = F.interpolate(
                        prediction.float(),
                        size=label.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

            # Compute loss per sample
            with torch.amp.autocast(device_type="cuda", enabled=False):
                if self.modality == "semantic":
                    prediction = prediction.float()
                    loss_type = str(self.config.semantic_loss_type).lower()
                    if loss_type == "focal":
                        loss_per_sample = self._semantic_focal_loss_per_pixel(
                            prediction,
                            label,
                        )
                    elif loss_type == "cross_entropy":
                        loss_per_sample = F.cross_entropy(
                            prediction,
                            label,
                            reduction="none",
                        )  # (B, H, W)
                    else:
                        raise ValueError(
                            f"Unknown semantic_loss_type: {self.config.semantic_loss_type!r}"
                        )
                    loss_per_sample = loss_per_sample.mean(dim=(1, 2))  # (B,)
                else:
                    loss_per_sample = F.l1_loss(
                        prediction.float(), label.float(), reduction="none"
                    )  # (B, H, W)
                    loss_per_sample = loss_per_sample.mean(dim=(1, 2))  # (B,)

                # Mask out losses from other data sources
                loss_value = (
                    loss_per_sample * source_mask
                ).sum() / source_mask.sum().clamp(min=1)

            if (
                data.get("compute_additional_metrics", False)
                and self.modality == "semantic"
            ):
                valid_prediction = prediction[source_mask.bool()]
                valid_label = label[source_mask.bool()]
                log["metric/semantic_miou"] = metrics.mean_iou_from_logits(
                    logits=valid_prediction,
                    label=valid_label,
                    num_classes=self.config.num_semantic_classes,
                )

            # Add dataset name prefix
            prefix = SOURCE_DATASET_NAME_MAP[self.source_data]
            if self.source_data == SourceDataset.CARLA:
                prefix = ""
            loss.update({f"{prefix}loss_{self.modality}": loss_value})

    def _semantic_focal_loss_per_pixel(
        self,
        prediction: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        alpha = float(self.config.semantic_focal_loss_alpha)
        gamma = float(self.config.semantic_focal_loss_gamma)
        return self._semantic_focal_loss(prediction, label, alpha, gamma)

    def forward(self, data: dict, image_feature_grid: torch.Tensor, log: dict):
        """Forward pass for the decoder.

        Args:
            data: dict containing the ground truth labels and masks
            image_feature_grid: (B, D, H, W) Image feature grid from the encoder
            log: dict to store computed metrics and logs

        Returns:
            (B, C, H, W) Prediction tensor
        """
        x = self.deconv1(image_feature_grid)
        x = F.interpolate(
            x, scale_factor=self.scale_factor_0, mode="bilinear", align_corners=False
        )
        x = self.deconv2(x)
        x = F.interpolate(
            x, scale_factor=self.scale_factor_1, mode="bilinear", align_corners=False
        )
        x = self.deconv3(x)

        # Ensure output size matches expected size
        expected_h = self.expected_h
        expected_w = self.expected_w
        if x.shape[2] != expected_h or x.shape[3] != expected_w:
            height_error = abs(x.shape[2] - expected_h) / expected_h * 100
            width_error = abs(x.shape[3] - expected_w) / expected_w * 100
            if max(height_error, width_error) > 10:
                raise ValueError(
                    f"Output size mismatch too large: got ({x.shape[2]}, {x.shape[3]}), expected ({expected_h}, {expected_w})"
                )
            x = F.interpolate(
                x, size=(expected_h, expected_w), mode="bilinear", align_corners=False
            )

        if self.is_depth:
            x = x.squeeze(1)
        return x
