from typing import Any
import math

import jaxtyping as jt
import timm
import torch
import torch.nn.functional as F
from beartype import beartype
from torch import nn

from lead.tfv6 import transfuser_utils as fn
from lead.training.config_training import TrainingConfig


def _config_get(
    config: TrainingConfig,
    names: str | tuple[str, ...],
    default: Any = None,
) -> Any:
    """Read a config value while keeping new dual-camera fields optional."""
    if isinstance(names, str):
        names = (names,)
    for name in names:
        if isinstance(config, dict) and name in config and config[name] is not None:
            return config[name]
        if hasattr(config, name):
            value = getattr(config, name)
            if value is not None:
                return value
    return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class TransfuserBackbone(nn.Module):
    """TransFuser backbone network for multi-modal sensor fusion.

    Default mode (``backbone_sensor_mode='lidar_rgb'``) is the original RGB
    panorama + LiDAR pseudo-image TransFuser. New mode
    (``backbone_sensor_mode='dual_front_camera'``) adds two independent front
    camera encoders, stage-wise GPT fusion, and a geometry-aware camera-to-BEV
    projector. The public output contract remains unchanged:

        return bev_features, image_features

    In the legacy mode ``bev_features`` are LiDAR-branch features. In the new
    mode they are camera-derived BEV features with the same shape contract used
    by the existing planning, CenterNet, and BEV semantic heads.
    """

    @beartype
    def __init__(self, device: torch.device, config: TrainingConfig) -> None:
        super().__init__()
        self.device = device
        self.config = config
        self.backbone_sensor_mode = str(
            _config_get(config, "backbone_sensor_mode", "lidar_rgb")
        ).lower()

        if self.backbone_sensor_mode in {"lidar_rgb", "rgb_lidar", "default"}:
            self.backbone_sensor_mode = "lidar_rgb"
            self._init_lidar_rgb_backbone()
        elif self.backbone_sensor_mode in {
            "dual_front_camera",
            "dual_camera",
            "two_front_cameras",
            "two_camera",
        }:
            self.backbone_sensor_mode = "dual_front_camera"
            self._init_dual_front_camera_backbone()
        else:
            raise ValueError(
                f"Unsupported backbone_sensor_mode={self.backbone_sensor_mode!r}. "
                "Expected 'lidar_rgb' or 'dual_front_camera'."
            )

        self._init_top_down_layers()

    @beartype
    def _init_lidar_rgb_backbone(self) -> None:
        """Initialize the original RGB panorama + LiDAR TransFuser path."""
        self.image_encoder = timm.create_model(
            self.config.image_architecture, pretrained=True, features_only=True
        )
        self.avgpool_img = nn.AdaptiveAvgPool2d(
            (self.config.img_vert_anchors, self.config.img_horz_anchors)
        )
        image_start_index = 0
        if len(self.image_encoder.return_layers) > 4:
            image_start_index += 1
        self.image_start_index = image_start_index
        self.num_image_features = self.image_encoder.feature_info.info[
            image_start_index + 3
        ]["num_chs"]

        self.lidar_encoder = timm.create_model(
            self.config.lidar_architecture,
            pretrained=False,
            in_chans=2 if self.config.LTF else 1,
            features_only=True,
        )
        lidar_start_index = 0
        if len(self.lidar_encoder.return_layers) > 4:
            lidar_start_index += 1
        self.lidar_start_index = lidar_start_index
        self.num_lidar_features = self.lidar_encoder.feature_info.info[
            lidar_start_index + 3
        ]["num_chs"]

        self.lidar_channel_to_img = nn.ModuleList(
            [
                nn.Conv2d(
                    self.lidar_encoder.feature_info.info[lidar_start_index + i][
                        "num_chs"
                    ],
                    self.image_encoder.feature_info.info[image_start_index + i][
                        "num_chs"
                    ],
                    kernel_size=1,
                )
                for i in range(4)
            ]
        )
        self.img_channel_to_lidar = nn.ModuleList(
            [
                nn.Conv2d(
                    self.image_encoder.feature_info.info[image_start_index + i][
                        "num_chs"
                    ],
                    self.lidar_encoder.feature_info.info[lidar_start_index + i][
                        "num_chs"
                    ],
                    kernel_size=1,
                )
                for i in range(4)
            ]
        )
        self.avgpool_lidar = nn.AdaptiveAvgPool2d(
            (self.config.lidar_vert_anchors, self.config.lidar_horz_anchors)
        )

        self.transformers = nn.ModuleList(
            [
                GPT(
                    n_embd=self.image_encoder.feature_info.info[image_start_index + i][
                        "num_chs"
                    ],
                    config=self.config,
                )
                for i in range(4)
            ]
        )

        self.perspective_upsample_factor = (
            self.image_encoder.feature_info.info[image_start_index + 3]["reduction"]
            // self.config.perspective_downsample_factor
        )

    @beartype
    def _init_dual_front_camera_backbone(self) -> None:
        """Initialize two front cameras + GPT fusion + camera-to-BEV projection."""
        self.left_camera_key = str(
            _config_get(
                self.config,
                ("left_camera_key", "dual_camera_left_key"),
                "rgb_left",
            )
        )
        self.right_camera_key = str(
            _config_get(
                self.config,
                ("right_camera_key", "dual_camera_right_key"),
                "rgb_right",
            )
        )
        left_arch = _config_get(
            self.config,
            ("left_camera_architecture", "dual_camera_left_architecture"),
            self.config.image_architecture,
        )
        right_arch = _config_get(
            self.config,
            ("right_camera_architecture", "dual_camera_right_architecture"),
            self.config.image_architecture,
        )
        dual_camera_pretrained = _as_bool(
            _config_get(self.config, "dual_camera_pretrained", True)
        )
        share_encoder = _as_bool(
            _config_get(
                self.config,
                ("share_dual_camera_encoder", "dual_camera_share_encoder"),
                False,
            )
        )

        self.left_camera_encoder = timm.create_model(
            left_arch, pretrained=dual_camera_pretrained, features_only=True
        )
        if share_encoder:
            self.right_camera_encoder = self.left_camera_encoder
        else:
            self.right_camera_encoder = timm.create_model(
                right_arch, pretrained=dual_camera_pretrained, features_only=True
            )

        left_start_index = 0
        if len(self.left_camera_encoder.return_layers) > 4:
            left_start_index += 1
        right_start_index = 0
        if len(self.right_camera_encoder.return_layers) > 4:
            right_start_index += 1
        self.left_camera_start_index = left_start_index
        self.right_camera_start_index = right_start_index

        left_stage_channels = [
            self.left_camera_encoder.feature_info.info[left_start_index + i]["num_chs"]
            for i in range(4)
        ]
        right_stage_channels = [
            self.right_camera_encoder.feature_info.info[right_start_index + i][
                "num_chs"
            ]
            for i in range(4)
        ]
        self.left_camera_stage_channels = left_stage_channels
        self.right_camera_stage_channels = right_stage_channels

        left_final_channels = left_stage_channels[3]
        right_final_channels = right_stage_channels[3]
        self.dual_camera_image_output = str(
            _config_get(self.config, "dual_camera_image_output", "left")
        ).lower()
        if self.dual_camera_image_output in {"left", "fused_left", "primary"}:
            self.num_image_features = left_final_channels
            selected_reduction = self.left_camera_encoder.feature_info.info[
                left_start_index + 3
            ]["reduction"]
        elif self.dual_camera_image_output in {"right", "fused_right"}:
            self.num_image_features = right_final_channels
            selected_reduction = self.right_camera_encoder.feature_info.info[
                right_start_index + 3
            ]["reduction"]
        else:
            raise ValueError(
                "dual_camera_image_output must be 'left', 'fused_left', 'primary', "
                f"'right', or 'fused_right', got {self.dual_camera_image_output!r}."
            )

        self.num_bev_features = int(
            _config_get(
                self.config,
                (
                    "dual_camera_bev_out_channels",
                    "dual_camera_bev_output_channels",
                    "camera_bev_out_channels",
                ),
                left_final_channels,
            )
        )
        # Compatibility alias: TFv6, planning, radar, and BEV heads expect this name.
        self.num_lidar_features = self.num_bev_features

        token_grid_left = _config_get(self.config, "left_camera_token_grid", None)
        token_grid_right = _config_get(self.config, "right_camera_token_grid", None)
        if token_grid_left is not None:
            default_left_vert, default_left_horz = token_grid_left
        else:
            default_left_vert = _config_get(self.config, "img_vert_anchors", 12)
            default_left_horz = max(1, _config_get(self.config, "img_horz_anchors", 72) // 2)
        if token_grid_right is not None:
            default_right_vert, default_right_horz = token_grid_right
        else:
            default_right_vert = default_left_vert
            default_right_horz = max(1, round(default_left_horz * 50.0 / 90.0))

        self.left_camera_vert_anchors = int(
            _config_get(
                self.config,
                ("left_camera_vert_anchors", "left_img_vert_anchors"),
                default_left_vert,
            )
        )
        self.left_camera_horz_anchors = int(
            _config_get(
                self.config,
                ("left_camera_horz_anchors", "left_img_horz_anchors"),
                default_left_horz,
            )
        )
        self.right_camera_vert_anchors = int(
            _config_get(
                self.config,
                ("right_camera_vert_anchors", "right_img_vert_anchors"),
                default_right_vert,
            )
        )
        self.right_camera_horz_anchors = int(
            _config_get(
                self.config,
                ("right_camera_horz_anchors", "right_img_horz_anchors"),
                default_right_horz,
            )
        )

        self.avgpool_left_camera = nn.AdaptiveAvgPool2d(
            (self.left_camera_vert_anchors, self.left_camera_horz_anchors)
        )
        self.avgpool_right_camera = nn.AdaptiveAvgPool2d(
            (self.right_camera_vert_anchors, self.right_camera_horz_anchors)
        )

        self.right_channel_to_left = nn.ModuleList(
            [
                nn.Conv2d(right_stage_channels[i], left_stage_channels[i], kernel_size=1)
                for i in range(4)
            ]
        )
        self.left_channel_to_right = nn.ModuleList(
            [
                nn.Conv2d(left_stage_channels[i], right_stage_channels[i], kernel_size=1)
                for i in range(4)
            ]
        )
        self.dual_camera_transformers = nn.ModuleList(
            [
                DualCameraGPT(
                    n_embd=left_stage_channels[i],
                    left_vert_anchors=self.left_camera_vert_anchors,
                    left_horz_anchors=self.left_camera_horz_anchors,
                    right_vert_anchors=self.right_camera_vert_anchors,
                    right_horz_anchors=self.right_camera_horz_anchors,
                    config=self.config,
                )
                for i in range(4)
            ]
        )

        self.camera_to_bev = DualCameraBEVProjector(
            config=self.config,
            left_channels=left_final_channels,
            right_channels=right_final_channels,
            out_channels=self.num_lidar_features,
        )
        self.perspective_upsample_factor = (
            selected_reduction // self.config.perspective_downsample_factor
        )

    @beartype
    def _init_top_down_layers(self) -> None:
        """Initialize the BEV top-down feature pyramid shared by both modes."""
        self.upsample = nn.Upsample(
            scale_factor=self.config.bev_upsample_factor,
            mode="bilinear",
            align_corners=False,
        )
        self.upsample2 = nn.Upsample(
            size=(
                self.config.lidar_height_pixel // self.config.bev_down_sample_factor,
                self.config.lidar_width_pixel // self.config.bev_down_sample_factor,
            ),
            mode="bilinear",
            align_corners=False,
        )
        self.up_conv5 = nn.Conv2d(
            self.config.bev_features_chanels,
            self.config.bev_features_chanels,
            (3, 3),
            padding=1,
        )
        self.up_conv4 = nn.Conv2d(
            self.config.bev_features_chanels,
            self.config.bev_features_chanels,
            (3, 3),
            padding=1,
        )
        self.c5_conv = nn.Conv2d(
            self.num_lidar_features, self.config.bev_features_chanels, (1, 1)
        )

    @jt.jaxtyped(typechecker=beartype)
    def top_down(
        self, x: jt.Float[torch.Tensor, "B C H W"]
    ) -> jt.Float[torch.Tensor, "B C2 H2 W2"]:
        """Apply top-down feature pyramid processing to BEV features."""
        p5 = F.relu(self.c5_conv(x), inplace=True)
        p4 = F.relu(self.up_conv5(self.upsample(p5)), inplace=True)
        p3 = F.relu(self.up_conv4(self.upsample2(p4)), inplace=True)
        return p3

    def forward(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the configured backbone mode."""
        if self.backbone_sensor_mode == "dual_front_camera":
            return self._forward_dual_front_camera(data)
        return self._forward_lidar_rgb(data)

    def _forward_lidar_rgb(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Original RGB panorama + LiDAR data loading path."""
        rgb = data["rgb"].to(
            self.device, dtype=self.config.torch_float_type, non_blocking=True
        )
        if self.config.LTF:
            x = torch.linspace(
                0,
                1,
                self.config.lidar_width_pixel,
                device=rgb.device,
                dtype=self.config.torch_float_type,
            )
            y = torch.linspace(
                0,
                1,
                self.config.lidar_height_pixel,
                device=rgb.device,
                dtype=self.config.torch_float_type,
            )
            y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
            lidar = torch.zeros(
                (
                    rgb.shape[0],
                    2,
                    self.config.lidar_height_pixel,
                    self.config.lidar_width_pixel,
                ),
                device=rgb.device,
                dtype=self.config.torch_float_type,
            )
            lidar[:, 0] = y_grid.unsqueeze(0)
            lidar[:, 1] = x_grid.unsqueeze(0)
        else:
            lidar = data["rasterized_lidar"].to(
                self.device, dtype=self.config.torch_float_type, non_blocking=True
            )
        return self._forward(rgb, lidar)

    def _forward_dual_front_camera(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Data loading path for the two-front-camera architecture."""
        if self.left_camera_key not in data:
            raise KeyError(
                f"Dual-camera backbone expected data[{self.left_camera_key!r}]."
            )
        if self.right_camera_key not in data:
            raise KeyError(
                f"Dual-camera backbone expected data[{self.right_camera_key!r}]."
            )
        left = data[self.left_camera_key].to(
            self.device, dtype=self.config.torch_float_type, non_blocking=True
        )
        right = data[self.right_camera_key].to(
            self.device, dtype=self.config.torch_float_type, non_blocking=True
        )
        return self._forward_dual_camera(left, right, data)

    @jt.jaxtyped(typechecker=beartype)
    def _forward(
        self,
        image: jt.Float[torch.Tensor, "B 3 img_h img_w"],
        lidar: jt.Float[torch.Tensor, "B 1 bev_h bev_w"]
        | jt.Float[torch.Tensor, "B 2 bev_h bev_w"]
        | None,
    ) -> tuple[
        jt.Float[torch.Tensor, "B D1 H1 W1"], jt.Float[torch.Tensor, "B D2 H2 W2"]
    ]:
        """Original image + LiDAR feature fusion using transformers."""
        image_features = fn.normalize_imagenet(image)
        lidar_features = lidar

        if self.config.channel_last:
            image = image.to(memory_format=torch.channels_last)
            if lidar is not None:
                lidar = lidar.to(memory_format=torch.channels_last)

        image_layers = iter(self.image_encoder.items())
        lidar_layers = iter(self.lidar_encoder.items())

        if len(self.image_encoder.return_layers) > 4:
            image_features = self.forward_layer_block(
                image_layers, self.image_encoder.return_layers, image_features
            )
        if len(self.lidar_encoder.return_layers) > 4:
            lidar_features = self.forward_layer_block(
                lidar_layers, self.lidar_encoder.return_layers, lidar_features
            )

        for i in range(4):
            image_features = self.forward_layer_block(
                image_layers, self.image_encoder.return_layers, image_features
            )
            lidar_features = self.forward_layer_block(
                lidar_layers, self.lidar_encoder.return_layers, lidar_features
            )
            image_features, lidar_features = self.fuse_features(
                image_features, lidar_features, i
            )
        return lidar_features, image_features

    def _forward_dual_camera(
        self,
        left_image: torch.Tensor,
        right_image: torch.Tensor,
        data: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Two independent camera encoders + multi-stage GPT fusion + BEV projection."""
        left_features = fn.normalize_imagenet(left_image)
        right_features = fn.normalize_imagenet(right_image)

        if self.config.channel_last:
            left_features = left_features.to(memory_format=torch.channels_last)
            right_features = right_features.to(memory_format=torch.channels_last)

        left_layers = iter(self.left_camera_encoder.items())
        right_layers = iter(self.right_camera_encoder.items())

        if len(self.left_camera_encoder.return_layers) > 4:
            left_features = self.forward_layer_block(
                left_layers, self.left_camera_encoder.return_layers, left_features
            )
        if len(self.right_camera_encoder.return_layers) > 4:
            right_features = self.forward_layer_block(
                right_layers, self.right_camera_encoder.return_layers, right_features
            )

        for i in range(4):
            left_features = self.forward_layer_block(
                left_layers, self.left_camera_encoder.return_layers, left_features
            )
            right_features = self.forward_layer_block(
                right_layers, self.right_camera_encoder.return_layers, right_features
            )
            left_features, right_features = self.fuse_dual_camera_features(
                left_features, right_features, i
            )

        if self.dual_camera_image_output in {"right", "fused_right"}:
            image_features = right_features
        else:
            image_features = left_features

        bev_features = self.camera_to_bev(
            left_features=left_features,
            right_features=right_features,
            left_image_hw=left_image.shape[2:4],
            right_image_hw=right_image.shape[2:4],
            data=data,
        )
        return bev_features, image_features

    @beartype
    def forward_layer_block(
        self, layers: Any, return_layers: dict[str, str], features: torch.Tensor
    ) -> torch.Tensor:
        """Run one forward pass to a block of layers from a TIMM neural network."""
        for name, module in layers:
            features = module(features)
            if name in return_layers:
                break
        return features

    @jt.jaxtyped(typechecker=beartype)
    def fuse_features(
        self,
        image_features: jt.Float[torch.Tensor, "B C H W"],
        lidar_features: jt.Float[torch.Tensor, "B C2 H2 W2"],
        layer_idx: int,
    ) -> tuple[jt.Float[torch.Tensor, "B C H W"], jt.Float[torch.Tensor, "B C2 H2 W2"]]:
        """Original TransFuser image/LiDAR feature fusion block."""
        image_embd_layer = self.avgpool_img(image_features)
        lidar_embd_layer = self.avgpool_lidar(lidar_features)
        lidar_embd_layer = self.lidar_channel_to_img[layer_idx](lidar_embd_layer)

        image_features_layer, lidar_features_layer = self.transformers[layer_idx](
            image_embd_layer, lidar_embd_layer
        )
        lidar_features_layer = self.img_channel_to_lidar[layer_idx](
            lidar_features_layer
        )
        image_features_layer = F.interpolate(
            image_features_layer,
            size=(image_features.shape[2], image_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        lidar_features_layer = F.interpolate(
            lidar_features_layer,
            size=(lidar_features.shape[2], lidar_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return image_features + image_features_layer, lidar_features + lidar_features_layer

    def fuse_dual_camera_features(
        self,
        left_features: torch.Tensor,
        right_features: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse left/right camera features at one encoder stage."""
        left_embd_layer = self.avgpool_left_camera(left_features)
        right_embd_layer = self.avgpool_right_camera(right_features)
        right_embd_layer = self.right_channel_to_left[layer_idx](right_embd_layer)

        left_delta, right_delta_left_channels = self.dual_camera_transformers[
            layer_idx
        ](left_embd_layer, right_embd_layer)
        right_delta = self.left_channel_to_right[layer_idx](right_delta_left_channels)

        left_delta = F.interpolate(
            left_delta,
            size=(left_features.shape[2], left_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        right_delta = F.interpolate(
            right_delta,
            size=(right_features.shape[2], right_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return left_features + left_delta, right_features + right_delta


class GPT(nn.Module):
    """Original GPT-style transformer for image + LiDAR feature fusion."""

    @beartype
    def __init__(self, n_embd: int, config: TrainingConfig) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.config = config
        self.pos_emb = nn.Parameter(
            torch.zeros(
                1,
                self.config.img_vert_anchors * self.config.img_horz_anchors
                + self.config.lidar_vert_anchors * self.config.lidar_horz_anchors,
                self.n_embd,
            )
        )
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd,
                    config.n_head,
                    config.block_exp,
                    config.attn_pdrop,
                    config.resid_pdrop,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.apply(self._init_weights)

    @beartype
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=self.config.gpt_linear_layer_init_mean,
                std=self.config.gpt_linear_layer_init_std,
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.gpt_layer_norm_init_weight)

    @jt.jaxtyped(typechecker=beartype)
    def forward(
        self,
        image_tensor: jt.Float[torch.Tensor, "B C img_h img_w"],
        lidar_tensor: jt.Float[torch.Tensor, "B C lidar_h lidar_w"],
    ) -> tuple[
        jt.Float[torch.Tensor, "B C img_h img_w"],
        jt.Float[torch.Tensor, "B C lidar_h lidar_w"],
    ]:
        bz = lidar_tensor.shape[0]
        lidar_h, lidar_w = lidar_tensor.shape[2:4]
        img_h, img_w = image_tensor.shape[2:4]

        image_tensor = (
            image_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)
        )
        lidar_tensor = (
            lidar_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)
        )
        token_embeddings = torch.cat((image_tensor, lidar_tensor), dim=1)
        x = self.drop(self.pos_emb + token_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)

        image_token_count = self.config.img_vert_anchors * self.config.img_horz_anchors
        image_tensor_out = (
            x[:, :image_token_count, :]
            .view(bz, img_h, img_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        lidar_tensor_out = (
            x[:, image_token_count:, :]
            .view(bz, lidar_h, lidar_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return image_tensor_out, lidar_tensor_out


class DualCameraGPT(nn.Module):
    """GPT-style transformer for left/right camera feature fusion."""

    @beartype
    def __init__(
        self,
        n_embd: int,
        left_vert_anchors: int,
        left_horz_anchors: int,
        right_vert_anchors: int,
        right_horz_anchors: int,
        config: TrainingConfig,
    ) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.config = config
        self.left_token_count = left_vert_anchors * left_horz_anchors
        self.right_token_count = right_vert_anchors * right_horz_anchors
        self.pos_emb = nn.Parameter(
            torch.zeros(1, self.left_token_count + self.right_token_count, n_embd)
        )
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd,
                    config.n_head,
                    config.block_exp,
                    config.attn_pdrop,
                    config.resid_pdrop,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.apply(self._init_weights)

    @beartype
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=self.config.gpt_linear_layer_init_mean,
                std=self.config.gpt_linear_layer_init_std,
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.gpt_layer_norm_init_weight)

    @jt.jaxtyped(typechecker=beartype)
    def forward(
        self,
        left_tensor: jt.Float[torch.Tensor, "B C left_h left_w"],
        right_tensor: jt.Float[torch.Tensor, "B C right_h right_w"],
    ) -> tuple[
        jt.Float[torch.Tensor, "B C left_h left_w"],
        jt.Float[torch.Tensor, "B C right_h right_w"],
    ]:
        bz = left_tensor.shape[0]
        left_h, left_w = left_tensor.shape[2:4]
        right_h, right_w = right_tensor.shape[2:4]

        left_tokens = (
            left_tensor.permute(0, 2, 3, 1)
            .contiguous()
            .view(bz, -1, self.n_embd)
        )
        right_tokens = (
            right_tensor.permute(0, 2, 3, 1)
            .contiguous()
            .view(bz, -1, self.n_embd)
        )
        if left_tokens.shape[1] != self.left_token_count:
            raise ValueError(
                f"Left token count mismatch: got {left_tokens.shape[1]}, "
                f"expected {self.left_token_count}."
            )
        if right_tokens.shape[1] != self.right_token_count:
            raise ValueError(
                f"Right token count mismatch: got {right_tokens.shape[1]}, "
                f"expected {self.right_token_count}."
            )

        x = torch.cat((left_tokens, right_tokens), dim=1)
        x = self.drop(self.pos_emb + x)
        x = self.blocks(x)
        x = self.ln_f(x)

        left_tensor_out = (
            x[:, : self.left_token_count, :]
            .view(bz, left_h, left_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        right_tensor_out = (
            x[:, self.left_token_count :, :]
            .view(bz, right_h, right_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return left_tensor_out, right_tensor_out


class DualCameraBEVProjector(nn.Module):
    """Geometry-aware camera-to-BEV projector for two forward cameras."""

    @beartype
    def __init__(
        self,
        config: TrainingConfig,
        left_channels: int,
        right_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.bev_h = int(config.lidar_vert_anchors)
        self.bev_w = int(config.lidar_horz_anchors)
        self.projection_eps = float(
            _config_get(config, "dual_camera_projection_eps", 1.0e-4)
        )
        hidden_channels = int(
            _config_get(config, "dual_camera_bev_hidden_channels", out_channels)
        )
        use_bn = _as_bool(
            _config_get(config, "dual_camera_bev_use_batch_norm", True)
        )
        in_channels = left_channels + right_channels + 4
        layers: list[nn.Module] = [nn.Conv2d(in_channels, hidden_channels, kernel_size=1)]
        if use_bn:
            layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=1))
        self.fuser = nn.Sequential(*layers)

    def forward(
        self,
        left_features: torch.Tensor,
        right_features: torch.Tensor,
        left_image_hw: tuple[int, int] | torch.Size,
        right_image_hw: tuple[int, int] | torch.Size,
        data: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size = left_features.shape[0]
        device = left_features.device
        calc_dtype = torch.float32
        bev_points = self._build_bev_points(batch_size, device, calc_dtype)

        left_grid, left_valid = self._project_bev_points_to_feature_grid(
            bev_points,
            "left",
            left_features.shape[2:4],
            left_image_hw,
            data,
            device,
            calc_dtype,
        )
        right_grid, right_valid = self._project_bev_points_to_feature_grid(
            bev_points,
            "right",
            right_features.shape[2:4],
            right_image_hw,
            data,
            device,
            calc_dtype,
        )

        left_bev = F.grid_sample(
            left_features,
            left_grid.to(dtype=left_features.dtype),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        right_bev = F.grid_sample(
            right_features,
            right_grid.to(dtype=right_features.dtype),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        xy = self._build_normalized_xy_channels(
            batch_size, device, left_features.dtype
        )
        bev_input = torch.cat(
            [
                left_bev,
                right_bev,
                left_valid.to(dtype=left_features.dtype),
                right_valid.to(dtype=left_features.dtype),
                xy,
            ],
            dim=1,
        )
        return self.fuser(bev_input)

    def _build_bev_points(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        x_min = float(self.config.min_x_meter)
        x_max = float(self.config.max_x_meter)
        y_min = float(self.config.min_y_meter)
        y_max = float(self.config.max_y_meter)
        z_ground = float(
            _config_get(
                self.config,
                ("dual_camera_bev_ground_z_m", "dual_camera_ground_z_m"),
                0.0,
            )
        )
        x_step = (x_max - x_min) / self.bev_w
        y_step = (y_max - y_min) / self.bev_h
        x_centers = torch.linspace(
            x_min + 0.5 * x_step,
            x_max - 0.5 * x_step,
            self.bev_w,
            device=device,
            dtype=dtype,
        )
        y_centers = torch.linspace(
            y_min + 0.5 * y_step,
            y_max - 0.5 * y_step,
            self.bev_h,
            device=device,
            dtype=dtype,
        )
        if _as_bool(_config_get(self.config, "dual_camera_bev_reverse_x", False)):
            x_centers = torch.flip(x_centers, dims=[0])
        if _as_bool(_config_get(self.config, "dual_camera_bev_reverse_y", False)):
            y_centers = torch.flip(y_centers, dims=[0])
        yy, xx = torch.meshgrid(y_centers, x_centers, indexing="ij")
        zz = torch.full_like(xx, z_ground)
        ones = torch.ones_like(xx)
        points = torch.stack([xx, yy, zz, ones], dim=-1).view(-1, 4)
        return points.unsqueeze(0).repeat(batch_size, 1, 1)

    def _build_normalized_xy_channels(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        x_min = float(self.config.min_x_meter)
        x_max = float(self.config.max_x_meter)
        y_min = float(self.config.min_y_meter)
        y_max = float(self.config.max_y_meter)
        x_step = (x_max - x_min) / self.bev_w
        y_step = (y_max - y_min) / self.bev_h
        x_centers = torch.linspace(
            x_min + 0.5 * x_step,
            x_max - 0.5 * x_step,
            self.bev_w,
            device=device,
            dtype=dtype,
        )
        y_centers = torch.linspace(
            y_min + 0.5 * y_step,
            y_max - 0.5 * y_step,
            self.bev_h,
            device=device,
            dtype=dtype,
        )
        if _as_bool(_config_get(self.config, "dual_camera_bev_reverse_x", False)):
            x_centers = torch.flip(x_centers, dims=[0])
        if _as_bool(_config_get(self.config, "dual_camera_bev_reverse_y", False)):
            y_centers = torch.flip(y_centers, dims=[0])
        yy, xx = torch.meshgrid(y_centers, x_centers, indexing="ij")
        x_norm = 2.0 * (xx - x_min) / max(x_max - x_min, 1.0e-6) - 1.0
        y_norm = 2.0 * (yy - y_min) / max(y_max - y_min, 1.0e-6) - 1.0
        return torch.stack([x_norm, y_norm], dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)

    def _project_bev_points_to_feature_grid(
        self,
        bev_points: torch.Tensor,
        camera_name: str,
        feature_hw: tuple[int, int] | torch.Size,
        image_hw: tuple[int, int] | torch.Size,
        data: dict[str, torch.Tensor] | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = bev_points.shape[0]
        image_h, image_w = int(image_hw[0]), int(image_hw[1])
        feature_h, feature_w = int(feature_hw[0]), int(feature_hw[1])
        t_cam_ego = self._get_camera_transform(
            camera_name, batch_size, data, device, dtype
        )
        intrinsics = self._get_camera_intrinsics(
            camera_name, batch_size, image_h, image_w, data, device, dtype
        )

        points_cam = torch.bmm(bev_points, t_cam_ego.transpose(1, 2))
        x_cam = points_cam[..., 0]
        y_cam = points_cam[..., 1]
        z_cam = points_cam[..., 2]
        fx = intrinsics[:, 0, 0].unsqueeze(1)
        fy = intrinsics[:, 1, 1].unsqueeze(1)
        cx = intrinsics[:, 0, 2].unsqueeze(1)
        cy = intrinsics[:, 1, 2].unsqueeze(1)

        z_safe = z_cam.clamp(min=self.projection_eps)
        u = fx * (x_cam / z_safe) + cx
        v = fy * (y_cam / z_safe) + cy
        valid = (
            (z_cam > self.projection_eps)
            & (u >= 0.0)
            & (u <= float(image_w - 1))
            & (v >= 0.0)
            & (v <= float(image_h - 1))
        )

        u_feat = u * (float(feature_w) / float(image_w))
        v_feat = v * (float(feature_h) / float(image_h))
        grid_x = 2.0 * (u_feat + 0.5) / float(feature_w) - 1.0
        grid_y = 2.0 * (v_feat + 0.5) / float(feature_h) - 1.0
        outside = torch.full_like(grid_x, 2.0)
        grid_x = torch.where(valid, grid_x, outside)
        grid_y = torch.where(valid, grid_y, outside)
        grid = torch.stack([grid_x, grid_y], dim=-1).view(
            batch_size, self.bev_h, self.bev_w, 2
        )
        valid_mask = valid.to(dtype=dtype).view(batch_size, 1, self.bev_h, self.bev_w)
        return grid, valid_mask

    def _get_camera_intrinsics(
        self,
        camera_name: str,
        batch_size: int,
        image_h: int,
        image_w: int,
        data: dict[str, torch.Tensor] | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if data is not None:
            for key in (
                f"{camera_name}_camera_intrinsics",
                f"{camera_name}_intrinsics",
                f"rgb_{camera_name}_intrinsics",
            ):
                if key in data:
                    return self._as_batched_matrix(data[key], batch_size, 3, 3, device, dtype)
        value = _config_get(
            self.config,
            (f"{camera_name}_camera_intrinsics", f"dual_camera_{camera_name}_intrinsics"),
            None,
        )
        if value is not None:
            return self._as_batched_matrix(value, batch_size, 3, 3, device, dtype)
        fov = float(
            _config_get(
                self.config,
                (f"{camera_name}_camera_fov_deg", f"dual_camera_{camera_name}_fov_deg"),
                90.0,
            )
        )
        fx = float(image_w) / (2.0 * math.tan(math.radians(fov) * 0.5))
        fy = fx
        cx = (float(image_w) - 1.0) * 0.5
        cy = (float(image_h) - 1.0) * 0.5
        matrix = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        return matrix.unsqueeze(0).repeat(batch_size, 1, 1)

    def _get_camera_transform(
        self,
        camera_name: str,
        batch_size: int,
        data: dict[str, torch.Tensor] | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if data is not None:
            for key in (
                f"{camera_name}_camera_T_cam_ego",
                f"{camera_name}_T_cam_ego",
                f"rgb_{camera_name}_T_cam_ego",
            ):
                if key in data:
                    return self._as_batched_matrix(data[key], batch_size, 4, 4, device, dtype)
        value = _config_get(
            self.config,
            (f"{camera_name}_camera_T_cam_ego", f"dual_camera_{camera_name}_T_cam_ego"),
            None,
        )
        if value is not None:
            return self._as_batched_matrix(value, batch_size, 4, 4, device, dtype)
        return self._build_default_camera_transform(camera_name, batch_size, device, dtype)

    def _build_default_camera_transform(
        self,
        camera_name: str,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        baseline = float(
            _config_get(self.config, ("camera_baseline_m", "dual_camera_baseline_m"), 0.15)
        )
        height = float(
            _config_get(
                self.config,
                (f"{camera_name}_camera_height_m", f"dual_camera_{camera_name}_height_m"),
                1.4,
            )
        )
        default_y = -0.5 * baseline if camera_name == "left" else 0.5 * baseline
        translation = _config_get(
            self.config,
            (f"{camera_name}_camera_translation_m", f"dual_camera_{camera_name}_translation_m"),
            [0.0, default_y, height],
        )
        tx, ty, tz = [float(v) for v in translation]
        yaw = float(
            _config_get(
                self.config,
                (f"{camera_name}_camera_yaw_deg", f"dual_camera_{camera_name}_yaw_deg"),
                0.0,
            )
        )
        pitch = float(
            _config_get(
                self.config,
                (f"{camera_name}_camera_pitch_deg", f"dual_camera_{camera_name}_pitch_deg"),
                0.0,
            )
        )
        roll = float(
            _config_get(
                self.config,
                (f"{camera_name}_camera_roll_deg", f"dual_camera_{camera_name}_roll_deg"),
                0.0,
            )
        )

        default_right_axis = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        default_down_axis = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=dtype)
        default_forward_axis = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        rotation_ego = self._ego_rotation_matrix(yaw, pitch, roll, device, dtype)
        right_axis = rotation_ego @ default_right_axis
        down_axis = rotation_ego @ default_down_axis
        forward_axis = rotation_ego @ default_forward_axis
        r_cam_ego = torch.stack([right_axis, down_axis, forward_axis], dim=0)
        t_ego = torch.tensor([tx, ty, tz], device=device, dtype=dtype)
        t_cam = -r_cam_ego @ t_ego
        transform = torch.eye(4, device=device, dtype=dtype)
        transform[:3, :3] = r_cam_ego
        transform[:3, 3] = t_cam
        return transform.unsqueeze(0).repeat(batch_size, 1, 1)

    def _ego_rotation_matrix(
        self,
        yaw_deg: float,
        pitch_deg: float,
        roll_deg: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        roll = math.radians(roll_deg)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        rz = torch.tensor(
            [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        ry = torch.tensor(
            [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
            device=device,
            dtype=dtype,
        )
        rx = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
            device=device,
            dtype=dtype,
        )
        return rz @ ry @ rx

    def _as_batched_matrix(
        self,
        value: Any,
        batch_size: int,
        rows: int,
        cols: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            matrix = value.to(device=device, dtype=dtype, non_blocking=True)
        else:
            matrix = torch.tensor(value, device=device, dtype=dtype)
        if matrix.ndim == 2:
            if matrix.shape != (rows, cols):
                raise ValueError(f"Expected matrix shape {(rows, cols)}, got {tuple(matrix.shape)}.")
            return matrix.unsqueeze(0).repeat(batch_size, 1, 1)
        if matrix.ndim == 3:
            if matrix.shape[1:] != (rows, cols):
                raise ValueError(
                    f"Expected batched matrix shape Bx{rows}x{cols}, got {tuple(matrix.shape)}."
                )
            if matrix.shape[0] == 1 and batch_size > 1:
                return matrix.repeat(batch_size, 1, 1)
            if matrix.shape[0] != batch_size:
                raise ValueError(f"Expected {batch_size} matrices, got {matrix.shape[0]}.")
            return matrix
        raise ValueError(f"Expected 2D or 3D matrix, got {matrix.ndim}D.")


class Block(nn.Module):
    """Transformer block with self-attention and feed-forward layers."""

    @beartype
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_exp: int,
        attn_pdrop: float,
        resid_pdrop: float,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    @jt.jaxtyped(typechecker=beartype)
    def forward(
        self, x: jt.Float[torch.Tensor, "B T C"]
    ) -> jt.Float[torch.Tensor, "B T C"]:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SelfAttention(nn.Module):
    """Multi-head self-attention module."""

    @beartype
    def __init__(
        self, n_embd: int, n_head: int, attn_pdrop: float, resid_pdrop: float
    ) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.dropout = attn_pdrop
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    @jt.jaxtyped(typechecker=beartype)
    def forward(
        self, x: jt.Float[torch.Tensor, "B T C"]
    ) -> jt.Float[torch.Tensor, "B T C"]:
        b, t, c = x.size()
        k = self.key(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        q = self.query(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = self.value(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_drop(self.proj(y))
