from __future__ import annotations

import collections.abc

import torch
import torch.nn.functional as F


def sigmoid_focal_loss_one_hot_label(
    pred: torch.Tensor,
    label: torch.Tensor,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    """One-hot sigmoid focal loss without materializing the one-hot target."""
    label = label.clamp(min=0)
    neg_loss = (1.0 - alpha) * F.softplus(pred) * torch.sigmoid(pred).pow(gamma)
    pred_true = pred.gather(1, label.unsqueeze(1)).squeeze(1)
    neg_true = neg_loss.gather(1, label.unsqueeze(1)).squeeze(1)
    pos_loss = alpha * F.softplus(-pred_true) * torch.sigmoid(-pred_true).pow(gamma)
    return neg_loss.sum(dim=1) - neg_true + pos_loss


def sigmoid_focal_loss_one_hot_label_ignore(
    pred: torch.Tensor,
    label: torch.Tensor,
    alpha: float,
    gamma: float,
    ignore_index: int,
) -> torch.Tensor:
    valid_mask = label != ignore_index
    focal_loss = sigmoid_focal_loss_one_hot_label(pred, label, alpha, gamma)
    return focal_loss * valid_mask.to(dtype=focal_loss.dtype)


_COMPILED_FOCAL_LOSS_CACHE: dict[
    tuple[collections.abc.Callable, str], collections.abc.Callable
] = {}


def maybe_compile_focal_loss(
    fn: collections.abc.Callable,
    *,
    enabled: bool,
    mode: str,
) -> collections.abc.Callable:
    if not enabled:
        return fn

    key = (fn, mode)
    compiled_fn = _COMPILED_FOCAL_LOSS_CACHE.get(key)
    if compiled_fn is None:
        compiled_fn = torch.compile(
            fn,
            fullgraph=True,
            dynamic=False,
            backend="inductor",
            mode=mode,
        )
        _COMPILED_FOCAL_LOSS_CACHE[key] = compiled_fn
    return compiled_fn


def maybe_compile_loss(
    fn: collections.abc.Callable,
    *,
    enabled: bool,
    mode: str,
) -> collections.abc.Callable:
    return maybe_compile_focal_loss(fn, enabled=enabled, mode=mode)


def _source_weighted_mean(loss_per_sample: torch.Tensor, source_mask: torch.Tensor):
    return (loss_per_sample * source_mask).sum() / source_mask.sum().clamp(min=1)


def semantic_focal_loss_scalar(
    pred: torch.Tensor,
    label: torch.Tensor,
    source_mask: torch.Tensor,
    class_weights: torch.Tensor,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    per_pixel = sigmoid_focal_loss_one_hot_label(pred.float(), label, alpha, gamma)
    per_pixel = per_pixel * class_weights.to(dtype=per_pixel.dtype)[label]
    return _source_weighted_mean(per_pixel.mean(dim=(1, 2)), source_mask.float())


def semantic_cross_entropy_loss_scalar(
    pred: torch.Tensor,
    label: torch.Tensor,
    source_mask: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    per_pixel = F.cross_entropy(
        pred.float(),
        label,
        weight=class_weights.to(dtype=pred.dtype),
        reduction="none",
    )
    return _source_weighted_mean(per_pixel.mean(dim=(1, 2)), source_mask.float())


def bev_focal_loss_scalar(
    pred: torch.Tensor,
    label: torch.Tensor,
    source_mask: torch.Tensor,
    valid_bev_pixels: torch.Tensor,
    class_weights: torch.Tensor,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    valid_mask = valid_bev_pixels > 0.5
    visible_label = torch.where(valid_mask, label, torch.full_like(label, -1))
    per_pixel = sigmoid_focal_loss_one_hot_label_ignore(
        pred.float(), visible_label, alpha, gamma, -1
    )
    safe_label = visible_label.clamp(min=0)
    per_pixel = per_pixel * class_weights.to(dtype=per_pixel.dtype)[safe_label]
    return _source_weighted_mean(per_pixel.mean(dim=(1, 2)), source_mask.float())


def bev_cross_entropy_loss_scalar(
    pred: torch.Tensor,
    label: torch.Tensor,
    source_mask: torch.Tensor,
    valid_bev_pixels: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    valid_mask = valid_bev_pixels > 0.5
    visible_label = torch.where(valid_mask, label, torch.full_like(label, -1))
    per_pixel = F.cross_entropy(
        pred.float(),
        visible_label.long(),
        weight=class_weights.to(dtype=pred.dtype),
        reduction="none",
        ignore_index=-1,
    )
    return _source_weighted_mean(per_pixel.mean(dim=(1, 2)), source_mask.float())


def gaussian_focal_loss_none(
    pred: torch.Tensor,
    gaussian_target: torch.Tensor,
    alpha: float = 2.0,
    gamma: float = 4.0,
) -> torch.Tensor:
    eps = 1e-12
    pos_weights = gaussian_target.eq(1)
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    return pos_loss + neg_loss


def center_net_loss_with_velocity(
    center_heatmap_pred: torch.Tensor,
    wh_pred: torch.Tensor,
    offset_pred: torch.Tensor,
    yaw_class_pred: torch.Tensor,
    yaw_res_pred: torch.Tensor,
    velocity_pred: torch.Tensor,
    center_heatmap_target: torch.Tensor,
    wh_target: torch.Tensor,
    offset_target: torch.Tensor,
    yaw_class_target: torch.Tensor,
    yaw_res_target: torch.Tensor,
    velocity_target: torch.Tensor,
    pixel_weight: torch.Tensor,
    avg_factor: torch.Tensor,
    source_mask: torch.Tensor,
    class_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source_mask = source_mask.float()
    avg_factor_clamped = avg_factor.float() + torch.finfo(torch.float32).eps
    source_count = source_mask.sum().clamp(min=1)
    class_weights = class_weights.to(dtype=torch.float32)
    heatmap_class_weights = class_weights.view(1, -1, 1, 1)
    pixel_class_weights = (
        center_heatmap_target.float() * heatmap_class_weights
    ).amax(dim=1, keepdim=True).clamp(min=1.0)

    heatmap_per_sample = gaussian_focal_loss_none(
        center_heatmap_pred.float(), center_heatmap_target.float()
    )
    heatmap_per_sample = (heatmap_per_sample * heatmap_class_weights).sum(
        dim=(1, 2, 3)
    )
    heatmap_per_sample = heatmap_per_sample / avg_factor_clamped
    loss_heatmap = (heatmap_per_sample * source_mask).sum() / source_count

    pixel_weight = pixel_weight.float()
    weighted_pixel_weight = pixel_weight * pixel_class_weights
    wh_channels = float(wh_pred.shape[1])
    wh_per_sample = (
        (wh_pred.float() - wh_target.float()).abs() * weighted_pixel_weight
    ).sum(dim=(1, 2, 3))
    wh_per_sample = wh_per_sample / (avg_factor_clamped * wh_channels)
    loss_wh = (wh_per_sample * source_mask).sum() / source_count

    offset_per_sample = (
        (offset_pred.float() - offset_target.float()).abs() * weighted_pixel_weight
    ).sum(dim=(1, 2, 3))
    offset_per_sample = offset_per_sample / (avg_factor_clamped * wh_channels)
    loss_offset = (offset_per_sample * source_mask).sum() / source_count

    yaw_class_per_sample = (
        F.cross_entropy(
            yaw_class_pred.float(),
            yaw_class_target.long(),
            reduction="none",
        )
        * weighted_pixel_weight[:, 0]
    ).sum(dim=(1, 2))
    yaw_class_per_sample = yaw_class_per_sample / avg_factor_clamped
    loss_yaw_class = (yaw_class_per_sample * source_mask).sum() / source_count

    yaw_res_per_sample = (
        F.smooth_l1_loss(
            yaw_res_pred.float(), yaw_res_target.float(), reduction="none"
        )
        * weighted_pixel_weight[:, 0:1]
    ).sum(dim=(1, 2, 3))
    yaw_res_per_sample = yaw_res_per_sample / avg_factor_clamped
    loss_yaw_res = (yaw_res_per_sample * source_mask).sum() / source_count

    velocity_per_sample = (
        (velocity_pred.float() - velocity_target.float()).abs()
        * weighted_pixel_weight[:, 0:1]
    ).sum(dim=(1, 2, 3))
    velocity_per_sample = velocity_per_sample / avg_factor_clamped
    loss_velocity = (velocity_per_sample * source_mask).sum() / source_count

    return (
        loss_heatmap,
        loss_wh,
        loss_offset,
        loss_yaw_class,
        loss_yaw_res,
        loss_velocity,
    )


def center_net_loss_no_velocity(
    center_heatmap_pred: torch.Tensor,
    wh_pred: torch.Tensor,
    offset_pred: torch.Tensor,
    yaw_class_pred: torch.Tensor,
    yaw_res_pred: torch.Tensor,
    center_heatmap_target: torch.Tensor,
    wh_target: torch.Tensor,
    offset_target: torch.Tensor,
    yaw_class_target: torch.Tensor,
    yaw_res_target: torch.Tensor,
    pixel_weight: torch.Tensor,
    avg_factor: torch.Tensor,
    source_mask: torch.Tensor,
    class_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    zero_velocity_pred = yaw_res_pred.new_zeros(yaw_res_pred.shape)
    zero_velocity_target = yaw_res_target.new_zeros(yaw_res_target.shape)
    return center_net_loss_with_velocity(
        center_heatmap_pred,
        wh_pred,
        offset_pred,
        yaw_class_pred,
        yaw_res_pred,
        zero_velocity_pred,
        center_heatmap_target,
        wh_target,
        offset_target,
        yaw_class_target,
        yaw_res_target,
        zero_velocity_target,
        pixel_weight,
        avg_factor,
        source_mask,
        class_weights,
    )
