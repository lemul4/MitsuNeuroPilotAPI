import numpy as np
import torch
import torch.nn.functional as F

from lead.common.constants import TransfuserBoundingBoxIndex
from lead.inference.inference_utils import iou_bbs


def mean_iou_from_logits(
    logits: torch.Tensor,
    label: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """Compute batch mIoU from semantic logits and integer labels."""
    if logits.shape[-2:] != label.shape[-2:]:
        logits = F.interpolate(
            logits.float(), size=label.shape[-2:], mode="bilinear", align_corners=False
        )

    pred = logits.argmax(dim=1)
    valid_mask = torch.ones_like(label, dtype=torch.bool)
    if ignore_index is not None:
        valid_mask = label != ignore_index

    ious = []
    for class_idx in range(num_classes):
        pred_mask = (pred == class_idx) & valid_mask
        label_mask = (label == class_idx) & valid_mask
        union = pred_mask | label_mask
        if union.any():
            intersection = pred_mask & label_mask
            ious.append(intersection.sum().float() / union.sum().float().clamp(min=1))

    if len(ious) == 0:
        return logits.new_tensor(0.0, dtype=torch.float32)
    return torch.stack(ious).mean()


def mean_average_precision_bev(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    num_classes: int,
    iou_threshold: float,
    score_threshold: float,
) -> float:
    """Compute batch AP averaged over classes for rotated BEV boxes."""
    aps = []
    for class_idx in range(num_classes):
        class_predictions = []
        class_gt_by_sample = []

        for sample_idx in range(gt_boxes.shape[0]):
            sample_gt = _valid_gt_boxes(gt_boxes[sample_idx])
            sample_gt = sample_gt[
                sample_gt[:, TransfuserBoundingBoxIndex.CLASS].astype(int)
                == int(class_idx)
            ]
            class_gt_by_sample.append(sample_gt)

            sample_pred = pred_boxes[sample_idx]
            sample_pred = sample_pred[
                sample_pred[:, TransfuserBoundingBoxIndex.SCORE] >= score_threshold
            ]
            sample_pred = sample_pred[
                (sample_pred[:, TransfuserBoundingBoxIndex.W] > 0.0)
                & (sample_pred[:, TransfuserBoundingBoxIndex.H] > 0.0)
            ]
            sample_pred = sample_pred[np.isfinite(sample_pred).all(axis=1)]
            sample_pred = sample_pred[
                sample_pred[:, TransfuserBoundingBoxIndex.CLASS].astype(int)
                == int(class_idx)
            ]
            for pred in sample_pred:
                class_predictions.append((sample_idx, pred))

        num_gt = sum(len(sample_gt) for sample_gt in class_gt_by_sample)
        if num_gt == 0:
            continue

        class_predictions.sort(
            key=lambda item: item[1][TransfuserBoundingBoxIndex.SCORE], reverse=True
        )
        matched = [
            np.zeros(len(sample_gt), dtype=bool) for sample_gt in class_gt_by_sample
        ]
        true_positive = np.zeros(len(class_predictions), dtype=np.float32)
        false_positive = np.zeros(len(class_predictions), dtype=np.float32)

        for pred_idx, (sample_idx, pred) in enumerate(class_predictions):
            sample_gt = class_gt_by_sample[sample_idx]
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(sample_gt):
                if matched[sample_idx][gt_idx]:
                    continue
                try:
                    iou = iou_bbs(
                        pred[
                            [
                                TransfuserBoundingBoxIndex.X,
                                TransfuserBoundingBoxIndex.Y,
                                TransfuserBoundingBoxIndex.W,
                                TransfuserBoundingBoxIndex.H,
                                TransfuserBoundingBoxIndex.YAW,
                            ]
                        ],
                        gt[
                            [
                                TransfuserBoundingBoxIndex.X,
                                TransfuserBoundingBoxIndex.Y,
                                TransfuserBoundingBoxIndex.W,
                                TransfuserBoundingBoxIndex.H,
                                TransfuserBoundingBoxIndex.YAW,
                            ]
                        ],
                    )
                except Exception:
                    iou = 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                true_positive[pred_idx] = 1.0
                matched[sample_idx][best_gt_idx] = True
            else:
                false_positive[pred_idx] = 1.0

        aps.append(_average_precision(true_positive, false_positive, num_gt))

    if len(aps) == 0:
        return 0.0
    return float(np.mean(aps))


def mean_average_precision_bev_torch(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    num_classes: int,
    iou_threshold: float,
    score_threshold: float,
    iou_mode: str = "aabb",
    metric_dtype: str = "float16",
    matching_mode: str = "max_iou",
) -> torch.Tensor:
    """Compute batch AP for rotated BEV boxes without GPU->CPU synchronization."""
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return pred_boxes.new_tensor(0.0, dtype=torch.float32)

    iou_mode = iou_mode.lower()
    if iou_mode not in {"aabb", "rotated"}:
        raise ValueError(
            f"Unknown CenterNet mAP IoU mode: {iou_mode!r}. "
            "Expected 'aabb' or 'rotated'."
        )
    matching_mode = matching_mode.lower()
    if matching_mode not in {"max_iou", "greedy"}:
        raise ValueError(
            f"Unknown CenterNet mAP matching mode: {matching_mode!r}. "
            "Expected 'max_iou' or 'greedy'."
        )

    metric_torch_dtype = _metric_torch_dtype(
        metric_dtype=metric_dtype,
        device=pred_boxes.device,
        iou_mode=iou_mode,
    )
    pred_boxes = pred_boxes.detach().to(dtype=metric_torch_dtype)
    gt_boxes = gt_boxes.detach().to(device=pred_boxes.device, dtype=metric_torch_dtype)
    iou_fn = aabb_envelope_iou_torch if iou_mode == "aabb" else rotated_box_iou_torch
    aps = []

    for class_idx in range(num_classes):
        class_predictions = []
        class_gt_by_sample = []

        for sample_idx in range(gt_boxes.shape[0]):
            sample_gt = _valid_gt_boxes_torch(gt_boxes[sample_idx])
            sample_gt = sample_gt[
                sample_gt[:, TransfuserBoundingBoxIndex.CLASS].long() == class_idx
            ]
            class_gt_by_sample.append(sample_gt)

            sample_pred = pred_boxes[sample_idx]
            sample_pred = sample_pred[
                sample_pred[:, TransfuserBoundingBoxIndex.SCORE] >= score_threshold
            ]
            sample_pred = sample_pred[
                (sample_pred[:, TransfuserBoundingBoxIndex.W] > 0.0)
                & (sample_pred[:, TransfuserBoundingBoxIndex.H] > 0.0)
            ]
            sample_pred = sample_pred[torch.isfinite(sample_pred).all(dim=1)]
            sample_pred = sample_pred[
                sample_pred[:, TransfuserBoundingBoxIndex.CLASS].long() == class_idx
            ]
            if sample_pred.numel() > 0:
                class_predictions.append(
                    (
                        torch.full(
                            (sample_pred.shape[0],),
                            sample_idx,
                            dtype=torch.long,
                            device=sample_pred.device,
                        ),
                        sample_pred,
                    )
                )

        num_gt = sum(int(sample_gt.shape[0]) for sample_gt in class_gt_by_sample)
        if num_gt == 0:
            continue
        if len(class_predictions) == 0:
            aps.append(pred_boxes.new_tensor(0.0, dtype=torch.float32))
            continue

        pred_sample_idx = torch.cat([item[0] for item in class_predictions])
        class_pred = torch.cat([item[1] for item in class_predictions])
        order = torch.argsort(
            class_pred[:, TransfuserBoundingBoxIndex.SCORE], descending=True
        )
        pred_sample_idx = pred_sample_idx[order]
        class_pred = class_pred[order]
        pred_sample_idx_list = pred_sample_idx.cpu().tolist()

        matched = [
            torch.zeros(sample_gt.shape[0], dtype=torch.bool, device=pred_boxes.device)
            for sample_gt in class_gt_by_sample
        ]
        true_positive = torch.zeros(
            class_pred.shape[0], dtype=class_pred.dtype, device=pred_boxes.device
        )
        false_positive = torch.zeros_like(true_positive)

        if matching_mode == "max_iou":
            best_iou_by_pred = torch.zeros_like(true_positive)
            for sample_idx, sample_gt in enumerate(class_gt_by_sample):
                if sample_gt.shape[0] == 0:
                    continue
                pred_indices = torch.where(pred_sample_idx == sample_idx)[0]
                if pred_indices.numel() == 0:
                    continue
                iou_matrix = iou_fn(class_pred[pred_indices], sample_gt)
                best_iou_by_pred[pred_indices] = iou_matrix.max(dim=1).values

            true_positive = (best_iou_by_pred >= iou_threshold).to(class_pred.dtype)
            true_positive = _cap_true_positives_torch(true_positive, num_gt)
            false_positive = 1.0 - true_positive
            aps.append(_average_precision_torch(true_positive, false_positive, num_gt))
            continue

        iou_rows_by_pred_idx: dict[int, torch.Tensor] = {}
        for sample_idx, sample_gt in enumerate(class_gt_by_sample):
            if sample_gt.shape[0] == 0:
                continue
            pred_indices = torch.where(pred_sample_idx == sample_idx)[0]
            if pred_indices.numel() == 0:
                continue
            iou_matrix = iou_fn(class_pred[pred_indices], sample_gt)
            for pred_idx, iou_row in zip(
                pred_indices.cpu().tolist(), iou_matrix, strict=False
            ):
                iou_rows_by_pred_idx[pred_idx] = iou_row

        for pred_idx in range(class_pred.shape[0]):
            sample_idx = pred_sample_idx_list[pred_idx]
            sample_gt = class_gt_by_sample[sample_idx]
            if sample_gt.shape[0] == 0:
                false_positive[pred_idx] = 1.0
                continue

            ious = iou_rows_by_pred_idx[pred_idx]
            ious = ious.masked_fill(matched[sample_idx], -1.0)
            best_iou, best_gt_idx = torch.max(ious, dim=0)
            if best_iou >= iou_threshold:
                true_positive[pred_idx] = 1.0
                matched[sample_idx][best_gt_idx] = True
            else:
                false_positive[pred_idx] = 1.0

        aps.append(_average_precision_torch(true_positive, false_positive, num_gt))

    if len(aps) == 0:
        return pred_boxes.new_tensor(0.0, dtype=torch.float32)
    return torch.stack(aps).float().mean()


def _metric_torch_dtype(
    metric_dtype: str,
    device: torch.device,
    iou_mode: str,
) -> torch.dtype:
    if iou_mode == "rotated":
        return torch.float32
    metric_dtype = metric_dtype.lower()
    if device.type != "cuda":
        return torch.float32
    if metric_dtype in {"float16", "fp16", "half"}:
        return torch.float16
    if metric_dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if metric_dtype in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(
        f"Unknown additional metrics dtype: {metric_dtype!r}. "
        "Expected float16, bfloat16, or float32."
    )


def aabb_envelope_iou_torch(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Fast IoU of axis-aligned envelopes around rotated boxes."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    xyxy1 = _rotated_box_envelope_xyxy_torch(boxes1)
    xyxy2 = _rotated_box_envelope_xyxy_torch(boxes2)

    left_top = torch.maximum(xyxy1[:, None, :2], xyxy2[None, :, :2])
    right_bottom = torch.minimum(xyxy1[:, None, 2:], xyxy2[None, :, 2:])
    wh = (right_bottom - left_top).clamp(min=0.0)
    intersection = wh[..., 0] * wh[..., 1]

    area1 = (
        (xyxy1[:, 2] - xyxy1[:, 0]).clamp(min=0.0)
        * (xyxy1[:, 3] - xyxy1[:, 1]).clamp(min=0.0)
    )[:, None]
    area2 = (
        (xyxy2[:, 2] - xyxy2[:, 0]).clamp(min=0.0)
        * (xyxy2[:, 3] - xyxy2[:, 1]).clamp(min=0.0)
    )[None, :]
    return intersection / (area1 + area2 - intersection).clamp(min=eps)


def rotated_box_iou_torch(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Pairwise IoU for rotated rectangles encoded as x, y, half_w, half_h, yaw."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    corners1 = _rotated_box_corners_torch(boxes1)
    corners2 = _rotated_box_corners_torch(boxes2)
    intersection = _convex_quad_intersection_area_torch(corners1, corners2, eps=eps)
    area1 = (
        4.0
        * boxes1[:, TransfuserBoundingBoxIndex.W].clamp(min=0.0)
        * boxes1[:, TransfuserBoundingBoxIndex.H].clamp(min=0.0)
    )[:, None]
    area2 = (
        4.0
        * boxes2[:, TransfuserBoundingBoxIndex.W].clamp(min=0.0)
        * boxes2[:, TransfuserBoundingBoxIndex.H].clamp(min=0.0)
    )[None, :]
    union = (area1 + area2 - intersection).clamp(min=eps)
    return intersection / union


def _valid_gt_boxes(gt_boxes: np.ndarray) -> np.ndarray:
    valid_mask = (
        (gt_boxes[:, TransfuserBoundingBoxIndex.X] != 0.0)
        | (gt_boxes[:, TransfuserBoundingBoxIndex.Y] != 0.0)
        | (gt_boxes[:, TransfuserBoundingBoxIndex.W] != 0.0)
        | (gt_boxes[:, TransfuserBoundingBoxIndex.H] != 0.0)
    )
    valid_mask &= gt_boxes[:, TransfuserBoundingBoxIndex.W] > 0.0
    valid_mask &= gt_boxes[:, TransfuserBoundingBoxIndex.H] > 0.0
    valid_mask &= np.isfinite(gt_boxes).all(axis=1)
    return gt_boxes[valid_mask]


def _valid_gt_boxes_torch(gt_boxes: torch.Tensor) -> torch.Tensor:
    valid_mask = (
        (gt_boxes[:, TransfuserBoundingBoxIndex.X] != 0.0)
        | (gt_boxes[:, TransfuserBoundingBoxIndex.Y] != 0.0)
        | (gt_boxes[:, TransfuserBoundingBoxIndex.W] != 0.0)
        | (gt_boxes[:, TransfuserBoundingBoxIndex.H] != 0.0)
    )
    valid_mask &= gt_boxes[:, TransfuserBoundingBoxIndex.W] > 0.0
    valid_mask &= gt_boxes[:, TransfuserBoundingBoxIndex.H] > 0.0
    valid_mask &= torch.isfinite(gt_boxes).all(dim=1)
    return gt_boxes[valid_mask]


def _average_precision(
    true_positive: np.ndarray, false_positive: np.ndarray, num_gt: int
) -> float:
    if len(true_positive) == 0:
        return 0.0

    tp_cumsum = np.cumsum(true_positive)
    fp_cumsum = np.cumsum(false_positive)
    recalls = tp_cumsum / max(num_gt, 1)
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-12)

    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for idx in range(len(precisions) - 1, 0, -1):
        precisions[idx - 1] = max(precisions[idx - 1], precisions[idx])

    changing_points = np.where(recalls[1:] != recalls[:-1])[0]
    return float(
        np.sum(
            (recalls[changing_points + 1] - recalls[changing_points])
            * precisions[changing_points + 1]
        )
    )


def _average_precision_torch(
    true_positive: torch.Tensor, false_positive: torch.Tensor, num_gt: int
) -> torch.Tensor:
    if true_positive.numel() == 0:
        return true_positive.new_tensor(0.0, dtype=torch.float32)

    tp_cumsum = torch.cumsum(true_positive, dim=0)
    fp_cumsum = torch.cumsum(false_positive, dim=0)
    recalls = tp_cumsum / max(num_gt, 1)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum).clamp(min=1e-12)

    recalls = torch.cat(
        [recalls.new_zeros(1), recalls, recalls.new_ones(1)],
        dim=0,
    )
    precisions = torch.cat(
        [precisions.new_zeros(1), precisions, precisions.new_zeros(1)],
        dim=0,
    )
    for idx in range(precisions.numel() - 1, 0, -1):
        precisions[idx - 1] = torch.maximum(precisions[idx - 1], precisions[idx])

    changing_points = torch.where(recalls[1:] != recalls[:-1])[0]
    if changing_points.numel() == 0:
        return true_positive.new_tensor(0.0, dtype=torch.float32)
    return torch.sum(
        (recalls[changing_points + 1] - recalls[changing_points])
        * precisions[changing_points + 1]
    )


def _cap_true_positives_torch(
    true_positive: torch.Tensor,
    num_gt: int,
) -> torch.Tensor:
    tp_cumsum = torch.cumsum(true_positive, dim=0)
    return true_positive * (tp_cumsum <= max(num_gt, 1)).to(true_positive.dtype)


def _rotated_box_corners_torch(boxes: torch.Tensor) -> torch.Tensor:
    half_w = boxes[:, TransfuserBoundingBoxIndex.W]
    half_h = boxes[:, TransfuserBoundingBoxIndex.H]
    x = boxes[:, TransfuserBoundingBoxIndex.X]
    y = boxes[:, TransfuserBoundingBoxIndex.Y]
    yaw = boxes[:, TransfuserBoundingBoxIndex.YAW]

    local = torch.stack(
        [
            torch.stack([-half_w, -half_h], dim=1),
            torch.stack([half_w, -half_h], dim=1),
            torch.stack([half_w, half_h], dim=1),
            torch.stack([-half_w, half_h], dim=1),
        ],
        dim=1,
    )
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    rot = torch.stack(
        [
            torch.stack([cos_yaw, -sin_yaw], dim=1),
            torch.stack([sin_yaw, cos_yaw], dim=1),
        ],
        dim=1,
    )
    return torch.bmm(local, rot.transpose(1, 2)) + torch.stack([x, y], dim=1)[:, None]


def _rotated_box_envelope_xyxy_torch(boxes: torch.Tensor) -> torch.Tensor:
    half_w = boxes[:, TransfuserBoundingBoxIndex.W].clamp(min=0.0)
    half_h = boxes[:, TransfuserBoundingBoxIndex.H].clamp(min=0.0)
    yaw = boxes[:, TransfuserBoundingBoxIndex.YAW]
    cos_yaw = torch.cos(yaw).abs()
    sin_yaw = torch.sin(yaw).abs()
    half_extent_x = cos_yaw * half_w + sin_yaw * half_h
    half_extent_y = sin_yaw * half_w + cos_yaw * half_h
    center = boxes[
        :,
        [
            TransfuserBoundingBoxIndex.X,
            TransfuserBoundingBoxIndex.Y,
        ],
    ]
    half_extent = torch.stack([half_extent_x, half_extent_y], dim=1)
    return torch.cat([center - half_extent, center + half_extent], dim=1)


def _convex_quad_intersection_area_torch(
    corners1: torch.Tensor,
    corners2: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    p = corners1[:, None]
    q = corners2[None]

    p_inside_q = _points_in_convex_quad_torch(p, q, eps=eps)
    q_inside_p = _points_in_convex_quad_torch(q, p, eps=eps)
    p_candidates = p.expand(-1, corners2.shape[0], -1, -1)
    q_candidates = q.expand(corners1.shape[0], -1, -1, -1)

    edge_points, edge_valid = _quad_edge_intersections_torch(corners1, corners2, eps)

    candidates = torch.cat([p_candidates, q_candidates, edge_points], dim=2)
    valid = torch.cat([p_inside_q, q_inside_p, edge_valid], dim=2)
    valid_count = valid.sum(dim=2)
    weighted_candidates = candidates * valid.unsqueeze(-1).to(candidates.dtype)
    centroid = weighted_candidates.sum(dim=2) / valid_count.clamp(min=1)[:, :, None]

    rel = candidates - centroid[:, :, None, :]
    angles = torch.atan2(rel[..., 1], rel[..., 0])
    angles = angles.masked_fill(~valid, torch.inf)
    order = torch.argsort(angles, dim=2)
    sorted_candidates = torch.gather(
        candidates,
        2,
        order[..., None].expand(-1, -1, -1, 2),
    )
    sorted_valid = torch.gather(valid, 2, order)

    first = sorted_candidates[:, :, 0]
    next_candidates = torch.roll(sorted_candidates, shifts=-1, dims=2)
    candidate_indices = torch.arange(
        sorted_candidates.shape[2],
        device=sorted_candidates.device,
    )
    is_last_valid = candidate_indices[None, None] == (valid_count[:, :, None] - 1)
    next_candidates = torch.where(
        is_last_valid[..., None],
        first[:, :, None, :],
        next_candidates,
    )

    cross = (
        sorted_candidates[..., 0] * next_candidates[..., 1]
        - sorted_candidates[..., 1] * next_candidates[..., 0]
    )
    area = 0.5 * torch.abs((cross * sorted_valid.to(cross.dtype)).sum(dim=2))
    return torch.where(valid_count >= 3, area, torch.zeros_like(area))


def _points_in_convex_quad_torch(
    points: torch.Tensor,
    quad: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    edges = torch.roll(quad, shifts=-1, dims=2) - quad
    rel = points[:, :, :, None, :] - quad[:, :, None, :, :]
    cross = (
        edges[:, :, None, :, 0] * rel[..., 1] - edges[:, :, None, :, 1] * rel[..., 0]
    )
    return torch.all(cross >= -eps, dim=-1)


def _quad_edge_intersections_torch(
    corners1: torch.Tensor,
    corners2: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    p = corners1[:, None, :, None, :]
    p_next = torch.roll(corners1, shifts=-1, dims=1)[:, None, :, None, :]
    q = corners2[None, :, None, :, :]
    q_next = torch.roll(corners2, shifts=-1, dims=1)[None, :, None, :, :]
    r = p_next - p
    s = q_next - q
    q_minus_p = q - p

    r_cross_s = _cross2d_torch(r, s)
    parallel = r_cross_s.abs() <= eps
    safe_denominator = torch.where(
        parallel,
        torch.ones_like(r_cross_s),
        r_cross_s,
    )
    t = _cross2d_torch(q_minus_p, s) / safe_denominator
    u = _cross2d_torch(q_minus_p, r) / safe_denominator
    valid = (
        (~parallel) & (t >= -eps) & (t <= 1.0 + eps) & (u >= -eps) & (u <= 1.0 + eps)
    )
    points = p + t[..., None] * r
    return points.reshape(corners1.shape[0], corners2.shape[0], 16, 2), valid.reshape(
        corners1.shape[0], corners2.shape[0], 16
    )


def _cross2d_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
