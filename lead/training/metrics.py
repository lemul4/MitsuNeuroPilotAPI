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
