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
