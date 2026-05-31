#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.metadata
import json
import sys
import types
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

try:
    import pkg_resources  # noqa: F401
except ModuleNotFoundError:
    pkg_resources_shim = types.ModuleType("pkg_resources")

    class DistributionNotFound(Exception):
        pass

    def get_distribution(name: str) -> SimpleNamespace:
        try:
            return SimpleNamespace(version=importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError as exc:
            raise DistributionNotFound(name) from exc

    pkg_resources_shim.DistributionNotFound = DistributionNotFound
    pkg_resources_shim.get_distribution = get_distribution
    sys.modules["pkg_resources"] = pkg_resources_shim

from lead.tfv6.tfv6 import TFv6
from lead.training.config_training import TrainingConfig


DEFAULT_CHECKPOINT = "outputs/local_training/posttrain/model_0011.pth"
DEFAULT_CONFIG = "outputs/model1/config_dual_front_camera.json"
DEFAULT_OUTPUT_DIR = "outputs/model1/planning_only"


@contextlib.contextmanager
def clean_argv() -> Iterator[None]:
    old_argv = sys.argv[:]
    sys.argv = [old_argv[0]]
    try:
        yield
    finally:
        sys.argv = old_argv


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def make_config(config_dict: dict[str, Any]) -> TrainingConfig:
    with clean_argv():
        return TrainingConfig(config_dict, raise_error_on_missing_key=False)


def strip_known_prefixes(state_dict: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    stripped: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        stripped[key] = value
    return stripped


def query_slices(config_dict: dict[str, Any]) -> dict[str, slice]:
    idx = 0
    slices: dict[str, slice] = {}
    if bool(config_dict.get("predict_spatial_path", False)):
        count = int(config_dict["num_route_points_prediction"])
        slices["route"] = slice(idx, idx + count)
        idx += count
    if bool(config_dict.get("predict_temporal_spatial_waypoints", False)):
        count = int(config_dict["num_way_points_prediction"])
        slices["waypoints"] = slice(idx, idx + count)
        idx += count
    if bool(config_dict.get("predict_target_speed", False)):
        slices["target_speed"] = slice(idx, idx + 1)
    return slices


def convert_planning_query(
    state_dict: OrderedDict[str, torch.Tensor],
    source_config: dict[str, Any],
    target_config: dict[str, Any],
) -> None:
    key = "planning_decoder.query"
    if key not in state_dict:
        return

    source_slices = query_slices(source_config)
    target_parts: list[torch.Tensor] = []
    for name in ("waypoints", "target_speed"):
        if name not in source_slices:
            raise KeyError(f"Source checkpoint has no planning query slice for {name!r}.")
        if name in query_slices(target_config):
            target_parts.append(state_dict[key][:, source_slices[name], :])

    if not target_parts:
        raise RuntimeError("Planning-only config produced no query tokens.")
    state_dict[key] = torch.cat(target_parts, dim=1).contiguous()


def make_planning_only_config(
    source_config: dict[str, Any],
    output_checkpoint: Path,
    device: str,
) -> dict[str, Any]:
    config = copy.deepcopy(source_config)
    config.update(
        {
            "load_file": str(output_checkpoint),
            "device": device,
            "continue_failed_training": False,
            "compile": False,
            "use_mixed_precision_training": False,
            "dual_camera_pretrained": False,
            "use_semantic": False,
            "use_depth": False,
            "use_bev_semantic": False,
            "detect_boxes": False,
            "radar_detection": False,
            "use_radar_detection": False,
            "debug_boxes_visualization": False,
            "use_planning_decoder": True,
            "predict_spatial_path": False,
            "predict_temporal_spatial_waypoints": True,
            "predict_target_speed": True,
            "use_navsim_data": False,
        }
    )
    return config


def filter_state_dict_for_model(
    source_state: OrderedDict[str, torch.Tensor],
    target_model: torch.nn.Module,
) -> tuple[OrderedDict[str, torch.Tensor], list[str], list[str]]:
    target_state = target_model.state_dict()
    converted: OrderedDict[str, torch.Tensor] = OrderedDict()
    dropped: list[str] = []

    for key, value in source_state.items():
        target_value = target_state.get(key)
        if target_value is None:
            dropped.append(key)
            continue
        if tuple(value.shape) != tuple(target_value.shape):
            dropped.append(key)
            continue
        converted[key] = value.detach().cpu()

    missing = [key for key in target_state if key not in converted]
    return converted, dropped, missing


def summarize_prefixes(keys: list[str]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for key in keys:
        prefix = key.split(".", 1)[0]
        summary[prefix] = summary.get(prefix, 0) + 1
    return dict(sorted(summary.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert dual-front-camera TFv6 weights to planning-only weights: "
            "future waypoints + target speed."
        )
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-checkpoint-name", default="model_planning_only.pth")
    parser.add_argument("--output-config-name", default="config_planning_only.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--strict-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output_checkpoint = output_dir / args.output_checkpoint_name
    output_config = output_dir / args.output_config_name

    source_config = load_json(config_path)
    target_config = make_planning_only_config(source_config, output_checkpoint, args.device)

    source_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    source_state = strip_known_prefixes(source_state)
    convert_planning_query(source_state, source_config, target_config)

    config = make_config(target_config)
    model = TFv6(config.device, config)

    converted_state, dropped_keys, missing_keys = filter_state_dict_for_model(
        source_state,
        model,
    )
    incompatible_missing, unexpected = model.load_state_dict(converted_state, strict=False)
    unexpected = list(unexpected)
    incompatible_missing = list(incompatible_missing)

    if args.strict_check and (incompatible_missing or unexpected):
        raise RuntimeError(
            "Converted state did not match target model. "
            f"missing={incompatible_missing}, unexpected={unexpected}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(converted_state, output_checkpoint)
    write_json(output_config, target_config)

    metadata = {
        "source_checkpoint": str(checkpoint),
        "source_config": str(config_path),
        "output_checkpoint": str(output_checkpoint),
        "output_config": str(output_config),
        "kept_tensors": len(converted_state),
        "dropped_tensors": len(dropped_keys),
        "missing_tensors_initialized_by_model": len(missing_keys),
        "dropped_prefixes": summarize_prefixes(dropped_keys),
        "missing_prefixes": summarize_prefixes(missing_keys),
        "outputs_left_enabled": [
            "pred_future_waypoints",
            "pred_target_speed_distribution",
            "pred_target_speed_scalar",
        ],
    }
    write_json(output_dir / "conversion_metadata.json", metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
