import argparse
import json
import os
import sys

import torch
from lead.data_loader.carla_dataset import CARLAData
from lead.training.config_training import TrainingConfig
from tqdm import tqdm


def _set_config_value(config: TrainingConfig, key: str, value):
    try:
        setattr(config, key, value)
    except Exception:
        pass
    if hasattr(config, "_loaded_config"):
        config._loaded_config[key] = value


def _dataset_has_modality(carla_data_root: str, modality: str) -> bool:
    if not os.path.isdir(carla_data_root):
        return False
    for scenario in os.listdir(carla_data_root):
        scenario_dir = os.path.join(carla_data_root, scenario)
        if not os.path.isdir(scenario_dir):
            continue
        for route in os.listdir(scenario_dir):
            route_dir = os.path.join(scenario_dir, route)
            if os.path.isdir(os.path.join(route_dir, modality)):
                return True
    return False


def _load_config(config_path: str | None) -> TrainingConfig:
    loaded_config = None
    if config_path:
        with open(config_path) as f:
            loaded_config = json.load(f)
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        return TrainingConfig(loaded_config=loaded_config)
    finally:
        sys.argv = original_argv


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    default=None,
    help="Optional JSON training config, e.g. outputs/model1/config_dual_front_camera.json.",
)
parser.add_argument(
    "--carla-root",
    default=None,
    help="CARLA dataset root containing the data/ folder.",
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=None,
    help="Override DataLoader worker count. Defaults to config.assigned_cpu_cores.",
)
args = parser.parse_args()

config = _load_config(args.config)
if args.carla_root:
    _set_config_value(config, "carla_root", args.carla_root)
elif not os.path.isdir(config.carla_data):
    dual_camera_root = "data/carla_leaderboard2_dual_cameras"
    if (
        "carla_leaderboard2_dual_cameras" in str(config.carla_root)
        and os.path.isdir(os.path.join(dual_camera_root, "data"))
    ):
        _set_config_value(config, "carla_root", dual_camera_root)

_set_config_value(config, "use_persistent_cache", True)
_set_config_value(config, "use_training_session_cache", False)
_set_config_value(config, "force_rebuild_data_cache", True)

if config.use_depth and not _dataset_has_modality(config.carla_data, "depth"):
    print(
        f"Depth is enabled in config, but no depth/ folders were found under {config.carla_data}. "
        "Disabling depth for cache build."
    )
    _set_config_value(config, "use_depth", False)

for k, v in config.training_dict().items():
    print(k, v)

data = CARLAData(
    root=config.carla_data,
    config=config,
    training_session_cache=None,
    build_cache=True
)

num_workers = (
    args.num_workers if args.num_workers is not None else config.assigned_cpu_cores
)
dataloader_kwargs = {
    "batch_size": max(1, num_workers),
    "shuffle": False,
    "num_workers": num_workers,
    "collate_fn": lambda samples: samples,
}
if num_workers > 0:
    dataloader_kwargs["prefetch_factor"] = 1
dataloader = torch.utils.data.DataLoader(data, **dataloader_kwargs)

for i, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
    pass
