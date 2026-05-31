from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lead.data_loader.carla_dataset import CARLAData
from lead.inference.config_open_loop import OpenLoopConfig
from lead.inference.open_loop_inference import OpenLoopInference
from lead.training.config_training import TrainingConfig
from lead.training.mixed_training_utils import mixed_data_collate_fn


LOG = logging.getLogger(__name__)


DEFAULT_CONFIG = "outputs/model1/planning_only/config_planning_only.json"
DEFAULT_DATA_DIR = (
    "data/carla_leaderboard2_dual_cameras_val/data/DynamicObjectCrossing/"
    "Town13_Rep0_1355_0_route0_05_18_23_40_37"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a dual-front-camera LEAD model over one CARLA route, save "
            "prediction visualizations, or benchmark model inference speed."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Training config JSON.")
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Route directory with rgb/metas/bboxes/lidar/hdmap folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Visualization output directory. Defaults to <data-dir>/viz.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Exact .pth checkpoint. Defaults to config load_file.",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--save-workers", type=int, default=4, help="PNG writer threads.")
    parser.add_argument(
        "--autocast-dtype",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="bf16",
        help=(
            "Autocast precision for inference. auto uses the training config, "
            "fp32 disables autocast."
        ),
    )
    parser.add_argument(
        "--speedtest",
        action="store_true",
        help=(
            "Run as inference speed test only: no visualization, no saving, "
            "only model forward timing."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Optional maximum number of frames to process.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup forwards excluded from speed stats.",
    )
    return parser.parse_args()


def normalize_path(path: str) -> Path:
    return Path(path.replace("\\", os.sep)).expanduser()


def load_config(config_path: Path, route_dir: Path) -> TrainingConfig:
    with config_path.open(encoding="utf-8") as f:
        loaded_config = json.load(f)

    validation_root = route_dir.parents[2]
    loaded_config.update(
        {
            "carla_root": str(validation_root),
            "validation_carla_root": str(validation_root),
            "use_color_aug": False,
            "use_sensor_perburtation": False,
            "use_sensor_perburtation_prob": 0.0,
            "carla_num_samples": -1,
            "carla_dataset_fraction": 1,
            "use_training_session_cache": False,
            "use_persistent_cache": True,
            "force_rebuild_data_cache": False,
            "force_rebuild_bucket": True,
            "visualize_dataset": False,
            "randomize_route_order": False,
            "validation_batch_size": 1,
            "validation_drop_last": False,
            "persistent_workers_val": False,
            "compile": False,
            "rank": 0,
        }
    )
    return TrainingConfig(loaded_config=loaded_config, raise_error_on_missing_key=False)


def filter_dataset_to_route(dataset: CARLAData, route_dir: Path) -> None:
    route_dir = route_dir.resolve()
    route_dirs = [
        Path(str(value, encoding="utf-8")).resolve() for value in dataset.route_dirs
    ]
    mask = np.array([path == route_dir for path in route_dirs], dtype=bool)
    if not mask.any():
        raise ValueError(f"No samples found for route: {route_dir}")

    for name in (
        "bev_3rd_person_images",
        "images",
        "images_perturbated",
        "semantics",
        "semantics_perturbated",
        "bev_semantics",
        "bev_semantics_perturbated",
        "depth",
        "depth_perturbated",
        "lidars",
        "radars",
        "radars_perturbated",
        "bboxes",
        "metas",
        "route_dirs",
        "route_indices",
        "sample_start",
        "bucket_identity",
        "global_indices",
    ):
        value = getattr(dataset, name, None)
        if isinstance(value, np.ndarray) and len(value) == len(mask):
            setattr(dataset, name, value[mask])


def make_dataloader(
    config: TrainingConfig, route_dir: Path, num_workers: int
) -> DataLoader:
    scenario_dir = route_dir.parent
    dataset = CARLAData(
        root=[str(scenario_dir)],
        config=config,
        training_session_cache=None,
        random=False,
    )
    filter_dataset_to_route(dataset, route_dir)
    LOG.info("Route dataset size: %d frames", len(dataset))

    kwargs = {
        "dataset": dataset,
        "batch_size": 1,
        "shuffle": False,
        "num_workers": max(0, num_workers),
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "collate_fn": mixed_data_collate_fn,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = False
    return DataLoader(**kwargs)


def resolve_checkpoint(config: TrainingConfig, checkpoint_arg: str | None) -> Path:
    checkpoint = normalize_path(checkpoint_arg) if checkpoint_arg else Path(config.load_file)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint


def make_inference(
    config: TrainingConfig, checkpoint: Path, device: torch.device
) -> OpenLoopInference:
    open_loop_config = OpenLoopConfig()
    inference = OpenLoopInference(
        config_training=config,
        config_open_loop=open_loop_config,
        model_path=str(checkpoint.parent),
        device=device,
        prefix=checkpoint.stem,
    )
    for net in inference.nets:
        if config.channel_last:
            net.to(memory_format=torch.channels_last)
        compile_strategy = str(getattr(config, "compile_strategy", "core")).lower()
        if not config.compile or compile_strategy == "none":
            net._compiled_forward_core = net._forward_core
            net._compiled_forward_core_is_dual_static = False
            continue
        if compile_strategy == "core":
            net.prepare_compile(
                fullgraph=False,
                dynamic=False,
                backend="inductor",
                mode=str(config.compile_mode).lower(),
            )
            LOG.info(
                "Using torch.compile on TFv6 forward core, mode=%s",
                str(config.compile_mode).lower(),
            )
        elif compile_strategy != "none":
            raise ValueError(
                f"Unsupported inference compile_strategy={compile_strategy!r}. "
                "Use 'core' or 'none'."
            )
    return inference


def prepare_viz_data(config: TrainingConfig, data: dict) -> dict:
    viz_data = dict(data)
    if viz_data.get("rgb") is None and viz_data.get(config.left_camera_key) is not None:
        viz_data["rgb"] = viz_data[config.left_camera_key]
    return viz_data


def prepare_inference_data(data: dict) -> dict[str, torch.Tensor]:
    return {key: value for key, value in data.items() if torch.is_tensor(value)}


def frame_name(data: dict, fallback_idx: int) -> str:
    frame_number = data.get("frame_number")
    if isinstance(frame_number, (list, tuple)) and frame_number:
        return str(frame_number[0])
    if torch.is_tensor(frame_number):
        return str(int(frame_number.flatten()[0].item())).zfill(4)
    return str(fallback_idx).zfill(4)


def save_png(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_autocast_dtype(
    config: TrainingConfig, autocast_dtype_arg: str
) -> torch.dtype | None:
    if autocast_dtype_arg == "auto":
        return (
            None
            if config.torch_float_type == torch.float32
            else config.torch_float_type
        )
    if autocast_dtype_arg == "fp32":
        return None
    if autocast_dtype_arg == "fp16":
        return torch.float16
    if autocast_dtype_arg == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported autocast dtype: {autocast_dtype_arg}")


def autocast_context(device: torch.device, autocast_dtype: torch.dtype | None):
    enabled = device.type == "cuda" and autocast_dtype is not None
    return torch.amp.autocast(
        device_type=device.type,
        dtype=autocast_dtype or torch.float32,
        enabled=enabled,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config_path = normalize_path(args.config)
    route_dir = normalize_path(args.data_dir)
    output_dir = normalize_path(args.output_dir) if args.output_dir else route_dir / "viz"

    if not route_dir.is_dir():
        raise FileNotFoundError(f"Route directory not found: {route_dir}")

    config = load_config(config_path, route_dir)
    checkpoint = resolve_checkpoint(config, args.checkpoint)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This model inference path expects CUDA. Use --device cuda:<id>.")

    dataloader = make_dataloader(config, route_dir, args.num_workers)
    inference = make_inference(config, checkpoint, device)
    autocast_dtype = resolve_autocast_dtype(config, args.autocast_dtype)
    autocast_label = (
        str(autocast_dtype).replace("torch.", "") if autocast_dtype else "fp32"
    )
    LOG.info("Autocast dtype: %s", autocast_label)

    forward_times: list[float] = []
    processed = 0

    total = len(dataloader.dataset)
    if args.limit > 0:
        total = min(total, args.limit)

    iterable = islice(dataloader, args.limit) if args.limit > 0 else dataloader
    if args.speedtest:
        for idx, data in enumerate(tqdm(iterable, total=total, desc="Speedtest")):
            if data is None:
                continue

            synchronize_if_cuda(device)
            start = time.perf_counter()
            with autocast_context(device, autocast_dtype):
                inference.forward(prepare_inference_data(data))
            synchronize_if_cuda(device)
            elapsed = time.perf_counter() - start
            if idx >= args.warmup:
                forward_times.append(elapsed)
            processed += 1
    else:
        from lead.visualization.visualizer import Visualizer

        output_dir.mkdir(parents=True, exist_ok=True)
        futures = []
        with ThreadPoolExecutor(max_workers=max(1, args.save_workers)) as executor:
            for idx, data in enumerate(tqdm(iterable, total=total, desc="Visualizing")):
                if data is None:
                    continue

                synchronize_if_cuda(device)
                start = time.perf_counter()
                with autocast_context(device, autocast_dtype):
                    prediction = inference.forward(prepare_inference_data(data))
                synchronize_if_cuda(device)
                elapsed = time.perf_counter() - start
                if idx >= args.warmup:
                    forward_times.append(elapsed)

                viz_data = prepare_viz_data(config, data)
                image = Visualizer(
                    config=config,
                    data=viz_data,
                    prediction=prediction,
                    training=False,
                ).visualize_inference_prediction()
                out_path = output_dir / f"{frame_name(data, idx)}.png"
                futures.append(executor.submit(save_png, out_path, image))

                if len(futures) >= args.save_workers * 4:
                    futures.pop(0).result()
                processed += 1

            for future in futures:
                future.result()

    measured = len(forward_times)
    saved_visualizations_line = (
        "" if args.speedtest else f"Saved visualizations: {output_dir}\n"
    )
    if measured:
        avg = sum(forward_times) / measured
        fps = 1.0 / avg
        sorted_times = sorted(forward_times)
        median = sorted_times[measured // 2]
        p95 = sorted_times[min(measured - 1, int(measured * 0.95))]
        print(
            f"Processed frames: {processed}\n"
            f"Mode: {'speedtest' if args.speedtest else 'visualization'}\n"
            f"{saved_visualizations_line}"
            f"Autocast dtype: {autocast_label}\n"
            f"Model batch size: 1\n"
            f"Measured forwards: {measured} (warmup skipped: {min(args.warmup, processed)})\n"
            f"Average model forward time: {avg * 1000:.2f} ms/frame\n"
            f"Median model forward time: {median * 1000:.2f} ms/frame\n"
            f"P95 model forward time: {p95 * 1000:.2f} ms/frame\n"
            f"Model throughput: {fps:.2f} FPS"
        )
    else:
        print(
            f"Processed frames: {processed}\n"
            f"Mode: {'speedtest' if args.speedtest else 'visualization'}\n"
            f"{saved_visualizations_line}"
            f"Autocast dtype: {autocast_label}\n"
            "Not enough frames after warmup to report speed."
        )


if __name__ == "__main__":
    main()
