from __future__ import annotations

import datetime
import json
import logging
import math
import os
import pathlib
import random
import typing
import warnings

import diskcache
import numpy as np
import torch
import torch.multiprocessing as mp
from beartype import beartype
from diskcache import Cache
from torch import optim
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LambdaLR,
)
from torch.utils.data import DataLoader

from lead.data_loader.carla_dataset import CARLAData
from lead.data_loader.navsim_dataset import NavsimData
from lead.data_loader.waymo_e2e_dataset import WODE2EData
from lead.tfv6 import transfuser_utils as fn
from lead.training import mixed_training_utils
from lead.training.config_training import TrainingConfig

LOG = logging.getLogger(__name__)


class DecayingPeakCosineAnnealingWarmRestarts(CosineAnnealingWarmRestarts):
    """Cosine restarts scheduler with a multiplicative decay on restart peaks."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        peak_decay: float = 1.0,
    ):
        self.peak_decay = float(peak_decay)
        super().__init__(
            optimizer=optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )

    def _restart_index(self) -> int:
        if self.T_mult == 1:
            return max(0, int(self.last_epoch // self.T_0))
        return max(0, int(round(math.log(self.T_i / self.T_0, self.T_mult))))

    def get_lr(self) -> list[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
                stacklevel=2,
            )

        peak_multiplier = self.peak_decay ** self._restart_index()
        return [
            self.eta_min
            + (base_lr * peak_multiplier - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i))
            / 2
            for base_lr in self.base_lrs
        ]


class DistributedEvalSampler(torch.utils.data.Sampler):
    """Distributed sampler for evaluation without padding or duplicate samples."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_replicas: int,
        rank: int,
    ):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.indices = list(range(rank, len(dataset), num_replicas))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def _set_config_value(config: TrainingConfig, key: str, value):
    try:
        setattr(config, key, value)
    except Exception:
        pass
    if hasattr(config, "_loaded_config"):
        config._loaded_config[key] = value


@beartype
def increase_limit_file_descriptors(n: int = 4096):
    # On some systems it is necessary to increase the limit on open file descriptors.
    try:
        import resource

        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (n, rlimit[1]))
    except (ModuleNotFoundError, ImportError) as e:
        LOG.error(str(e))


@beartype
def initialize_config(config_path: str | None = None) -> TrainingConfig:
    config = TrainingConfig()
    if config_path is not None:
        with open(config_path) as f:
            loaded_config = json.load(f)
        config = TrainingConfig(loaded_config, raise_error_on_missing_key=False)
    elif config.load_file is not None:
        with open(
            os.path.join("/".join(config.load_file.split("/")[:-1]), "config.json")
        ) as f:
            loaded_config = json.load(f)
        config = TrainingConfig(loaded_config, raise_error_on_missing_key=False)
    return config


@beartype
def initialize_training_session_cache(config: TrainingConfig) -> Cache | None:
    training_session_cache = None
    if config.use_training_session_cache:
        LOG.info(
            "Initializing training session cache at %s",
            config.training_session_cache_path,
        )
        training_session_cache = Cache(
            directory=config.training_session_cache_path, size_limit=int(2048 * 1024**3)
        )
    return training_session_cache


@beartype
def initialize_torch(config: TrainingConfig) -> int:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    ngpus_per_node = torch.cuda.device_count()
    ncpus_per_node = config.assigned_cpu_cores
    num_workers = int(ncpus_per_node / ngpus_per_node) * config.workers_per_cpu_cores

    if torch.cuda.device_count() > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=config.world_size,
            rank=config.rank,
            timeout=datetime.timedelta(minutes=120),
        )

    torch.cuda.device(config.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    return num_workers


@beartype
def initialize_model(
    config: TrainingConfig,
) -> tuple[typing.Any | torch.nn.parallel.distributed.DistributedDataParallel, int]:
    from lead.tfv6.tfv6 import TFv6

    model = TFv6(config.device, config)

    model.cuda(device=config.device)
    if config.sync_batchnorm:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        LOG.info("Using sync_batch_norm")

    # Convert all norm layers to use fp32
    fn.patch_norm_fp32(model)

    start_epoch = 0  # Epoch to continue training from
    if config.load_file is not None:
        LOG.info(f"Loading model from {config.load_file}")
        # Add +1 because the epoch before that was already trained
        load_name = str(pathlib.Path(config.load_file).stem)
        if config.continue_failed_training:
            start_epoch = int("".join(filter(str.isdigit, load_name))) + 1
        model.load_state_dict(
            torch.load(config.load_file, map_location=config.device, weights_only=True),
            strict=config.continue_failed_training,
        )

    model.backbone.requires_grad_(not config.freeze_backbone)
    LOG.info(
        f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters"
    )
    if config.channel_last:
        model = model.to(memory_format=torch.channels_last)
        LOG.info("Using channel last memory format")

    compile_strategy = "none"
    if config.compile:
        compile_strategy = str(getattr(config, "compile_strategy", "core")).lower()
    compile_mode = str(
        getattr(config, "compile_mode", "max-autotune-no-cudagraphs")
    ).lower()
    if compile_strategy == "core":
        model.prepare_compile(
            fullgraph=True,
            dynamic=False,
            backend="inductor",
            mode=compile_mode,
        )
        LOG.info(f"Using torch.compile on TFv6 forward core, mode={compile_mode}")
    elif compile_strategy == "module":
        model = torch.compile(
            model,
            fullgraph=False,
            dynamic=False,  # aggressively specialize to current input shapes
            backend="inductor",
            mode=compile_mode,
        )
        LOG.info(f"Using torch.compile on full TFv6 module, mode={compile_mode}")
    elif compile_strategy != "none":
        raise ValueError(
            f"Unknown compile_strategy={compile_strategy!r}. "
            "Expected 'none', 'core', or 'module'."
        )

    if torch.cuda.device_count() > 1:
        model_wrapper = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=None,
            output_device=None,
            broadcast_buffers=False,
            bucket_cap_mb=config.bucket_cap_mb,
        )
    else:
        model_wrapper = model
    return model_wrapper, start_epoch


@beartype
def initialize_optimizer(
    model_wrapper: typing.Any | torch.nn.parallel.DistributedDataParallel,
    model: torch.nn.Module,
    config: TrainingConfig,
    gradient_steps_per_epoch: int,
) -> tuple[
    ZeroRedundancyOptimizer | torch.optim.AdamW,
    DecayingPeakCosineAnnealingWarmRestarts | LambdaLR | CosineAnnealingLR,
    torch.amp.GradScaler,
    int,
]:
    params = model_wrapper.parameters()
    if config.use_zero_redundancy and torch.cuda.device_count() > 1:
        optimizer = ZeroRedundancyOptimizer(
            params,
            optimizer_class=torch.optim.AdamW,
            lr=config.lr,
            amsgrad=True,
            weight_decay=config.weight_decay,
            fused=True,
        )
    else:
        optimizer = optim.AdamW(
            params,
            lr=config.lr,
            amsgrad=True,
            weight_decay=config.weight_decay,
            fused=True,
        )

    if config.use_cosine_annealing_with_restarts:
        scheduler = DecayingPeakCosineAnnealingWarmRestarts(
            optimizer,
            T_0=gradient_steps_per_epoch,
            T_mult=2,
            eta_min=1e-9,
            peak_decay=config.cosine_annealing_restart_peak_decay,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=gradient_steps_per_epoch * config.epochs
        )

    if config.load_file is not None and config.continue_failed_training:
        scheduler.load_state_dict(
            torch.load(
                config.load_file.replace("model_", "scheduler_"),
                map_location=config.device,
                weights_only=True,
            )
        )

    if config.load_file is not None and config.continue_failed_training:
        optimizer.load_state_dict(
            torch.load(
                config.load_file.replace("model_", "optimizer_"),
                map_location=config.device,
                weights_only=True,
            )
        )

    scaler = torch.amp.GradScaler(
        init_scale=config.grad_scaler_init_scale,
        growth_factor=config.grad_scaler_growth_factor,
        backoff_factor=config.grad_scaler_backoff_factor,
        growth_interval=config.grad_scaler_growth_interval,
        enabled=config.need_grad_scaler,
    )
    if config.load_file is not None and config.continue_failed_training:
        scaler.load_state_dict(
            torch.load(
                config.load_file.replace("model_", "scaler_"),
                map_location=config.device,
                weights_only=True,
            )
        )

    gradient_steps_skipped = 0
    if config.load_file is not None and config.continue_failed_training:
        gradient_steps_skipped_path = config.load_file.replace(
            "model_", "gradient_steps_skipped_"
        ).replace(".pth", ".txt")
        if os.path.exists(gradient_steps_skipped_path):
            with open(gradient_steps_skipped_path) as f:
                gradient_steps_skipped = int(f.read().strip())

    return optimizer, scheduler, scaler, gradient_steps_skipped


@beartype
def initialize_dataloader(
    config: TrainingConfig,
    ssd_cache: dict | diskcache.core.Cache | None,
    num_workers: int,
):
    g_cuda = torch.Generator(device="cpu")
    g_cuda.manual_seed(config.seed)

    datasets, samplers = [], []
    if config.use_carla_data:
        datasets.append(
            CARLAData(
                root=config.carla_data,
                config=config,
                training_session_cache=ssd_cache,
            )
        )
        assert not datasets[-1].build_cache and not datasets[-1].build_buckets
        samplers.append(
            torch.utils.data.DistributedSampler(
                datasets[-1],
                shuffle=True,
                num_replicas=config.world_size,
                rank=config.rank,
                drop_last=config.train_drop_last,
            )
        )
    if config.use_navsim_data:
        datasets.append(
            NavsimData(
                root=config.navsim_data_root,
                config=config,
                training_session_cache=ssd_cache,
            )
        )
        samplers.append(
            torch.utils.data.DistributedSampler(
                datasets[-1],
                shuffle=True,
                num_replicas=config.world_size,
                rank=config.rank,
                drop_last=config.train_drop_last,
            )
        )
    if config.use_waymo_e2e_data:
        datasets.append(
            WODE2EData(
                root=config.waymo_e2e_training_data_root,
                config=config,
                training_session_cache=ssd_cache,
                training=True,
            )
        )
        samplers.append(
            torch.utils.data.DistributedSampler(
                datasets[-1],
                shuffle=True,
                num_replicas=config.world_size,
                rank=config.rank,
                drop_last=config.train_drop_last,
            )
        )

    assert len(datasets) > 0, "No datasets selected for training!"

    for ds in datasets:
        LOG.info(f"Dataset size: {len(ds)} samples")

    if config.schedule_carla_num_samples:
        assert config.use_carla_data and config.mixed_data_training
        sample_scheduler = mixed_training_utils.Sim2RealSampleScheduler(
            config, datasets
        )
    else:
        sample_scheduler = mixed_training_utils.UniformSampleScheduler(config, datasets)

    train_dataset = mixed_training_utils.MixedDataset(
        config=config,
        datasets=datasets,
    )

    mixed_sampler = mixed_training_utils.MixedSampler(
        samplers=samplers,
        sample_scheduler=sample_scheduler,
        config=config,
    )

    dataloader_kwargs = {
        "dataset": train_dataset,
        "batch_sampler": mixed_sampler,
        "worker_init_fn": seed_worker,
        "generator": g_cuda,
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": mixed_training_utils.mixed_data_collate_fn,
    }
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = config.prefetch_factor
        dataloader_kwargs["persistent_workers"] = config.persistent_workers_train
        dataloader_context = _dataloader_context(config)
        if dataloader_context is not None:
            dataloader_kwargs["multiprocessing_context"] = dataloader_context

    dataloader_train = DataLoader(**dataloader_kwargs)
    return dataloader_train, mixed_sampler


@beartype
def initialize_validation_dataloader(
    config: TrainingConfig,
    num_workers: int,
) -> tuple[DataLoader | None, TrainingConfig | None]:
    if not config.use_validation:
        return None, None

    validation_data_root = os.path.join(config.validation_carla_root, "data")
    if not os.path.isdir(validation_data_root):
        raise FileNotFoundError(
            "Validation dataset is enabled, but the expected CARLA validation "
            f"data directory does not exist: {validation_data_root}"
        )

    loaded_config = dict(getattr(config, "_loaded_config", {}))
    for derived_key in (
        "carla_data",
        "bucket_collection_path",
        "training_session_cache_path",
    ):
        loaded_config.pop(derived_key, None)
    validation_overrides = {
        "carla_root": config.validation_carla_root,
        "use_color_aug": False,
        "use_sensor_perburtation": False,
        "use_sensor_perburtation_prob": 0.0,
        "carla_dataset_fraction": config.val_dataset_fraction,
        "carla_num_samples": -1,
        "use_training_session_cache": False,
        "use_persistent_cache": True,
        "force_rebuild_data_cache": False,
        "visualize_dataset": False,
        "hard_sample_oversampling": False,
    }
    loaded_config.update(validation_overrides)
    validation_config = TrainingConfig(
        loaded_config=loaded_config,
        raise_error_on_missing_key=False,
    )
    for derived_key in (
        "carla_data",
        "bucket_collection_path",
        "training_session_cache_path",
    ):
        validation_config._loaded_config.pop(derived_key, None)
    for key, value in validation_overrides.items():
        _set_config_value(validation_config, key, value)

    validation_dataset = CARLAData(
        root=validation_config.carla_data,
        config=validation_config,
        training_session_cache=None,
        random=True,
    )
    if len(validation_dataset) == 0:
        raise ValueError(
            f"Validation dataset contains no trainable samples: {validation_data_root}"
        )

    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        world_size = 1
        rank = 0

    sampler = DistributedEvalSampler(
        validation_dataset,
        num_replicas=world_size,
        rank=rank,
    )
    batch_size = max(1, config.validation_batch_size // max(1, world_size))
    validation_num_workers = (
        num_workers
        if config.validation_num_workers < 0
        else config.validation_num_workers
    )
    validation_num_workers = max(0, validation_num_workers)
    dataloader_kwargs = {
        "dataset": validation_dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "worker_init_fn": seed_worker,
        "num_workers": validation_num_workers,
        "pin_memory": True,
        "drop_last": config.validation_drop_last,
        "collate_fn": mixed_training_utils.mixed_data_collate_fn,
    }
    if validation_num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = config.validation_prefetch_factor
        dataloader_kwargs["persistent_workers"] = config.persistent_workers_val
        dataloader_context = _dataloader_context(config)
        if dataloader_context is not None:
            dataloader_kwargs["multiprocessing_context"] = dataloader_context

    dataloader_validation = DataLoader(**dataloader_kwargs)
    if config.validation_drop_last:
        dropped_local_samples = len(sampler) % batch_size
        if dropped_local_samples:
            LOG.info(
                "Validation drop_last=True drops %d local tail samples on rank %d "
                "to keep a fixed compiled batch shape.",
                dropped_local_samples,
                rank,
            )
    LOG.info(
        "Validation dataset size: %d samples, %d local samples on rank %d.",
        len(validation_dataset),
        len(sampler),
        rank,
    )
    return dataloader_validation, validation_config


@beartype
def save_config(config: TrainingConfig, rank: int):
    def is_json_serializable(v):
        try:
            json.dumps(v)
            return True
        except (TypeError, OverflowError):
            return False

    if rank == 0 and config.logdir is not None:
        os.makedirs(config.logdir, exist_ok=True)
        json_config = {
            k: v
            for k, v in config.training_dict().items()
            if is_json_serializable(v)
            and not k.startswith("_")
            and not k.endswith("__")
        }
        json_config = json.dumps(json_config, indent=4)
        # LOG.info(json_config)
        with open(os.path.join(config.logdir, "config.json"), "w") as f2:
            f2.write(json_config)


def seed_worker(_):
    # We need to seed the workers individually otherwise random processes in the
    # dataloader return the same values across workers!
    worker_seed = (
        torch.initial_seed()
    ) % 2**32  # this is different across workers, but not gpus when setting config.seed
    rank = int(os.environ.get("RANK", "0"))
    worker_seed = worker_seed + rank * 1000
    # if config.seed is not None, torch.initial_seed is the same across different gpus,
    # so we need to combine it with the rank to get different rng seeds on different gpus.
    # multiply with 1000 because the last digit is already incremented across workers
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _dataloader_context(config: TrainingConfig):
    start_method = str(getattr(config, "dataloader_start_method", "fork"))
    if start_method == "":
        return None
    available_start_methods = mp.get_all_start_methods()
    if start_method not in available_start_methods:
        LOG.warning(
            "DataLoader start method %r is unavailable. Available methods: %s. "
            "Falling back to PyTorch default.",
            start_method,
            available_start_methods,
        )
        return None
    return mp.get_context(start_method)


def set_start_method():
    # Select how the threads in the data loader are spawned
    # fork is the lightest option for DataLoader workers in this training setup.
    if mp.get_start_method(allow_none=True) is not None:
        return
    available_start_methods = mp.get_all_start_methods()
    for start_method in ("fork", "spawn", "forkserver"):
        if start_method in available_start_methods:
            mp.set_start_method(start_method)
            LOG.info("Using multiprocessing start method: %s", start_method)
            return
