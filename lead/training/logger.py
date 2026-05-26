import json
import logging
import os

import numpy as np
import torch
import wandb
from beartype import beartype
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from lead.tfv6.tfv6 import Prediction
from lead.training.config_training import TrainingConfig
from lead.visualization.visualizer import visualize_sample

LOG = logging.getLogger(__name__)


def to_scalar_for_wandb(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().float().cpu().item()
        return value.detach().float().cpu().mean().item()

    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.reshape(-1)[0])
        return float(np.nanmean(value))

    if isinstance(value, (np.integer, np.floating, np.bool_)):  # noqa: UP038
        return value.item()

    if isinstance(value, (int, float, bool, str)):  # noqa: UP038
        return value

    return None


def is_valid_wandb_scalar(value):
    if isinstance(value, float):
        return np.isfinite(value)
    return True


def is_tensorboard_scalar(value):
    if isinstance(value, torch.Tensor):
        return (
            value.numel() == 1 and torch.isfinite(value.detach().float()).all().item()
        )
    if isinstance(value, np.ndarray):
        return value.size == 1 and np.isfinite(value.reshape(-1)[0])
    if isinstance(value, int | float | np.integer | np.floating):
        return np.isfinite(value)
    return False


class Logger:
    @beartype
    def __init__(
        self,
        config: TrainingConfig,
        model: torch.nn.Module | torch.nn.parallel.distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        continue_step: int,
        dataset: Dataset,
        dataloader: DataLoader,
        total_gradient_steps: int,
    ):
        """
        Initialize the Logger for training.

        Args:
            config: The configuration dictionary.
            model: The model to log.
            optimizer: The optimizer to log.
            scaler: The gradient scaler for mixed precision training.
            continue_step: The step to continue from if training is resumed.
            dataset: The dataset used for training.
            dataloader: The dataloader used for training.
            total_gradient_steps: Total number of gradient steps for the training.
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.step = continue_step
        self.dataset = dataset
        self.total_gradient_steps = total_gradient_steps
        self.dataloader = dataloader
        self.scaler = scaler
        self.wandb_failed = False
        self.validation_history = []
        self.validation_metrics_path = None
        if self.config.logdir is not None:
            self.validation_metrics_path = os.path.join(
                self.config.logdir,
                "validation_metrics.jsonl",
            )
            self.validation_history = self._load_validation_history()

        # Initialize TensorBoard and WandB loggers
        self.tensorboard_writer = None
        if self.config.rank == 0 and self.config.logdir is not None:
            self.tensorboard_writer = SummaryWriter(
                log_dir=config.logdir,
            )
            if self.config.log_wandb:
                try:
                    wandb_id = None
                    if self.config.wandb_resume != "never":
                        wandb_id = config.wandb_id
                    wandb.init(
                        project=self.config.wandb_project_name,
                        name=config.description,
                        config=config.training_dict(),
                        id=wandb_id,
                        resume=config.wandb_resume,
                        dir=self.config.logdir,
                        settings=wandb.Settings(_service_wait=300),
                    )
                    self.step = max(self.step, wandb.run.step)
                    LOG.info(
                        f"WandB logger will log scalar every {self.config.log_scalars_frequency} steps"
                    )
                    LOG.info(
                        f"WandB logger will log images every {self.config.log_images_frequency} steps"
                    )
                    if self.config.use_validation:
                        wandb.define_metric("validation_epoch")
                        wandb.define_metric("val/*", step_metric="validation_epoch")
                    if self.config.use_additional_metrics:
                        LOG.info("Additional perception metrics logging is enabled.")
                except BrokenPipeError:
                    self.wandb_failed = True
                    LOG.exception(
                        "W&B initialization failed with BrokenPipeError. Disabling W&B for this run."
                    )
                except Exception:
                    self.wandb_failed = True
                    LOG.exception(
                        "W&B initialization failed. Disabling W&B for this run."
                    )

    @staticmethod
    def _log_exception_no_raise(message: str):
        try:
            LOG.exception(message)
        except Exception:
            pass

    def __del__(self):
        config = getattr(self, "config", None)
        if config is not None and config.rank == 0:
            if config.log_wandb and not getattr(self, "wandb_failed", False):
                try:
                    wandb.finish()
                except BrokenPipeError:
                    self.wandb_failed = True
                    self._log_exception_no_raise(
                        "W&B finish failed with BrokenPipeError. Disabling W&B for this run."
                    )
                except Exception:
                    self.wandb_failed = True
                    self._log_exception_no_raise(
                        "W&B finish failed. Disabling W&B for this run."
                    )
            tensorboard_writer = getattr(self, "tensorboard_writer", None)
            if tensorboard_writer is not None:
                try:
                    tensorboard_writer.close()
                except Exception:
                    self._log_exception_no_raise(
                        "TensorBoard writer close failed. Continuing shutdown."
                    )

    def _safe_wandb_log(self, message: dict, **kwargs):
        if not self.config.log_wandb or self.wandb_failed:
            return
        try:
            wandb.log(message, **kwargs)
        except BrokenPipeError:
            self.wandb_failed = True
            LOG.exception(
                "W&B logging failed with BrokenPipeError. Disabling W&B for this run."
            )
        except Exception:
            self.wandb_failed = True
            LOG.exception("W&B logging failed. Disabling W&B for this run.")

    def _safe_wandb_visualize_sample(self, **kwargs):
        if not self.config.log_wandb or self.wandb_failed:
            return
        try:
            visualize_sample(**kwargs)
        except BrokenPipeError:
            self.wandb_failed = True
            LOG.exception(
                "W&B visualization logging failed with BrokenPipeError. Disabling W&B for this run."
            )
        except Exception:
            self.wandb_failed = True
            LOG.exception(
                "W&B visualization logging failed. Disabling W&B for this run."
            )

    def _safe_logger_call(self, fn, description: str):
        try:
            return fn()
        except Exception:
            LOG.exception("%s failed. Continuing training.", description)
            return None

    @staticmethod
    def _add_safe_scalar(message: dict, key: str, value):
        safe_value = to_scalar_for_wandb(value)
        if safe_value is not None and is_valid_wandb_scalar(safe_value):
            message[key] = safe_value

    def _log_train_scalars(
        self,
        cur_epoch: int,
        unscaled_loss: dict,
        scaled_loss: dict,
        data: dict,
        step: int,
        gradient_steps_skipped: int,
        log: dict,
    ):
        self.step = max(self.step, step)

        message = {}

        # General logs
        self._add_safe_scalar(message, "debug/epoch", cur_epoch)
        for g in range(len(self.optimizer.param_groups)):
            self._add_safe_scalar(
                message, f"debug/lr_{g}", self.optimizer.param_groups[g]["lr"]
            )
        self._add_safe_scalar(
            message,
            "debug/batch_size_per_gpu",
            data["source_dataset"].shape[0],
        )
        self._add_safe_scalar(message, "debug/num_gpu", torch.cuda.device_count())
        self._add_safe_scalar(
            message,
            "debug/model_size",
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        )
        self._add_safe_scalar(message, "debug/dataset_size", len(self.dataset))
        self._add_safe_scalar(
            message, "debug/num_gradient_steps", self.total_gradient_steps
        )
        self._add_safe_scalar(
            message, "debug/finished_percentage", step / self.total_gradient_steps
        )
        self._add_safe_scalar(
            message, "debug/steps_left", self.total_gradient_steps - step
        )
        self._add_safe_scalar(message, "debug/dataloader_size", len(self.dataloader))
        self._add_safe_scalar(
            message, "debug/allocated_cpus", self.config.assigned_cpu_cores
        )
        self._add_safe_scalar(
            message, "debug/gradient_steps_skipped", gradient_steps_skipped
        )
        self._add_safe_scalar(
            message,
            "debug/max_gpu_mem",
            torch.cuda.max_memory_allocated(self.config.device) / (1024**3),
        )
        self._add_safe_scalar(
            message,
            "debug/average_loading_time",
            data["loading_time"].detach().float().cpu().mean(),
        )
        self._add_safe_scalar(
            message,
            "debug/average_loading_meta_time",
            data["loading_meta_time"].detach().float().cpu().mean(),
        )
        self._add_safe_scalar(
            message,
            "debug/average_loading_sensor_time",
            data["loading_sensor_time"].detach().float().cpu().mean(),
        )
        self._add_safe_scalar(
            message,
            "debug/source_dataset",
            data["source_dataset"].detach().float().cpu().mean(),
        )
        if self.scaler is not None:
            self._add_safe_scalar(message, "debug/grad_scale", self.scaler.get_scale())

        # Loss and metrics logs
        for key, value in log.items():
            self._add_safe_scalar(message, key, value)
        for loss_name, loss_value in unscaled_loss.items():
            self._add_safe_scalar(message, f"unscaled_loss/{loss_name}", loss_value)
        for loss_name, loss_value in scaled_loss.items():
            self._add_safe_scalar(message, f"scaled_loss/{loss_name}", loss_value)

        self._safe_wandb_log(message, step=self.step)

        if self.tensorboard_writer is not None:
            for key, value in message.items():
                if not is_tensorboard_scalar(value):
                    continue
                self.tensorboard_writer.add_scalar(
                    key,
                    value,
                    self.step,
                )

    @beartype
    def log_train(
        self,
        epoch_iteration: int,
        cur_epoch: int,
        unscaled_loss: dict,
        scaled_loss: dict,
        data: dict,
        step: int,
        gradient_steps_skipped: int,
        predictions: Prediction,
        log: dict,
    ):
        """
        Log training information.

        Args:
            epoch_iteration: Current iteration number of training in the epoch.
            cur_epoch: Current epoch number.
            unscaled_loss: Dictionary of unscaled losses for the current epoch.
            scaled_loss: Dictionary of scaled losses for the current epoch.
            data: Dictionary containing data used for training.
            step: Current step in the training process.
            gradient_steps_skipped: Number of gradient steps skipped due to inf/nan gradients.
            predictions: Model predictions for the current batch.
            log: Dictionary containing debug information.
        """
        if (
            self.config.rank == 0
            and not self.config.is_on_slurm
            and self.config.visualize_training
            and self.config.carla_leaderboard_mode
            and (
                (epoch_iteration + 1) % self.config.log_images_frequency == 0
                or epoch_iteration <= 1
            )
        ):
            LOG.info(f"Visualizing training sample at step {step}.")
            self._safe_logger_call(
                lambda: visualize_sample(
                    config=self.config,
                    predictions=predictions,
                    data=data,
                    save_image=True,
                    save_path=os.path.join("outputs", "training_viz"),
                    postfix=str(self.step).zfill(5),
                    prefix="train",
                ),
                "Local training visualization",
            )

        if self.config.rank == 0:
            if (
                self.config.log_wandb
                and not self.wandb_failed
                and (
                    (epoch_iteration + 1) % self.config.log_images_frequency == 0
                    or epoch_iteration <= 1
                )
                and self.config.carla_leaderboard_mode
            ):
                LOG.info(f"Logging training sample to WandB at step {step}.")
                self._safe_wandb_visualize_sample(
                    config=self.config,
                    predictions=predictions,
                    data=data,
                    prefix="train",
                    log_wandb=True,
                )

            if (epoch_iteration + 1) % self.config.log_scalars_frequency == 0:
                self._safe_logger_call(
                    lambda: self._log_train_scalars(
                        cur_epoch=cur_epoch,
                        unscaled_loss=unscaled_loss,
                        scaled_loss=scaled_loss,
                        data=data,
                        step=step,
                        gradient_steps_skipped=gradient_steps_skipped,
                        log=log,
                    ),
                    "Training scalar logging",
                )
            self.step += 1

    def _load_validation_history(self) -> list[dict]:
        if self.validation_metrics_path is None:
            return []
        if not os.path.isfile(self.validation_metrics_path):
            return []
        history = []
        with open(self.validation_metrics_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    LOG.warning(
                        "Skipping invalid validation metrics JSONL line: %s",
                        line,
                    )
        return history

    def _write_validation_history(self):
        if self.validation_metrics_path is None:
            return
        os.makedirs(os.path.dirname(self.validation_metrics_path), exist_ok=True)
        with open(self.validation_metrics_path, "w") as f:
            for record in self.validation_history:
                f.write(json.dumps(record, sort_keys=True) + "\n")

    @staticmethod
    def _clean_validation_metrics(metrics: dict[str, float]) -> dict[str, float]:
        clean_metrics = {}
        for key, value in metrics.items():
            try:
                scalar_value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(scalar_value):
                clean_metrics[key] = scalar_value
        return clean_metrics

    @staticmethod
    def _validation_plot_name(metric_name: str) -> str:
        return (
            metric_name.replace("/", "__")
            .replace(" ", "_")
            .replace(":", "_")
            .replace("\\", "_")
        )

    def _save_validation_plots(self):
        if not self.config.validation_save_plots or self.config.logdir is None:
            return
        import matplotlib.pyplot as plt

        plot_dir = os.path.join(self.config.logdir, "validation_curves")
        os.makedirs(plot_dir, exist_ok=True)
        metric_names = sorted(
            {
                key
                for record in self.validation_history
                for key in record
                if key != "epoch"
            }
        )
        for metric_name in metric_names:
            points = [
                (record["epoch"], record[metric_name])
                for record in self.validation_history
                if metric_name in record and np.isfinite(record[metric_name])
            ]
            if len(points) == 0:
                continue
            epochs, values = zip(*points, strict=False)
            plt.figure(figsize=(8, 4.5))
            plt.plot(epochs, values, marker="o", linewidth=1.8)
            plt.xlabel("epoch")
            plt.ylabel(metric_name)
            plt.title(metric_name)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    plot_dir,
                    f"{self._validation_plot_name(metric_name)}.png",
                )
            )
            plt.close()

    @beartype
    def log_validation_epoch(self, epoch: int, metrics: dict[str, float]):
        self._safe_logger_call(
            lambda: self._log_validation_epoch(epoch, metrics),
            "Validation epoch logging",
        )

    def _log_validation_epoch(self, epoch: int, metrics: dict[str, float]):
        if self.config.rank != 0:
            return
        clean_metrics = self._clean_validation_metrics(metrics)
        if len(clean_metrics) == 0:
            return

        validation_epoch = epoch + 1
        message = {"validation_epoch": validation_epoch, **clean_metrics}

        self._safe_wandb_log(message)

        if self.tensorboard_writer is not None:
            for key, value in clean_metrics.items():
                self.tensorboard_writer.add_scalar(
                    key,
                    value,
                    validation_epoch,
                )

        record = {"epoch": validation_epoch, **clean_metrics}
        self.validation_history = [
            item
            for item in self.validation_history
            if item.get("epoch") != validation_epoch
        ]
        self.validation_history.append(record)
        self.validation_history.sort(key=lambda item: item["epoch"])
        self._write_validation_history()
        self._save_validation_plots()

    def logs(self, msg: dict):
        self._safe_logger_call(lambda: self._logs(msg), "Ad-hoc logging")

    def _logs(self, msg: dict):
        if self.config.rank == 0:
            self._safe_wandb_log(msg, step=self.step)

            if self.tensorboard_writer is not None:
                for key, value in msg.items():
                    if not is_tensorboard_scalar(value):
                        continue
                    self.tensorboard_writer.add_scalar(
                        key,
                        value,
                        self.step,
                    )
