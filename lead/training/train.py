import argparse
import logging
import os
import sys
import warnings
from itertools import islice

import matplotlib
import torch
from beartype import beartype
from torch.distributed.elastic.multiprocessing.errors import record
from tqdm import tqdm

from lead.common.constants import SourceDataset
from lead.common.logging_config import setup_logging
from lead.data_loader.waymo_e2e_dataset import evaluate_waymo_e2e
from lead.training import training_utils
from lead.training.logger import Logger

matplotlib.use("Agg")  # non-GUI backend for headless servers

setup_logging()
LOG = logging.getLogger(__name__)

warnings.filterwarnings("error")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The CUDA Graph is empty.*",
)
warnings.filterwarnings(
    "ignore", message="Grad strides do not match bucket view strides"
)
warnings.filterwarnings("ignore", category=UserWarning, message=".*epoch parameter.*")
warnings.filterwarnings(
    "ignore",
    category=ResourceWarning,
    message="Implicitly cleaning up.*TemporaryDirectory.*",
)


class Trainer:
    def __init__(self, config_path: str | None = None):
        self.config = training_utils.initialize_config(config_path)

        self.ssd_cache = training_utils.initialize_training_session_cache(self.config)
        self.num_worker = training_utils.initialize_torch(self.config)
        self.model_wrapper, self.cur_epoch = training_utils.initialize_model(
            self.config
        )
        self.model = self.model_wrapper
        if isinstance(self.model_wrapper, torch.nn.parallel.DistributedDataParallel):
            self.model = self.model_wrapper.module
        self.dataloader, self.sampler = training_utils.initialize_dataloader(
            self.config, self.ssd_cache, self.num_worker
        )
        self.validation_dataloader, self.validation_config = (
            training_utils.initialize_validation_dataloader(
                self.config,
                self.num_worker,
            )
        )

        # Predict how long the training will take
        self.gradient_steps_per_epoch = int(len(self.dataloader))
        self.total_gradient_steps = self.gradient_steps_per_epoch * self.config.epochs

        if self.config.rank == 0:
            LOG.info(
                f"Training for {self.config.epochs} epochs, "
                f"{self.gradient_steps_per_epoch} gradient steps per epoch, "
                f"total {self.total_gradient_steps} gradient steps."
            )

        # Set up optimizer
        self.optimizer, self.scheduler, self.scaler, self.gradient_steps_skipped = (
            training_utils.initialize_optimizer(
                model_wrapper=self.model_wrapper,
                model=self.model,
                config=self.config,
                gradient_steps_per_epoch=self.gradient_steps_per_epoch,
            )
        )

        training_utils.save_config(self.config, self.config.rank)

        self.step = 0

        self.logger = Logger(
            config=self.config,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            continue_step=int(self.cur_epoch * self.gradient_steps_per_epoch),
            dataset=self.dataloader.dataset,
            dataloader=self.dataloader,
            total_gradient_steps=self.total_gradient_steps,
        )

    @beartype
    def schedule_loss_weights(self, epoch: int):
        """Schedule loss weights for different datasets based on the epoch."""
        self.detailed_loss_weights = {}
        if self.config.use_carla_data:
            carla_loss_weights = self.config.detailed_loss_weights(
                SourceDataset.CARLA, epoch
            )
            self.detailed_loss_weights.update(carla_loss_weights)
        if self.config.use_navsim_data:
            navsim_loss_weights = self.config.detailed_loss_weights(
                SourceDataset.NAVSIM, epoch
            )
            self.detailed_loss_weights.update(navsim_loss_weights)
        if self.config.use_waymo_e2e_data:
            waymo_e2e_loss_weights = self.config.detailed_loss_weights(
                SourceDataset.WAYMO_E2E_2025, epoch
            )
            self.detailed_loss_weights.update(waymo_e2e_loss_weights)

        total_loss_weight = sum(self.detailed_loss_weights.values())
        for key in self.detailed_loss_weights:
            self.detailed_loss_weights[key] /= (
                total_loss_weight  # Normalize to sum to 1
            )

    def train_loop(self):
        for epoch in range(self.cur_epoch, self.config.epochs):
            self.dataloader.dataset.shuffle(epoch)

            # Update sampler epoch
            self.sampler.update_batch_sizes(epoch)

            # Update loss weights
            self.schedule_loss_weights(epoch)

            # Training
            rfm_score = self.train()

            # Save model immediately after the training epoch finishes, before
            # validation starts.
            if self.config.rank == 0:
                self.save(rfm_score)

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            if self.should_validate_epoch(epoch):
                validation_metrics = self.validate()
                if self.config.rank == 0:
                    self.logger.log_validation_epoch(
                        epoch=epoch,
                        metrics=validation_metrics,
                    )

            if torch.distributed.is_initialized():
                torch.distributed.barrier()  # Ensure all processes sync here before next epoch

            self.cur_epoch += 1

    @beartype
    def train(self) -> float | None:
        self.model_wrapper.train()

        # Train loop
        for epoch_iteration, data in enumerate(
            tqdm(
                islice(self.dataloader, self.gradient_steps_per_epoch),
                total=self.gradient_steps_per_epoch,
                disable=self.config.rank != 0,
                desc=f"Train epoch {self.cur_epoch}/{self.config.epochs}",
            )
        ):
            loss = torch.zeros(
                1, dtype=self.config.torch_float_type, device=self.config.device
            )
            data["iteration"] = epoch_iteration
            data["training_step"] = self.step
            should_compute_additional_metrics = (
                self.config.additional_metrics_frequency > 0
                and (epoch_iteration + 1) % self.config.additional_metrics_frequency
                == 0
            )
            data["compute_additional_metrics"] = (
                self.config.use_additional_metrics and should_compute_additional_metrics
            )
            with torch.amp.autocast(
                device_type="cuda",
                dtype=self.config.torch_float_type,
                enabled=self.config.use_mixed_precision_training,
            ):
                # Forward pass
                predictions = self.model_wrapper(data=data)
                losses, log = self.model.compute_loss(
                    predictions=predictions, data=data
                )
                self.step += 1

                # Sum up losses
                for key, value in losses.items():
                    loss += self.detailed_loss_weights[key] * value.reshape(
                        1
                    )  # Reshape as sanity check if the loss is a scalar
                scaled_loss = {
                    key: self.detailed_loss_weights[key] * value
                    for key, value in losses.items()
                }
            # Important to backprop outside the autocast context
            self.scaler.scale(loss).backward()

            # Gradient step
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.gradient_steps_skipped += int(
                scale_before > self.scaler.get_scale()
            )  # Count how many times the scale changed
            if not (scale_before > self.scaler.get_scale()):
                self.scheduler.step(
                    self.gradient_steps_per_epoch * self.cur_epoch + epoch_iteration
                )
            if self.scaler.get_scale() > self.config.grad_scaler_max_grad_scale:
                new_scale = torch.tensor(
                    self.config.grad_scaler_max_grad_scale,
                    device=self.config.device,
                    dtype=self.scaler.get_scale().dtype,
                )
                self.scaler.update(new_scale=new_scale)

            self.logger.log_train(
                epoch_iteration=epoch_iteration,
                cur_epoch=self.cur_epoch,
                unscaled_loss=losses,
                scaled_loss=scaled_loss,
                data=data,
                step=int(self.cur_epoch * self.gradient_steps_per_epoch)
                + epoch_iteration,
                gradient_steps_skipped=self.gradient_steps_skipped,
                log=log,
                predictions=predictions,
            )
            self.optimizer.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=self.config.torch_float_type,
            enabled=self.config.use_mixed_precision_training,
        ):
            rfm_score = None
            if self.config.use_waymo_e2e_data:
                rfm_score = evaluate_waymo_e2e(self.model, self.config)
                self.logger.logs({"rfm": rfm_score})
        return rfm_score

    @beartype
    def should_validate_epoch(self, epoch: int) -> bool:
        return (
            self.validation_dataloader is not None
            and self.config.validation_frequency_epochs > 0
            and (epoch + 1) % self.config.validation_frequency_epochs == 0
        )

    @staticmethod
    def _scalar_tensor(value, device: torch.device) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().float().mean()
        return torch.tensor(float(value), dtype=torch.float32, device=device)

    def _accumulate_metric(
        self,
        stats: dict[str, list[torch.Tensor]],
        name: str,
        value,
        weight: float,
    ):
        if weight <= 0:
            return
        scalar_value = Trainer._scalar_tensor(value, self.config.device)
        weight_tensor = torch.tensor(
            float(weight), dtype=torch.float32, device=scalar_value.device
        )
        if name not in stats:
            stats[name] = [
                torch.zeros((), dtype=torch.float32, device=scalar_value.device),
                torch.zeros((), dtype=torch.float32, device=scalar_value.device),
            ]
        stats[name][0] = stats[name][0] + scalar_value * weight_tensor
        stats[name][1] = stats[name][1] + weight_tensor

    @beartype
    def _aggregate_validation_stats(
        self,
        local_stats: dict[str, list[torch.Tensor]],
    ) -> dict[str, float]:
        if not torch.distributed.is_initialized():
            return {
                key: float((value_sum / value_count.clamp(min=1.0)).cpu().item())
                for key, (value_sum, value_count) in local_stats.items()
                if value_count.item() > 0
            }

        gathered_keys = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered_keys, sorted(local_stats))
        metric_keys = sorted({key for keys in gathered_keys for key in keys})

        aggregated_metrics = {}
        for key in metric_keys:
            default = torch.zeros((), dtype=torch.float64, device=self.config.device)
            value_sum, value_count = local_stats.get(key, [default, default])
            tensor = torch.stack(
                [
                    value_sum.to(device=self.config.device, dtype=torch.float64),
                    value_count.to(device=self.config.device, dtype=torch.float64),
                ]
            )
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            if self.config.rank == 0 and tensor[1].item() > 0:
                aggregated_metrics[key] = float((tensor[0] / tensor[1]).item())
        return aggregated_metrics

    @beartype
    def validate(self) -> dict[str, float]:
        assert self.validation_dataloader is not None

        self.model_wrapper.eval()
        local_stats: dict[str, list[torch.Tensor]] = {}

        with torch.no_grad():
            for validation_iteration, data in enumerate(
                tqdm(
                    self.validation_dataloader,
                    total=len(self.validation_dataloader),
                    disable=self.config.rank != 0,
                    desc=f"Validation epoch {self.cur_epoch}/{self.config.epochs}",
                )
            ):
                batch_size = float(data["source_dataset"].shape[0])
                data["iteration"] = validation_iteration
                data["training_step"] = self.step
                should_compute_additional_metrics = (
                    self.config.val_additional_metrics_frequency > 0
                    and (validation_iteration + 1)
                    % self.config.val_additional_metrics_frequency
                    == 0
                )
                data["compute_additional_metrics"] = (
                    self.config.validation_compute_additional_metrics
                    and should_compute_additional_metrics
                )

                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=self.config.torch_float_type,
                    enabled=self.config.use_mixed_precision_training,
                ):
                    predictions = self.model_wrapper(data=data)
                    losses, log = self.model.compute_loss(
                        predictions=predictions,
                        data=data,
                    )

                    loss_total = torch.zeros(
                        1,
                        dtype=self.config.torch_float_type,
                        device=self.config.device,
                    )
                    scaled_losses = {}
                    for key, value in losses.items():
                        scaled_value = self.detailed_loss_weights[key] * value
                        loss_total += scaled_value.reshape(1)
                        scaled_losses[key] = scaled_value

                self._accumulate_metric(
                    local_stats,
                    "val/loss_total",
                    loss_total,
                    batch_size,
                )
                for loss_name, loss_value in losses.items():
                    self._accumulate_metric(
                        local_stats,
                        f"val/unscaled_loss/{loss_name}",
                        loss_value,
                        batch_size,
                    )
                for loss_name, loss_value in scaled_losses.items():
                    self._accumulate_metric(
                        local_stats,
                        f"val/scaled_loss/{loss_name}",
                        loss_value,
                        batch_size,
                    )
                for metric_name, metric_value in log.items():
                    if metric_name.startswith("metric/"):
                        self._accumulate_metric(
                            local_stats,
                            f"val/metric/{metric_name.removeprefix('metric/')}",
                            metric_value,
                            1.0,
                        )

        return self._aggregate_validation_stats(local_stats)

    @beartype
    def save(self, rfm_score: float | None = None):
        if not self.config.save_model_checkpoint:
            return

        # Save best Waymo E2E model based on RFM score
        if self.config.use_waymo_e2e_data:
            potential_path = os.path.join(
                self.config.logdir,
                f"model_best_rfm_{self.cur_epoch}_{rfm_score:04f}.pth",
            )
            current_bests = os.listdir(self.config.logdir)
            current_bests = [f for f in current_bests if f.startswith("model_best_rfm")]
            new_best = False
            if len(current_bests) == 0 or all(
                rfm_score > float(f.split("_")[-1][:-4]) for f in current_bests
            ):
                torch.save(
                    self.model.state_dict(),
                    potential_path,
                )
                LOG.info(
                    f"New best Waymo E2E RFM score: {rfm_score:.4f}, saved model to {potential_path}"
                )
                new_best = True
            if new_best:
                for f in current_bests:
                    os.remove(os.path.join(self.config.logdir, f))

        if self.config.use_zero_redundancy and torch.cuda.device_count() > 1:
            # To save the whole optimizer we need to gather it on GPU 0.
            self.optimizer.consolidate_state_dict(0)

        # The parallel weights are named differently with the module.
        # We remove that, so that we can load the model with the same code.
        torch.save(
            self.model.state_dict(),
            os.path.join(self.config.logdir, f"model_{self.cur_epoch:04d}.pth"),
        )
        LOG.info(f"Saved model checkpoint for epoch {self.cur_epoch}.")
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(self.config.logdir, f"optimizer_{self.cur_epoch:04d}.pth"),
        )
        LOG.info(f"Saved optimizer checkpoint for epoch {self.cur_epoch}.")
        torch.save(
            self.scheduler.state_dict(),
            os.path.join(self.config.logdir, f"scheduler_{self.cur_epoch:04d}.pth"),
        )
        LOG.info(f"Saved scheduler checkpoint for epoch {self.cur_epoch}.")
        if self.scaler is not None:
            torch.save(
                self.scaler.state_dict(),
                os.path.join(self.config.logdir, f"scaler_{self.cur_epoch:04d}.pth"),
            )
            LOG.info(f"Saved scaler checkpoint for epoch {self.cur_epoch}.")
        with open(
            os.path.join(
                self.config.logdir, f"gradient_steps_skipped_{self.cur_epoch:04d}.txt"
            ),
            "w",
        ) as f:
            f.write(str(self.gradient_steps_skipped))
        LOG.info(f"Saved gradient steps skipped for epoch {self.cur_epoch}.")

        # Remove last epochs files to avoid accumulating storage
        if self.cur_epoch > 0:
            last_model_file = os.path.join(
                self.config.logdir, f"model_{self.cur_epoch - 1:04d}.pth"
            )
            last_optimizer_file = os.path.join(
                self.config.logdir, f"optimizer_{self.cur_epoch - 1:04d}.pth"
            )
            last_scheduler_file = os.path.join(
                self.config.logdir, f"scheduler_{self.cur_epoch - 1:04d}.pth"
            )
            last_scaler_file = os.path.join(
                self.config.logdir, f"scaler_{self.cur_epoch - 1:04d}.pth"
            )
            last_gradient_steps_skipped_file = os.path.join(
                self.config.logdir,
                f"gradient_steps_skipped_{self.cur_epoch - 1:04d}.txt",
            )

            if (
                os.path.isfile(last_model_file)
                and self.cur_epoch not in self.config.epoch_checkpoints_keep
            ):
                os.remove(last_model_file)
                LOG.info(f"Removed model checkpoint for epoch {self.cur_epoch - 1}.")
            if os.path.isfile(last_optimizer_file):
                os.remove(last_optimizer_file)
                LOG.info(
                    f"Removed optimizer checkpoint for epoch {self.cur_epoch - 1}."
                )
            if os.path.isfile(last_scheduler_file):
                os.remove(last_scheduler_file)
                LOG.info(
                    f"Removed scheduler checkpoint for epoch {self.cur_epoch - 1}."
                )
            if os.path.isfile(last_scaler_file):
                os.remove(last_scaler_file)
                LOG.info(f"Removed scaler checkpoint for epoch {self.cur_epoch - 1}.")
            if os.path.isfile(last_gradient_steps_skipped_file):
                os.remove(last_gradient_steps_skipped_file)
                LOG.info(
                    f"Removed gradient steps skipped for epoch {self.cur_epoch - 1}."
                )


@record  # Records error and tracebacks in case of failure
def main(config_path: str | None = None):
    training_utils.increase_limit_file_descriptors()
    Trainer(config_path=config_path).train_loop()


def parse_args() -> argparse.Namespace:
    def parse_fraction(value: str) -> float:
        return float(value.replace(",", "."))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help="Path to a training config JSON file.",
    )
    parser.add_argument(
        "--carla-dataset-fraction",
        dest="carla_dataset_fraction",
        type=parse_fraction,
        default=None,
        help="Fraction of the CARLA dataset to train on, e.g. 0.1 or 0.001.",
    )
    args, remaining = parser.parse_known_args()
    if args.config_path is None and remaining and remaining[0].endswith(".json"):
        args.config_path = remaining.pop(0)
    if args.carla_dataset_fraction is not None:
        remaining.append(f"carla_dataset_fraction={args.carla_dataset_fraction}")
    remaining = [
        arg.replace(",", ".") if arg.startswith("carla_dataset_fraction=") else arg
        for arg in remaining
    ]
    sys.argv = [sys.argv[0], *remaining]
    return args


if __name__ == "__main__":
    training_utils.set_start_method()
    args = parse_args()
    main(config_path=args.config_path)
