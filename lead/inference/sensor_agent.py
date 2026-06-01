import json
import logging
import os
import pathlib
import shutil
import time
from collections import deque
from copy import deepcopy

import carla
import cv2
import jaxtyping as jt
import matplotlib
import numpy as np
import numpy.typing as npt
import torch
from beartype import beartype
from leaderboard.autoagents import autonomous_agent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from lead.common import common_utils
from lead.common.base_agent import BaseAgent
from lead.common.constants import TransfuserBoundingBoxClass
from lead.common.logging_config import setup_logging
from lead.common.route_planner import RoutePlanner
from lead.common.sensor_setup import av_sensor_setup
from lead.data_loader import carla_dataset_utils, training_cache
from lead.data_loader.carla_dataset_utils import rasterize_lidar
from lead.expert import expert_utils
from lead.inference.closed_loop_inference import (
    ClosedLoopInference,
    ClosedLoopPrediction,
)
from lead.inference.config_closed_loop import ClosedLoopConfig
from lead.inference.video_recorder import VideoRecorder
from lead.training.config_training import TrainingConfig
from lead.visualization.visualizer import Visualizer

matplotlib.use("Agg")  # non-GUI backend for headless servers

setup_logging()
LOG = logging.getLogger(__name__)

# Configure pytorch for maximum performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True


def get_entry_point():  # dead: disable
    return "SensorAgent"


def resolve_checkpoint_config_path(checkpoint_path: str | os.PathLike) -> pathlib.Path:
    checkpoint_path = pathlib.Path(checkpoint_path)
    if checkpoint_path.is_file():
        return checkpoint_path

    config_path = checkpoint_path / "config.json"
    if config_path.exists():
        return config_path

    candidates = sorted(checkpoint_path.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No JSON config file found in checkpoint directory: {checkpoint_path}"
        )
    return candidates[0]


class SensorAgent(BaseAgent, autonomous_agent.AutonomousAgent):
    @beartype
    def setup(self, path_to_conf_file: str, _=None, __=None):
        # Set up test time training default parameters
        self.config_closed_loop = ClosedLoopConfig()
        super().setup(sensor_agent=True)
        self.config_path = path_to_conf_file
        self.step = -1
        self.initialized = False
        self.device = torch.device("cuda:0")
        self._sensors_cache = None

        # Load the config saved during training
        if self.config_closed_loop.is_bench2drive:
            path_to_conf_file = path_to_conf_file.split("+")[0]
        config_path = resolve_checkpoint_config_path(path_to_conf_file)
        LOG.info("[SensorAgent] Loading model config from %s", config_path)
        with open(config_path, encoding="utf-8") as f:
            json_config = f.read()
            json_config = json.loads(json_config)

        # Generate new config for the case that it has new variables.
        self.training_config = TrainingConfig(json_config)
        self._apply_model_runtime_overrides()

        # Store training config in base class for Kalman filter decision
        # This is accessed by BaseAgent._use_kalman_filter()

        # Load model files
        self.closed_loop_inference = ClosedLoopInference(
            config_training=self.training_config,
            config_closed_loop=self.config_closed_loop,
            config_expert=self.config_expert,
            model_path=path_to_conf_file,
            device=self.device,
            prefix="model",
        )

        # Post-processing heuristics
        self.bb_buffer = deque(maxlen=1)
        self.stop_sign_post_processor = StopSignPostProcessor(
            config=self.training_config,
            config_test_time=self.config_closed_loop,
            bb_buffer=self.bb_buffer,
        )
        self.force_move_post_processor = ForceMovePostProcessor(
            config=self.training_config,
            config_test_time=self.config_closed_loop,
            lidar_queue=self.lidar_pc_queue,
        )
        self.metric_info = {}
        self.meters_travelled = 0.0
        self._model_inference_timing_enabled = bool(
            getattr(self.training_config, "model_inference_timing", False)
        )
        self._model_inference_timing_warmup_steps = int(
            getattr(self.training_config, "model_inference_timing_warmup_steps", 0)
        )
        self._model_inference_timing_samples: list[float] = []
        self._model_inference_timing_seen = 0
        self._inference_dataset_save_enabled = bool(
            getattr(self.training_config, "save_inference_dataset", False)
        )
        self._inference_dataset_frame = 0
        self._inference_dataset_root: pathlib.Path | None = None
        self._setup_inference_dataset_saving()

        # Infraction tracking
        self.infractions_log = []  # List of {"step": int, "infraction": str}
        self.tracked_infraction_ids = (
            set()
        )  # Track which infractions we've already logged
        self.scenario = None  # Will be set by set_scenario() method

        self.track = autonomous_agent.Track.SENSORS

        if self._uses_video_output() and not shutil.which("ffmpeg"):
            raise RuntimeError(
                "ffmpeg is not installed or not found in PATH. Please install ffmpeg to use video compression."
            )

        self._telemetry_file = None
        self._telemetry_enabled = False
        self._setup_raw_telemetry()

    def _setup_inference_dataset_saving(self) -> None:
        if not self._inference_dataset_save_enabled:
            return
        if self.config_closed_loop.save_path is None:
            LOG.warning(
                "[InferenceDataset] save_inference_dataset is enabled, but SAVE_PATH is not set."
            )
            self._inference_dataset_save_enabled = False
            return

        scenario_type = os.environ.get("SCENARIO_TYPE", "noScenarios")
        route_name = (
            os.environ.get("ROUTE_INDEX")
            or os.environ.get("BENCHMARK_ROUTE_ID")
            or os.environ.get("ROUTE_NUMBER")
            or "route"
        )
        route_name = str(route_name).replace(os.sep, "_")
        self._inference_dataset_root = (
            self.config_closed_loop.save_path
            / "inference_dataset"
            / scenario_type
            / route_name
        )
        self._inference_dataset_root.mkdir(parents=True, exist_ok=True)

        if bool(getattr(self.training_config, "save_inference_dataset_rgb", True)):
            rgb_dir = self._inference_dataset_root / "rgb"
            rgb_dir.mkdir(exist_ok=True)
            if bool(
                getattr(
                    self.training_config,
                    "save_inference_dataset_in_subfolders",
                    True,
                )
            ):
                for camera_id in self._inference_dataset_camera_ids():
                    (rgb_dir / f"cam{camera_id}").mkdir(exist_ok=True)

        if bool(getattr(self.training_config, "save_inference_dataset_metas", True)):
            (self._inference_dataset_root / "metas").mkdir(exist_ok=True)

        LOG.info(
            "[InferenceDataset] Saving inference dataset to %s",
            self._inference_dataset_root,
        )

    def _inference_dataset_camera_ids(self) -> tuple[int, ...]:
        camera_ids = tuple(
            int(camera_id)
            for camera_id in getattr(
                self.training_config,
                "rgb_camera_ids",
                tuple(range(1, self.training_config.num_cameras + 1)),
            )
        )
        if len(camera_ids) == 0:
            camera_ids = tuple(range(1, self.training_config.num_cameras + 1))
        return camera_ids

    def _tensor_to_list(self, value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().float().numpy().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _prediction_field(self, prediction, *names):
        for name in names:
            if hasattr(prediction, name):
                return getattr(prediction, name)
        return None

    def _build_inference_dataset_meta(
        self,
        input_data: dict,
        closed_loop_prediction: ClosedLoopPrediction | None,
        frame: int,
    ) -> dict:
        control = getattr(self, "control", carla.VehicleControl())
        meta = {
            "frame": int(frame),
            "step": int(self.step),
            "route_index": os.environ.get("ROUTE_INDEX"),
            "benchmark_route_id": os.environ.get("BENCHMARK_ROUTE_ID"),
            "scenario_type": os.environ.get("SCENARIO_TYPE"),
            "town": self._world.get_map().name.split("/")[-1],
            "speed": float(np.asarray(input_data.get("speed", 0.0)).item()),
            "steer": float(control.steer),
            "throttle": float(control.throttle),
            "brake": float(control.brake),
            "theta": float(np.asarray(input_data.get("theta", 0.0)).item()),
            "pos_global": self._tensor_to_list(input_data.get("noisy_state")),
            "noisy_pos_global": self._tensor_to_list(input_data.get("noisy_state")),
            "filtered_pos_global": self._tensor_to_list(
                input_data.get("filtered_state")
            ),
            "target_point_previous": self._tensor_to_list(
                input_data.get("target_point_previous")
            ),
            "target_point": self._tensor_to_list(input_data.get("target_point")),
            "target_point_next": self._tensor_to_list(
                input_data.get("target_point_next")
            ),
            "command": self._tensor_to_list(input_data.get("command")),
            "next_command": self._tensor_to_list(input_data.get("next_command")),
            "gps": self._tensor_to_list(input_data.get("gps")),
            "compass": self._tensor_to_list(input_data.get("compass")),
            "accel": self._tensor_to_list(input_data.get("accel")),
            "angular_velocity": self._tensor_to_list(
                input_data.get("angular_velocity")
            ),
            "meters_travelled": float(self.meters_travelled),
            "camera_ids": list(self._inference_dataset_camera_ids()),
            "left_camera_key": str(
                getattr(self.training_config, "left_camera_key", "rgb_left")
            ),
            "right_camera_key": str(
                getattr(self.training_config, "right_camera_key", "rgb_right")
            ),
        }

        if closed_loop_prediction is not None:
            meta.update(
                {
                    "pred_route": self._tensor_to_list(
                        self._prediction_field(closed_loop_prediction, "pred_route")
                    ),
                    "pred_future_waypoints": self._tensor_to_list(
                        self._prediction_field(
                            closed_loop_prediction,
                            "pred_future_waypoints",
                        )
                    ),
                    "pred_target_speed": self._tensor_to_list(
                        self._prediction_field(
                            closed_loop_prediction,
                            "pred_target_speed",
                            "pred_target_speed_scalar",
                        )
                    ),
                    "pred_target_speed_distribution": self._tensor_to_list(
                        self._prediction_field(
                            closed_loop_prediction,
                            "pred_target_speed_distribution",
                        )
                    ),
                    "model_steer": float(closed_loop_prediction.steer),
                    "model_throttle": float(closed_loop_prediction.throttle),
                    "model_brake": float(closed_loop_prediction.brake),
                }
            )

        return meta

    def _save_inference_dataset_frame(
        self,
        input_data: dict,
        closed_loop_prediction: ClosedLoopPrediction | None = None,
    ) -> None:
        if (
            not self._inference_dataset_save_enabled
            or self._inference_dataset_root is None
        ):
            return

        frequency = max(
            1,
            int(getattr(self.training_config, "save_inference_dataset_frequency", 1)),
        )
        if self.step % frequency != 0:
            return

        frame = self._inference_dataset_frame
        self._inference_dataset_frame += 1

        if bool(getattr(self.training_config, "save_inference_dataset_rgb", True)):
            rgb_dir = self._inference_dataset_root / "rgb"
            quality = int(
                getattr(self.training_config, "save_inference_dataset_jpeg_quality", 95)
            )
            params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            camera_ids = self._inference_dataset_camera_ids()
            images = []
            for camera_id in camera_ids:
                sensor_key = f"rgb_{camera_id}"
                if sensor_key not in input_data:
                    LOG.warning(
                        "[InferenceDataset] Missing %s while saving frame %04d.",
                        sensor_key,
                        frame,
                    )
                    continue
                image = input_data[sensor_key]
                images.append(image)
                if bool(
                    getattr(
                        self.training_config,
                        "save_inference_dataset_in_subfolders",
                        True,
                    )
                ):
                    cv2.imwrite(
                        str(rgb_dir / f"cam{camera_id}" / f"{frame:04}.jpg"),
                        image,
                        params,
                    )

            if (
                bool(
                    getattr(
                        self.training_config,
                        "save_inference_dataset_as_panorama",
                        False,
                    )
                )
                and images
            ):
                cv2.imwrite(
                    str(rgb_dir / f"{frame:04}.jpg"),
                    np.concatenate(images, axis=1),
                    params,
                )

        if bool(getattr(self.training_config, "save_inference_dataset_metas", True)):
            meta = self._build_inference_dataset_meta(
                input_data=input_data,
                closed_loop_prediction=closed_loop_prediction,
                frame=frame,
            )
            common_utils.write_pickle(
                path=self._inference_dataset_root / "metas" / f"{frame:04}.pkl",
                data=meta,
            )

    def _sync_inference_timer(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _record_model_inference_time(self, elapsed_s: float) -> None:
        if not self._model_inference_timing_enabled:
            return
        self._model_inference_timing_seen += 1
        if self._model_inference_timing_seen <= self._model_inference_timing_warmup_steps:
            return
        self._model_inference_timing_samples.append(float(elapsed_s))

    def _model_inference_timing_summary(self) -> dict:
        samples = np.array(self._model_inference_timing_samples, dtype=np.float64)
        summary = {
            "enabled": bool(self._model_inference_timing_enabled),
            "warmup_steps": int(self._model_inference_timing_warmup_steps),
            "seen_steps": int(self._model_inference_timing_seen),
            "measured_steps": int(samples.size),
        }
        if samples.size == 0:
            return summary
        summary.update(
            {
                "mean_s": float(samples.mean()),
                "mean_ms": float(samples.mean() * 1000.0),
                "min_ms": float(samples.min() * 1000.0),
                "max_ms": float(samples.max() * 1000.0),
                "median_ms": float(np.median(samples) * 1000.0),
                "p95_ms": float(np.percentile(samples, 95) * 1000.0),
                "fps": float(1.0 / samples.mean()) if samples.mean() > 0 else 0.0,
            }
        )
        return summary

    def _save_model_inference_timing_summary(self) -> None:
        if not self._model_inference_timing_enabled:
            return
        summary = self._model_inference_timing_summary()
        LOG.info(
            "[ModelTiming] measured_steps=%s mean=%.3f ms median=%.3f ms p95=%.3f ms fps=%.2f",
            summary.get("measured_steps", 0),
            summary.get("mean_ms", 0.0),
            summary.get("median_ms", 0.0),
            summary.get("p95_ms", 0.0),
            summary.get("fps", 0.0),
        )
        if self.config_closed_loop.save_path is None:
            return
        timing_path = self.config_closed_loop.save_path / "model_inference_timing.json"
        with open(timing_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

    def _apply_model_runtime_overrides(self) -> None:
        """Apply inference-time switches stored in the model training config."""
        if not bool(getattr(self.training_config, "disable_visual_artifacts", False)):
            return

        disabled_outputs = {
            "produce_demo_image": False,
            "produce_demo_video": False,
            "produce_debug_image": False,
            "produce_debug_video": False,
            "produce_input_image": False,
            "produce_input_video": False,
            "produce_grid_image": False,
            "produce_grid_video": False,
        }
        loaded_config = getattr(self.config_closed_loop, "_loaded_config", None)
        if loaded_config is None:
            loaded_config = {}
            self.config_closed_loop._loaded_config = loaded_config
        loaded_config.update(disabled_outputs)

    def _uses_video_output(self) -> bool:
        return any(
            bool(getattr(self.config_closed_loop, flag, False))
            for flag in (
                "produce_demo_video",
                "produce_debug_video",
                "produce_input_video",
                "produce_grid_video",
            )
        )

    def _setup_raw_telemetry(self) -> None:
        emit_flag = os.environ.get("LEAD_EMIT_TELEMETRY", "0").lower()
        telemetry_file = os.environ.get("LEAD_TELEMETRY_FILE", "")
        expert_mode = os.environ.get("LEAD_EXPERT_MODE", "").lower()

        should_emit = emit_flag in {"1", "true", "yes", "on"}
        mode_matches = expert_mode in {"", "0", "false", "no", "off"}
        if not should_emit or not telemetry_file or not mode_matches:
            return

        try:
            telemetry_dir = os.path.dirname(telemetry_file)
            if telemetry_dir:
                os.makedirs(telemetry_dir, exist_ok=True)
            self._telemetry_file = open(telemetry_file, "a", encoding="utf-8")
            self._telemetry_enabled = True
            LOG.info(f"[Telemetry] Enabled raw telemetry writer: {telemetry_file}")
        except Exception as e:
            LOG.error(
                f"[Telemetry] Failed to open telemetry file '{telemetry_file}': {e}"
            )
            self._telemetry_enabled = False
            self._telemetry_file = None

    def _emit_raw_telemetry(self, input_data: dict | None) -> None:
        if not self._telemetry_enabled or self._telemetry_file is None:
            return
        if not hasattr(self, "control") or self.control is None:
            return

        frame = int(self.step)
        try:
            frame = int(self._world.get_snapshot().frame)
        except Exception:
            pass

        raw_speed = 0.0
        if isinstance(input_data, dict):
            raw_speed = input_data.get("speed", 0.0)
        try:
            raw_speed = float(raw_speed)
        except Exception:
            raw_speed = 0.0

        payload = {
            "timestamp": time.time(),
            "frame": frame,
            "step": int(self.step),
            "speed": raw_speed,
            "steer": float(self.control.steer),
            "throttle": float(self.control.throttle),
            "brake": float(self.control.brake),
            "source": "sensor_agent",
            "expert": False,
        }

        try:
            self._telemetry_file.write(json.dumps(payload, ensure_ascii=True) + "\n")
            self._telemetry_file.flush()
        except Exception as e:
            LOG.error(f"[Telemetry] Failed to write telemetry payload: {e}")

    def _close_raw_telemetry(self) -> None:
        if self._telemetry_file is not None:
            try:
                self._telemetry_file.close()
            except Exception:
                pass
        self._telemetry_file = None
        self._telemetry_enabled = False

    def set_scenario(self, scenario):
        """Set the scenario reference to track infractions.

        This should be called by the leaderboard after loading the scenario.
        """
        self.scenario = scenario
        LOG.info("[SensorAgent] Scenario reference set for infraction tracking")

    def _init(self):
        # Get the hero vehicle and the CARLA world
        self._vehicle: carla.Actor = CarlaDataProvider.get_hero_actor()
        self._world: carla.World = self._vehicle.get_world()

        # Set up video recorder
        self.video_recorder = VideoRecorder(
            config_closed_loop=self.config_closed_loop,
            vehicle=self._vehicle,
            world=self._world,
            step_counter=self.step,
            training_config=self.training_config,
        )

        self.set_weather()
        self.initialized = True

    def set_weather(self):
        weather_name = None

        if self.config_closed_loop.random_weather:
            weathers = self.config_expert.weather_settings.keys()
            weather_name = np.random.choice(list(weathers))

        if self.config_closed_loop.custom_weather is not None:
            weather_name = self.config_closed_loop.custom_weather

        if weather_name is not None:
            weather = carla.WeatherParameters(
                **self.config_expert.weather_settings[weather_name]
            )
            self._world.set_weather(weather)
            LOG.info(f"Set weather to: {weather_name}")
            # night mode
            vehicles = self._world.get_actors().filter("*vehicle*")
            if expert_utils.get_night_mode(weather):
                for vehicle in vehicles:
                    vehicle.set_light_state(
                        carla.VehicleLightState(
                            carla.VehicleLightState.Position
                            | carla.VehicleLightState.LowBeam
                        )
                    )
            else:
                for vehicle in vehicles:
                    vehicle.set_light_state(carla.VehicleLightState.NONE)

    def _uses_dual_front_camera_input(self) -> bool:
        return bool(getattr(self.training_config, "dual_front_camera_mode", False))

    def _uses_lidar_input(self) -> bool:
        return bool(getattr(self.training_config, "use_lidar", True))

    def _uses_radar_input(self) -> bool:
        return bool(getattr(self.training_config, "use_radars", False))

    @beartype
    def sensors(self) -> list[dict]:
        if self._sensors_cache is None:
            self._sensors_cache = av_sensor_setup(
                config=self.training_config,
                lidar=self._uses_lidar_input(),
                radar=self._uses_radar_input(),
                sensor_agent=True,
                perturbate=False,
                perturbation_rotation=0.0,
                perturbation_translation=0.0,
            )
        return deepcopy(self._sensors_cache)

    def check_infractions(self) -> None:
        """Check for infractions that occurred in the current step and log them.

        Handles two types of infractions:
        1. Discrete events (e.g., collisions): Each event is logged once
        2. Continuous infractions (e.g., OutsideRouteLanesTest): Only logged when first detected

        This matches the behavior of scenario_runner where continuous infractions create
        a single event that gets updated, while discrete infractions create new events.
        """
        if self.scenario is None:
            return

        try:
            criteria = self.scenario.get_criteria()

            for criterion in criteria:
                if hasattr(criterion, "events") and criterion.events:
                    # Track which criterion is currently active
                    # For continuous infractions (like OutsideRouteLanesTest), we only log once
                    # until the infraction is cleared
                    criterion_key = criterion.name

                    for event in criterion.events:
                        # For discrete events (collisions), use frame as unique identifier
                        # For continuous events (route lanes), use only criterion name
                        # Continuous infractions typically have only 1 event that gets updated
                        is_continuous = len(criterion.events) == 1

                        if is_continuous:
                            # Continuous infraction: only log if not currently tracked
                            event_id = criterion_key
                        else:
                            # Discrete infraction: log each unique frame
                            event_id = (criterion_key, event.get_frame())

                        # Only log if we haven't seen this infraction before
                        if event_id not in self.tracked_infraction_ids:
                            self.tracked_infraction_ids.add(event_id)

                            # Map criterion name to readable infraction type
                            infraction_info = {
                                "step": self.step,
                                "infraction": criterion.name,
                                "frame": event.get_frame(),
                                "message": event.get_message()
                                if hasattr(event, "get_message")
                                else "",
                                "event_type": str(event.get_type())
                                if hasattr(event, "get_type")
                                else "",
                                "meters_travelled": round(self.meters_travelled, 2),
                            }

                            self.infractions_log.append(infraction_info)
                            LOG.info(
                                f"[SensorAgent] Infraction detected at step {self.step}: {criterion.name}"
                            )

                    # If no events remain for this criterion, remove it from tracking
                    # This allows continuous infractions to be logged again if they reoccur
                    if (
                        not criterion.events
                        and criterion_key in self.tracked_infraction_ids
                    ):
                        self.tracked_infraction_ids.discard(criterion_key)

            # Save infractions log to JSON
            if self.config_closed_loop.save_path is not None and hasattr(
                self, "infractions_log"
            ):
                infractions_path = (
                    self.config_closed_loop.save_path / "infractions.json"
                )
                infractions_data = {
                    "infractions": self.infractions_log,
                    "video_fps": self.config_closed_loop.video_fps,
                }
                with open(infractions_path, "w") as f:
                    json.dump(infractions_data, f, indent=4)
                LOG.info(
                    f"[SensorAgent] Saved {len(self.infractions_log)} infractions to {infractions_path}"
                )

        except Exception as e:
            LOG.warning(f"[SensorAgent] Error checking infractions: {e}")

    @beartype
    def set_target_points(self, input_data: dict, pop_distance: float):
        """Defines local planning signals based on the input data.

        Args:
            input_data: The input data containing sensor information and state. Will be fed into model.
            pop_distance: Distance threshold to pop waypoints from the route planner.
        """
        planner: RoutePlanner = self.gps_waypoint_planners_dict[pop_distance]

        @beartype
        def transform(point: list[float]) -> jt.Float[npt.NDArray, " 2"]:
            # Use filtered or noisy position based on training config
            ego_position = (
                self.filtered_state[:2]
                if self.config_closed_loop.use_kalman_filter
                else input_data["noisy_state"][:2]
            )
            return common_utils.inverse_conversion_2d(
                np.array(point), np.array(ego_position), self.compass
            )

        next_target_points = [tp[0].tolist() for tp in planner.route]
        next_commands = [int(planner.route[i][1]) for i in range(len(planner.route))]

        # Merge duplicate consecutive target points
        filtered_tp_list = []
        filtered_command_list = []
        for pt, cmd in zip(next_target_points, next_commands, strict=False):
            if (
                len(next_target_points) == 2
                or not filtered_tp_list
                or not np.allclose(pt[:2], filtered_tp_list[-1][:2])
            ):
                filtered_tp_list.append(pt)
                filtered_command_list.append(cmd)
        next_target_points = filtered_tp_list
        next_commands = filtered_command_list

        if len(next_target_points) > 2:
            input_data["target_point_next"] = transform(next_target_points[2][:2])
            input_data["target_point"] = transform(next_target_points[1][:2])
            input_data["target_point_previous"] = transform(next_target_points[0][:2])
        else:
            assert len(next_target_points) == 2
            input_data["target_point_next"] = transform(next_target_points[1][:2])
            input_data["target_point"] = transform(next_target_points[1][:2])
            input_data["target_point_previous"] = transform(next_target_points[0][:2])

        input_data["command"] = carla_dataset_utils.command_to_one_hot(next_commands[0])
        input_data["next_command"] = carla_dataset_utils.command_to_one_hot(
            next_commands[1]
        )

    @beartype
    @torch.inference_mode()
    def tick(self, input_data: dict) -> dict:
        """Pre-processes sensor data"""
        input_data = super().tick(
            input_data, use_kalman_filter=self.training_config.use_kalman_filter_for_gps
        )

        def process_rgb(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            original_rgb = rgb.copy()
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            _, rgb = cv2.imencode(
                ".jpg",
                rgb,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.config_closed_loop.jpeg_quality],
            )
            rgb = cv2.imdecode(rgb, cv2.IMREAD_UNCHANGED)
            rgb = np.transpose(rgb, (2, 0, 1))

            if self.training_config.horizontal_fov_reduction > 0:
                crop_pixels = self.training_config.horizontal_fov_reduction
                _, h, w = rgb.shape
                rgb = rgb[:, :, crop_pixels:-crop_pixels]
                rgb = np.transpose(rgb, (1, 2, 0))
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
                rgb = np.transpose(rgb, (2, 0, 1))

                h, w = original_rgb.shape[:2]
                original_rgb = original_rgb[:, crop_pixels:-crop_pixels, :]
                original_rgb = cv2.resize(
                    original_rgb, (w, h), interpolation=cv2.INTER_LINEAR
                )

            return rgb, original_rgb

        if self._uses_dual_front_camera_input():
            camera_ids = tuple(
                int(camera_id)
                for camera_id in getattr(
                    self.training_config,
                    "rgb_camera_ids",
                    tuple(range(1, self.training_config.num_cameras + 1)),
                )
            )
            if len(camera_ids) != 2:
                camera_ids = tuple(range(1, self.training_config.num_cameras + 1))
            if len(camera_ids) != 2:
                raise ValueError(
                    f"Dual-front inference expects exactly 2 cameras, got {camera_ids}."
                )

            left_key = str(getattr(self.training_config, "left_camera_key", "rgb_left"))
            right_key = str(
                getattr(self.training_config, "right_camera_key", "rgb_right")
            )
            original_images = []
            for output_key, camera_id in (
                (left_key, camera_ids[0]),
                (right_key, camera_ids[1]),
            ):
                sensor_key = f"rgb_{camera_id}"
                if sensor_key not in input_data:
                    raise KeyError(
                        f"Missing camera sensor {sensor_key!r} for dual-front input "
                        f"{output_key!r}. Available keys: {sorted(input_data.keys())}"
                    )
                input_data[output_key], original_rgb = process_rgb(
                    input_data[sensor_key]
                )
                original_images.append(original_rgb)

            input_data["original_rgb"] = np.concatenate(original_images, axis=1)
        else:
            # Simulate JPEG compression to avoid train-test mismatch
            rgb = input_data["rgb"]
            input_data["rgb"], input_data["original_rgb"] = process_rgb(rgb)

            # Cut cameras down to only used cameras
            for modality in ["rgb", "original_rgb"]:
                if (
                    self.training_config.num_used_cameras
                    != self.training_config.num_available_cameras
                ):
                    n = self.training_config.num_available_cameras
                    w = input_data[modality].shape[2] // n

                    rgb_slices = []
                    for i, use in enumerate(self.training_config.used_cameras):
                        if use:
                            s, e = i * w, (i + 1) * w
                            rgb_slices.append(input_data[modality][:, :, s:e])

                    input_data[modality] = np.concatenate(rgb_slices, axis=2)

        # Plan next target point and command.
        self.set_target_points(
            input_data, pop_distance=self.config_closed_loop.route_planner_min_distance
        )
        if self.config_closed_loop.sensor_agent_pop_distance_adaptive:
            dense_points = (
                np.linalg.norm(
                    input_data["target_point"] - input_data["target_point_next"]
                )
                < 10.0
                and min(
                    np.linalg.norm(input_data["target_point_previous"]),
                    np.linalg.norm(input_data["target_point"]),
                )
                < 10.0
            )
            dense_points = dense_points or (
                np.linalg.norm(
                    input_data["target_point_previous"] - input_data["target_point"]
                )
                < 10.0
                and min(
                    np.linalg.norm(input_data["target_point_previous"]),
                    np.linalg.norm(input_data["target_point"]),
                )
                < 10.0
            )
            if dense_points:
                self.set_target_points(input_data, pop_distance=4.0)

        # Ignore the next target point if it's too far away
        if (
            self.config_closed_loop.sensor_agent_skip_distant_target_point
            and np.linalg.norm(input_data["target_point_next"])
            > self.config_closed_loop.sensor_agent_skip_distant_target_point_threshold
        ):
            # Skip the next target point if it's too far away
            input_data["target_point_next"] = input_data["target_point"]

        # Lidar input
        if self._uses_lidar_input():
            lidar = self.accumulate_lidar()
            # Use only part of the lidar history we trained on
            lidar = lidar[lidar[:, -1] < self.training_config.training_used_lidar_steps]

            # At inference time, simulate laspy quantization to avoid train-test mismatch
            lidar[:, 0] = (
                np.round(lidar[:, 0] / self.config_expert.point_precision_x)
                * self.config_expert.point_precision_x
            )
            lidar[:, 1] = (
                np.round(lidar[:, 1] / self.config_expert.point_precision_y)
                * self.config_expert.point_precision_y
            )
            lidar[:, 2] = (
                np.round(lidar[:, 2] / self.config_expert.point_precision_z)
                * self.config_expert.point_precision_z
            )

            # Convert to pseudo image
            input_data["rasterized_lidar"] = rasterize_lidar(
                config=self.training_config, lidar=lidar[:, :3]
            )[..., None]

            # Simulate training time compression to avoid train-test mismatch
            input_data["rasterized_lidar"] = training_cache.compress_float_image(
                input_data["rasterized_lidar"], self.training_config
            )
            input_data["rasterized_lidar"] = training_cache.decompress_float_image(
                input_data["rasterized_lidar"]
            ).squeeze()[None, None]

        # Radar input preprocessing
        if self._uses_radar_input():
            # Preprocess radar input using the same function as during training
            input_data["radar"] = np.concatenate(
                carla_dataset_utils.preprocess_radar_input(
                    self.training_config, input_data
                ),
                axis=0,
            )

        return input_data

    @beartype
    @torch.inference_mode()
    def run_step(self, input_data: dict, _, __=None) -> carla.VehicleControl:
        self.step += 1

        if not self.initialized:
            self._init()
            self.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            input_data = self.tick(input_data)
            input_data["meters_travelled"] = self.meters_travelled
            self._save_inference_dataset_frame(input_data)
            self._emit_raw_telemetry(input_data)
            return self.control

        # Update video recorder step and demo cameras
        if hasattr(self, "video_recorder"):
            self.video_recorder.update_step(self.step)
            self.video_recorder.move_demo_cameras_with_ego()

        # Need to run this every step for GPS filtering
        input_data = self.tick(input_data)

        # Transform the data into torch tensor comforting with data loader's format.
        input_data_tensors = {
            "target_point_previous": torch.Tensor(input_data["target_point_previous"])
            .to(self.device, dtype=torch.float32)
            .view(1, 2),
            "target_point": torch.Tensor(input_data["target_point"])
            .to(self.device, dtype=torch.float32)
            .view(1, 2),
            "target_point_next": (
                torch.Tensor(input_data["target_point_next"]).to(
                    self.device, dtype=torch.float32
                )
            ).view(1, 2),
            "speed": torch.Tensor([input_data["speed"]])
            .to(self.device, dtype=torch.float32)
            .view(1),
            "command": torch.Tensor(input_data["command"])
            .to(self.device, dtype=torch.float32)
            .view(1, 6),
            "next_command": torch.Tensor(input_data["next_command"])
            .to(self.device, dtype=torch.float32)
            .view(1, 6),
            "town": np.array([self._world.get_map().name.split("/")[-1]]),
        }

        if self._uses_dual_front_camera_input():
            for key in (
                str(getattr(self.training_config, "left_camera_key", "rgb_left")),
                str(getattr(self.training_config, "right_camera_key", "rgb_right")),
            ):
                input_data_tensors[key] = torch.Tensor(input_data[key]).to(
                    self.device, dtype=torch.float32
                )[None]
        else:
            input_data_tensors["rgb"] = torch.Tensor(input_data["rgb"]).to(
                self.device, dtype=torch.float32
            )[None]

        if self._uses_lidar_input():
            input_data_tensors["rasterized_lidar"] = torch.Tensor(
                input_data["rasterized_lidar"]
            ).to(self.device, dtype=torch.float32)

        # Add radar data if available
        if self._uses_radar_input() and "radar" in input_data:
            input_data_tensors["radar"] = torch.Tensor(input_data["radar"]).to(
                self.device, dtype=torch.float32
            )[None]

        # Save input log if need
        if (
            self.config_closed_loop.save_path is not None
            and self.config_closed_loop.produce_input_log
        ):
            torch.save(
                {
                    k: v.to(torch.device("cpu")) if isinstance(v, torch.Tensor) else v
                    for k, v in input_data_tensors.items()
                },
                os.path.join(
                    self.config_closed_loop.input_log_path, str(self.step).zfill(5)
                )
                + ".pth",
            )

        # Forward pass
        if self._model_inference_timing_enabled:
            self._sync_inference_timer()
            inference_start = time.perf_counter()
            closed_loop_prediction: ClosedLoopPrediction = (
                self.closed_loop_inference.forward(data=input_data_tensors)
            )
            self._sync_inference_timer()
            self._record_model_inference_time(time.perf_counter() - inference_start)
        else:
            closed_loop_prediction: ClosedLoopPrediction = (
                self.closed_loop_inference.forward(data=input_data_tensors)
            )
        # Update bounding boxes
        if (
            closed_loop_prediction.pred_bounding_box_vehicle_system is not None
            and len(closed_loop_prediction.pred_bounding_box_vehicle_system) > 0
        ):
            self.bb_buffer.append(
                closed_loop_prediction.pred_bounding_box_vehicle_system
            )

        # Post-processing heuristic
        self.stop_sign_post_processor.update_stop_box(
            self.ego_past_positions[-2][0],
            self.ego_past_positions[-2][1],
            self.ego_past_yaws[-2],
            0.0,
            0.0,
            0.0,
        )
        closed_loop_prediction.throttle, closed_loop_prediction.brake = (
            self.force_move_post_processor.adjust(
                input_data["speed"].item(),
                closed_loop_prediction.throttle,
                closed_loop_prediction.brake,
            )
        )
        closed_loop_prediction.throttle, closed_loop_prediction.brake = (
            self.stop_sign_post_processor.adjust(
                input_data["speed"].item(),
                closed_loop_prediction.throttle,
                closed_loop_prediction.brake,
            )
        )
        self.meters_travelled += (
            input_data["speed"].item() * self.config_closed_loop.carla_frame_rate
        )
        input_data["meters_travelled"] = self.meters_travelled

        self.control = carla.VehicleControl(
            steer=float(closed_loop_prediction.steer),
            throttle=float(closed_loop_prediction.throttle),
            brake=float(closed_loop_prediction.brake),
        )

        # CARLA will not let the car drive in the initial frames. This help the filter not get confused.
        if self.step < self.training_config.inital_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)

        self._save_inference_dataset_frame(input_data, closed_loop_prediction)

        # Check for infractions at this step
        self.check_infractions()

        # Visualization of prediction for debugging and video recording
        input_data_tensors.update(
            {
                "steer": torch.Tensor([self.control.steer]),
                "throttle": torch.Tensor([self.control.throttle]),
                "brake": torch.Tensor([self.control.brake]).bool(),
                "distance_to_stop_sign": torch.Tensor(
                    [
                        self.stop_sign_post_processor.stop_sign_buffer[0].norm
                        if len(self.stop_sign_post_processor.stop_sign_buffer) > 0
                        else np.inf
                    ]
                ),
                "stuck_detector": torch.Tensor(
                    [int(self.force_move_post_processor.stuck_detector)]
                ).int(),
                "force_move": torch.Tensor(
                    [int(self.force_move_post_processor.force_move)]
                ).int(),
                "route_curvature": torch.Tensor(
                    [
                        common_utils.waypoints_curvature(
                            closed_loop_prediction.pred_route.squeeze()
                        )
                    ]
                ),
                "meters_travelled": torch.Tensor([self.meters_travelled]),
            }
        )

        # Save input images as PNG and video
        if (
            self.config_closed_loop.save_path is not None
            and self.step % self.config_closed_loop.produce_frame_frequency == 0
        ):
            # Get the RGB image for visualization (before JPEG compression)
            input_image = input_data["original_rgb"].copy()

            # Save input image and video using VideoRecorder
            if hasattr(self, "video_recorder"):
                self.video_recorder.save_input_image(input_image)
                self.video_recorder.save_input_video_frame(input_image)

        # Save demo images
        if (
            self.config_closed_loop.save_path is not None
            and self.step % self.config_closed_loop.produce_frame_frequency == 0
        ):
            # Get predicted route and waypoints (if available)
            pred_waypoints = (
                closed_loop_prediction.pred_future_waypoints[0]
                if closed_loop_prediction.pred_future_waypoints is not None
                else None
            )

            # Prepare target points dictionary for BEV visualization
            target_points = {
                "previous": input_data.get("target_point_previous"),
                "current": input_data.get("target_point"),
                "next": input_data.get("target_point_next"),
            }

            # Save demo cameras with visualization using VideoRecorder
            if hasattr(self, "video_recorder"):
                self.video_recorder.save_demo_cameras(pred_waypoints, target_points)
                # Save grid (demo + input stacked vertically) with planning visualization
                self.video_recorder.save_grid_image_and_video(
                    pred_waypoints=pred_waypoints,
                    target_points=target_points,
                )

        # Save abstract debug images
        if (
            self.config_closed_loop.save_path is not None
            and (
                self.config_closed_loop.produce_debug_video
                or self.config_closed_loop.produce_debug_image
            )
            and self.step % self.config_closed_loop.produce_frame_frequency == 0
        ):
            # Produce image
            image = Visualizer(
                config=self.training_config,
                data=input_data_tensors,
                prediction=closed_loop_prediction,
                config_test_time=self.config_closed_loop,
                test_time=True,
            ).visualize_inference_prediction()
            image = np.array(image).astype(np.uint8)

            # Save debug image and video using VideoRecorder
            if hasattr(self, "video_recorder"):
                self.video_recorder.save_debug_video_frame(image)
                self.video_recorder.save_debug_image(image)

        # Save metric info if in Bench2Drive mode
        if self.config_closed_loop.is_bench2drive and hasattr(self, "get_metric_info"):
            metric = self.get_metric_info()
            self.metric_info[self.step] = metric
            with open(
                f"{self.config_closed_loop.save_path}/metric_info.json", "w"
            ) as outfile:
                json.dump(self.metric_info, outfile, indent=4)
        self._emit_raw_telemetry(input_data)
        return self.control

    def destroy(self, _=None):
        # Clean up video recorder
        self._save_model_inference_timing_summary()
        if hasattr(self, "video_recorder"):
            self.video_recorder.cleanup_and_compress()
        self._close_raw_telemetry()


class StopSignPostProcessor:
    """Heuristics to obey stop sign law."""

    @beartype
    def __init__(
        self,
        config: TrainingConfig,
        config_test_time: ClosedLoopConfig,
        bb_buffer: deque,
    ):
        self.config = config
        self.config_test_time = config_test_time
        self.bb_buffer = bb_buffer
        self.stop_sign_buffer: deque = deque(maxlen=1)
        self.clear_stop_sign_cool_down = 0  # Counter if we recently cleared a stop sign
        self.slower_stop_sign_count = 0
        self.slower_for_stop_sign_cool_down = 0

    @beartype
    def adjust(self, ego_speed: float, current_throttle: float, current_brake: float):
        """Checks whether the car is intersecting with one of the detected stop signs"""
        if not self.config_test_time.slower_for_stop_sign or len(self.bb_buffer) == 0:
            # LOG.info("No bounding box")
            return current_throttle, current_brake

        if self.clear_stop_sign_cool_down > 0:
            self.clear_stop_sign_cool_down -= 1
        if self.slower_for_stop_sign_cool_down > 0:
            self.slower_for_stop_sign_cool_down -= 1
        stop_sign_stop_predicted = False

        for bb in self.bb_buffer[-1]:
            if bb.clazz == TransfuserBoundingBoxClass.STOP_SIGN:  # Stop sign detected
                # LOG.info("Stop sign detected.")
                self.stop_sign_buffer.append(bb)

        if len(self.stop_sign_buffer) > 0:
            # Check if we need to stop
            stop_box = self.stop_sign_buffer[0]
            stop_origin = carla.Location(x=stop_box.x, y=stop_box.y, z=0.0)
            stop_extent = carla.Vector3D(stop_box.w, stop_box.h, 1.0)
            stop_carla_box = carla.BoundingBox(stop_origin, stop_extent)
            stop_carla_box.rotation = carla.Rotation(0.0, np.rad2deg(stop_box.yaw), 0.0)

            stop_sign_distance = np.linalg.norm([stop_box.x, stop_box.y])
            boxes_intersect = (
                stop_sign_distance
                < self.config_test_time.slower_for_stop_sign_dist_threshold
            )
            if boxes_intersect and self.clear_stop_sign_cool_down <= 0:
                if ego_speed > 0.01:
                    # LOG.info("Stop sign intersection detected.")
                    stop_sign_stop_predicted = True
                else:
                    # LOG.info("Stop sign intersection detected but car is already stopped.")
                    # We have cleared the stop sign
                    stop_sign_stop_predicted = False
                    self.stop_sign_buffer.pop()
                    # Stop signs don't come in herds, so we know we don't need to clear one for a while.
                    self.clear_stop_sign_cool_down = (
                        self.config_test_time.slower_for_stop_sign_cool_down
                    )
                    self.slower_stop_sign_count = 0
            elif (
                self.slower_for_stop_sign_cool_down <= 0
                and stop_sign_distance
                < self.config_test_time.slower_for_stop_sign_dist_threshold
            ):
                # LOG.info("Stop sign in range for slower.")
                self.slower_stop_sign_count = (
                    self.config_test_time.slower_for_stop_sign_count
                )
                self.slower_for_stop_sign_cool_down = (
                    self.config_test_time.slower_for_stop_sign_cool_down
                )

        if len(self.stop_sign_buffer) > 0:
            # Remove boxes that are too far away
            if self.stop_sign_buffer[0].norm > abs(self.config.max_x_meter):
                # LOG.info("Stop sign removed")
                self.stop_sign_buffer.pop()

        if stop_sign_stop_predicted:
            # LOG.info("Stopping for stop sign.")
            current_throttle = 0.0
            current_brake = True

        if (
            self.config_test_time.slower_for_stop_sign
            and self.slower_stop_sign_count > 0
        ):
            # LOG.info("Slowing down for stop sign.")
            current_throttle = np.clip(
                current_throttle,
                0.0,
                self.config_test_time.slower_for_stop_sign_throttle_threshold,
            )
            self.slower_stop_sign_count -= 1

        return current_throttle, current_brake

    @beartype
    def update_stop_box(
        self,
        x: float,
        y: float,
        orientation: float,
        x_target: float,
        y_target: float,
        orientation_target: float,
    ):
        if not self.config_test_time.slower_for_stop_sign:
            return
        if len(self.stop_sign_buffer) != 0:
            self.stop_sign_buffer.append(
                self.stop_sign_buffer[0].update(
                    x, y, orientation, x_target, y_target, orientation_target
                )
            )


class ForceMovePostProcessor:
    """Forces the agent to move after a certain time of being stuck."""

    @beartype
    def __init__(
        self,
        config: TrainingConfig,
        config_test_time: ClosedLoopConfig,
        lidar_queue: deque,
    ):
        self.config = config
        self.config_test_time = config_test_time
        self.stuck_detector = 0
        self.force_move = 0
        self.lidar_buffer = lidar_queue

    @beartype
    def adjust(
        self, ego_speed: float, current_throttle: float, current_brake: float
    ) -> tuple[float, float]:
        if (
            ego_speed < 0.1
        ):  # 0.1 is just an arbitrary low number to threshold when the car is stopped
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0

        # If last red light was encountered a long time ago, we can assume it was cleared
        stuck_threshold = self.config_test_time.sensor_agent_stuck_threshold

        if self.stuck_detector > stuck_threshold:
            self.force_move = self.config_test_time.sensor_agent_stuck_move_duration

        if self.force_move > 0:
            emergency_stop = False
            # safety check
            safety_box = deepcopy(self.lidar_buffer[-1])

            # z-axis
            safety_box = safety_box[safety_box[..., 2] > self.config.safety_box_z_min]
            safety_box = safety_box[safety_box[..., 2] < self.config.safety_box_z_max]

            # y-axis
            safety_box = safety_box[safety_box[..., 1] > self.config.safety_box_y_min]
            safety_box = safety_box[safety_box[..., 1] < self.config.safety_box_y_max]

            # x-axis
            safety_box = safety_box[safety_box[..., 0] > self.config.safety_box_x_min]
            safety_box = safety_box[safety_box[..., 0] < self.config.safety_box_x_max]
            if len(safety_box) > 0:  # Checks if the List is empty
                emergency_stop = True
                LOG.info("Creeping overriden by safety box.")
            if not emergency_stop:
                LOG.info("Detected agent being stuck.")
                current_throttle = max(
                    self.config_test_time.sensor_agent_stuck_throttle, current_throttle
                )
                current_brake = 0.0
                self.force_move -= 1
            else:
                LOG.info("Forced moving stopped by safety box.")
                current_throttle = 0.0
                current_brake = 1.0
                self.force_move = self.config_test_time.sensor_agent_stuck_move_duration
        return current_throttle, current_brake


if __name__ == "__main__":
    sensor_agent = SensorAgent()
