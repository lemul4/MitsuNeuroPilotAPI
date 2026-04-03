"""Class that provides data collection functionalities for expert agents."""

import json
import logging
import math
import os
import pathlib
import pickle
import queue
import random
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import carla
import cv2
import laspy
import numpy as np
import torch
from agents.navigation.local_planner import LocalPlanner
from beartype import beartype
from leaderboard.utils.statistics_manager_local import RouteRecord

import lead.common.common_utils as common_utils
import lead.expert.expert_utils as expert_utils
from lead.common import constants, ransac, weathers
from lead.common.constants import (
    CameraPointCloudIndex,
    CarlaSemanticSegmentationClass,
    TargetDataset,
    TransfuserSemanticSegmentationClass,
)
from lead.common.pid_controller import ExpertLongitudinalController
from lead.common.route_planner import RoutePlanner
from lead.common.sensor_setup import av_sensor_setup
from lead.expert.expert_base import ExpertBase
from lead.expert.hdmap.chauffeurnet import ObsManager
from lead.expert.hdmap.run_stop_sign import RunStopSign
from lead.expert.kinematic_bicycle_model import KinematicBicycleModel

LOG = logging.getLogger(__name__)

LOS_RAY_ORIGIN_Z_OFFSET_METERS = 0.3
LOS_NEAR_ORIGIN_NOISE_METERS = 0.25
LOS_TARGET_DISTANCE_EPS_METERS = 0.2
LOS_FRONT_FACE_CENTERS_TO_CHECK = 3

LOS_LOCAL_SAMPLE_CENTER_UNIT = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
LOS_LOCAL_FACE_CENTER_UNIT = np.array(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float32,
)
LOS_LOCAL_CORNERS_UNIT = np.array(
    [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)


class SemanticPanoramaStitcher:
    """Geometric semantic panorama stitcher using camera calibration from config."""

    def __init__(self, config):
        self.config = config
        self.W_pano = int(config.semantic_panorama_width)
        self.H_pano = int(config.semantic_panorama_height)
        self.pano_fov = np.deg2rad(float(config.semantic_panorama_fov_deg))
        self.vertical_fov = np.deg2rad(float(config.semantic_panorama_vertical_fov_deg))
        self.projection_dist = float(config.semantic_panorama_projection_dist)

        configured_camera_ids = tuple(config.semantic_panorama_camera_ids)
        if len(configured_camera_ids) == 0:
            configured_camera_ids = tuple(sorted(config.camera_calibration.keys()))
        self.camera_ids = list(configured_camera_ids)

        configured_center_id = int(config.semantic_panorama_center_camera_id)
        self.center_camera_id = (
            configured_center_id
            if configured_center_id in self.camera_ids
            else self.camera_ids[len(self.camera_ids) // 2]
        )

        configured_render_order = tuple(config.semantic_panorama_render_order)
        if len(configured_render_order) > 0:
            self.render_order = [
                camera_id
                for camera_id in configured_render_order
                if camera_id in self.camera_ids
            ]
        else:
            self.render_order = [
                camera_id
                for camera_id in self.camera_ids
                if camera_id != self.center_camera_id
            ] + [self.center_camera_id]

        self.center_dilate_kernel = max(
            1, int(config.semantic_panorama_center_dilate_kernel)
        )
        self.center_dilate_iterations = max(
            0, int(config.semantic_panorama_center_dilate_iterations)
        )

        center_camera_cfg = config.camera_calibration[self.center_camera_id]
        self.reference_camera_z = float(center_camera_cfg["pos"][2])

        self.maps: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.valid_masks: dict[int, np.ndarray] = {}
        self._precompute_maps()

    def _precompute_maps(self) -> None:
        theta = np.linspace(-self.pano_fov / 2.0, self.pano_fov / 2.0, self.W_pano)
        phi = np.linspace(
            -self.vertical_fov / 2.0,
            self.vertical_fov / 2.0,
            self.H_pano,
        )
        theta_grid, phi_grid = np.meshgrid(theta, phi)

        x_world = self.projection_dist * np.sin(theta_grid)
        y_world = self.projection_dist * np.tan(phi_grid)
        z_world = self.projection_dist * np.cos(theta_grid)

        for camera_id in self.camera_ids:
            camera_cfg = self.config.camera_calibration[camera_id]
            camera_pos = np.asarray(camera_cfg["pos"], dtype=np.float32)
            yaw = np.deg2rad(float(camera_cfg["rot"][2]))
            width = int(camera_cfg["width"])
            height = int(camera_cfg["height"])
            camera_fov = np.deg2rad(float(camera_cfg["fov"]))

            focal = (width / 2.0) / np.tan(camera_fov / 2.0)

            dx = x_world - camera_pos[1]
            dy = y_world - (camera_pos[2] - self.reference_camera_z)
            dz = z_world - camera_pos[0]

            cos_yaw = np.cos(-yaw)
            sin_yaw = np.sin(-yaw)
            x_local = cos_yaw * dx + sin_yaw * dz
            y_local = dy
            z_local = -sin_yaw * dx + cos_yaw * dz

            z_positive = z_local > 0.1
            z_safe = np.where(z_positive, z_local, 0.1)
            u = focal * (x_local / z_safe) + width / 2.0
            v = focal * (y_local / z_safe) + height / 2.0

            valid = z_positive & (u >= 0) & (u < width) & (v >= 0) & (v < height)

            self.maps[camera_id] = (u.astype(np.float32), v.astype(np.float32))

            if (
                camera_id == self.center_camera_id
                and self.center_dilate_iterations > 0
                and self.center_dilate_kernel > 1
            ):
                kernel = np.ones(
                    (self.center_dilate_kernel, self.center_dilate_kernel),
                    dtype=np.uint8,
                )
                valid = cv2.dilate(
                    valid.astype(np.uint8),
                    kernel,
                    iterations=self.center_dilate_iterations,
                ).astype(bool)

            self.valid_masks[camera_id] = valid

    def stitch(self, img_dict: dict[int, np.ndarray]) -> np.ndarray:
        if len(img_dict) == 0:
            raise ValueError("img_dict must contain at least one camera image")

        first_image = next(iter(img_dict.values()))
        if first_image.ndim == 2:
            panorama = np.zeros((self.H_pano, self.W_pano), dtype=first_image.dtype)
        else:
            panorama = np.zeros(
                (self.H_pano, self.W_pano, first_image.shape[2]),
                dtype=first_image.dtype,
            )

        warped_images: dict[int, np.ndarray] = {}
        for camera_id in self.camera_ids:
            if camera_id not in img_dict:
                continue
            map_u, map_v = self.maps[camera_id]
            warped_images[camera_id] = cv2.remap(
                img_dict[camera_id],
                map_u,
                map_v,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REPLICATE,
            )

        for camera_id in self.render_order:
            if camera_id not in warped_images:
                continue
            valid_mask = self.valid_masks[camera_id]
            panorama[valid_mask] = warped_images[camera_id][valid_mask]

        return panorama


class ExpertData(ExpertBase):
    def expert_setup(
        self,
        path_to_conf_file: str,
        route_index: str | None = None,
        traffic_manager: carla.TrafficManager | None = None,
    ):
        """
        Set up the autonomous agent for the CARLA simulation.

        Args:
            path_to_conf_file: Path to the configuration file.
            route_index: Index of the route to follow.
            traffic_manager: The traffic manager object.
        """
        super().expert_setup(path_to_conf_file, route_index, traffic_manager)

        # Dynamics models
        self.ego_model = KinematicBicycleModel(self.config_expert)
        self.vehicle_model = KinematicBicycleModel(self.config_expert)

        # To avoid failing the ActorBlockedTest, the agent has to move at least 0.1 m/s every 179 ticks
        self.waiting_ticks_at_stop_sign = 0
        self.ego_blocked_for_ticks = 0

        # Controllers
        self.perturbation_translation = 0
        self.perturbation_rotation = 0

        # Set up the save path if specified
        if os.environ.get("SAVE_PATH", None) is not None:
            self.save_path = (
                pathlib.Path(os.environ["SAVE_PATH"])
                / f"{self.town}_Rep{self.rep}_{self.route_index}"
            )
            self.save_path.mkdir(parents=True, exist_ok=False)

            if self.config_expert.datagen:
                (self.save_path / "metas").mkdir()

        # Store metas to update acceleration with forward difference after finishing route
        self.metas = []
        self.transform_queue = deque(
            maxlen=self.config_expert.ego_num_temporal_data_points_saved + 1
        )
        self.negative_id_counter = expert_utils.NegativeIdCounter()
        self.traffic_manager = traffic_manager

        self.cutin_vehicle_starting_position = None

        # FOV settings for bounding box filtering
        self.target_fov = float(self.config_expert.target_fov)
        self.half_fov_rad = np.deg2rad(self.target_fov / 2.0)

        # Pre-initialize stitcher once so remap grids are not recomputed per frame.
        self.semantic_panorama_stitcher = self._init_semantic_panorama_stitcher()

        # Threading for async data saving
        self._save_queue_maxsize = max(1, int(self.config_expert.save_queue_maxsize))
        self._save_queue = queue.Queue(maxsize=self._save_queue_maxsize)
        self._save_queue_sentinel = object()
        self._save_thread = None
        self._save_thread_stop = threading.Event()
        self._image_write_executor: ThreadPoolExecutor | None = None
        self._save_queue_high_watermark = max(
            1,
            int(
                self._save_queue_maxsize
                * float(self.config_expert.save_queue_high_watermark_ratio)
            ),
        )
        self._camera_image_saving_disabled_warning_logged = False
        self._sensor_save_conflict_warnings: set[str] = set()
        self._sync_sensor_frame_origin: int | None = None
        self._sensor_frame_mismatch_warning_logged = False
        self._history_cache_dir: pathlib.Path | None = None
        self._meta_stage_dir: pathlib.Path | None = None
        self._bboxes_stage_dir: pathlib.Path | None = None
        self._staged_meta_paths: dict[int, pathlib.Path] = {}
        self._staged_bboxes_paths: dict[int, pathlib.Path] = {}
        self._cached_level_bbs_by_class: dict[carla.CityObjectLabel, tuple] = {}

        if (
            self.save_path is not None
            and self.config_expert.datagen
            and (
                not self.config_expert.py123d_data_format
                or self.config_expert.save_legacy_outputs_with_py123d
            )
        ):
            if self.config_expert.use_lidar:
                (self.save_path / "lidar").mkdir()
            (self.save_path / "rgb").mkdir()
            if (
                self.config_expert.save_camera_pc
                and self.config_expert.compute_camera_pc
            ):
                (self.save_path / "camera_pc").mkdir()
                if self.config_expert.perturbate_sensors:
                    (self.save_path / "camera_pc_perturbated").mkdir()
            if self.config_expert.perturbate_sensors:
                (self.save_path / "rgb_perturbated").mkdir()
            if (
                self.config_expert.enable_semantic_sensor
                and self.config_expert.save_semantic_segmentation
            ):
                (self.save_path / "semantics").mkdir()
                if self.config_expert.perturbate_sensors:
                    (self.save_path / "semantics_perturbated").mkdir()
            if self.config_expert.enable_depth_sensor and self.config_expert.save_depth:
                (self.save_path / "depth").mkdir()
                if self.config_expert.perturbate_sensors:
                    (self.save_path / "depth_perturbated").mkdir()
            (self.save_path / "hdmap").mkdir()
            if self.config_expert.perturbate_sensors:
                (self.save_path / "hdmap_perturbated").mkdir()
            (self.save_path / "bboxes").mkdir()
            if (
                self.config_expert.enable_instance_sensor
                and self.config_expert.save_instance_segmentation
            ):
                (self.save_path / "instance").mkdir()
                if self.config_expert.perturbate_sensors:
                    (self.save_path / "instance_perturbated").mkdir()
            if self.config_expert.use_radars:
                (self.save_path / "radar").mkdir()
                if self.config_expert.perturbate_sensors:
                    (self.save_path / "radar_perturbated").mkdir()
            if self.config_expert.SAVE_CAM_IMAGES_IN_SUBFOLDERS:
                camera_modalities = ["rgb"]
                if (
                    self.config_expert.enable_semantic_sensor
                    and self.config_expert.save_semantic_segmentation
                ):
                    camera_modalities.append("semantics")
                if (
                    self.config_expert.enable_depth_sensor
                    and self.config_expert.save_depth
                ):
                    camera_modalities.append("depth")
                if (
                    self.config_expert.enable_instance_sensor
                    and self.config_expert.save_instance_segmentation
                ):
                    camera_modalities.append("instance")

                all_camera_modalities = list(camera_modalities)
                if self.config_expert.perturbate_sensors:
                    all_camera_modalities.extend(
                        [f"{modality}_perturbated" for modality in camera_modalities]
                    )

                for modality in all_camera_modalities:
                    for camera_idx in range(1, self.config_expert.num_cameras + 1):
                        (self.save_path / modality / f"cam{camera_idx}").mkdir()

            if self.config_expert.stage_history_to_disk:
                self._history_cache_dir = self.save_path / ".history_cache"
                self._meta_stage_dir = self._history_cache_dir / "metas"
                self._bboxes_stage_dir = self._history_cache_dir / "bboxes"
                self._meta_stage_dir.mkdir(parents=True, exist_ok=True)
                self._bboxes_stage_dir.mkdir(parents=True, exist_ok=True)

        self.weather_setting = "ClearNoon"
        self.semantics_converter = np.uint8(
            list(constants.SEMANTIC_SEGMENTATION_CONVERTER.values())
        )

        # Start background saving thread if saving is enabled
        if self.save_path is not None and self.config_expert.datagen:
            if self.config_expert.sensor_image_write_threads > 1:
                self._image_write_executor = ThreadPoolExecutor(
                    max_workers=self.config_expert.sensor_image_write_threads
                )
            self._save_thread = threading.Thread(target=self._save_worker, daemon=True)
            self._save_thread.start()

    def expert_init(self, hd_map: carla.Map | None):
        """
        Initialize the agent by setting up the route planner, longitudinal controller,
        command planner, and other necessary components.

        Args:
            hd_map: The map object of the CARLA world.
        """
        super().expert_init(hd_map)

        # Set up the longitudinal controller and command planner
        self._longitudinal_controller = ExpertLongitudinalController(self.config_expert)
        self._command_planner = RoutePlanner(
            self.config_expert.route_planner_min_distance,
            self.config_expert.route_planner_max_distance,
        )
        self._command_planner.set_route(self._global_plan_world_coord)

        self._command_planners_dict = {}
        for dist in self.config_expert.tp_distances:
            planner = RoutePlanner(dist, self.config_expert.route_planner_max_distance)
            planner.set_route(self._global_plan_world_coord)
            self._command_planners_dict[dist] = planner

        # Debug camera setup
        if self.config_expert.save_3rd_person_camera and (
            not self.config_expert.is_on_slurm or self.save_path is not None
        ):
            bp_lib = self.carla_world.get_blueprint_library()
            camera_bp = bp_lib.find("sensor.camera.rgb")
            camera_bp.set_attribute(
                "image_size_x",
                self.config_expert.camera_3rd_person_calibration["image_size_x"],
            )
            camera_bp.set_attribute(
                "image_size_y",
                self.config_expert.camera_3rd_person_calibration["image_size_y"],
            )
            camera_bp.set_attribute(
                "fov", self.config_expert.camera_3rd_person_calibration["fov"]
            )
            self._3rd_person_camera = self.carla_world.spawn_actor(
                camera_bp, self.transform_3rd_person_camera
            )

            def _save_image(image):
                frame = self.step // self.config_expert.data_save_freq
                if self.config_expert.is_on_slurm or self.save_path is not None:
                    if self.save_path is None:
                        return
                    save_path_3rd_person = str(self.save_path / "3rd_person")
                    os.makedirs(save_path_3rd_person, exist_ok=True)
                    use_png = bool(self.config_expert.save_jpeg_images_as_png)
                    extension = "png" if use_png else "jpg"
                    image_path = os.path.join(
                        save_path_3rd_person, f"{str(frame).zfill(4)}.{extension}"
                    )
                    array = (
                        np.frombuffer(image.raw_data, dtype=np.uint8)
                        .reshape(image.height, image.width, 4)
                        .copy()
                    )
                    bgr = array[:, :, :3]
                    third_person_params: list[int] | None = None
                    if use_png:
                        third_person_params = [
                            int(cv2.IMWRITE_PNG_COMPRESSION),
                            self.config_expert.png_storage_compression_level
                            if self.config_expert.compress_images
                            else 0,
                        ]
                    if (
                        not self._save_thread_stop.is_set()
                        and self._save_thread is not None
                        and self._save_thread.is_alive()
                    ):
                        try:
                            self._save_queue.put(
                                {
                                    "kind": "third_person",
                                    "path": image_path,
                                    "image": bgr,
                                    "params": third_person_params,
                                },
                                timeout=self.config_expert.save_queue_put_timeout_third_person_sec,
                            )
                        except queue.Full:
                            # Keep callback non-blocking to avoid stalling CARLA sensor threads.
                            if third_person_params is None:
                                cv2.imwrite(image_path, bgr)
                            else:
                                cv2.imwrite(image_path, bgr, third_person_params)
                    else:
                        if third_person_params is None:
                            cv2.imwrite(image_path, bgr)
                        else:
                            cv2.imwrite(image_path, bgr, third_person_params)

            self._3rd_person_camera.listen(_save_image)
        if self.config_expert.datagen:
            self.shuffle_weather()
        jpeg_storage_quality_distribution = (
            self.config_expert.weather_jpeg_compression_quality[self.weather_setting]
        )  # key value: quality maps to probability
        if self.config_expert.jpeg_compression:
            self.jpeg_storage_quality = int(
                np.random.choice(
                    list(jpeg_storage_quality_distribution.keys()),
                    p=list(jpeg_storage_quality_distribution.values()),
                )
            )
        else:
            self.jpeg_storage_quality = (
                100 if not self.config_expert.compress_images else 90
            )
        LOG.info(f"[DataAgent] Chose JPEG storage quality {self.jpeg_storage_quality}")

        obs_config = {
            "width_in_pixels": 256,  # self.config.lidar_resolution_width,
            "pixels_ev_to_bottom": 32 * self.config_expert.pixels_per_meter,
            "pixels_per_meter": self.config_expert.pixels_per_meter_collection,
            "history_idx": [-1],
            "scale_bbox": True,
            "scale_mask_col": 1.0,
            "map_folder": "maps_2ppm_cv",
        }
        if obs_config["width_in_pixels"] != self.config_expert.lidar_width_pixel:
            LOG.warning(
                "The BEV resolution is not the same as the LiDAR resolution. This might lead to unexpected results"
            )

        self.stop_sign_criteria = RunStopSign(self.carla_world)
        self.ss_bev_manager = ObsManager(obs_config, self.config_expert)
        self.ss_bev_manager.attach_ego_vehicle(
            self.ego_vehicle, criteria_stop=self.stop_sign_criteria
        )

        if self.config_expert.perturbate_sensors:
            self.ss_bev_manager_perturbated = ObsManager(obs_config, self.config_expert)
            bb_copy = carla.BoundingBox(
                self.ego_vehicle.bounding_box.location,
                self.ego_vehicle.bounding_box.extent,
            )
            transform_copy = carla.Transform(
                self.ego_vehicle.get_transform().location,
                self.ego_vehicle.get_transform().rotation,
            )
            # Can't clone the carla vehicle object, so I use a dummy class with similar attributes.
            self.perturbated_vehicle_dummy = expert_utils.CarlaActorDummy(
                self.ego_vehicle.get_world(),
                bb_copy,
                transform_copy,
                self.ego_vehicle.id,
            )
            self.ss_bev_manager_perturbated.attach_ego_vehicle(
                self.perturbated_vehicle_dummy, criteria_stop=self.stop_sign_criteria
            )

        self._local_planner = LocalPlanner(
            self.ego_vehicle, opt_dict={}, map_inst=self.carla_world_map
        )
        self.bounding_boxes = []
        ransac.remove_ground(
            np.random.rand(1000, 3), self.config_expert, parallel=True
        )  # Pre-compile numba code
        try:
            self._cached_level_bbs_by_class[carla.CityObjectLabel.Car] = tuple(
                self.carla_world.get_level_bbs(carla.CityObjectLabel.Car)
            )
        except Exception as e:
            LOG.warning(f"Failed to pre-cache static level bbs: {e}")
            self._cached_level_bbs_by_class[carla.CityObjectLabel.Car] = tuple()

        self.initialized = True

    @beartype
    def shuffle_weather(self) -> None:
        LOG.info("Shuffling weather settings")
        # change weather for visual diversity
        weather = self.carla_world.get_weather()

        if self.config_expert.shuffle_weather or self.config_expert.nice_weather:
            if self.config_expert.nice_weather:
                self.weather_setting = "ClearNoon"
                LOG.info(f"Chose nice weather {self.weather_setting}")
            else:
                self.weather_setting = random.choice(
                    list(weathers.WEATHER_SETTINGS.keys())
                )
                LOG.info(f"Chose random weather {self.weather_setting}")
            LOG.info(f"Chose weather {self.weather_setting}")
            self.weather_parameters: dict[str, float] = weathers.WEATHER_SETTINGS[
                self.weather_setting
            ]

            if "Noon" in self.weather_setting:
                self.weather_parameters["sun_altitude_angle"] += np.random.uniform(
                    -45.0, 45.0
                )
            elif "Custom" not in self.weather_setting:
                self.weather_parameters["sun_altitude_angle"] += np.random.uniform(
                    -15.0, 15.0
                )

            for randomizing_parameter in ["wind_intensity", "fog_density", "wetness"]:
                if self.weather_parameters[randomizing_parameter] < 30:
                    self.weather_parameters[randomizing_parameter] += np.random.uniform(
                        -5.0, 5.0
                    )
                elif self.weather_parameters[randomizing_parameter] < 80:
                    self.weather_parameters[randomizing_parameter] += np.random.uniform(
                        -10.0, 10.0
                    )
                else:
                    self.weather_parameters[randomizing_parameter] += np.random.uniform(
                        -5.0, 5.0
                    )
                self.weather_parameters[randomizing_parameter] = np.clip(
                    self.weather_parameters[randomizing_parameter], 0.0, 100.0
                )

            weather = carla.WeatherParameters(**self.weather_parameters)

            self.carla_world.set_weather(weather)

            # night mode
            vehicles = self.carla_world.get_actors().filter("*vehicle*")
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
        else:
            self.weather_setting = expert_utils.get_weather_name(
                weather, self.config_expert
            )
            self.weather_parameters = expert_utils.weather_parameter_to_dict(weather)

        LOG.info(f"Current weather setting: {self.weather_setting}")
        self.visual_visibility = int(
            weathers.WEATHER_VISIBILITY_MAPPING[self.weather_setting]
        )

    @beartype
    def is_actor_inside_bev(self, actor: carla.Actor) -> bool:
        """
        Check if actor is visible in TransFuser++'s planning visible range.
        This is used to filter out actors that are not visible to TransFuser++'s
        planning module even though they might be visible in the camera.
        """
        actor_in_ego = common_utils.get_relative_transform(
            self.ego_matrix, np.array(actor.get_transform().get_matrix())
        )
        x_ego, y_ego, _ = actor_in_ego
        return bool(
            self.config_expert.min_x_meter - 2
            < x_ego
            < self.config_expert.max_x_meter + 2
            and self.config_expert.min_y_meter - 2
            < y_ego
            < self.config_expert.max_y_meter + 2
            and np.linalg.norm(actor_in_ego) < self.config_expert.bb_save_radius
        )

    def update_3rd_person_camera(self):
        """
        Track ego with 3rd person camera.
        """
        if hasattr(self, "_3rd_person_camera") and self._3rd_person_camera.is_alive:
            self._3rd_person_camera.set_transform(self.transform_3rd_person_camera)

    def sensors(self):
        """
        Returns a list of sensor specifications for the ego vehicle.

        Each sensor specification is a dictionary containing the sensor type,
        reading frequency, position, and other relevant parameters.

        Returns:
            list: A list of sensor specification dictionaries.
        """
        result = []
        if not self.config_expert.datagen:
            result = [
                {
                    "type": "sensor.opendrive_map",
                    "reading_frequency": 1e-6,
                    "id": "hd_map",
                },
                {
                    "type": "sensor.other.imu",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "sensor_tick": self.config_expert.carla_frame_rate,
                    "id": "imu",
                },
                {
                    "type": "sensor.speedometer",
                    "reading_frequency": self.config_expert.carla_fps,
                    "id": "speed",
                },
                {
                    "type": "sensor.other.gnss",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "sensor_tick": 0.01,
                    "id": "gps",
                },
            ]

        self.perturbation_translation, self.perturbation_rotation = (
            expert_utils.sample_sensor_perturbation_parameters(
                config=self.config_expert,
                max_speed_limit_route=self.max_speed_limit_route,
                min_lane_width_route=self.min_lane_width_route,
            )
        )

        # --- Set up sensor rig ---
        if self.save_path is not None and self.config_expert.datagen:
            result += av_sensor_setup(
                self.config_expert,
                perturbation_rotation=self.perturbation_rotation,
                perturbation_translation=self.perturbation_translation,
                lidar=self.config_expert.use_lidar,
                perturbate=self.config_expert.perturbate_sensors,
                sensor_agent=False,
                radar=self.config_expert.use_radars,
            )
        else:
            if self.config_expert.use_lidar:
                result.append(
                    {
                        "type": "sensor.lidar.ray_cast",
                        "x": self.config_expert.lidar_pos_1[0],
                        "y": self.config_expert.lidar_pos_1[1],
                        "z": self.config_expert.lidar_pos_1[2],
                        "roll": self.config_expert.lidar_rot_1[0],
                        "pitch": self.config_expert.lidar_rot_1[1],
                        "yaw": self.config_expert.lidar_rot_1[2],
                        "id": "lidar1",
                    },
                )
                if self.config_expert.use_two_lidars:
                    result.append(
                        {
                            "type": "sensor.lidar.ray_cast",
                            "x": self.config_expert.lidar_pos_2[0],
                            "y": self.config_expert.lidar_pos_2[1],
                            "z": self.config_expert.lidar_pos_2[2],
                            "roll": self.config_expert.lidar_rot_2[0],
                            "pitch": self.config_expert.lidar_rot_2[1],
                            "yaw": self.config_expert.lidar_rot_2[2],
                            "id": "lidar2",
                        },
                    )
        return result

    @beartype
    def get_nearby_object(self, actors: carla.ActorList, search_radius: float) -> list:
        """
        Find actors, who's trigger boxes are within a specified radius around the ego vehicle.

        Args:
            actors: A list of actors to search through.
            search_radius: The radius (in meters) around the ego vehicle to search for nearby actors.

        Returns:
            A list of actors within the specified search radius.
        """
        nearby_objects = []
        for actor in actors:
            try:
                trigger_box_global_pos = actor.get_transform().transform(
                    actor.trigger_volume.location
                )
            except:
                LOG.info(
                    "Warning! Error caught in get_nearby_objects. (probably AttributeError: actor.trigger_volume)"
                )
                LOG.info("Skipping this object.")
                continue

            # Convert the vector to a carla.Location for distance calculation
            trigger_box_global_pos = carla.Location(
                x=trigger_box_global_pos.x,
                y=trigger_box_global_pos.y,
                z=trigger_box_global_pos.z,
            )

            # Check if the actor's trigger volume is within the search radius
            if trigger_box_global_pos.distance(self.ego_location) < search_radius:
                nearby_objects.append(actor)

        return nearby_objects

    @staticmethod
    def _empty_camera_pc_numpy() -> np.ndarray:
        return np.zeros((0, 5), dtype=np.float32)

    @staticmethod
    def _empty_camera_pc_torch() -> torch.Tensor:
        return torch.zeros((0, 5), dtype=torch.float32)

    def _set_empty_camera_pc(self, input_data: dict) -> None:
        for camera_idx in range(1, self.config_expert.num_cameras + 1):
            input_data[f"semantics_camera_pc_{camera_idx}"] = (
                self._empty_camera_pc_torch()
            )
        input_data["semantics_camera_pc"] = self._empty_camera_pc_numpy()
        input_data["semantics_camera_pc_all"] = input_data["semantics_camera_pc"]

    def _extract_sensor_frames(self, input_data: dict) -> dict[str, int]:
        sensor_frames: dict[str, int] = {}

        raw_sensor_frames = input_data.get("_raw_sensor_frames")
        if isinstance(raw_sensor_frames, dict):
            for sensor_id, frame in raw_sensor_frames.items():
                if isinstance(frame, (int, np.integer)):
                    sensor_frames[str(sensor_id)] = int(frame)

        if len(sensor_frames) == 0:
            for sensor_id, sensor_value in input_data.items():
                if (
                    isinstance(sensor_value, tuple)
                    and len(sensor_value) == 2
                    and isinstance(sensor_value[0], (int, np.integer))
                ):
                    sensor_frames[str(sensor_id)] = int(sensor_value[0])

        return sensor_frames

    def _pick_reference_sensor_frame(self, sensor_frames: dict[str, int]) -> int | None:
        preferred_sensor_ids = ["rgb_1", "lidar1", "imu", "gps", "speed"]
        for sensor_id in preferred_sensor_ids:
            if sensor_id in sensor_frames:
                return int(sensor_frames[sensor_id])

        if len(sensor_frames) == 0:
            return None

        return int(next(iter(sensor_frames.values())))

    def _resolve_sync_state_from_input_data(
        self, input_data: dict
    ) -> tuple[int, bool, int | None]:
        freq = max(1, int(self.config_expert.data_save_freq))
        fallback_frame = self.step // freq
        fallback_is_saved = self.step % freq == 0

        if not isinstance(input_data, dict):
            return fallback_frame, fallback_is_saved, None

        sensor_frames = self._extract_sensor_frames(input_data)
        reference_sensor_frame = self._pick_reference_sensor_frame(sensor_frames)
        if reference_sensor_frame is None:
            return fallback_frame, fallback_is_saved, None

        if (
            not self._sensor_frame_mismatch_warning_logged
            and len(set(sensor_frames.values())) > 1
        ):
            LOG.warning(
                "Sensor frame mismatch detected at step %d. Frames=%s. "
                "Using reference sensor frame %d for save synchronization.",
                self.step,
                sensor_frames,
                reference_sensor_frame,
            )
            self._sensor_frame_mismatch_warning_logged = True

        if self._sync_sensor_frame_origin is None:
            self._sync_sensor_frame_origin = int(reference_sensor_frame)

        relative_frame = int(reference_sensor_frame) - int(
            self._sync_sensor_frame_origin
        )
        if relative_frame < 0:
            LOG.warning(
                "Reference sensor frame moved backwards (origin=%d, current=%d). "
                "Falling back to step-based frame numbering for this step.",
                int(self._sync_sensor_frame_origin),
                int(reference_sensor_frame),
            )
            return fallback_frame, fallback_is_saved, int(reference_sensor_frame)

        frame = relative_frame // freq
        is_saved_step = relative_frame % freq == 0
        input_data["_sync_sensor_frame"] = int(reference_sensor_frame)
        input_data["_sync_saved_frame"] = int(frame)
        input_data["_sync_is_saved_step"] = bool(is_saved_step)
        return int(frame), bool(is_saved_step), int(reference_sensor_frame)

    @beartype
    def tick(self, input_data: dict) -> dict:
        """
        Get the current state of the vehicle from the input data and the vehicle's sensors.

        Args:
            input_data: Input data containing sensor information.

        Returns:
            A dictionary containing the vehicle's position (GPS), speed, and compass heading.
        """
        input_data = super().tick(input_data)
        self.transform_queue.append(self.ego_vehicle.get_transform())
        if self.config_expert.use_radars and self.config_expert.datagen:
            radar_arrays = []
            for i in range(1, self.config_expert.num_radar_sensors + 1):
                radar_arrays.append(input_data[f"radar{i}"])
            input_data["radar"] = np.concatenate(radar_arrays, axis=0)

        _, synced_saved_step, _ = self._resolve_sync_state_from_input_data(input_data)

        process_sensor_payload_this_step = True
        if self.config_expert.sync_sensor_processing_with_data_save_freq:
            process_sensor_payload_this_step = synced_saved_step

        if self.save_path is not None and self.config_expert.datagen:
            if process_sensor_payload_this_step:
                if self.config_expert.perturbate_sensors:
                    # Process perturbated RGB images for each camera
                    rgb_perturbated_cameras = []
                    for camera_idx in range(1, self.config_expert.num_cameras + 1):
                        rgb_perturbated = input_data[f"rgb_{camera_idx}_perturbated"][
                            1
                        ][:, :, :3]
                        input_data[f"rgb_{camera_idx}_perturbated"] = rgb_perturbated
                        rgb_perturbated_cameras.append(rgb_perturbated)
                    input_data["rgb_perturbated"] = np.concatenate(
                        rgb_perturbated_cameras, axis=1
                    )

                if (
                    self.config_expert.use_radars
                    and self.config_expert.perturbate_sensors
                ):
                    radar_perturbated_dict = {}
                    for i in range(1, self.config_expert.num_radar_sensors + 1):
                        radar_perturbated = common_utils.radar_points_to_ego(
                            input_data[f"radar{i}_perturbated"][1],
                            sensor_pos=self.config_expert.radar_calibration[str(i)][
                                "pos"
                            ],
                            sensor_rot=self.config_expert.radar_calibration[str(i)][
                                "rot"
                            ],
                        )
                        radar_perturbated_dict[f"radar{i}_perturbated"] = (
                            radar_perturbated
                        )

                    input_data.update(radar_perturbated_dict)

                can_process_instance = self._can_process_instance()
                can_process_semantic = self._can_process_semantic()
                can_process_depth = self._can_process_depth()

                if can_process_instance:
                    # Instance segmentation - flexible camera processing
                    instances = []
                    converted_instances = []
                    for camera_idx in range(1, self.config_expert.num_cameras + 1):
                        instance = cv2.cvtColor(
                            input_data[f"instance_{camera_idx}"][1][:, :, :3],
                            cv2.COLOR_BGR2RGB,
                        )
                        converted_instance = expert_utils.convert_instance_segmentation(
                            instance
                        )

                        input_data[f"instance_{camera_idx}"] = instance
                        input_data[f"converted_instance_{camera_idx}"] = (
                            converted_instance
                        )
                        instances.append(instance)
                        converted_instances.append(converted_instance)

                    input_data["instance"] = np.concatenate(instances, axis=1)

                    if self.config_expert.perturbate_sensors:
                        instances_perturbated = []
                        for camera_idx in range(1, self.config_expert.num_cameras + 1):
                            instance_perturbated = cv2.cvtColor(
                                input_data[f"instance_{camera_idx}_perturbated"][1][
                                    :, :, :3
                                ],
                                cv2.COLOR_BGR2RGB,
                            )
                            converted_instance_perturbated = (
                                expert_utils.convert_instance_segmentation(
                                    instance_perturbated
                                )
                            )

                            input_data[f"instance_{camera_idx}_perturbated"] = (
                                instance_perturbated
                            )
                            input_data[
                                f"converted_instance_{camera_idx}_perturbated"
                            ] = converted_instance_perturbated
                            instances_perturbated.append(instance_perturbated)

                        input_data["instance_perturbated"] = np.concatenate(
                            instances_perturbated, axis=1
                        )
                else:
                    self._warn_sensor_save_conflict_once(
                        "instance_sensor_disabled",
                        "Instance sensor is disabled. Skipping instance processing/saving.",
                    )

                if can_process_semantic:
                    # Standard semantics with some details we don't learn but will be useful to enhance the depth map
                    semantics_standard_by_camera = {}
                    for camera_idx in range(1, self.config_expert.num_cameras + 1):
                        semantics_standard = input_data[f"semantics_{camera_idx}"][1][
                            :, :, 2
                        ]
                        input_data[f"semantics_{camera_idx}"] = semantics_standard
                        semantics_standard_by_camera[camera_idx] = semantics_standard
                    input_data["semantics"] = self._build_semantic_panorama(
                        semantics_standard_by_camera
                    )

                    if self.config_expert.perturbate_sensors:
                        semantics_perturbated_standard_by_camera = {}
                        for camera_idx in range(1, self.config_expert.num_cameras + 1):
                            semantics_perturbated_standard = input_data[
                                f"semantics_{camera_idx}_perturbated"
                            ][1][:, :, 2]
                            input_data[f"semantics_{camera_idx}_perturbated"] = (
                                semantics_perturbated_standard
                            )
                            semantics_perturbated_standard_by_camera[camera_idx] = (
                                semantics_perturbated_standard
                            )
                        input_data["semantics_perturbated"] = (
                            self._build_semantic_panorama(
                                semantics_perturbated_standard_by_camera
                            )
                        )
                else:
                    self._warn_sensor_save_conflict_once(
                        "semantic_sensor_disabled",
                        "Semantic sensor is disabled. Skipping semantic processing/saving.",
                    )

                if can_process_depth:
                    # Depth - flexible camera processing
                    depth_cameras = []
                    for camera_idx in range(1, self.config_expert.num_cameras + 1):
                        depth = expert_utils.convert_depth(
                            input_data[f"depth_{camera_idx}"][1][:, :, :3]
                        )
                        depth = expert_utils.enhance_depth(
                            depth,
                            input_data[f"semantics_{camera_idx}"],
                            input_data[f"converted_instance_{camera_idx}"],
                        )
                        input_data[f"depth_{camera_idx}"] = depth
                        depth_cameras.append(depth)
                    input_data["depth"] = np.concatenate(depth_cameras, axis=1)

                    if self.config_expert.perturbate_sensors:
                        depth_perturbated_cameras = []
                        for camera_idx in range(1, self.config_expert.num_cameras + 1):
                            perturbated_depth = expert_utils.convert_depth(
                                input_data[f"depth_{camera_idx}_perturbated"][1][
                                    :, :, :3
                                ]
                            )
                            perturbated_depth = expert_utils.enhance_depth(
                                perturbated_depth,
                                input_data[f"semantics_{camera_idx}_perturbated"],
                                input_data[
                                    f"converted_instance_{camera_idx}_perturbated"
                                ],
                            )
                            input_data[f"depth_{camera_idx}_perturbated"] = (
                                perturbated_depth
                            )
                            depth_perturbated_cameras.append(perturbated_depth)
                        input_data["depth_perturbated"] = np.concatenate(
                            depth_perturbated_cameras, axis=1
                        )

                # Semantics segmentation using first channel of instance segmentation
                # After enhancing the depth map, we use the first channel of instance segmentation which has cleaner labels
                if can_process_instance:
                    semantics_by_camera = {}
                    for camera_idx in range(1, self.config_expert.num_cameras + 1):
                        semantics = input_data[f"converted_instance_{camera_idx}"][
                            ..., 0
                        ]
                        input_data[f"semantics_{camera_idx}"] = semantics
                        semantics_by_camera[camera_idx] = semantics
                    input_data["semantics"] = self._build_semantic_panorama(
                        semantics_by_camera
                    )

                    if self.config_expert.perturbate_sensors:
                        semantics_perturbated_by_camera = {}
                        for camera_idx in range(1, self.config_expert.num_cameras + 1):
                            semantics_perturbated = input_data[
                                f"converted_instance_{camera_idx}_perturbated"
                            ][..., 0]
                            input_data[f"semantics_{camera_idx}_perturbated"] = (
                                semantics_perturbated
                            )
                            semantics_perturbated_by_camera[camera_idx] = (
                                semantics_perturbated
                            )
                        input_data["semantics_perturbated"] = (
                            self._build_semantic_panorama(
                                semantics_perturbated_by_camera
                            )
                        )

                if self._can_compute_camera_pc():
                    camera_pcs = []
                    for camera_idx in range(1, self.config_expert.num_cameras + 1):
                        cam_config = self.config_expert.camera_calibration[camera_idx]
                        camera_pc = expert_utils.semantics_camera_pc(
                            input_data[f"depth_{camera_idx}"],
                            instance=input_data[f"converted_instance_{camera_idx}"],
                            camera_fov=cam_config["fov"],
                            camera_width=cam_config["width"],
                            camera_height=cam_config["height"],
                            camera_pos=cam_config["pos"],
                            camera_rot=cam_config["rot"],
                            perturbation_rotation=0.0,
                            perturbation_translation=0.0,
                            config=self.config_expert,
                        )
                        input_data[f"semantics_camera_pc_{camera_idx}"] = camera_pc
                        camera_pcs.append(camera_pc)

                    if self.config_expert.perturbate_sensors:
                        for camera_idx in range(1, self.config_expert.num_cameras + 1):
                            cam_config = self.config_expert.camera_calibration[
                                camera_idx
                            ]
                            input_data[
                                f"semantics_camera_pc_{camera_idx}_perturbated"
                            ] = expert_utils.semantics_camera_pc(
                                input_data[f"depth_{camera_idx}_perturbated"],
                                instance=input_data[
                                    f"converted_instance_{camera_idx}_perturbated"
                                ],
                                camera_fov=cam_config["fov"],
                                camera_width=cam_config["width"],
                                camera_height=cam_config["height"],
                                camera_pos=cam_config["pos"],
                                camera_rot=cam_config["rot"],
                                perturbation_rotation=self.perturbation_rotation,
                                perturbation_translation=self.perturbation_translation,
                                config=self.config_expert,
                            )
                    input_data["semantics_camera_pc"] = (
                        torch.cat(camera_pcs, dim=0).cpu().numpy()
                    )
                    input_data["semantics_camera_pc_all"] = input_data[
                        "semantics_camera_pc"
                    ]
                else:
                    self._set_empty_camera_pc(input_data)
            else:
                self._set_empty_camera_pc(input_data)

        # Bounding box
        actors_snapshot = self.actors
        self._actors = actors_snapshot
        input_data["bounding_boxes"] = self.get_bounding_boxes(input_data=input_data)
        self.stored_bounding_boxes_of_this_step = input_data["bounding_boxes"]
        self.id2bb_map = {bb["id"]: bb for bb in input_data["bounding_boxes"]}
        self.id2actor_map = {
            actor.id: actor
            for actor in actors_snapshot
            if actor.is_alive and actor.id in self.id2bb_map
        }
        # BEV Semantic
        self.stop_sign_criteria.tick(self.ego_vehicle)
        bev_data = self.ss_bev_manager.get_observation(self.close_traffic_lights)
        # применяем маску
        input_data["hdmap"] = self._apply_bev_mask(bev_data["hdmap_classes"])

        if self.config_expert.perturbate_sensors:
            bev_data_pert = self.ss_bev_manager_perturbated.get_observation(
                self.close_traffic_lights
            )
            input_data["hdmap_perturbated"] = self._apply_bev_mask(
                bev_data_pert["hdmap_classes"]
            )

        can_enhance_semantics = (
            process_sensor_payload_this_step
            and self.config_expert.compute_camera_pc
            and "converted_instance_1" in input_data
        )
        if can_enhance_semantics:
            # --- Update semantic segmentation to make cones, traffic warning and special vehicles labels ---
            construction_meshes_id_map = (
                expert_utils.match_unreal_engine_ids_to_carla_bounding_boxes_ids(
                    self.ego_matrix,
                    CarlaSemanticSegmentationClass.Dynamic,
                    [
                        box
                        for box in input_data["bounding_boxes"]
                        if box.get("type_id") in constants.CONSTRUCTION_MESHES
                    ],
                    input_data["semantics_camera_pc_all"],
                )
            )
            emergency_meshes_id_map = (
                expert_utils.match_unreal_engine_ids_to_carla_bounding_boxes_ids(
                    self.ego_matrix,
                    CarlaSemanticSegmentationClass.Car,
                    [
                        box
                        for box in input_data["bounding_boxes"]
                        if box.get("type_id") in constants.EMERGENCY_MESHES
                    ],
                    input_data["semantics_camera_pc_all"],
                    penalize_points_outside=True,
                )
            )
            emergency_meshes_id_map.update(
                expert_utils.match_unreal_engine_ids_to_carla_bounding_boxes_ids(
                    self.ego_matrix,
                    CarlaSemanticSegmentationClass.Truck,
                    [
                        box
                        for box in input_data["bounding_boxes"]
                        if box.get("type_id") in constants.EMERGENCY_MESHES
                    ],
                    input_data["semantics_camera_pc_all"],
                )
            )
            stop_sign_meshes_id_map = (
                expert_utils.match_unreal_engine_ids_to_carla_actors_ids(
                    self.ego_matrix,
                    CarlaSemanticSegmentationClass.TrafficSign,
                    self.get_nearby_object(
                        actors_snapshot.filter("*traffic.stop*"),
                        self.config_expert.light_radius,
                    ),
                    input_data["semantics_camera_pc_all"],
                )
            )
        else:
            construction_meshes_id_map = {}
            emergency_meshes_id_map = {}
            stop_sign_meshes_id_map = {}

        if (
            len(construction_meshes_id_map) > 0
            or len(emergency_meshes_id_map) > 0
            or len(stop_sign_meshes_id_map) > 0
        ):
            for camera_idx in range(1, self.config_expert.num_cameras + 1):
                input_data[f"semantics_{camera_idx}"] = (
                    expert_utils.enhance_semantics_segmentation(
                        input_data[f"converted_instance_{camera_idx}"],
                        input_data.get(f"semantics_{camera_idx}"),
                        construction_meshes_id_map,
                        CarlaSemanticSegmentationClass.ConeAndTrafficWarning,
                    )
                )
                input_data[f"semantics_{camera_idx}"] = (
                    expert_utils.enhance_semantics_segmentation(
                        input_data[f"converted_instance_{camera_idx}"],
                        input_data.get(f"semantics_{camera_idx}"),
                        emergency_meshes_id_map,
                        CarlaSemanticSegmentationClass.SpecialVehicles,
                    )
                )
                input_data[f"semantics_{camera_idx}"] = (
                    expert_utils.enhance_semantics_segmentation(
                        input_data[f"converted_instance_{camera_idx}"],
                        input_data.get(f"semantics_{camera_idx}"),
                        stop_sign_meshes_id_map,
                        CarlaSemanticSegmentationClass.StopSign,
                    )
                )

            semantics_by_camera = {
                camera_idx: input_data[f"semantics_{camera_idx}"]
                for camera_idx in range(1, self.config_expert.num_cameras + 1)
            }
            input_data["semantics"] = self._build_semantic_panorama(semantics_by_camera)

            if self.config_expert.perturbate_sensors:
                for camera_idx in range(1, self.config_expert.num_cameras + 1):
                    input_data[f"semantics_{camera_idx}_perturbated"] = (
                        expert_utils.enhance_semantics_segmentation(
                            input_data[f"converted_instance_{camera_idx}_perturbated"],
                            input_data.get(f"semantics_{camera_idx}_perturbated"),
                            construction_meshes_id_map,
                            CarlaSemanticSegmentationClass.ConeAndTrafficWarning,
                        )
                    )
                    input_data[f"semantics_{camera_idx}_perturbated"] = (
                        expert_utils.enhance_semantics_segmentation(
                            input_data[f"converted_instance_{camera_idx}_perturbated"],
                            input_data.get(f"semantics_{camera_idx}_perturbated"),
                            emergency_meshes_id_map,
                            CarlaSemanticSegmentationClass.SpecialVehicles,
                        )
                    )
                    input_data[f"semantics_{camera_idx}_perturbated"] = (
                        expert_utils.enhance_semantics_segmentation(
                            input_data[f"converted_instance_{camera_idx}_perturbated"],
                            input_data.get(f"semantics_{camera_idx}_perturbated"),
                            stop_sign_meshes_id_map,
                            CarlaSemanticSegmentationClass.StopSign,
                        )
                    )

                semantics_perturbated_by_camera = {
                    camera_idx: input_data[f"semantics_{camera_idx}_perturbated"]
                    for camera_idx in range(1, self.config_expert.num_cameras + 1)
                }
                input_data["semantics_perturbated"] = self._build_semantic_panorama(
                    semantics_perturbated_by_camera
                )

        self.tick_data = input_data

        return input_data

    @beartype
    def encode_depth(self, depth: np.ndarray) -> np.ndarray:
        if self.config_expert.save_depth_bits == 8:
            return common_utils.encode_depth_8bit(depth)
        return common_utils.encode_depth_16bit(depth)

    def _apply_bev_mask(self, bev_image):
        """
        Applies an inverted FOV mask to BEV map:
        keeps only the sector inside self.target_fov,
        masks everything outside it.

        Mask is cached after first creation.
        """
        if bev_image.dtype != np.uint8:
            bev_image = bev_image.astype(np.uint8)

        h, w = bev_image.shape[:2]
        bev_manager = getattr(self, "ss_bev_manager", None)
        if bev_manager is not None and hasattr(bev_manager, "_pixels_ev_to_bottom"):
            # ObsManager rotates BEV by 90 degrees clockwise.
            # After this rotation ego center becomes (pixels_ev_to_bottom, height / 2).
            ego_x = int(round(float(bev_manager._pixels_ev_to_bottom)))
            ego_y = h // 2
        else:
            ego_x = w // 2
            ego_y = h // 2

        ego_x = int(np.clip(ego_x, 0, w - 1))
        ego_y = int(np.clip(ego_y, 0, h - 1))

        need_rebuild = (
            not hasattr(self, "_bev_mask")
            or self._bev_mask.shape != (h, w)
            or not hasattr(self, "_bev_mask_fov")
            or self._bev_mask_fov != self.target_fov
            or not hasattr(self, "_bev_mask_center")
            or self._bev_mask_center != (ego_x, ego_y)
        )

        if need_rebuild:
            mask = np.zeros((h, w), dtype=np.uint8)
            radius = int(np.ceil(np.hypot(h, w)))
            half_fov = self.target_fov / 2.0

            # In rotated BEV forward direction points to the right (+x).
            center_angle = 0.0

            start_angle = center_angle - half_fov
            end_angle = center_angle + half_fov

            angles = np.deg2rad(np.linspace(start_angle, end_angle, num=300))
            points = [(ego_x, ego_y)]

            for angle in angles:
                x = int(round(ego_x + radius * np.cos(angle)))
                y = int(round(ego_y + radius * np.sin(angle)))
                points.append((x, y))

            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)

            self._bev_mask = mask
            self._bev_mask_fov = self.target_fov
            self._bev_mask_center = (ego_x, ego_y)

        return cv2.bitwise_and(bev_image, bev_image, mask=self._bev_mask)

    def _should_save_camera_panorama(self) -> bool:
        return bool(self.config_expert.SAVE_CAM_IMAGES_AS_PANORAMA)

    def _should_save_camera_subfolders(self) -> bool:
        return bool(self.config_expert.SAVE_CAM_IMAGES_IN_SUBFOLDERS)

    def _use_png_for_jpeg_images(self) -> bool:
        return bool(self.config_expert.save_jpeg_images_as_png)

    def _rgb_image_extension(self) -> str:
        return "png" if self._use_png_for_jpeg_images() else "jpg"

    def _rgb_image_write_params(
        self, jpeg_quality: int, png_params: list[int]
    ) -> list[int]:
        if self._use_png_for_jpeg_images():
            return png_params
        return [
            int(cv2.IMWRITE_JPEG_QUALITY),
            jpeg_quality if self.config_expert.compress_images else 100,
        ]

    def _should_process_semantic_panorama(self) -> bool:
        if not bool(self.config_expert.enable_semantic_panorama_stitching):
            return False
        # Panorama processing should be disabled when camera panorama/subfolder
        # pipelines are fully disabled.
        return (
            self._should_save_camera_panorama() or self._should_save_camera_subfolders()
        )

    def _semantic_save_modes(self) -> tuple[bool, bool]:
        """Return (save_panorama, save_subfolders) for semantic outputs.

        Rules:
        - Default follows global camera save modes.
                - If semantic panorama stitching is enabled, semantic subfolder saving
                    is always disabled.
        """
        save_panorama = self._should_save_camera_panorama()
        save_subfolders = self._should_save_camera_subfolders()
        semantic_stitching = bool(self.config_expert.enable_semantic_panorama_stitching)

        save_semantic_panorama = bool(save_panorama)
        save_semantic_subfolders = bool(save_subfolders)

        if semantic_stitching:
            # Keep semantic panorama enabled whenever at least one camera save mode is on.
            save_semantic_panorama = bool(save_panorama or save_subfolders)
            save_semantic_subfolders = False

        return save_semantic_panorama, save_semantic_subfolders

    def _warn_if_camera_image_saving_disabled(self) -> None:
        if (
            not self._should_save_camera_panorama()
            and not self._should_save_camera_subfolders()
            and not self._camera_image_saving_disabled_warning_logged
        ):
            LOG.warning(
                "Both SAVE_CAM_IMAGES_AS_PANORAMA and "
                "SAVE_CAM_IMAGES_IN_SUBFOLDERS are False. "
                "Skipping camera image saving for rgb/semantics/depth/instance."
            )
            self._camera_image_saving_disabled_warning_logged = True

    def _camera_frame_path(
        self, modality: str, frame: int, extension: str, camera_idx: int | None = None
    ) -> pathlib.Path:
        frame_name = f"{frame:04}.{extension}"
        if camera_idx is None:
            return self.save_path / modality / frame_name
        return self.save_path / modality / f"cam{camera_idx}" / frame_name

    def _warn_sensor_save_conflict_once(self, key: str, message: str) -> None:
        if key in self._sensor_save_conflict_warnings:
            return
        LOG.warning(message)
        self._sensor_save_conflict_warnings.add(key)

    def _can_process_instance(self) -> bool:
        return bool(self.config_expert.enable_instance_sensor)

    def _can_process_semantic(self) -> bool:
        return bool(self.config_expert.enable_semantic_sensor)

    def _can_process_depth(self) -> bool:
        if not self.config_expert.enable_depth_sensor:
            return False
        if not self._can_process_instance() or not self._can_process_semantic():
            self._warn_sensor_save_conflict_once(
                "depth_dependencies",
                "Depth processing requires instance+semantic sensors. "
                "Skipping depth processing because one dependency is disabled.",
            )
            return False
        return True

    def _can_compute_camera_pc(self) -> bool:
        if not self.config_expert.compute_camera_pc:
            return False
        if not self._can_process_depth() or not self._can_process_instance():
            self._warn_sensor_save_conflict_once(
                "camera_pc_dependencies",
                "Camera point cloud computation requires depth+instance sensors. "
                "Falling back to empty camera point clouds.",
            )
            return False
        return True

    def _init_semantic_panorama_stitcher(self) -> SemanticPanoramaStitcher | None:
        if not self._should_process_semantic_panorama():
            return None

        if self.config_expert.num_cameras < 2:
            LOG.warning(
                "Semantic panorama stitching is enabled but num_cameras=%d. "
                "Falling back to horizontal concatenation.",
                self.config_expert.num_cameras,
            )
            return None

        try:
            return SemanticPanoramaStitcher(self.config_expert)
        except Exception as e:
            LOG.warning(
                "Failed to initialize SemanticPanoramaStitcher (%s). "
                "Falling back to horizontal concatenation.",
                e,
            )
            return None

    def _build_semantic_panorama(
        self, semantics_by_camera: dict[int, np.ndarray]
    ) -> np.ndarray:
        if self.semantic_panorama_stitcher is None:
            return np.concatenate(
                [
                    semantics_by_camera[idx]
                    for idx in sorted(semantics_by_camera.keys())
                ],
                axis=1,
            )
        return self.semantic_panorama_stitcher.stitch(semantics_by_camera)

    @beartype
    def _prepare_semantics_for_saving(self, semantics: np.ndarray) -> np.ndarray:
        if self.config_expert.save_grouped_semantic:
            return self.semantics_converter[semantics]
        return semantics

    @beartype
    def _prepare_depth_for_saving(self, depth: np.ndarray) -> np.ndarray:
        if self.config_expert.save_depth_lower_resolution:
            depth = cv2.resize(
                depth,
                (
                    depth.shape[1] // self.config_expert.save_depth_resolution_ratio,
                    depth.shape[0] // self.config_expert.save_depth_resolution_ratio,
                ),
                interpolation=cv2.INTER_AREA,
            )
        return self.encode_depth(depth)

    @beartype
    def _build_save_payload(self, tick_data: dict) -> dict:
        payload = {}
        save_panorama = self._should_save_camera_panorama()
        save_subfolders = self._should_save_camera_subfolders()
        save_semantic_panorama, save_semantic_subfolders = self._semantic_save_modes()

        if save_panorama:
            if "rgb" in tick_data:
                payload["rgb"] = tick_data["rgb"]
            if self.config_expert.perturbate_sensors and "rgb_perturbated" in tick_data:
                payload["rgb_perturbated"] = tick_data["rgb_perturbated"]

        if save_subfolders:
            for camera_idx in range(1, self.config_expert.num_cameras + 1):
                rgb_key = f"rgb_{camera_idx}"
                if rgb_key in tick_data:
                    payload[rgb_key] = tick_data[rgb_key]
                if self.config_expert.perturbate_sensors:
                    rgb_perturbated_key = f"rgb_{camera_idx}_perturbated"
                    if rgb_perturbated_key in tick_data:
                        payload[rgb_perturbated_key] = tick_data[rgb_perturbated_key]

        if self.config_expert.save_camera_pc and self.config_expert.compute_camera_pc:
            if "semantics_camera_pc" in tick_data:
                payload["semantics_camera_pc"] = tick_data["semantics_camera_pc"]

        if (
            self.config_expert.enable_semantic_sensor
            and self.config_expert.save_semantic_segmentation
        ):
            if save_semantic_panorama and "semantics" in tick_data:
                payload["semantics"] = tick_data["semantics"]
            if (
                save_semantic_panorama
                and self.config_expert.perturbate_sensors
                and "semantics_perturbated" in tick_data
            ):
                payload["semantics_perturbated"] = tick_data["semantics_perturbated"]
            if save_semantic_subfolders:
                for camera_idx in range(1, self.config_expert.num_cameras + 1):
                    semantics_key = f"semantics_{camera_idx}"
                    if semantics_key in tick_data:
                        payload[semantics_key] = tick_data[semantics_key]
                    if self.config_expert.perturbate_sensors:
                        semantics_perturbated_key = (
                            f"semantics_{camera_idx}_perturbated"
                        )
                        if semantics_perturbated_key in tick_data:
                            payload[semantics_perturbated_key] = tick_data[
                                semantics_perturbated_key
                            ]

        if self.config_expert.enable_depth_sensor and self.config_expert.save_depth:
            if "depth" in tick_data:
                payload["depth"] = tick_data["depth"]
            if (
                self.config_expert.perturbate_sensors
                and "depth_perturbated" in tick_data
            ):
                payload["depth_perturbated"] = tick_data["depth_perturbated"]
            if save_subfolders:
                for camera_idx in range(1, self.config_expert.num_cameras + 1):
                    depth_key = f"depth_{camera_idx}"
                    if depth_key in tick_data:
                        payload[depth_key] = tick_data[depth_key]
                    if self.config_expert.perturbate_sensors:
                        depth_perturbated_key = f"depth_{camera_idx}_perturbated"
                        if depth_perturbated_key in tick_data:
                            payload[depth_perturbated_key] = tick_data[
                                depth_perturbated_key
                            ]

        if (
            self.config_expert.enable_instance_sensor
            and self.config_expert.save_instance_segmentation
        ):
            if "instance" in tick_data:
                payload["instance"] = tick_data["instance"]
            if (
                self.config_expert.perturbate_sensors
                and "instance_perturbated" in tick_data
            ):
                payload["instance_perturbated"] = tick_data["instance_perturbated"]
            if save_subfolders:
                for camera_idx in range(1, self.config_expert.num_cameras + 1):
                    instance_key = f"instance_{camera_idx}"
                    if instance_key in tick_data:
                        payload[instance_key] = tick_data[instance_key]
                    if self.config_expert.perturbate_sensors:
                        instance_perturbated_key = f"instance_{camera_idx}_perturbated"
                        if instance_perturbated_key in tick_data:
                            payload[instance_perturbated_key] = tick_data[
                                instance_perturbated_key
                            ]

        if "hdmap" in tick_data:
            payload["hdmap"] = tick_data["hdmap"]
        if self.config_expert.perturbate_sensors and "hdmap_perturbated" in tick_data:
            payload["hdmap_perturbated"] = tick_data["hdmap_perturbated"]

        if self.config_expert.use_radars:
            for sensor_idx in range(1, self.config_expert.num_radar_sensors + 1):
                radar_key = f"radar{sensor_idx}"
                if radar_key in tick_data:
                    payload[radar_key] = tick_data[radar_key]
                if self.config_expert.perturbate_sensors:
                    radar_perturbated_key = f"radar{sensor_idx}_perturbated"
                    if radar_perturbated_key in tick_data:
                        payload[radar_perturbated_key] = tick_data[
                            radar_perturbated_key
                        ]

        return payload

    def _queue_image_write(
        self,
        tasks: list[tuple[str, np.ndarray, list[int]]],
        path: pathlib.Path,
        image: np.ndarray,
        params: list[int],
    ) -> None:
        tasks.append((str(path), image, params))

    def _flush_image_write_tasks(
        self, image_write_tasks: list[tuple[str, np.ndarray, list[int]]]
    ) -> None:
        if len(image_write_tasks) == 0:
            return

        if self._image_write_executor is None:
            for path, image, params in image_write_tasks:
                cv2.imwrite(path, image, params)
            return

        futures = [
            self._image_write_executor.submit(cv2.imwrite, path, image, params)
            for path, image, params in image_write_tasks
        ]
        for future in futures:
            try:
                _ = future.result()
            except Exception as e:
                LOG.error(f"Image write task failed: {e}")

    def _stage_saved_meta(self, frame: int, data: dict) -> None:
        if self._meta_stage_dir is None:
            return
        stage_path = self._meta_stage_dir / f"{frame:04}.pkl"
        with open(stage_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._staged_meta_paths[frame] = stage_path

    def _load_staged_meta(self, frame: int, fallback: dict) -> dict:
        stage_path = self._staged_meta_paths.get(frame)
        if stage_path is None:
            return dict(fallback)
        try:
            with open(stage_path, "rb") as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            LOG.warning(f"Failed to load staged meta for frame {frame:04}: {e}")
            return dict(fallback)

    def _stage_saved_bounding_boxes(self, frame: int, boxes: list[dict]) -> None:
        if self._bboxes_stage_dir is None:
            return
        stage_path = self._bboxes_stage_dir / f"{frame:04}.pkl"
        with open(stage_path, "wb") as f:
            pickle.dump(boxes, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._staged_bboxes_paths[frame] = stage_path

    def _load_staged_bounding_boxes(
        self, frame: int, fallback: list[dict]
    ) -> list[dict]:
        stage_path = self._staged_bboxes_paths.get(frame)
        if stage_path is None:
            return fallback
        try:
            with open(stage_path, "rb") as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            LOG.warning(f"Failed to load staged bboxes for frame {frame:04}: {e}")
            return fallback

    @beartype
    def save_sensors(self, tick_data: dict, frame: int | None = None) -> None:
        if frame is None:
            frame, _, _ = self._resolve_sync_state_from_input_data(tick_data)

        if self.config_expert.eval_expert:
            # Just save RGB synchronously for eval
            jpeg_quality = (
                self.jpeg_storage_quality if self.config_expert.compress_images else 100
            )
            png_params = [
                int(cv2.IMWRITE_PNG_COMPRESSION),
                self.config_expert.png_storage_compression_level
                if self.config_expert.compress_images
                else 0,
            ]
            rgb_extension = self._rgb_image_extension()
            rgb_params = self._rgb_image_write_params(jpeg_quality, png_params)
            save_panorama = self._should_save_camera_panorama()
            save_subfolders = self._should_save_camera_subfolders()
            self._warn_if_camera_image_saving_disabled()

            if save_panorama:
                cv2.imwrite(
                    str(self._camera_frame_path("rgb", frame, rgb_extension)),
                    tick_data["rgb"],
                    rgb_params,
                )
            if save_subfolders:
                for camera_idx in range(1, self.config_expert.num_cameras + 1):
                    cv2.imwrite(
                        str(
                            self._camera_frame_path(
                                "rgb", frame, rgb_extension, camera_idx
                            )
                        ),
                        tick_data[f"rgb_{camera_idx}"],
                        rgb_params,
                    )
            LOG.info("Evaluation mode: not saving more sensor data.")
            return

        # Queue data for background thread.
        # Keep only keys used by saving code path to avoid retaining large per-tick state in queue.
        lidar_points = self.accumulate_lidar()
        if (not self.config_expert.use_lidar) and lidar_points.shape[0] == 0:
            lidar_points = None
        payload = self._build_save_payload(tick_data)
        save_data = {
            "frame": frame,
            "tick_data": payload,
            "jpeg_quality": self.jpeg_storage_quality,
            "lidar_points": lidar_points,
        }

        queue_size_before_put = self._save_queue.qsize()
        put_timeout = self.config_expert.save_queue_put_timeout_sec
        if queue_size_before_put >= self._save_queue_high_watermark:
            put_timeout = max(put_timeout, 0.2)

        # Never drop frames. If queue is congested, fallback to synchronous save.
        try:
            self._save_queue.put(
                save_data,
                timeout=put_timeout,
            )
        except queue.Full:
            LOG.warning(
                "Save queue full after %.3fs; falling back to synchronous save for this frame",
                put_timeout,
            )
            try:
                self._save_sensors_sync(save_data)
            except Exception as e:
                LOG.error(
                    f"Error while saving sensors synchronously on queue fallback: {e}"
                )

        # Log queue depth occasionally for monitoring
        if frame % 10 == 0:
            queue_size = self._save_queue.qsize()
            if queue_size >= self._save_queue_high_watermark:
                LOG.warning(
                    "Save queue depth: %d/%d - disk may be falling behind",
                    queue_size,
                    self._save_queue_maxsize,
                )

    def _save_worker(self):
        """Background thread worker for saving sensor data."""
        while True:
            try:
                save_data = self._save_queue.get(timeout=0.2)
            except queue.Empty:
                if self._save_thread_stop.is_set() and self._save_queue.empty():
                    return
                continue
            try:
                if save_data is self._save_queue_sentinel:
                    return
                self._save_sensors_sync(save_data)
            except Exception as e:
                LOG.error(f"Error in save worker thread: {e}")
            finally:
                self._save_queue.task_done()

    def _save_sensors_sync(self, save_data: dict) -> None:
        """Synchronous sensor saving - runs in background thread."""
        if save_data.get("kind") == "third_person":
            params = save_data.get("params")
            if params is None:
                cv2.imwrite(save_data["path"], save_data["image"])
            else:
                cv2.imwrite(save_data["path"], save_data["image"], params)
            return

        frame = save_data["frame"]
        tick_data = save_data["tick_data"]
        jpeg_quality = save_data["jpeg_quality"]
        points = save_data["lidar_points"]
        png_params = [
            int(cv2.IMWRITE_PNG_COMPRESSION),
            self.config_expert.png_storage_compression_level
            if self.config_expert.compress_images
            else 0,
        ]
        rgb_extension = self._rgb_image_extension()
        rgb_params = self._rgb_image_write_params(jpeg_quality, png_params)
        save_panorama = self._should_save_camera_panorama()
        save_subfolders = self._should_save_camera_subfolders()
        save_semantic_panorama, save_semantic_subfolders = self._semantic_save_modes()
        self._warn_if_camera_image_saving_disabled()
        image_write_tasks: list[tuple[str, np.ndarray, list[int]]] = []

        # CARLA images are already in opencv's BGR format.
        if save_panorama:
            self._queue_image_write(
                image_write_tasks,
                self._camera_frame_path("rgb", frame, rgb_extension),
                tick_data["rgb"],
                rgb_params,
            )
            if self.config_expert.perturbate_sensors:
                self._queue_image_write(
                    image_write_tasks,
                    self._camera_frame_path("rgb_perturbated", frame, rgb_extension),
                    tick_data["rgb_perturbated"],
                    rgb_params,
                )
        if save_subfolders:
            for camera_idx in range(1, self.config_expert.num_cameras + 1):
                self._queue_image_write(
                    image_write_tasks,
                    self._camera_frame_path("rgb", frame, rgb_extension, camera_idx),
                    tick_data[f"rgb_{camera_idx}"],
                    rgb_params,
                )
                if self.config_expert.perturbate_sensors:
                    self._queue_image_write(
                        image_write_tasks,
                        self._camera_frame_path(
                            "rgb_perturbated", frame, rgb_extension, camera_idx
                        ),
                        tick_data[f"rgb_{camera_idx}_perturbated"],
                        rgb_params,
                    )

        # Store camera point clouds
        if self.config_expert.save_camera_pc and self.config_expert.compute_camera_pc:
            np.savez_compressed(
                str(self.save_path / "camera_pc" / (f"{frame:04}.npz")),
                tick_data["semantics_camera_pc"],
            )

        save_semantic = (
            self.config_expert.enable_semantic_sensor
            and self.config_expert.save_semantic_segmentation
        )
        if self.config_expert.save_semantic_segmentation and (
            not self.config_expert.enable_semantic_sensor
        ):
            self._warn_sensor_save_conflict_once(
                "semantic_save_without_sensor",
                "save_semantic_files=True but enable_semantic_sensor=False. "
                "Skipping semantic file saving.",
            )

        if save_semantic and save_semantic_panorama and ("semantics" not in tick_data):
            self._warn_sensor_save_conflict_once(
                "semantic_missing_payload",
                "Semantic saving is enabled but semantic payload is missing. "
                "Skipping semantic file saving for this run.",
            )
            save_semantic = False

        if save_semantic and save_semantic_panorama:
            self._queue_image_write(
                image_write_tasks,
                self._camera_frame_path("semantics", frame, "png"),
                self._prepare_semantics_for_saving(tick_data["semantics"]),
                png_params,
            )
            if self.config_expert.perturbate_sensors:
                self._queue_image_write(
                    image_write_tasks,
                    self._camera_frame_path("semantics_perturbated", frame, "png"),
                    self._prepare_semantics_for_saving(
                        tick_data["semantics_perturbated"]
                    ),
                    png_params,
                )
        if save_semantic and save_semantic_subfolders:
            for camera_idx in range(1, self.config_expert.num_cameras + 1):
                self._queue_image_write(
                    image_write_tasks,
                    self._camera_frame_path("semantics", frame, "png", camera_idx),
                    self._prepare_semantics_for_saving(
                        tick_data[f"semantics_{camera_idx}"]
                    ),
                    png_params,
                )
                if self.config_expert.perturbate_sensors:
                    self._queue_image_write(
                        image_write_tasks,
                        self._camera_frame_path(
                            "semantics_perturbated", frame, "png", camera_idx
                        ),
                        self._prepare_semantics_for_saving(
                            tick_data[f"semantics_{camera_idx}_perturbated"]
                        ),
                        png_params,
                    )

        save_depth = (
            self.config_expert.enable_depth_sensor and self.config_expert.save_depth
        )
        if self.config_expert.save_depth and (
            not self.config_expert.enable_depth_sensor
        ):
            self._warn_sensor_save_conflict_once(
                "depth_save_without_sensor",
                "save_depth_files=True but enable_depth_sensor=False. "
                "Skipping depth file saving.",
            )
        if save_depth and ("depth" not in tick_data):
            self._warn_sensor_save_conflict_once(
                "depth_missing_payload",
                "Depth saving is enabled but depth payload is missing. "
                "Skipping depth file saving for this run.",
            )
            save_depth = False

        if save_depth:
            if save_panorama:
                self._queue_image_write(
                    image_write_tasks,
                    self._camera_frame_path("depth", frame, "png"),
                    self._prepare_depth_for_saving(tick_data["depth"]),
                    png_params,
                )
                if self.config_expert.perturbate_sensors:
                    self._queue_image_write(
                        image_write_tasks,
                        self._camera_frame_path("depth_perturbated", frame, "png"),
                        self._prepare_depth_for_saving(tick_data["depth_perturbated"]),
                        png_params,
                    )
            if save_subfolders:
                for camera_idx in range(1, self.config_expert.num_cameras + 1):
                    self._queue_image_write(
                        image_write_tasks,
                        self._camera_frame_path("depth", frame, "png", camera_idx),
                        self._prepare_depth_for_saving(
                            tick_data[f"depth_{camera_idx}"]
                        ),
                        png_params,
                    )
                    if self.config_expert.perturbate_sensors:
                        self._queue_image_write(
                            image_write_tasks,
                            self._camera_frame_path(
                                "depth_perturbated", frame, "png", camera_idx
                            ),
                            self._prepare_depth_for_saving(
                                tick_data[f"depth_{camera_idx}_perturbated"]
                            ),
                            png_params,
                        )

        save_instance = (
            self.config_expert.enable_instance_sensor
            and self.config_expert.save_instance_segmentation
        )
        if self.config_expert.save_instance_segmentation and (
            not self.config_expert.enable_instance_sensor
        ):
            self._warn_sensor_save_conflict_once(
                "instance_save_without_sensor",
                "save_instance_files=True but enable_instance_sensor=False. "
                "Skipping instance file saving.",
            )
        if save_instance and ("instance" not in tick_data):
            self._warn_sensor_save_conflict_once(
                "instance_missing_payload",
                "Instance saving is enabled but instance payload is missing. "
                "Skipping instance file saving for this run.",
            )
            save_instance = False

        if save_instance:
            if save_panorama:
                self._queue_image_write(
                    image_write_tasks,
                    self._camera_frame_path("instance", frame, "png"),
                    tick_data["instance"],
                    png_params,
                )
                if self.config_expert.perturbate_sensors:
                    self._queue_image_write(
                        image_write_tasks,
                        self._camera_frame_path("instance_perturbated", frame, "png"),
                        tick_data["instance_perturbated"],
                        png_params,
                    )
            if save_subfolders:
                for camera_idx in range(1, self.config_expert.num_cameras + 1):
                    self._queue_image_write(
                        image_write_tasks,
                        self._camera_frame_path("instance", frame, "png", camera_idx),
                        tick_data[f"instance_{camera_idx}"],
                        png_params,
                    )
                    if self.config_expert.perturbate_sensors:
                        self._queue_image_write(
                            image_write_tasks,
                            self._camera_frame_path(
                                "instance_perturbated",
                                frame,
                                "png",
                                camera_idx,
                            ),
                            tick_data[f"instance_{camera_idx}_perturbated"],
                            png_params,
                        )

        self._queue_image_write(
            image_write_tasks,
            self.save_path / "hdmap" / (f"{frame:04}.png"),
            tick_data["hdmap"].astype(np.uint8),
            png_params,
        )
        if self.config_expert.perturbate_sensors:
            self._queue_image_write(
                image_write_tasks,
                self.save_path / "hdmap_perturbated" / (f"{frame:04}.png"),
                tick_data["hdmap_perturbated"].astype(np.uint8),
                png_params,
            )
        self._flush_image_write_tasks(image_write_tasks)

        if self.config_expert.use_radars:
            # Prepare radar data for saving dynamically
            radar_save_dict = {}
            for i in range(1, self.config_expert.num_radar_sensors + 1):
                radar_save_dict[f"radar{i}"] = tick_data[f"radar{i}"].astype(np.float16)

            np.savez_compressed(
                self.save_path / "radar" / (f"{frame:04}.npz"), **radar_save_dict
            )
            if self.config_expert.perturbate_sensors:
                # Prepare perturbated radar data for saving dynamically
                radar_perturbated_save_dict = {}
                for i in range(1, self.config_expert.num_radar_sensors + 1):
                    radar_perturbated_save_dict[f"radar{i}"] = tick_data[
                        f"radar{i}_perturbated"
                    ].astype(np.float16)

                np.savez_compressed(
                    self.save_path / "radar_perturbated" / (f"{frame:04}.npz"),
                    **radar_perturbated_save_dict,
                )

        if self.config_expert.use_lidar and points is not None:
            # Specialized LiDAR compression format (using pre-computed points)
            header = laspy.LasHeader(point_format=self.config_expert.point_format)
            if points.shape[0] > 0:
                header.offsets = np.min(points, axis=0)[:3]
            else:
                header.offsets = np.zeros(3, dtype=np.float64)
            header.scales = np.array(
                [
                    self.config_expert.point_precision_x,
                    self.config_expert.point_precision_y,
                    self.config_expert.point_precision_z,
                ]
            )
            # Add extra dimension for time
            header.add_extra_dim(laspy.ExtraBytesParams(name="time", type=np.uint8))

            with laspy.open(
                self.save_path / "lidar" / (f"{frame:04}.laz"), mode="w", header=header
            ) as writer:
                point_record = laspy.ScaleAwarePointRecord.zeros(
                    points.shape[0], header=header
                )
                if points.shape[0] > 0:
                    point_record.x = points[:, 0]
                    point_record.y = points[:, 1]
                    point_record.z = points[:, 2]
                    point_record["time"] = points[:, 3].astype(np.uint8)
                writer.write_points(point_record)

    @beartype
    def destroy(self, results: RouteRecord = None) -> None:
        """
        Save the collected data and statistics to files, and clean up the data structures.
        This method should be called at the end of the data collection process.

        Args:
            results: Any additional results to be processed or saved.
        """
        self._save_thread_stop.set()

        if hasattr(self, "_3rd_person_camera"):
            try:
                self._3rd_person_camera.stop()
                self._3rd_person_camera.destroy()
            except Exception as e:
                LOG.warning(f"Failed to cleanly destroy 3rd person camera: {e}")

        # Clean shutdown of background save thread
        if hasattr(self, "_save_thread") and self._save_thread is not None:
            LOG.info("Shutting down background save thread...")
            sentinel_enqueued = False
            sentinel_deadline = time.monotonic() + 5.0
            while not sentinel_enqueued and time.monotonic() < sentinel_deadline:
                try:
                    self._save_queue.put(self._save_queue_sentinel, timeout=0.1)
                    sentinel_enqueued = True
                except queue.Full:
                    continue

            if not sentinel_enqueued:
                LOG.warning("Could not enqueue save queue sentinel before timeout")

            queue_deadline = time.monotonic() + 20.0
            while (
                getattr(self._save_queue, "unfinished_tasks", 0) > 0
                and time.monotonic() < queue_deadline
            ):
                time.sleep(0.1)

            unfinished = getattr(self._save_queue, "unfinished_tasks", 0)
            if unfinished > 0:
                LOG.warning(
                    "Save queue still has %d unfinished tasks during shutdown",
                    unfinished,
                )

            self._save_thread.join(timeout=10)
            if self._save_thread.is_alive():
                LOG.warning("Save thread did not shutdown cleanly within timeout")

        if self._image_write_executor is not None:
            self._image_write_executor.shutdown(wait=True)
            self._image_write_executor = None

        if not self.config_expert.eval_expert and (
            not self.config_expert.py123d_data_format
            or self.config_expert.save_legacy_outputs_with_py123d
        ):
            self._offline_process_data()

        if results is not None and self.save_path is not None:
            with open(
                os.path.join(self.save_path, "results.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(results.__dict__, f, indent=2)

        if self._history_cache_dir is not None:
            for _path in self._staged_meta_paths.values():
                try:
                    _path.unlink(missing_ok=True)
                except Exception:
                    pass
            for _path in self._staged_bboxes_paths.values():
                try:
                    _path.unlink(missing_ok=True)
                except Exception:
                    pass

    def _offline_process_data(self) -> None:
        """
        Offline process the collected data for additional annotations or processing.
        This method is called after the data collection is completed.
        """

        # Re-save metas with privileged information for data filtering later
        # This step is necessary so we can obtain higher qualitative data
        # If any logic is changed, this step should be kept an eye on.
        if (
            (self.save_path is not None)
            and self.config_expert.datagen
            and not self.config_expert.eval_expert
        ):
            metas_dir = self.save_path / "metas"
            delta_t = 1.0 / self.config_expert.fps
            N = len(self.metas)

            # Enhance metas
            for i in range(N):
                step, frame, state_now = self.metas[i]

                # --- Privileged acceleration and angular velocity, mostly for data filtering ---
                if (
                    step % self.config_expert.data_save_freq != 0
                ):  # Very important. Only override data that was saved.
                    continue

                data = self._load_staged_meta(frame, state_now)
                speed_now = state_now.get("speed", 0.0)
                yaw_now = state_now.get("theta", 0.0)

                if i < N - 1:
                    _, _, state_next = self.metas[i + 1]

                    speed_next = state_next.get("speed", 0.0)
                    accel = (speed_next - speed_now) / delta_t

                    yaw_next = state_next.get("theta", 0.0)
                    dyaw = (yaw_next - yaw_now + np.pi) % (2 * np.pi) - np.pi
                    rot_speed = dyaw / delta_t
                else:
                    # No future data; make final frame zero
                    accel = 0.0
                    rot_speed = 0.0

                data["privileged_acceleration"] = accel
                data["privileged_rotation_speed"] = rot_speed

                # --- Future speeds: from one step after current to furthest future ---
                future_speeds = []
                for offset in range(
                    0, self.config_expert.ego_num_temporal_data_points_saved + 1
                ):
                    idx = i + offset
                    if idx < N:
                        _, _, future_state = self.metas[idx]
                        future_speeds.append(future_state.get("speed", 0.0))
                data["future_speeds"] = np.array(future_speeds, dtype=np.float32)

                # --- Future yaws: from one step after current to furthest future ---
                future_yaws = []
                for offset in range(
                    0, self.config_expert.ego_num_temporal_data_points_saved + 1
                ):
                    idx = i + offset
                    if idx < N:
                        _, _, future_state = self.metas[idx]
                        yaw_future = future_state.get("theta", 0.0)
                        dyaw = (yaw_future - yaw_now + np.pi) % (2 * np.pi) - np.pi
                        future_yaws.append(dyaw)
                data["future_yaws"] = np.array(future_yaws, dtype=np.float32)

                # --- Future positions: from one step after current to furthest future ---
                T_world_to_current_ego = np.linalg.inv(np.array(data["ego_matrix"]))
                future_positions = []
                for offset in range(
                    0, self.config_expert.ego_num_temporal_data_points_saved + 1
                ):
                    idx = i + offset
                    if idx < N:
                        _, _, future_state = self.metas[idx]
                        T_future = np.array(future_state["ego_matrix"])
                        pos_world = np.append(T_future[:3, 3], 1.0)
                        pos_current_ego = T_world_to_current_ego @ pos_world
                        future_positions.append(pos_current_ego[:3].tolist())
                data["future_positions"] = np.array(future_positions, dtype=np.float32)

                # --- Save metas ---
                common_utils.write_pickle(path=metas_dir / f"{frame:04}.pkl", data=data)

            # Enhance bounding boxes with temporal information
            bbox_index_cache: dict[int, dict[int, dict]] = {}
            for i in range(N):
                step, frame, compact_boxes = self.bounding_boxes[i]

                if step % self.config_expert.data_save_freq != 0:
                    continue

                _, _, state_now = self.metas[i]
                ego_matrix_current = np.array(state_now["ego_matrix"])
                T_world_to_current_ego = np.linalg.inv(ego_matrix_current)
                bounding_boxes = self._load_staged_bounding_boxes(frame, compact_boxes)

                for box in bounding_boxes:
                    box_id = box["id"]

                    if box["class"] not in ["car", "walker"]:
                        continue

                    # --- Future positions and yaws: from one step after current to furthest future ---
                    future_positions = []
                    future_yaws = []
                    for offset in range(
                        0,
                        self.config_expert.other_vehicles_num_temporal_data_points_saved
                        + 1,
                    ):
                        idx = i + offset
                        if idx < N:
                            if idx not in bbox_index_cache:
                                _, future_frame, future_boxes = self.bounding_boxes[idx]
                                if (
                                    self.metas[idx][0]
                                    % self.config_expert.data_save_freq
                                    == 0
                                ):
                                    future_boxes = self._load_staged_bounding_boxes(
                                        future_frame, future_boxes
                                    )
                                bbox_index_cache[idx] = {
                                    b["id"]: b for b in future_boxes
                                }
                            future_box = bbox_index_cache[idx].get(box_id)
                            if future_box:
                                T_future = np.array(future_box["matrix"])
                                pos_world = np.append(T_future[:3, 3], 1.0)
                                pos_current_ego = T_world_to_current_ego @ pos_world
                                future_positions.append(pos_current_ego[:2].tolist())

                                rot_world = T_future[:3, :3]
                                heading_vector_world = rot_world @ np.array(
                                    [1.0, 0.0, 0.0]
                                )
                                heading_vector_world = np.append(
                                    heading_vector_world, 0.0
                                )
                                heading_vector_ego = (
                                    T_world_to_current_ego @ heading_vector_world
                                )
                                yaw = np.arctan2(
                                    heading_vector_ego[1], heading_vector_ego[0]
                                )
                                future_yaws.append(yaw)

                    box["future_positions"] = np.array(
                        future_positions, dtype=np.float16
                    )
                    box["future_yaws"] = np.array(future_yaws, dtype=np.float16)

                common_utils.write_pickle(
                    path=self.save_path / "bboxes" / f"{frame:04}.pkl",
                    data=bounding_boxes,
                )

    def _build_xy_tree(self, points: np.ndarray | None):
        if points is None or points.shape[0] == 0:
            return None

        try:
            from scipy.spatial import cKDTree
        except Exception:
            return None

        try:
            return cKDTree(points[:, :2])
        except Exception:
            return None

    def _count_points_in_actor_prefiltered(
        self,
        ego_matrix: np.ndarray,
        actor_matrix: np.ndarray,
        actor_extent_xyz: np.ndarray,
        point_cloud: np.ndarray | None,
        xy_tree,
        pad: bool = False,
    ) -> int:
        if point_cloud is None:
            return -1
        if point_cloud.shape[0] == 0:
            return 0

        candidate_points = point_cloud
        if xy_tree is not None:
            ego_rot = ego_matrix[:3, :3]
            ego_pos = ego_matrix[:3, 3]
            actor_pos = actor_matrix[:3, 3]
            actor_center_ego = ego_rot.T @ (actor_pos - ego_pos)

            margin_xy = 0.25 if pad else 0.0
            query_radius = (
                np.hypot(
                    actor_extent_xyz[0] + margin_xy,
                    actor_extent_xyz[1] + margin_xy,
                )
                + 1.0
            )
            try:
                candidate_indices = xy_tree.query_ball_point(
                    actor_center_ego[:2], r=float(query_radius)
                )
            except Exception:
                candidate_indices = []

            if len(candidate_indices) == 0:
                return 0
            candidate_points = point_cloud[candidate_indices]

        points_in_bbox = expert_utils.get_points_in_actor_frame_and_in_bbox(
            ego_matrix=ego_matrix,
            actor_matrix=actor_matrix,
            actor_extent=actor_extent_xyz,
            ego_point_cloud=candidate_points,
            pad=pad,
        )
        return int(len(points_in_bbox))

    def _precompute_existing_actor_bbs(
        self,
        existed_bboxes_ids: set[int],
        vehicle_list,
        static_list,
    ) -> list[tuple[carla.BoundingBox, float]]:
        existing_bbs: list[tuple[carla.BoundingBox, float]] = []
        for actor in list(vehicle_list) + list(static_list):
            if actor.id not in existed_bboxes_ids:
                continue

            bb = actor.bounding_box
            try:
                global_location = actor.get_transform().transform(bb.location)
            except Exception:
                continue

            global_bb = carla.BoundingBox(
                carla.Location(
                    x=global_location.x,
                    y=global_location.y,
                    z=global_location.z,
                ),
                bb.extent,
            )
            global_bb.rotation = bb.rotation
            radius = float(np.sqrt(bb.extent.x**2 + bb.extent.y**2 + bb.extent.z**2))
            existing_bbs.append((global_bb, radius))
        return existing_bbs

    def _intersects_any_with_broadphase(
        self,
        candidate_bb: carla.BoundingBox,
        existing_bbs: list[tuple[carla.BoundingBox, float]],
    ) -> bool:
        candidate_radius = float(
            np.sqrt(
                candidate_bb.extent.x**2
                + candidate_bb.extent.y**2
                + candidate_bb.extent.z**2
            )
        )
        candidate_location = candidate_bb.location

        for other_bb, other_radius in existing_bbs:
            dx = candidate_location.x - other_bb.location.x
            dy = candidate_location.y - other_bb.location.y
            dz = candidate_location.z - other_bb.location.z
            radius_sum = candidate_radius + other_radius
            if dx * dx + dy * dy + dz * dz > radius_sum * radius_sum:
                continue
            if expert_utils.check_obb_intersection(candidate_bb, other_bb):
                return True
        return False

    def _is_in_fov(self, relative_pos, extent):
        x = relative_pos[0]
        y = relative_pos[1]

        # объект полностью позади машины
        if x <= -extent[0]:
            return False

        # ограничение дальности впереди (85 м + половина длины объекта)
        if x > 85 + extent[0]:
            return False

        angle = abs(np.arctan2(y, x))

        # расширяем угол на половину длины объекта
        extra = np.arctan2(extent[0] * 1.2, max(x, 0.001))

        return angle <= (self.half_fov_rad + extra)

    def _get_los_ignore_labels(self) -> frozenset:
        labels = getattr(self, "_los_ignore_labels", None)
        if labels is not None:
            return labels

        labels = frozenset(
            label
            for name in ("RoadLines", "Roads", "Sidewalks", "Sky", "Ground", "Terrain")
            if (label := getattr(carla.CityObjectLabel, name, None)) is not None
        )
        self._los_ignore_labels = labels
        return labels

    def _get_or_build_actor_bbox_cache(
        self, actor: carla.Actor
    ) -> tuple[np.ndarray, np.ndarray]:
        bbox_local_cache = getattr(self, "_bbox_local_cache", None)
        if bbox_local_cache is None:
            bbox_local_cache = {}
            self._bbox_local_cache = bbox_local_cache

        actor_id = int(actor.id)
        cached = bbox_local_cache.get(actor_id)
        if cached is not None:
            return cached["local_bbox_matrix"], cached["extent_xyz"]

        bbox = actor.bounding_box
        local_bbox_matrix = np.asarray(
            carla.Transform(bbox.location, bbox.rotation).get_matrix(),
            dtype=np.float32,
        )
        extent_xyz = np.array(
            [bbox.extent.x, bbox.extent.y, bbox.extent.z],
            dtype=np.float32,
        )

        bbox_local_cache[actor_id] = {
            "local_bbox_matrix": local_bbox_matrix,
            "extent_xyz": extent_xyz,
        }
        return local_bbox_matrix, extent_xyz

    def _get_los_ray_origin_world(self, ego_transform: carla.Transform) -> np.ndarray:
        ego_bbox_center = ego_transform.transform(
            self.ego_vehicle.bounding_box.location
        )
        return np.array(
            [ego_bbox_center.x, ego_bbox_center.y, ego_bbox_center.z + 2.3],
            dtype=np.float32,
        )

    def _numpy_point_to_carla_location(self, point_world: np.ndarray) -> carla.Location:
        return carla.Location(
            x=float(point_world[0]),
            y=float(point_world[1]),
            z=float(point_world[2]),
        )

    def _world_location_to_relative_position(
        self, ego_inverse_matrix: np.ndarray, world_location: carla.Location
    ) -> np.ndarray:
        x = float(world_location.x)
        y = float(world_location.y)
        z = float(world_location.z)

        return np.array(
            [
                ego_inverse_matrix[0, 0] * x
                + ego_inverse_matrix[0, 1] * y
                + ego_inverse_matrix[0, 2] * z
                + ego_inverse_matrix[0, 3],
                ego_inverse_matrix[1, 0] * x
                + ego_inverse_matrix[1, 1] * y
                + ego_inverse_matrix[1, 2] * z
                + ego_inverse_matrix[1, 3],
                ego_inverse_matrix[2, 0] * x
                + ego_inverse_matrix[2, 1] * y
                + ego_inverse_matrix[2, 2] * z
                + ego_inverse_matrix[2, 3],
            ],
            dtype=np.float32,
        )

    def _squared_distance_locations(
        self, a: carla.Location, b: carla.Location
    ) -> float:
        dx = float(a.x) - float(b.x)
        dy = float(a.y) - float(b.y)
        dz = float(a.z) - float(b.z)
        return dx * dx + dy * dy + dz * dz

    def _compose_actor_bbox_world_matrix_from_local(
        self, actor_matrix: np.ndarray, local_bbox_matrix: np.ndarray
    ) -> np.ndarray:
        return actor_matrix @ local_bbox_matrix

    def _transform_bbox_local_point_to_world(
        self, bbox_matrix_world: np.ndarray, local_point: np.ndarray
    ) -> np.ndarray:
        rotation = bbox_matrix_world[:3, :3]
        translation = bbox_matrix_world[:3, 3]
        return local_point @ rotation.T + translation

    def _transform_bbox_local_points_to_world(
        self, bbox_matrix_world: np.ndarray, local_points: np.ndarray
    ) -> np.ndarray:
        rotation = bbox_matrix_world[:3, :3]
        translation = bbox_matrix_world[:3, 3]
        return local_points @ rotation.T + translation[None, :]

    def _get_observer_face_check_order(
        self,
        ray_origin_world: np.ndarray,
        bbox_matrix_world: np.ndarray,
    ) -> tuple[list[int], list[int]]:
        rotation = bbox_matrix_world[:3, :3]
        translation = bbox_matrix_world[:3, 3]

        # local_row = (world_row - translation) @ rotation
        observer_local = (ray_origin_world - translation) @ rotation

        x_front, x_back = (0, 1) if observer_local[0] >= 0.0 else (1, 0)
        y_front, y_back = (2, 3) if observer_local[1] >= 0.0 else (3, 2)
        z_front, z_back = (4, 5) if observer_local[2] >= 0.0 else (5, 4)

        ordered = [x_front, y_front, z_front, x_back, y_back, z_back]
        front_count = max(0, min(int(LOS_FRONT_FACE_CENTERS_TO_CHECK), len(ordered)))
        return ordered[:front_count], ordered[front_count:]

    def _is_sample_point_visible_by_cast_ray(
        self,
        ray_origin_world: np.ndarray,
        ray_origin_world_carla: carla.Location,
        target_point_world: np.ndarray,
        ignore_labels: frozenset,
    ) -> bool:
        ox = float(ray_origin_world[0])
        oy = float(ray_origin_world[1])
        oz = float(ray_origin_world[2])

        tx = float(target_point_world[0])
        ty = float(target_point_world[1])
        tz = float(target_point_world[2])

        dx = tx - ox
        dy = ty - oy
        dz = tz - oz

        target_dist_sq = dx * dx + dy * dy + dz * dz
        if target_dist_sq <= 1e-12:
            return True

        target_dist = math.sqrt(target_dist_sq)
        visible_threshold = max(
            target_dist - float(LOS_TARGET_DISTANCE_EPS_METERS), 0.0
        )
        visible_threshold_sq = visible_threshold * visible_threshold
        near_origin_noise_sq = float(LOS_NEAR_ORIGIN_NOISE_METERS) * float(
            LOS_NEAR_ORIGIN_NOISE_METERS
        )

        ray_target = carla.Location(x=tx, y=ty, z=tz)
        ray_hits = self.carla_world.cast_ray(ray_origin_world_carla, ray_target)

        for hit in ray_hits:
            hit_label = getattr(hit, "label", None)
            if hit_label in ignore_labels:
                continue

            hit_location = hit.location
            hx = float(hit_location.x) - ox
            hy = float(hit_location.y) - oy
            hz = float(hit_location.z) - oz
            hit_dist_sq = hx * hx + hy * hy + hz * hz

            if hit_dist_sq <= near_origin_noise_sq:
                continue

            return hit_dist_sq >= visible_threshold_sq

        # If no relevant hit exists, line of sight is considered clear.
        return True

    def _is_bbox_partially_visible_by_cast_ray(
        self,
        ray_origin_world: np.ndarray,
        ray_origin_world_carla: carla.Location,
        bbox_matrix_world: np.ndarray,
        bbox_extent_xyz: np.ndarray | list[float],
        ignore_labels: frozenset,
    ) -> bool:
        try:
            extent_xyz = np.asarray(bbox_extent_xyz, dtype=np.float32)

            # 1) center
            center_local = LOS_LOCAL_SAMPLE_CENTER_UNIT[0] * extent_xyz
            center_world = self._transform_bbox_local_point_to_world(
                bbox_matrix_world=bbox_matrix_world,
                local_point=center_local,
            )
            if self._is_sample_point_visible_by_cast_ray(
                ray_origin_world=ray_origin_world,
                ray_origin_world_carla=ray_origin_world_carla,
                target_point_world=center_world,
                ignore_labels=ignore_labels,
            ):
                return True

            # 2) front-facing face centers first, based on observer in bbox-local space
            face_centers_local = LOS_LOCAL_FACE_CENTER_UNIT * extent_xyz
            front_face_indices, remaining_face_indices = (
                self._get_observer_face_check_order(
                    ray_origin_world=ray_origin_world,
                    bbox_matrix_world=bbox_matrix_world,
                )
            )

            for idx in front_face_indices:
                face_world = self._transform_bbox_local_point_to_world(
                    bbox_matrix_world=bbox_matrix_world,
                    local_point=face_centers_local[idx],
                )
                if self._is_sample_point_visible_by_cast_ray(
                    ray_origin_world=ray_origin_world,
                    ray_origin_world_carla=ray_origin_world_carla,
                    target_point_world=face_world,
                    ignore_labels=ignore_labels,
                ):
                    return True

            for idx in remaining_face_indices:
                face_world = self._transform_bbox_local_point_to_world(
                    bbox_matrix_world=bbox_matrix_world,
                    local_point=face_centers_local[idx],
                )
                if self._is_sample_point_visible_by_cast_ray(
                    ray_origin_world=ray_origin_world,
                    ray_origin_world_carla=ray_origin_world_carla,
                    target_point_world=face_world,
                    ignore_labels=ignore_labels,
                ):
                    return True

            # 3) corners only if needed
            corners_local = LOS_LOCAL_CORNERS_UNIT * extent_xyz
            corners_world = self._transform_bbox_local_points_to_world(
                bbox_matrix_world=bbox_matrix_world,
                local_points=corners_local,
            )
            for corner_world in corners_world:
                if self._is_sample_point_visible_by_cast_ray(
                    ray_origin_world=ray_origin_world,
                    ray_origin_world_carla=ray_origin_world_carla,
                    target_point_world=corner_world,
                    ignore_labels=ignore_labels,
                ):
                    return True

            return False
        except Exception:
            # Fail-open on bbox-level to keep data generation robust.
            return True

    @beartype
    def get_bounding_boxes(self, input_data: dict) -> list[dict]:
        boxes = []

        ego_transform = self.ego_vehicle.get_transform()
        ego_control = self.ego_vehicle.get_control()
        ego_velocity = self.ego_vehicle.get_velocity()
        ego_matrix = np.asarray(ego_transform.get_matrix(), dtype=np.float32)
        ego_inverse_matrix = np.asarray(
            ego_transform.get_inverse_matrix(), dtype=np.float32
        )
        ego_rotation = ego_transform.rotation
        ego_location = ego_transform.location
        ego_extent = self.ego_vehicle.bounding_box.extent
        ego_speed = self._get_forward_speed(
            transform=ego_transform, velocity=ego_velocity
        )
        ego_dx = np.array([ego_extent.x, ego_extent.y, ego_extent.z], dtype=np.float32)
        ego_yaw = np.deg2rad(ego_rotation.yaw)
        ego_brake = ego_control.brake
        ego_matrix_list = ego_matrix.tolist()

        ego_wp = self.carla_world_map.get_waypoint(
            ego_location,
            project_to_road=True,
            lane_type=carla.libcarla.LaneType.Any,
        )

        self._actors = self.actors
        vehicle_list = self._actors.filter("*vehicle*")

        result = {
            "class": "ego_car",
            "transfuser_semantics_id": int(
                TransfuserSemanticSegmentationClass.UNLABELED
            ),
            "extent": [float(ego_dx[0]), float(ego_dx[1]), float(ego_dx[2])],
            "position": [0, 0, 0],
            "yaw": 0.0,
            "num_points": -1,
            "distance": 0,
            "speed": ego_speed,
            "brake": ego_brake,
            "id": int(self.ego_vehicle.id),
            "matrix": ego_matrix_list,
            "visible_pixels": -1,
        }
        boxes.append(result)

        transfuser_camera_semantics_pc = input_data.get("semantics_camera_pc")
        if transfuser_camera_semantics_pc is None:
            transfuser_camera_semantics_pc = self._empty_camera_pc_numpy()

        transfuser_camera_semantics_pc_semantics_id = self.semantics_converter[
            transfuser_camera_semantics_pc[
                :, CameraPointCloudIndex.UNREAL_SEMANTICS_ID
            ].astype(np.int32)
        ]
        global_camera_pc = {
            TransfuserSemanticSegmentationClass.VEHICLE: transfuser_camera_semantics_pc[
                (
                    transfuser_camera_semantics_pc_semantics_id
                    == TransfuserSemanticSegmentationClass.VEHICLE
                )
                | (
                    transfuser_camera_semantics_pc_semantics_id
                    == TransfuserSemanticSegmentationClass.SPECIAL_VEHICLE
                )
                | (
                    transfuser_camera_semantics_pc_semantics_id
                    == TransfuserSemanticSegmentationClass.BIKER
                )
            ][:, : CameraPointCloudIndex.Z + 1],
            TransfuserSemanticSegmentationClass.PEDESTRIAN: transfuser_camera_semantics_pc[
                transfuser_camera_semantics_pc_semantics_id
                == TransfuserSemanticSegmentationClass.PEDESTRIAN
            ][:, : CameraPointCloudIndex.Z + 1],
            TransfuserSemanticSegmentationClass.OBSTACLE: transfuser_camera_semantics_pc[
                transfuser_camera_semantics_pc_semantics_id
                == TransfuserSemanticSegmentationClass.OBSTACLE
            ][:, : CameraPointCloudIndex.Z + 1],
        }

        lidar_points = input_data.get("lidar")
        radar_points = input_data.get("radar")
        lidar_tree = self._build_xy_tree(lidar_points)
        radar_tree = self._build_xy_tree(radar_points)

        los_ray_origin_world = self._get_los_ray_origin_world(ego_transform)
        los_ray_origin_world_carla = self._numpy_point_to_carla_location(
            los_ray_origin_world
        )
        los_ignore_labels = self._get_los_ignore_labels()

        bb_save_radius = float(self.config_expert.bb_save_radius)
        bb_save_radius_sq = bb_save_radius * bb_save_radius

        # -------------------------------------------------------------------------
        # Vehicles
        # -------------------------------------------------------------------------
        for vehicle in vehicle_list:
            if vehicle.id == self.ego_vehicle.id:
                continue

            vehicle_transform = vehicle.get_transform()
            vehicle_location = vehicle_transform.location

            if (
                self._squared_distance_locations(vehicle_location, ego_location)
                >= bb_save_radius_sq
            ):
                continue

            vehicle_rotation = vehicle_transform.rotation
            yaw = np.deg2rad(vehicle_rotation.yaw)
            relative_yaw = common_utils.normalize_angle(yaw - ego_yaw)

            relative_pos = self._world_location_to_relative_position(
                ego_inverse_matrix=ego_inverse_matrix,
                world_location=vehicle_location,
            )

            local_bbox_matrix, vehicle_extent_xyz = self._get_or_build_actor_bbox_cache(
                vehicle
            )
            vehicle_extent_list = vehicle_extent_xyz.tolist()

            if not self._is_in_fov(relative_pos, vehicle_extent_list):
                continue

            vehicle_matrix = np.asarray(
                vehicle_transform.get_matrix(), dtype=np.float32
            )
            vehicle_bbox_matrix = self._compose_actor_bbox_world_matrix_from_local(
                actor_matrix=vehicle_matrix,
                local_bbox_matrix=local_bbox_matrix,
            )
            if self.config_expert.cast_ray:
                if not self._is_bbox_partially_visible_by_cast_ray(
                    ray_origin_world=los_ray_origin_world,
                    ray_origin_world_carla=los_ray_origin_world_carla,
                    bbox_matrix_world=vehicle_bbox_matrix,
                    bbox_extent_xyz=vehicle_extent_xyz,
                    ignore_labels=los_ignore_labels,
                ):
                    continue

            # Survivors only
            vehicle_control: carla.VehicleControl = vehicle.get_control()
            vehicle_velocity = vehicle.get_velocity()
            vehicle_id = vehicle.id

            vehicle_wp = self.carla_world_map.get_waypoint(
                vehicle_location,
                project_to_road=True,
                lane_type=carla.libcarla.LaneType.Any,
            )

            vehicle_speed = self._get_forward_speed(
                transform=vehicle_transform,
                velocity=vehicle_velocity,
            )
            vehicle_brake = vehicle_control.brake
            vehicle_steer = vehicle_control.steer
            vehicle_throttle = vehicle_control.throttle

            rel_x = float(relative_pos[0])
            rel_y = float(relative_pos[1])
            rel_z = float(relative_pos[2])
            distance = math.sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z)

            try:
                next_action = self.traffic_manager.get_next_action(vehicle)[0]
            except Exception:
                next_action = None

            vehicle_cuts_in = False
            if (
                self.scenario_name == "ParkingCutIn"
                and vehicle.attributes["role_name"] == "scenario"
            ):
                if self.cutin_vehicle_starting_position is None:
                    self.cutin_vehicle_starting_position = vehicle_location

                moved_distance = vehicle_location.distance(
                    self.cutin_vehicle_starting_position
                )
                if 0.2 < moved_distance < 8.0:
                    vehicle_cuts_in = True

            elif (
                self.scenario_name == "StaticCutIn"
                and vehicle.attributes["role_name"] == "scenario"
            ):
                if vehicle_speed > 1.0 and abs(vehicle_steer) > 0.2:
                    vehicle_cuts_in = True

            if lidar_points is not None:
                num_in_bbox_points = self._count_points_in_actor_prefiltered(
                    ego_matrix=ego_matrix,
                    actor_matrix=vehicle_matrix,
                    actor_extent_xyz=vehicle_extent_xyz,
                    point_cloud=lidar_points,
                    xy_tree=lidar_tree,
                    pad=True,
                )
            else:
                num_in_bbox_points = -1

            if radar_points is not None:
                num_in_bb_radar_points = self._count_points_in_actor_prefiltered(
                    ego_matrix=ego_matrix,
                    actor_matrix=vehicle_matrix,
                    actor_extent_xyz=vehicle_extent_xyz,
                    point_cloud=radar_points,
                    xy_tree=radar_tree,
                    pad=True,
                )
            else:
                num_in_bb_radar_points = -1

            if num_in_bbox_points == 0 or num_in_bb_radar_points == 0:
                continue

            result = {
                "class": "car",
                "ego_velocity": expert_utils.get_vehicle_velocity_in_ego_frame(
                    self.ego_vehicle, vehicle
                ),
                "next_action": next_action,
                "vehicle_cuts_in": vehicle_cuts_in,
                "road_id": vehicle_wp.road_id,
                "lane_id": vehicle_wp.lane_id,
                "lane_type_str": str(vehicle_wp.lane_type),
                "base_type": vehicle.attributes["base_type"],
                "transfuser_semantics_id": int(
                    TransfuserSemanticSegmentationClass.VEHICLE
                ),
                "extent": vehicle_extent_list,
                "position": [rel_x, rel_y, rel_z],
                "yaw": relative_yaw,
                "num_points": int(num_in_bbox_points),
                "num_radar_points": int(num_in_bb_radar_points),
                "distance": distance,
                "speed": vehicle_speed,
                "brake": vehicle_brake,
                "steer": vehicle_steer,
                "throttle": vehicle_throttle,
                "id": int(vehicle_id),
                "role_name": vehicle.attributes["role_name"],
                "type_id": vehicle.type_id,
                "matrix": vehicle_matrix.tolist(),
                "speed_limit": vehicle.get_speed_limit(),
                "visible_pixels": expert_utils.get_num_points_in_actor(
                    self.ego_vehicle,
                    vehicle,
                    global_camera_pc[TransfuserSemanticSegmentationClass.VEHICLE],
                    pad=True,
                ),
            }
            boxes.append(result)

        # -------------------------------------------------------------------------
        # Walkers
        # -------------------------------------------------------------------------
        walkers = self._actors.filter("*walker*")
        for walker in walkers:
            walker_transform = walker.get_transform()
            walker_location = walker_transform.location

            if (
                self._squared_distance_locations(walker_location, ego_location)
                >= bb_save_radius_sq
            ):
                continue

            walker_rotation = walker_transform.rotation
            yaw = np.deg2rad(walker_rotation.yaw)
            relative_yaw = common_utils.normalize_angle(yaw - ego_yaw)

            relative_pos = self._world_location_to_relative_position(
                ego_inverse_matrix=ego_inverse_matrix,
                world_location=walker_location,
            )

            local_bbox_matrix, walker_extent_xyz = self._get_or_build_actor_bbox_cache(
                walker
            )
            walker_extent_list = walker_extent_xyz.tolist()

            if not self._is_in_fov(relative_pos, walker_extent_list):
                continue

            walker_matrix = np.asarray(walker_transform.get_matrix(), dtype=np.float32)
            walker_bbox_matrix = self._compose_actor_bbox_world_matrix_from_local(
                actor_matrix=walker_matrix,
                local_bbox_matrix=local_bbox_matrix,
            )
            if self.config_expert.cast_ray:
                if not self._is_bbox_partially_visible_by_cast_ray(
                    ray_origin_world=los_ray_origin_world,
                    ray_origin_world_carla=los_ray_origin_world_carla,
                    bbox_matrix_world=walker_bbox_matrix,
                    bbox_extent_xyz=walker_extent_xyz,
                    ignore_labels=los_ignore_labels,
                ):
                    continue

            # Survivors only
            walker_velocity = walker.get_velocity()
            walker_id = walker.id
            walker_speed = self._get_forward_speed(
                transform=walker_transform,
                velocity=walker_velocity,
            )

            if lidar_points is not None:
                num_in_bbox_points = self._count_points_in_actor_prefiltered(
                    ego_matrix=ego_matrix,
                    actor_matrix=walker_matrix,
                    actor_extent_xyz=walker_extent_xyz,
                    point_cloud=lidar_points,
                    xy_tree=lidar_tree,
                    pad=True,
                )
            else:
                num_in_bbox_points = -1

            if radar_points is not None:
                num_in_bb_radar_points = self._count_points_in_actor_prefiltered(
                    ego_matrix=ego_matrix,
                    actor_matrix=walker_matrix,
                    actor_extent_xyz=walker_extent_xyz,
                    point_cloud=radar_points,
                    xy_tree=radar_tree,
                    pad=True,
                )
            else:
                num_in_bb_radar_points = -1

            if num_in_bbox_points == 0 or num_in_bb_radar_points == 0:
                continue

            rel_x = float(relative_pos[0])
            rel_y = float(relative_pos[1])
            rel_z = float(relative_pos[2])
            distance = math.sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z)

            result = {
                "class": "walker",
                "ego_velocity": expert_utils.get_vehicle_velocity_in_ego_frame(
                    self.ego_vehicle, walker
                ),
                "role_name": walker.attributes["role_name"],
                "transfuser_semantics_id": int(
                    TransfuserSemanticSegmentationClass.PEDESTRIAN
                ),
                "extent": walker_extent_list,
                "position": [rel_x, rel_y, rel_z],
                "yaw": relative_yaw,
                "num_points": int(num_in_bbox_points),
                "num_radar_points": int(num_in_bb_radar_points),
                "distance": distance,
                "speed": walker_speed,
                "id": int(walker_id),
                "matrix": walker_matrix.tolist(),
                "visible_pixels": expert_utils.get_num_points_in_actor(
                    self.ego_vehicle,
                    walker,
                    global_camera_pc[TransfuserSemanticSegmentationClass.PEDESTRIAN],
                    pad=True,
                ),
            }
            boxes.append(result)

        # -------------------------------------------------------------------------
        # Static actors (без изменений по LOS, но с float32 и без лишних get_transform())
        # -------------------------------------------------------------------------
        if self.config_expert.eval_expert:
            if self.step % max(self.config_expert.log_info_freq, 1) == 0:
                LOG.debug("Skipping static actor bounding boxes in eval_expert mode.")
            static_list = []
        else:
            static_list = self._actors.filter("*static*")
            if LOG.isEnabledFor(logging.DEBUG) and (
                self.step % max(self.config_expert.log_info_freq, 1) == 0
            ):
                LOG.debug("Found %d static actors in the scene.", len(static_list))

            for static in static_list:
                static_transform = static.get_transform()
                static_location = static_transform.location

                if (
                    self._squared_distance_locations(static_location, ego_location)
                    >= bb_save_radius_sq
                ):
                    continue

                static_velocity = static.get_velocity()
                static_rotation = static_transform.rotation
                static_matrix = np.asarray(
                    static_transform.get_matrix(), dtype=np.float32
                )
                static_id = static.id
                type_id = static.type_id
                mesh_path = static.attributes.get("mesh_path", None)

                static_extent_raw = static.bounding_box.extent
                static_extent_raw_xyz = np.array(
                    [static_extent_raw.x, static_extent_raw.y, static_extent_raw.z],
                    dtype=np.float32,
                )
                static_extent = [
                    float(static_extent_raw.x),
                    float(static_extent_raw.y),
                    float(static_extent_raw.z),
                ]

                if mesh_path is not None and mesh_path in constants.LOOKUP_TABLE:
                    static_extent = constants.LOOKUP_TABLE[mesh_path]
                if type_id == "static.prop.trafficwarning":
                    static_extent[0], static_extent[1] = (
                        self.config_expert.traffic_warning_bb_size[0],
                        self.config_expert.traffic_warning_bb_size[1],
                    )
                elif type_id == "static.prop.constructioncone":
                    static_extent[0], static_extent[1] = (
                        self.config_expert.construction_cone_bb_size[0],
                        self.config_expert.construction_cone_bb_size[1],
                    )

                yaw = np.deg2rad(static_rotation.yaw)
                relative_yaw = common_utils.normalize_angle(yaw - ego_yaw)
                relative_pos = common_utils.get_relative_transform(
                    ego_matrix, static_matrix
                )
                if not self._is_in_fov(relative_pos, static_extent):
                    continue

                static_speed = self._get_forward_speed(
                    transform=static_transform,
                    velocity=static_velocity,
                )

                if lidar_points is not None:
                    num_in_bbox_points = self._count_points_in_actor_prefiltered(
                        ego_matrix=ego_matrix,
                        actor_matrix=static_matrix,
                        actor_extent_xyz=static_extent_raw_xyz,
                        point_cloud=lidar_points,
                        xy_tree=lidar_tree,
                        pad=True,
                    )
                else:
                    num_in_bbox_points = -1

                if num_in_bbox_points == 0:
                    continue

                rel_x = float(relative_pos[0])
                rel_y = float(relative_pos[1])
                rel_z = float(relative_pos[2])
                distance = math.sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z)

                result = {
                    "class": "static",
                    "transfuser_semantics_id": int(
                        TransfuserSemanticSegmentationClass.UNLABELED
                    ),
                    "extent": static_extent,
                    "position": [rel_x, rel_y, rel_z],
                    "yaw": relative_yaw,
                    "num_points": int(num_in_bbox_points),
                    "distance": distance,
                    "speed": static_speed,
                    "id": int(static_id),
                    "matrix": static_matrix.tolist(),
                    "type_id": type_id,
                    "mesh_path": mesh_path,
                }
                if result["mesh_path"] is not None and "Car" in result["mesh_path"]:
                    result["transfuser_semantics_id"] = int(
                        TransfuserSemanticSegmentationClass.VEHICLE
                    )
                    result["visible_pixels"] = expert_utils.get_num_points_in_bbox(
                        self.ego_vehicle,
                        result,
                        global_camera_pc[TransfuserSemanticSegmentationClass.VEHICLE],
                        pad=True,
                    )
                else:
                    result["transfuser_semantics_id"] = int(
                        TransfuserSemanticSegmentationClass.OBSTACLE
                    )
                    result["visible_pixels"] = expert_utils.get_num_points_in_bbox(
                        self.ego_vehicle,
                        result,
                        global_camera_pc[TransfuserSemanticSegmentationClass.OBSTACLE],
                        pad=True,
                    )

                boxes.append(result)

        # -------------------------------------------------------------------------
        # Traffic lights
        # -------------------------------------------------------------------------
        for (
            traffic_light,
            original_traffic_light_bounding_box,
            traffic_light_state,
            traffic_light_id,
            traffic_light_affects_ego,
        ) in self.close_traffic_lights:
            original_waypoint = self.carla_world_map.get_waypoint(
                original_traffic_light_bounding_box.location
            )
            waypoint_transform_matrix = np.asarray(
                original_waypoint.transform.get_matrix(),
                dtype=np.float32,
            )
            traffic_light_transform_matrix = np.asarray(
                traffic_light.get_transform().get_matrix(),
                dtype=np.float32,
            )

            traffic_light_in_waypoint = common_utils.get_relative_transform(
                ego_matrix=waypoint_transform_matrix,
                vehicle_matrix=traffic_light_transform_matrix,
            )

            is_over_head_traffic_light = (
                self.town in ["Town11", "Town12", "Town13", "Town15"]
                and abs(traffic_light_in_waypoint[0]) < 4.0
            )
            is_europe_traffic_light = (
                self.town
                in [
                    "Town01",
                    "Town02",
                    "Town03",
                    "Town04",
                    "Town05",
                    "Town06",
                    "Town07",
                    "Town10HD",
                ]
                and abs(traffic_light_in_waypoint[0]) < 4.0
            )

            if traffic_light_affects_ego:
                red_light_stop_waypoints = expert_utils.get_stop_waypoints(
                    ego_wp, traffic_light
                )

                for i, red_light_stop_waypoint in enumerate(red_light_stop_waypoints):
                    duplicated_traffic_light_bounding_box = (
                        expert_utils.create_bounding_box_for_waypoint(
                            original_traffic_light_bounding_box,
                            red_light_stop_waypoint,
                        )
                    )

                    traffic_light_extent = np.array(
                        [
                            duplicated_traffic_light_bounding_box.extent.x,
                            duplicated_traffic_light_bounding_box.extent.y,
                            duplicated_traffic_light_bounding_box.extent.z,
                        ],
                        dtype=np.float32,
                    )
                    traffic_light_extent_list = traffic_light_extent.tolist()

                    traffic_light_transform = carla.Transform(
                        duplicated_traffic_light_bounding_box.location,
                        original_traffic_light_bounding_box.rotation,
                    )
                    traffic_light_rotation = traffic_light_transform.rotation
                    traffic_light_matrix = np.asarray(
                        traffic_light_transform.get_matrix(),
                        dtype=np.float32,
                    )
                    yaw = np.deg2rad(traffic_light_rotation.yaw)

                    relative_yaw = common_utils.normalize_angle(yaw - ego_yaw)
                    relative_pos = common_utils.get_relative_transform(
                        ego_matrix, traffic_light_matrix
                    )

                    if not self._is_in_fov(relative_pos, traffic_light_extent_list):
                        continue
                    if self.config_expert.cast_ray:
                        if not self._is_bbox_partially_visible_by_cast_ray(
                            ray_origin_world=los_ray_origin_world,
                            ray_origin_world_carla=los_ray_origin_world_carla,
                            bbox_matrix_world=traffic_light_matrix,
                            bbox_extent_xyz=traffic_light_extent,
                            ignore_labels=los_ignore_labels,
                        ):
                            continue

                    rel_x = float(relative_pos[0])
                    rel_y = float(relative_pos[1])
                    rel_z = float(relative_pos[2])
                    distance = math.sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z)

                    traffic_light_loc = traffic_light.get_transform().location
                    bbox_loc = duplicated_traffic_light_bounding_box.location
                    tl_dx = float(traffic_light_loc.x) - float(bbox_loc.x)
                    tl_dy = float(traffic_light_loc.y) - float(bbox_loc.y)
                    distance_traffic_light_to_bounding_box = math.sqrt(
                        tl_dx * tl_dx + tl_dy * tl_dy
                    )

                    if i == 0:
                        result = {
                            "class": "traffic_light",
                            "transfuser_semantics_id": int(
                                TransfuserSemanticSegmentationClass.TRAFFIC_LIGHT
                            ),
                            "extent": traffic_light_extent_list,
                            "position": [rel_x, rel_y, rel_z],
                            "yaw": relative_yaw,
                            "distance": distance,
                            "state": str(traffic_light_state),
                            "id": int(traffic_light_id),
                            "affects_ego": traffic_light_affects_ego,
                            "matrix": traffic_light_matrix.tolist(),
                            "distance_to_physical_traffic_light": distance_traffic_light_to_bounding_box,
                            "dummy_traffic_light_bounding_box": False,
                            "same_lane_as_ego": red_light_stop_waypoint.lane_id
                            == ego_wp.lane_id,
                            "is_over_head_traffic_light": is_over_head_traffic_light,
                            "is_europe_traffic_light": is_europe_traffic_light,
                        }
                    else:
                        result = {
                            "class": "traffic_light",
                            "transfuser_semantics_id": int(
                                TransfuserSemanticSegmentationClass.TRAFFIC_LIGHT
                            ),
                            "extent": traffic_light_extent_list,
                            "position": [rel_x, rel_y, rel_z],
                            "yaw": relative_yaw,
                            "distance": distance,
                            "state": str(traffic_light_state),
                            "id": self.negative_id_counter(traffic_light.id),
                            "affects_ego": traffic_light_affects_ego,
                            "matrix": traffic_light_matrix.tolist(),
                            "distance_to_physical_traffic_light": distance_traffic_light_to_bounding_box,
                            "dummy_traffic_light_bounding_box": True,
                            "same_lane_as_ego": red_light_stop_waypoint.lane_id
                            == ego_wp.lane_id,
                            "is_over_head_traffic_light": is_over_head_traffic_light,
                            "is_europe_traffic_light": is_europe_traffic_light,
                        }
                    boxes.append(result)

        # -------------------------------------------------------------------------
        # Stop signs
        # -------------------------------------------------------------------------
        for stop_sign in self.close_stop_signs:
            stop_sign_extent = np.array(
                [stop_sign[0].extent.x, stop_sign[0].extent.y, stop_sign[0].extent.z],
                dtype=np.float32,
            )
            stop_sign_extent_list = stop_sign_extent.tolist()

            stop_sign_transform = carla.Transform(
                stop_sign[0].location, stop_sign[0].rotation
            )
            stop_sign_rotation = stop_sign_transform.rotation
            stop_sign_matrix = np.asarray(
                stop_sign_transform.get_matrix(), dtype=np.float32
            )
            yaw = np.deg2rad(stop_sign_rotation.yaw)

            relative_yaw = common_utils.normalize_angle(yaw - ego_yaw)
            relative_pos = common_utils.get_relative_transform(
                ego_matrix, stop_sign_matrix
            )

            if not self._is_in_fov(relative_pos, stop_sign_extent_list):
                continue
            if self.config_expert.cast_ray:
                if not self._is_bbox_partially_visible_by_cast_ray(
                    ray_origin_world=los_ray_origin_world,
                    ray_origin_world_carla=los_ray_origin_world_carla,
                    bbox_matrix_world=stop_sign_matrix,
                    bbox_extent_xyz=stop_sign_extent,
                    ignore_labels=los_ignore_labels,
                ):
                    continue

            rel_x = float(relative_pos[0])
            rel_y = float(relative_pos[1])
            rel_z = float(relative_pos[2])
            distance = math.sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z)

            result = {
                "class": "stop_sign",
                "transfuser_semantics_id": int(
                    TransfuserSemanticSegmentationClass.UNLABELED
                ),
                "extent": stop_sign_extent_list,
                "position": [rel_x, rel_y, rel_z],
                "yaw": relative_yaw,
                "distance": distance,
                "id": int(stop_sign[1]),
                "affects_ego": stop_sign[2],
                "matrix": stop_sign_matrix.tolist(),
            }
            boxes.append(result)

        # -------------------------------------------------------------------------
        # Static meshes from level bbs (static_prop_car)
        # -------------------------------------------------------------------------
        if self.config_expert.eval_expert:
            if self.step % max(self.config_expert.log_info_freq, 1) == 0:
                LOG.debug("Skipping static actor bounding boxes in eval_expert mode.")
        else:
            static_parking_id_start = 1e8
            found = 0
            existed_bboxes_ids = {box["id"] for box in boxes}
            existing_actor_bbs = self._precompute_existing_actor_bbs(
                existed_bboxes_ids=existed_bboxes_ids,
                vehicle_list=vehicle_list,
                static_list=static_list,
            )

            cached_level_bbs = self._cached_level_bbs_by_class.get(
                carla.CityObjectLabel.Car, tuple()
            )
            if len(cached_level_bbs) == 0:
                cached_level_bbs = tuple(
                    self.carla_world.get_level_bbs(carla.CityObjectLabel.Car)
                )

            for i, vehicle_bounding_box in enumerate(cached_level_bbs):
                extent = vehicle_bounding_box.extent
                location = vehicle_bounding_box.location
                dx = float(location.x) - float(ego_location.x)
                dy = float(location.y) - float(ego_location.y)
                dz = float(location.z) - float(ego_location.z)
                if dx * dx + dy * dy + dz * dz > bb_save_radius_sq:
                    continue

                rotation = vehicle_bounding_box.rotation
                matrix = np.asarray(
                    carla.Transform(location, rotation).get_matrix(),
                    dtype=np.float32,
                )
                relative_pos = common_utils.get_relative_transform(ego_matrix, matrix)

                if not self._is_in_fov(relative_pos, [extent.x, extent.y, extent.z]):
                    continue

                rel_x = float(relative_pos[0])
                rel_y = float(relative_pos[1])
                rel_z = float(relative_pos[2])
                distance = math.sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z)

                if distance > bb_save_radius:
                    continue

                relative_yaw = common_utils.normalize_angle(
                    np.deg2rad(rotation.yaw) - ego_yaw
                )
                result = {
                    "class": "static_prop_car",
                    "transfuser_semantics_id": int(
                        TransfuserSemanticSegmentationClass.VEHICLE
                    ),
                    "extent": [extent.x, extent.y, extent.z],
                    "position": [rel_x, rel_y, rel_z],
                    "yaw": relative_yaw,
                    "distance": distance,
                    "id": int(static_parking_id_start + i),
                    "matrix": matrix.tolist(),
                }

                if self._intersects_any_with_broadphase(
                    vehicle_bounding_box, existing_actor_bbs
                ):
                    continue
                num_in_bbox_points = -1
                num_in_bb_radar_points = -1

                if lidar_points is not None:
                    result["num_points"] = expert_utils.get_num_points_in_bbox(
                        self.ego_vehicle,
                        result,
                        lidar_points,
                        pad=True,
                    )
                else:
                    result["num_points"] = -1

                if num_in_bbox_points == 0:
                    continue

                result["visible_pixels"] = expert_utils.get_num_points_in_bbox(
                    self.ego_vehicle,
                    result,
                    global_camera_pc[TransfuserSemanticSegmentationClass.VEHICLE],
                    pad=True,
                )

                boxes.append(result)
                found += 1

        # -------------------------------------------------------------------------
        # Deduplicate and sort
        # -------------------------------------------------------------------------
        bounding_box_ids = set()
        filtered_bounding_boxes = []
        for box in boxes:
            if box["id"] not in bounding_box_ids:
                bounding_box_ids.add(box["id"])
                filtered_bounding_boxes.append(box)

        filtered_bounding_boxes = sorted(
            filtered_bounding_boxes,
            key=lambda x: x["distance"],
        )
        return filtered_bounding_boxes

    @beartype
    def visualize_ego_bb(self, ego_bb_global: carla.BoundingBox):
        ego_vehicle_transform = self.ego_vehicle.get_transform()
        # Calculate the global bounding box of the ego vehicle
        center_ego_bb_global = ego_vehicle_transform.transform(
            self.ego_vehicle.bounding_box.location
        )
        ego_bb_global = carla.BoundingBox(
            center_ego_bb_global, self.ego_vehicle.bounding_box.extent
        )
        ego_bb_global.rotation = ego_vehicle_transform.rotation

        if self.config_expert.visualize_bounding_boxes:
            self.carla_world.debug.draw_box(
                box=ego_bb_global,
                rotation=ego_bb_global.rotation,
                thickness=0.1,
                color=self.config_expert.ego_vehicle_bb_color,
                life_time=self.config_expert.draw_life_time,
            )

    @beartype
    def visualize_lead_and_trailing_vehicles(self):
        if self.config_expert.visualize_internal_data:
            vehicle_list = ...

            leading_vehicle_ids = (
                self.privileged_route_planner.compute_leading_vehicles(
                    vehicle_list, self.ego_vehicle.id
                )
            )
            trailing_vehicle_ids = (
                self.privileged_route_planner.compute_trailing_vehicles(
                    vehicle_list, self.ego_vehicle.id
                )
            )

            for vehicle in vehicle_list:
                if vehicle.id in leading_vehicle_ids:
                    self.carla_world.debug.draw_string(
                        vehicle.get_location(),
                        f"Leading Vehicle: {vehicle.get_velocity().length():.2f} m/s",
                        life_time=self.config_expert.draw_life_time,
                        color=self.config_expert.leading_vehicle_color,
                    )
                elif vehicle.id in trailing_vehicle_ids:
                    self.carla_world.debug.draw_string(
                        vehicle.get_location(),
                        f"Trailing Vehicle: {vehicle.get_velocity().length():.2f} m/s",
                        life_time=self.config_expert.draw_life_time,
                        color=self.config_expert.trailing_vehicle_color,
                    )

    @beartype
    def visualize_forecasted_bounding_boxes(
        self,
        predicted_bounding_boxes: dict[int, list[carla.BoundingBox]],
    ):
        if self.config_expert.visualize_bounding_boxes:
            (
                dangerous_adversarial_actors_ids,
                safe_adversarial_actors_ids,
                ignored_adversarial_actors_ids,
            ) = self.adversarial_actors_ids
            for (
                _actor_idx,
                actors_forecasted_bounding_boxes,
            ) in predicted_bounding_boxes.items():
                for bb in actors_forecasted_bounding_boxes:
                    color = self.config_expert.other_vehicles_forecasted_bbs_color
                    if (
                        _actor_idx in dangerous_adversarial_actors_ids
                        or _actor_idx in safe_adversarial_actors_ids
                    ):
                        color = self.config_expert.adversarial_color
                    self.carla_world.debug.draw_box(
                        box=bb,
                        rotation=bb.rotation,
                        thickness=0.1,
                        color=color,
                        life_time=self.config_expert.draw_life_time,
                    )

                for vehicle_id in predicted_bounding_boxes.keys():
                    # check if vehicle is in front of the ego vehicle
                    if (
                        vehicle_id in self.leading_vehicle_ids
                        and not self.near_lane_change
                    ):
                        vehicle = self.carla_world.get_actor(vehicle_id)
                        extent = vehicle.bounding_box.extent
                        bb = carla.BoundingBox(vehicle.get_location(), extent)
                        bb.rotation = carla.Rotation(
                            pitch=0, yaw=vehicle.get_transform().rotation.yaw, roll=0
                        )
                        self.carla_world.debug.draw_box(
                            box=bb,
                            rotation=bb.rotation,
                            thickness=0.5,
                            color=self.config_expert.leading_vehicle_color,
                            life_time=self.config_expert.draw_life_time,
                        )
                    elif vehicle_id in self.trailing_vehicle_ids:
                        vehicle = self.carla_world.get_actor(vehicle_id)
                        extent = vehicle.bounding_box.extent
                        bb = carla.BoundingBox(vehicle.get_location(), extent)
                        bb.rotation = carla.Rotation(
                            pitch=0, yaw=vehicle.get_transform().rotation.yaw, roll=0
                        )
                        self.carla_world.debug.draw_box(
                            box=bb,
                            rotation=bb.rotation,
                            thickness=0.5,
                            color=self.config_expert.trailing_vehicle_color,
                            life_time=self.config_expert.draw_life_time,
                        )

    @beartype
    def visualize_pedestrian_bounding_boxes(
        self, nearby_pedestrians_bbs: list[list[carla.BoundingBox]]
    ):
        # Visualize the future bounding boxes of pedestrians (if enabled)
        if self.config_expert.visualize_bounding_boxes:
            for bbs in nearby_pedestrians_bbs:
                for bbox in bbs:
                    self.carla_world.debug.draw_box(
                        box=bbox,
                        rotation=bbox.rotation,
                        thickness=0.1,
                        color=self.config_expert.pedestrian_forecasted_bbs_color,
                        life_time=self.config_expert.draw_life_time,
                    )

    @beartype
    def visualize_traffic_lights(
        self,
        traffic_light: carla.TrafficLight,
        wp: carla.Waypoint,
        bounding_box: carla.BoundingBox,
    ):
        if self.config_expert.visualize_traffic_lights_bounding_boxes:
            if traffic_light.state == carla.TrafficLightState.Red:
                color = self.config_expert.red_traffic_light_color
            elif traffic_light.state == carla.TrafficLightState.Yellow:
                color = self.config_expert.yellow_traffic_light_color
            elif traffic_light.state == carla.TrafficLightState.Green:
                color = self.config_expert.green_traffic_light_color
            elif traffic_light.state == carla.TrafficLightState.Off:
                color = self.config_expert.off_traffic_light_color
            else:  # unknown
                color = self.config_expert.unknown_traffic_light_color

            self.carla_world.debug.draw_box(
                box=bounding_box,
                rotation=bounding_box.rotation,
                thickness=0.1,
                color=color,
                life_time=0.051,
            )

            self.carla_world.debug.draw_point(
                wp.transform.location
                + carla.Location(z=traffic_light.trigger_volume.location.z),
                size=0.1,
                color=color,
                life_time=(1.0 / self.config_expert.carla_fps) + 1e-6,
            )

            self.carla_world.debug.draw_box(
                box=traffic_light.bounding_box,
                rotation=traffic_light.bounding_box.rotation,
                thickness=0.1,
                color=color,
                life_time=0.051,
            )

    @beartype
    def visualize_stop_signs(
        self, bounding_box_stop_sign: carla.BoundingBox, affects_ego: bool
    ):
        if self.config_expert.visualize_bounding_boxes:
            color = carla.Color(0, 1, 0) if affects_ego else carla.Color(1, 0, 0)
            self.carla_world.debug.draw_box(
                box=bounding_box_stop_sign,
                rotation=bounding_box_stop_sign.rotation,
                thickness=0.1,
                color=color,
                life_time=(1.0 / self.config_expert.carla_fps) + 1e-6,
            )

    @expert_utils.step_cached_property
    def visibility_range_camera_1(self):
        if not self.config_expert.compute_camera_pc:
            return 1.0
        if self.config_expert.target_dataset in [
            TargetDataset.CARLA_LEADERBOARD2_3CAMERAS,
            TargetDataset.CARLA_LEADERBOARD2_ONLY3CAMERAS,
            TargetDataset.CARLA_LEADERBOARD2_6CAMERAS,
        ]:
            if "semantics_camera_pc_1" not in self.tick_data:
                return 1.0
            pc = self.tick_data["semantics_camera_pc_1"]
            if pc.shape[0] == 0:
                return 1.0
            return expert_utils.compute_camera_occlusion_score(pc)
        return 1.0

    @expert_utils.step_cached_property
    def visibility_range_camera_2(self):
        if not self.config_expert.compute_camera_pc:
            return 1.0
        if self.config_expert.target_dataset in [
            TargetDataset.CARLA_LEADERBOARD2_3CAMERAS,
            TargetDataset.CARLA_LEADERBOARD2_ONLY3CAMERAS,
            TargetDataset.CARLA_LEADERBOARD2_6CAMERAS,
        ]:
            if "semantics_camera_pc_2" not in self.tick_data:
                return 1.0
            pc = self.tick_data["semantics_camera_pc_2"]
            if pc.shape[0] == 0:
                return 1.0
            return expert_utils.compute_camera_occlusion_score(pc)
        return 1.0

    @expert_utils.step_cached_property
    def visibility_range_camera_3(self):
        if not self.config_expert.compute_camera_pc:
            return 1.0
        if self.config_expert.target_dataset in [
            TargetDataset.CARLA_LEADERBOARD2_3CAMERAS,
            TargetDataset.CARLA_LEADERBOARD2_ONLY3CAMERAS,
            TargetDataset.CARLA_LEADERBOARD2_6CAMERAS,
        ]:
            if "semantics_camera_pc_3" not in self.tick_data:
                return 1.0
            pc = self.tick_data["semantics_camera_pc_3"]
            if pc.shape[0] == 0:
                return 1.0
            return expert_utils.compute_camera_occlusion_score(pc)
        return 1.0

    @expert_utils.step_cached_property
    def visibility_range_camera_4(self):
        if not self.config_expert.compute_camera_pc:
            return 1.0
        if self.config_expert.target_dataset in [
            TargetDataset.CARLA_LEADERBOARD2_6CAMERAS
        ]:
            if "semantics_camera_pc_4" not in self.tick_data:
                return 1.0
            pc = self.tick_data["semantics_camera_pc_4"]
            if pc.shape[0] == 0:
                return 1.0
            return expert_utils.compute_camera_occlusion_score(pc)
        return 1.0
