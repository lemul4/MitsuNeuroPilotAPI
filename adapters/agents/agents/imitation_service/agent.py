#!/usr/bin/env python
from __future__ import print_function

import os
import math
import queue
import threading
from collections import deque
from datetime import datetime

import numpy as np
import torch
from torch.backends import cudnn

import carla
from infrastructure.carla.agents.navigation.behavior_agent import BehaviorAgent
from data.carla_autopilot_net import ImprovedCarlaAutopilotNet
from services.evaluation_service.scenario_runner.srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from services.evaluation_service.leaderboard.leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

from .constants import DEBUG, SENSOR_CONFIG, stats
from .sensors import get_sensors_config
from .data_saver import DataSaver
from .vehicle_utils import is_vehicle_hazard


def get_entry_point():
    return 'AutopilotAgent'

class AutopilotAgent(AutonomousAgent):
    _agent = None
    _route_assigned = False
    _initialized_behavior_agent = False

    def __init__(self, carla_host, carla_port, debug=False):
        super().__init__(carla_host, carla_port, debug)
        self.angle_to_far_waypoint_deg = 0
        self.angle_to_near_waypoint_deg = 0
        self.debug = debug or DEBUG

        self.step = -1
        self.data_count = 0
        self.initialized_data_saving = False
        self.dataset_save_path = ""
        self._save_queue = queue.Queue()
        self._worker_thread = None
        self.is_collision = False
        self.subfolder_paths = []

        self.last_brake = 0.0
        self.last_throttle = 0.0
        self.last_steer = 0.0
        self.last_speed = 0.0
        self.last_light = 0

        self.speed_sequence = deque(np.zeros(20), maxlen=20)
        self.brake_sequence = deque(np.zeros(20), maxlen=20)
        self.throttle_sequence = deque(np.zeros(20), maxlen=20)
        self.steer_sequence = deque(np.zeros(20), maxlen=20)
        self.light_sequence = deque(np.zeros(20), maxlen=20)

        self.angle_far_unnorm = 0.0
        self.is_red_light_present_log = 0
        self.is_stops_present_log = 0
        self.hero_vehicle = None

        self.behavior_name = "cautious"
        self.opt_dict = {}

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        cudnn.deterministic = False
        cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.multiprocessing.set_start_method('spawn', force=True)
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = ImprovedCarlaAutopilotNet(
            depth_channels=1,
            seg_channels=2,
            img_emb_dim=1024,
            rnn_input=5,
            rnn_hidden=512,
            cont_feat_dim=10,
            signal_dim=2,
            near_cmd_dim=7,
            far_cmd_dim=7,
            mlp_hidden=1024
        ).to(self.DEVICE)

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",".."))
        model_path = os.path.join(base_dir, "model_2.pth")
        self.model.load_state_dict(torch.load(model_path, map_location=self.DEVICE))
        self.model.eval()

        self._data_saver = None
        self.collision_sensor = None

    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS
        self._sensor_data_config = SENSOR_CONFIG
        self.config_path = path_to_conf_file

    def _init_behavior_agent_if_needed(self):
        if not self._initialized_behavior_agent:
            self.hero_vehicle = None
            for actor in CarlaDataProvider.get_world().get_actors():
                if actor.attributes.get('role_name', '') == 'hero':
                    self.hero_vehicle = actor
                    break

            if not self.hero_vehicle:
                print("AutopilotAgent: Hero vehicle not found. Cannot initialize BehaviorAgent.")
                return False

            self._agent = BehaviorAgent(self.hero_vehicle, behavior=self.behavior_name, opt_dict=self.opt_dict)
            print(f"AutopilotAgent: BehaviorAgent initialized with behavior: {self.behavior_name}")

            route = []
            if self._global_plan_world_coord:
                map_api = CarlaDataProvider.get_map()
                for transform, road_option in self._global_plan_world_coord:
                    wp = map_api.get_waypoint(transform.location)
                    route.append((wp, road_option))
                self._agent.set_global_plan(route, False)
                print("AutopilotAgent: Global plan set for BehaviorAgent.")
            else:
                print("AutopilotAgent: Warning - Global plan not available when initializing BehaviorAgent.")

            self._initialized_behavior_agent = True
            return True
        return True

    def _init_data_saving_components(self):
        if not self.hero_vehicle:
            self.hero_vehicle = CarlaDataProvider.get_hero_actor()
            if not self.hero_vehicle:
                print("AutopilotAgent: Hero vehicle not found. Cannot initialize privileged sensors.")
                return

        self.world = self.hero_vehicle.get_world()
        self._map = self.world.get_map()

        if not self.initialized_data_saving:
            today = datetime.today()
            now = datetime.now()
            current_date = today.strftime("%b_%d_%Y")
            current_time = now.strftime("%H_%M_%S")
            time_info = f"{current_date}-{current_time}"

            map_name = self._map.name.split('/')[-1]
            self.dataset_save_path = os.path.join("dataset", "autopilot_behavior_data", map_name, time_info)

            self._data_saver = DataSaver(self.dataset_save_path)
            self.initialized_data_saving = True

        from .collizions import CollisionSensor
        if not self.collision_sensor:
            self.collision_sensor = CollisionSensor(self.world, self.hero_vehicle, None)
            self.collision_sensor.initialize()
            print("AutopilotAgent: Collision sensor initialized.")

    def sensors(self):
        return get_sensors_config(self._sensor_data_config)

    def run_step(self, input_data, timestamp):
        self.step += 1

        if not self._initialized_behavior_agent:
            if not self._init_behavior_agent_if_needed():
                print("AutopilotAgent: BehaviorAgent initialization failed. Returning empty control.")
                return carla.VehicleControl()
            self._init_data_saving_components()

        if not self._agent:
            print("AutopilotAgent: _agent (BehaviorAgent) not initialized. Returning empty control.")
            return carla.VehicleControl()

        control = carla.VehicleControl()

        bh_control = carla.VehicleControl()
        if self._agent:
            bh_control = self._agent.run_step(debug=self.debug)

        current_speed_m_s = input_data['speed'][1]['speed']
        imu_sensor_data = input_data['imu'][1]
        compass_rad = imu_sensor_data[-1]

        vehicle_transform = self.hero_vehicle.get_transform()
        vehicle_location = vehicle_transform.location

        log_near_node_coords, log_near_node_cmd = (None, None)
        log_far_node_coords, log_far_node_cmd = (None, None)

        if self._agent and hasattr(self._agent, 'get_local_planner') and self._agent.get_local_planner():
            upcoming_waypoints = self._agent.get_local_planner().get_plan()
            if upcoming_waypoints and len(upcoming_waypoints) > 0:
                near_wp, near_cmd = upcoming_waypoints[0]
                log_near_node_coords = (near_wp.transform.location.x, near_wp.transform.location.y)
                log_near_node_cmd = near_cmd.name
                if len(upcoming_waypoints) > 5:
                    far_wp, far_cmd = upcoming_waypoints[5]
                    log_far_node_coords = (far_wp.transform.location.x, far_wp.transform.location.y)
                    log_far_node_cmd = far_cmd.name

            far_dist = is_vehicle_hazard(self.world, self.hero_vehicle, self._map, max_distance=25.0)
            far_dist = float(far_dist) if far_dist > 0.0 else -1.0

            if self._data_saver:
                depth = input_data['depth_front'][1]
                seg = input_data['instance_segmentation_front'][1]
                if depth.ndim == 3:
                    img_depth = depth[:, :, 0]
                else:
                    img_depth = depth  # уже 2D

                #img_depth = depth[:, :, 0]
                img_seg = seg[:, :, 0]

                data_to_save = {
                    "location": {
                        "x": vehicle_location.x,
                        "y": vehicle_location.y,
                        "z": vehicle_location.z
                    },
                    "speed": current_speed_m_s,
                    "near_node": log_near_node_coords,
                    "far_node": log_far_node_coords,
                    "near_cmd": log_near_node_cmd,
                    "far_cmd": log_far_node_cmd,
                    "is_red_light_present_log": self.is_red_light_present_log,
                    "is_stops_present_log": self.is_stops_present_log,
                    "far_distance_to_object": far_dist,
                    "target_control": {
                        "steer": float(bh_control.steer),
                        "throttle": float(bh_control.throttle),
                        "brake": float(bh_control.brake)
                    },
                    "timestamp": timestamp
                }

                self._data_saver.save_async(img_depth, img_seg, data_to_save)

            return bh_control

        def destroy(self):
            print("AutopilotAgent: Cleaning up agent.")
            if self.collision_sensor:
                self.collision_sensor.stop()
                self.collision_sensor.destroy()
                self.collision_sensor = None
            if self._data_saver:
                self._data_saver.shutdown()
                self._data_saver = None

