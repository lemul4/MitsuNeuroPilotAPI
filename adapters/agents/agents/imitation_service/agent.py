#!/usr/bin/env python

from __future__ import print_function

import math
import os
import weakref
import numpy as np
import torch
from torch.backends import cudnn
import queue
import carla
import threading
from collections import deque
from datetime import datetime

from infrastructure.carla.agents.navigation.behavior_agent import BehaviorAgent
from data.carla_autopilot_net import ImprovedCarlaAutopilotNet
from services.evaluation_service.scenario_runner.srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from services.evaluation_service.leaderboard.leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

from services.training_service.data_directly import DirectlyCarlaDatasetLoader

from .sensors import sensors
from .data_saver import save_data_async, save_worker
from .collizions import CollisionHandler
from .vehicle_utils import HazardDetector
import carla
from .constants import DEBUG, stats, SENSOR_CONFIG

def get_entry_point():
    return 'AutopilotAgent'

class AutopilotAgent(AutonomousAgent):
    """
    Autopilot agent that controls the ego vehicle using BehaviorAgent.
    """
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
        self._worker_thread = threading.Thread(target=lambda: save_worker(self), daemon=True)
        self._worker_thread.start()
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

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters.
        """
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
                route = []
                print(self._global_plan_world_coord)
                for transform, road_option in self._global_plan_world_coord:
                    print(transform)
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

            self.init_dataset_folders(self.dataset_save_path)
            self._worker_thread = threading.Thread(target=lambda: save_worker(self), daemon=True)
            self._worker_thread.start()
            self.initialized_data_saving = True

        self.init_privileged_sensors()

    def init_dataset_folders(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.subfolder_paths = []
        subfolders = [
            "depth_front", "instance_segmentation_front", "measurements"
        ]
        for sub in subfolders:
            path = os.path.join(output_dir, sub)
            os.makedirs(path, exist_ok=True)
            self.subfolder_paths.append(path)
        print(f"AutopilotAgent: Dataset folders initialized at {output_dir}")

    def init_privileged_sensors(self):
        bp_lib = self.world.get_blueprint_library()
        bp_collision = bp_lib.find('sensor.other.collision')
        self.is_collision = False
        self.collision_sensor = self.world.spawn_actor(bp_collision, carla.Transform(), attach_to=self.hero_vehicle)
        self.collision_sensor.listen(lambda event: CollisionHandler.on_collision(weakref.ref(self), event))
        print("AutopilotAgent: Collision sensor initialized.")

    def sensors(self):
        return sensors(self._sensor_data_config)

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
                log_near_node_cmd = near_cmd.value
                far_idx = min(3, len(upcoming_waypoints) - 1)
                if far_idx >= 0:
                    far_wp, far_cmd = upcoming_waypoints[far_idx]
                    log_far_node_coords = (far_wp.transform.location.x, far_wp.transform.location.y)
                    log_far_node_cmd = far_cmd.value

        self.steer_sequence.append(self.last_steer)
        self.throttle_sequence.append(self.last_throttle)
        self.brake_sequence.append(self.last_brake)
        self.speed_sequence.append(self.last_speed)
        self.light_sequence.append(self.last_light)

        self.is_red_light_present_log = 1 if self._agent.is_red_light_present_log else 0

        vehicle_list = self.world.get_actors().filter("*vehicle*")
        max_dist = 25
        distanse = HazardDetector.detect_vehicle_hazard(self, vehicle_list, max_dist)
        vehicle_detected_flag = 1 if distanse >= 0 else 0
        if vehicle_detected_flag != 1:
            distanse = max_dist

        vehicle_forward_vector = vehicle_transform.get_forward_vector()
        vehicle_heading_rad = math.atan2(vehicle_forward_vector.y, vehicle_forward_vector.x)

        vec_to_near_wp_x = log_near_node_coords[0] - vehicle_location.x
        vec_to_near_wp_y = log_near_node_coords[1] - vehicle_location.y
        angle_to_near_wp_rad = math.atan2(vec_to_near_wp_y, vec_to_near_wp_x)
        angle_diff_near_rad = angle_to_near_wp_rad - vehicle_heading_rad
        normalized_angle_near_rad = math.atan2(math.sin(angle_diff_near_rad), math.cos(angle_diff_near_rad))
        self.angle_to_near_waypoint_deg = math.degrees(normalized_angle_near_rad)

        vec_to_far_wp_x = log_far_node_coords[0] - vehicle_location.x
        vec_to_far_wp_y = log_far_node_coords[1] - vehicle_location.y
        angle_to_far_wp_rad = math.atan2(vec_to_far_wp_y, vec_to_far_wp_x)
        angle_diff_far_rad = angle_to_far_wp_rad - vehicle_heading_rad
        normalized_angle_far_rad = math.atan2(math.sin(angle_diff_far_rad), math.cos(angle_diff_far_rad))
        self.angle_to_far_waypoint_deg = math.degrees(normalized_angle_far_rad)

        depth_front_data = input_data['depth_front'][1]
        inst_seg_raw = input_data['instance_segmentation_front'][1]
        inst_seg_data = inst_seg_raw[:, :, :-1]

        measurement_data = {
            'timestamp': timestamp,

            'x': vehicle_location.x,
            'y': vehicle_location.y,

            'speed': current_speed_m_s,
            'theta': compass_rad,

            'near_node_x': log_near_node_coords[0],
            'near_node_y': log_near_node_coords[1],
            'near_command': log_near_node_cmd,

            'far_node_x': log_far_node_coords[0],
            'far_node_y': log_far_node_coords[1],
            'far_command': log_far_node_cmd,

            'angle_near': self.angle_to_near_waypoint_deg,
            'angle_far': self.angle_to_far_waypoint_deg,

            'steer': control.steer,
            'throttle': control.throttle,
            'brake': control.brake,

            'is_red_light_present': self.is_red_light_present_log,
            'vehicle_detected_flag': vehicle_detected_flag,
            'distanse': distanse,

            'steer_sequence': np.array(self.steer_sequence, dtype=np.float32).tolist(),
            'throttle_sequence': np.array(self.throttle_sequence, dtype=np.float32).tolist(),
            'brake_sequence': np.array(self.brake_sequence, dtype=np.float32).tolist(),
            'speed_sequence': np.array(self.speed_sequence, dtype=np.float32).tolist(),
            'light_sequence': np.array(self.light_sequence, dtype=np.float32).tolist(),
            'is_collision_event': self.is_collision,
        }

        input_data_model = {
            'depth_front_data': depth_front_data,
            'instance_segmentation_front_data': inst_seg_data,
            'measurement_data': measurement_data
        }

        dataset = DirectlyCarlaDatasetLoader(
            input_batch_data=[input_data_model],
            num_near_commands=7,
            num_far_commands=7,
            stats=stats
        )

        sample = dataset[0]

        depth = sample['depth'].unsqueeze(0).to(self.DEVICE)
        seg = sample['segmentation'].unsqueeze(0).to(self.DEVICE)
        hist = torch.stack([
            sample['speed_sequence'],
            sample['steer_sequence'],
            sample['throttle_sequence'],
            sample['brake_sequence'],
            sample['light_sequence']
        ], dim=-1).unsqueeze(0).to(self.DEVICE)
        cont = sample['cont_feats'].unsqueeze(0).to(self.DEVICE)
        sig = sample['signal_vec'].unsqueeze(0).to(self.DEVICE)
        near = sample['near_cmd_oh'].unsqueeze(0).to(self.DEVICE)
        far = sample['far_cmd_oh'].unsqueeze(0).to(self.DEVICE)

        with torch.no_grad():
            pred_steer, pred_throttle, pred_brake = self.model(depth, seg, hist, cont, sig, near, far)
            steer = pred_steer.item()
            throttle = pred_throttle.item()
            brake = pred_brake.item()

        if throttle > 0.1:
            brake = 0

        control.steer = steer
        control.throttle = throttle
        control.brake = brake

        d = False
        if d:
            print(f"Predicted steer: {steer:.4f}")
            print(f"Predicted throttle: {throttle:.4f}")
            print(f"Predicted brake: {brake:.4f}")
            measurement_data_updated = {
                'timestamp': timestamp,

                'x': vehicle_location.x,
                'y': vehicle_location.y,

                'speed': current_speed_m_s,
                'theta': compass_rad,

                'near_node_x': log_near_node_coords[0],
                'near_node_y': log_near_node_coords[1],
                'near_command': log_near_node_cmd,

                'far_node_x': log_far_node_coords[0],
                'far_node_y': log_far_node_coords[1],
                'far_command': log_far_node_cmd,

                'angle_near': self.angle_to_near_waypoint_deg,
                'angle_far': self.angle_to_far_waypoint_deg,

                'steer': control.steer,
                'throttle': control.throttle,
                'brake': control.brake,

                'is_red_light_present': self.is_red_light_present_log,
                'vehicle_detected_flag': vehicle_detected_flag,
                'distanse': distanse,

                'steer_sequence': np.array(self.steer_sequence, dtype=np.float32).tolist(),
                'throttle_sequence': np.array(self.throttle_sequence, dtype=np.float32).tolist(),
                'brake_sequence': np.array(self.brake_sequence, dtype=np.float32).tolist(),
                'speed_sequence': np.array(self.speed_sequence, dtype=np.float32).tolist(),
                'light_sequence': np.array(self.light_sequence, dtype=np.float32).tolist(),

                'is_collision_event': self.is_collision,
            }

            save_data_async(
                self,
                image_depth=depth_front_data,
                image_seg=inst_seg_data,
                data=measurement_data_updated
            )
        self.last_steer = float(control.steer)
        self.last_throttle = float(control.throttle)
        self.last_brake = float(control.brake)
        self.last_speed = float(current_speed_m_s)
        self.last_light = int(self.is_red_light_present_log)

        return control

    def set_global_plan(self, global_plan, clean=False):
        self._global_plan_world_coord = global_plan
        if self._agent:
            route = []
            map_api = CarlaDataProvider.get_map()
            for transform, road_option in global_plan:
                wp = map_api.get_waypoint(transform.location)
                route.append((wp, road_option))
            self._agent.set_global_plan(route, clean)
            self._route_assigned = True

    def destroy(self):
        if self.collision_sensor is not None:
            self.collision_sensor.stop()
            self.collision_sensor.destroy()
        self._save_queue.put(None)
        self._worker_thread.join()
        if self._agent:
            self._agent.done()
