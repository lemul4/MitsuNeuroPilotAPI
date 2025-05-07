#!/usr/bin/env python

from __future__ import print_function

import math
import carla
from agents.navigation.behavior_agent import BehaviorAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from agent_utils import base_utils
import os
import weakref
import numpy as np
import cv2
import json
import threading
import queue
from collections import deque
from datetime import datetime

DEBUG = False
SENSOR_CONFIG = {
    'width': 592,
    'height': 333,
    'fov': 135
}

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
        self._worker_thread = None
        self.is_collision = False
        self.subfolder_paths = []

        self.last_brake = 0.0
        self.last_throttle = 0.0
        self.last_steer = 0.0
        self.last_speed = 0.0
        self.speed_sequence = deque(np.zeros(20), maxlen=20)
        self.brake_sequence = deque(np.zeros(20), maxlen=20)
        self.throttle_sequence = deque(np.zeros(20), maxlen=20)
        self.steer_sequence = deque(np.zeros(20), maxlen=20)
        self.angle_far_unnorm = 0.0
        self.is_red_light_present_log = 0
        self.is_stops_present_log = 0
        self.hero_vehicle = None

        self.behavior_name = "cautious"
        self.opt_dict = {}

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters.
        """
        self.track = Track.SENSORS
        self._sensor_data_config = SENSOR_CONFIG
        self.config_path = path_to_conf_file

        if not self.initialized_data_saving:
            today = datetime.today()
            now = datetime.now()
            current_date = today.strftime("%b_%d_%Y")
            current_time = now.strftime("%H_%M_%S")
            time_info = f"/{current_date}-{current_time}/"
            self.dataset_save_path = os.path.join("dataset/autopilot_behavior_data" + time_info)

            self._worker_thread = threading.Thread(target=self._save_worker, daemon=True)
            self._worker_thread.start()
            self.init_dataset_folders(self.dataset_save_path)
            self.initialized_data_saving = True

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
                self._agent.set_global_plan(route)
                print("AutopilotAgent: Global plan set for BehaviorAgent.")
            else:
                print("AutopilotAgent: Warning - Global plan not available when initializing BehaviorAgent.")

            self._initialized_behavior_agent = True
            return True
        return True

    def _init_data_saving_components(self):
        """Initializes components needed for data saving that depend on the world/vehicle being ready."""
        if not self.hero_vehicle:
            self.hero_vehicle = CarlaDataProvider.get_hero_actor()
            if not self.hero_vehicle:
                print("AutopilotAgent: Hero vehicle not found. Cannot initialize privileged sensors.")
                return

        self.world = self.hero_vehicle.get_world()
        self.init_privileged_sensors()

        if self.debug:
            pass

    def init_dataset_folders(self, output_dir):
        """
        Initialize dataset folders for saving data.
        """
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
        """
        Initialize privileged sensors for data collection.
        """
        bp_lib = self.world.get_blueprint_library()
        bp_collision = bp_lib.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(bp_collision, carla.Transform(), attach_to=self.hero_vehicle)
        self.collision_sensor.listen(lambda event: AutopilotAgent._on_collision(weakref.ref(self), event))
        print("AutopilotAgent: Collision sensor initialized.")

    def sensors(self):
        """
        Define the sensor suite required by the agent.
        """
        sens = [
            {'type': 'sensor.camera.depth', 'x':1.3,'y':0.0,'z':2.3,
             'roll':0.0,'pitch':0.0,'yaw':0.0,
             'width':self._sensor_data_config['width'],'height':self._sensor_data_config['height'],'fov':self._sensor_data_config['fov'],
             'sensor_tick': 0.1,
             'id':'depth_front'},
            {'type': 'sensor.camera.instance_segmentation', 'x':1.3,'y':0.0,'z':2.3,
             'roll':0.0,'pitch':0.0,'yaw':0.0,
             'width':self._sensor_data_config['width'],'height':self._sensor_data_config['height'],'fov':self._sensor_data_config['fov'],
             'sensor_tick': 0.1,
             'id':'instance_segmentation_front'},
            {'type':'sensor.other.imu','x':0.0,'y':0.0,'z':0.0,'roll':0.0,'pitch':0.0,'yaw':0.0,
             'sensor_tick':0.1,'id':'imu'},
            {'type':'sensor.speedometer','reading_frequency': 20,'id':'speed'}
        ]
        return sens

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

        control = self._agent.run_step(debug=self.debug)

        is_agent_affected_by_red_light = False
        if hasattr(self._agent, '_last_traffic_light') and \
                self._agent._last_traffic_light is not None and \
                hasattr(self._agent._last_traffic_light, 'state') and \
                self._agent._last_traffic_light.state == carla.TrafficLightState.Red:
            is_agent_affected_by_red_light = True
        self.is_red_light_present_log = 1 if is_agent_affected_by_red_light else 0

        is_stop_sign_affecting_ego = self._is_stop_sign_affecting_ego_for_logging()
        self.is_stops_present_log = 1 if is_stop_sign_affecting_ego else 0

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

        self.last_steer = float(control.steer)
        self.last_throttle = float(control.throttle)
        self.last_brake = float(control.brake)
        self.last_speed = float(current_speed_m_s)

        if self.step % 10 == 0:
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
                'z': vehicle_location.z,

                'speed': current_speed_m_s,
                'theta': compass_rad,

                'near_node_x': log_near_node_coords[0] if log_near_node_coords else None,
                'near_node_y': log_near_node_coords[1] if log_near_node_coords else None,
                'near_command': log_near_node_cmd,

                'far_node_x': log_far_node_coords[0] if log_far_node_coords else None,
                'far_node_y': log_far_node_coords[1] if log_far_node_coords else None,
                'far_command': log_far_node_cmd,

                'angle_near': self.angle_to_near_waypoint_deg,
                'angle_far': self.angle_to_far_waypoint_deg,

                'steer': control.steer,
                'throttle': control.throttle,
                'brake': control.brake,

                'is_red_light_present': self.is_red_light_present_log,

                'steer_sequence': np.array(self.steer_sequence, dtype=np.float32).tolist(),
                'throttle_sequence': np.array(self.throttle_sequence, dtype=np.float32).tolist(),
                'brake_sequence': np.array(self.brake_sequence, dtype=np.float32).tolist(),
                'speed_sequence': np.array(self.speed_sequence, dtype=np.float32).tolist(),

                'is_collision_event': self.is_collision,
            }
            self.save_data_async(
                image_depth=depth_front_data.copy(),
                image_seg=inst_seg_data.copy(),
                data=measurement_data
            )
        self.is_collision = False
        return control

    def _is_stop_sign_affecting_ego_for_logging(self):
        """
        Check if stop sign is affecting ego vehicle for logging purposes.
        """
        if not self.hero_vehicle or not self.world:
            return False

        if base_utils:
            try:
                all_stop_signs = self.world.get_actors().filter('*traffic.stop*')
                nearby_relevant_stops = base_utils.get_nearby_lights(self.hero_vehicle, all_stop_signs)
                return len(nearby_relevant_stops) > 0
            except Exception as e:
                if self.debug:
                    print(f"AutopilotAgent: Error using base_utils.get_nearby_lights for stop signs: {e}")
                return self._manual_stop_sign_check_for_logging()
        else:
            return self._manual_stop_sign_check_for_logging()

    def _manual_stop_sign_check_for_logging(self):
        """
        Manual check for stop signs affecting ego vehicle.
        """
        if not self.hero_vehicle or not self.world:
            return False

        stop_signs = self.world.get_actors().filter('*.stop')
        ego_transform = self.hero_vehicle.get_transform()
        ego_location = ego_transform.location
        ego_fwd_vec = ego_transform.get_forward_vector()
        vehicle_waypoint = CarlaDataProvider.get_map().get_waypoint(ego_location)

        for stop_sign in stop_signs:
            stop_location = stop_sign.get_location()
            distance_to_stop = ego_location.distance(stop_location)

            if distance_to_stop < 25.0:
                vec_to_stop = (stop_location - ego_location).make_unit_vector()
                if ego_fwd_vec.dot(vec_to_stop) > 0.707:
                    stop_waypoint = CarlaDataProvider.get_map().get_waypoint(stop_location, project_to_road=True,
                                                                             lane_type=carla.LaneType.Any)
                    if stop_waypoint.road_id == vehicle_waypoint.road_id or vehicle_waypoint.is_junction:
                        return True
        return False

    def save_data_async(self, image_depth, image_seg, data):
        """
        Save data asynchronously using worker thread.
        """
        idx = self.data_count
        paths = self.subfolder_paths
        task = (image_depth, image_seg, data, paths, idx)
        self._save_queue.put(task)
        self.data_count += 1

    def _save_worker(self):
        """
        Worker thread for saving data.
        """
        while True:
            task = self._save_queue.get()
            if task is None:
                self._save_queue.task_done()
                break
            img_depth, img_seg, data, paths, idx = task
            try:
                max_depth_val = np.nanmax(img_depth)
                if max_depth_val > 0:
                    depth_vis = (img_depth / max_depth_val * 65535).astype(np.uint16)
                else:
                    depth_vis = np.zeros_like(img_depth, dtype=np.uint16)
                cv2.imwrite(os.path.join(paths[0], f"{idx:06d}.png"), depth_vis)

                seg_vis = img_seg
                cv2.imwrite(os.path.join(paths[1], f"{idx:06d}.png"), seg_vis)

                with open(os.path.join(paths[2], f"{idx:06d}.json"), 'w+', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"AutopilotAgent: Error saving task {idx} to {paths}: {e}")
            finally:
                self._save_queue.task_done()

    @staticmethod
    def _on_collision(weak_self, event):
        """Callback on collision sensor event"""
        self = weak_self()
        if not self:
            return
        self.is_collision = True

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

        if self._agent and self._initialized_behavior_agent:
            route = []
            map_api = CarlaDataProvider.get_map()
            for transform, road_option in self._global_plan_world_coord:
                wp = map_api.get_waypoint(transform.location)
                route.append((wp, road_option))
            self._agent.set_global_plan(route)
            print("AutopilotAgent: Global plan updated for already initialized BehaviorAgent.")

    def destroy(self):
        """Clean up resources"""
        print("AutopilotAgent: Destroying agent...")
        if self._agent:
            pass

        if hasattr(self, 'collision_sensor') and self.collision_sensor and self.collision_sensor.is_alive:
            self.collision_sensor.stop()
            self.collision_sensor.destroy()
            self.collision_sensor = None
            print("AutopilotAgent: Collision sensor destroyed.")

        if self._worker_thread and self._worker_thread.is_alive():
            self._save_queue.put(None)
            self._save_queue.join()
            self._worker_thread.join(timeout=5.0)
            if self._worker_thread.is_alive():
                print("AutopilotAgent: Warning - Save worker thread did not terminate cleanly.")
            else:
                print("AutopilotAgent: Save worker thread terminated.")
        if self.debug:
            pass
        print("AutopilotAgent: Destroyed.")
