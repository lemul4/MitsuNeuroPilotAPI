#!/usr/bin/env python

from __future__ import print_function
import math
import carla
import numpy as np
import queue
from collections import deque
from datetime import datetime
import os
import threading
import weakref

from infrastructure.carla.agents.navigation.behavior_agent import BehaviorAgent
from services.evaluation_service.scenario_runner.srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from services.evaluation_service.leaderboard.leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

from .constants import DEBUG, SENSOR_CONFIG
from .collizions import CollisionHandler
from .data_saver import DataSaver
from .sensors import Sensors
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
        self.data_saver = DataSaver()
        self.step = -1
        self.data_count = 0

        self.speed_sequence = deque(np.zeros(20), maxlen=20)
        self.brake_sequence = deque(np.zeros(20), maxlen=20)
        self.throttle_sequence = deque(np.zeros(20), maxlen=20)
        self.steer_sequence = deque(np.zeros(20), maxlen=20)
        self.light_sequence = deque(np.zeros(20), maxlen=20)

        self.last_brake = 0.0
        self.last_throttle = 0.0
        self.last_steer = 0.0
        self.last_speed = 0.0
        self.last_light = 0

        self.is_collision = False
        self.subfolder_paths = []

        self.hero_vehicle = None
        self._map = None

        self.behavior_name = "cautious"
        self.opt_dict = {}

        # Data saving
        self.data_saver = DataSaver()

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

        if not self.data_saver.initialized_data_saving:
            today = datetime.today()
            now = datetime.now()
            current_date = today.strftime("%b_%d_%Y")
            current_time = now.strftime("%H_%M_%S")
            time_info = f"{current_date}-{current_time}"

            map_name = self._map.name.split('/')[-1]

            dataset_save_path = os.path.join("dataset", "autopilot_behavior_data", map_name, time_info)
            self.data_saver.init_save_worker(dataset_save_path)

        self.collision_sensor = Sensors.init_collision_sensor(self.world, self.hero_vehicle, lambda event: CollisionHandler.on_collision(weakref.ref(self), event))

    def sensors(self):
        return Sensors.get_sensor_config()

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

        self.is_red_light_present_log = 1 if self._agent.is_red_light_present_log else 0

        self.steer_sequence.append(self.last_steer)
        self.throttle_sequence.append(self.last_throttle)
        self.brake_sequence.append(self.last_brake)
        self.speed_sequence.append(self.last_speed)
        self.light_sequence.append(self.last_light)

        if self.step % 1 == 0:
            vehicle_list = self.world.get_actors().filter("*vehicle*")
            max_dist = 25
            distanse = is_vehicle_hazard(self, vehicle_list, max_dist)
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

                'steer_sequence': list(self.steer_sequence),
                'throttle_sequence': list(self.throttle_sequence),
                'brake_sequence': list(self.brake_sequence),
                'speed_sequence': list(self.speed_sequence),
                'light_sequence': list(self.light_sequence),

                'is_collision_event': self.is_collision,
            }
            self.data_saver.save_data_async(
                image_depth=depth_front_data.copy(),
                image_seg=inst_seg_data.copy(),
                data=measurement_data
            )

        self.last_steer = float(control.steer)
        self.last_throttle = float(control.throttle)
        self.last_brake = float(control.brake)
        self.last_speed = float(current_speed_m_s)
        self.last_light = self.is_red_light_present_log
        self.is_collision = False
        return control

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

        if self._agent and self._initialized_behavior_agent:
            route = []
            map_api = CarlaDataProvider.get_map()
            for transform, road_option in self._global_plan_world_coord:
                wp = map_api.get_waypoint(transform.location)
                route.append((wp, road_option))
            self._agent.set_global_plan(route, False)
            print("AutopilotAgent: Global plan updated for already initialized BehaviorAgent.")

    def destroy(self):
        """Clean up resources"""
        print("AutopilotAgent: Destroying agent...")

        # Можно добавить завершение BehaviorAgent, если есть метод cleanup
        if self._agent:
            # В оригинальном коде было pass — можно оставить или добавить очистку, если есть
            pass

        # Удаление сенсора столкновений
        if hasattr(self, 'collision_sensor') and self.collision_sensor and self.collision_sensor.is_alive:
            self.collision_sensor.stop()
            self.collision_sensor.destroy()
            self.collision_sensor = None
            print("AutopilotAgent: Collision sensor destroyed.")

        # Завершение потока сохранения данных
        if self.data_saver and self.data_saver._worker_thread and self.data_saver._worker_thread.is_alive():
            self.data_saver.save_queue.put(None)  # Отправляем сигнал остановки воркеру
            self.data_saver.save_queue.join()  # Ждём выполнения всех заданий
            self.data_saver.worker_thread.join(timeout=5.0)
            if self.data_saver.worker_thread.is_alive():
                print("AutopilotAgent: Warning - Save worker thread did not terminate cleanly.")
            else:
                print("AutopilotAgent: Save worker thread terminated.")

        if self.debug:
            pass

        print("AutopilotAgent: Destroyed.")

