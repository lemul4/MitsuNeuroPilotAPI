#!/usr/bin/env python3
import math
import os
import queue
import threading
import weakref
from collections import deque
from datetime import datetime

import carla
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from run_scenario import router as scenario_router

from infrastructure.carla.agents.navigation.behavior_agent import BehaviorAgent
from services.evaluation_service.scenario_runner.srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from collizions import CollisionHandler
from sensors import Sensors
from vehicle_utils import is_vehicle_hazard
from data_saver import DataSaver  # адаптированный модуль сохранения
from logger import setup_logger
from car import spawn_hero_vehicle
logger = setup_logger()

DEBUG = False
SENSOR_CONFIG = Sensors.get_sensor_config()

app = FastAPI()
app.include_router(scenario_router)


# Pydantic модели для валидации входных данных
class SensorIMU(BaseModel):
    imu_values: list  # например [ax, ay, az, gx, gy, gz, compass]


class SensorInput(BaseModel):
    speed: float
    imu: list  # формат: [timestamp, [ax, ay, az, gx, gy, gz, compass]]
    depth_front: list  # 2-ой элемент — np.ndarray (будем передавать base64, пока упрощаем)
    instance_segmentation_front: list  # аналогично


class ControlOutput(BaseModel):
    steer: float
    throttle: float
    brake: float
    hand_brake: bool = False
    reverse: bool = False


class AutopilotAgentMicroservice:
    def __init__(self, carla_host='localhost', carla_port=2000):
        logger.info("AutopilotAgent: initialized")
        self._agent = None
        self._initialized_behavior_agent = False
        self.step = -1
        logger.info("Connecting to CARLA at %s:%d", carla_host, carla_port)
        self.client = carla.Client(carla_host, carla_port)
        self.client.set_timeout(10.0)

        self.world = self.client.get_world()
        CarlaDataProvider.set_world(self.world)  # установим мир в провайдер
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

        self.hero_vehicle = self.hero_vehicle = spawn_hero_vehicle(self.client)
        self.world = None
        self._map = None

        self.behavior_name = "cautious"
        self.opt_dict = {}

        self.data_saver = DataSaver()
        self.collision_sensor = None

    def initialize_behavior_agent(self):
        logger.info("Initializing AutopilotAgent...")

        self.hero_vehicle = None
        for actor in CarlaDataProvider.get_world().get_actors():
            if actor.attributes.get('role_name', '') == 'hero':
                self.hero_vehicle = actor
                break

        if not self.hero_vehicle:
            raise RuntimeError("Hero vehicle not found in CARLA world")

        # Создаём BehaviorAgent с найденным героем
        self._agent = BehaviorAgent(self.hero_vehicle, behavior=self.behavior_name, opt_dict=self.opt_dict)

        # Устанавливаем глобальный план
        route = []
        if hasattr(self, '_global_plan_world_coord') and self._global_plan_world_coord:
            map_api = CarlaDataProvider.get_map()
            for transform, road_option in self._global_plan_world_coord:
                wp = map_api.get_waypoint(transform.location)
                route.append((wp, road_option))
            self._agent.set_global_plan(route, False)

        self._initialized_behavior_agent = True

        self.world = self.hero_vehicle.get_world()
        self._map = self.world.get_map()

        self.collision_sensor = Sensors.init_collision_sensor(
            self.world,
            self.hero_vehicle,
            lambda event: CollisionHandler.on_collision(weakref.ref(self), event)
        )

        # Запускаем воркер сохранения данных
        if not self.data_saver.initialized_data_saving:
            today = datetime.today()
            now = datetime.now()
            current_date = today.strftime("%b_%d_%Y")
            current_time = now.strftime("%H_%M_%S")
            time_info = f"{current_date}-{current_time}"

            map_name = self._map.name.split('/')[-1]
            dataset_save_path = os.path.join("dataset", "autopilot_behavior_data", map_name, time_info)
            self.data_saver.init_save_worker(dataset_save_path)
            logger.info("Agent successfully initialized.")

    def run_step(self, input_data: dict, timestamp: float) -> carla.VehicleControl:
        logger.info(f"Running step at timestamp: {timestamp}")
        self.step += 1

        if not self._initialized_behavior_agent:
            self.initialize_behavior_agent()
        logger.debug("Calling behavior_agent.run_step...")
        control = self._agent.run_step(debug=DEBUG)
        logger.debug(f"Control command issued: {control}")
        current_speed_m_s = input_data['speed']
        imu_sensor_data = input_data['imu'][1]  # например [ax, ay, az, gx, gy, gz, compass]
        compass_rad = imu_sensor_data[-1]

        vehicle_transform = self.hero_vehicle.get_transform()
        vehicle_location = vehicle_transform.location

        log_near_node_coords, log_near_node_cmd = (None, None)
        log_far_node_coords, log_far_node_cmd = (None, None)

        if self._agent.get_local_planner():
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

        self.is_red_light_present_log = 1 if getattr(self._agent, 'is_red_light_present_log', False) else 0

        self.steer_sequence.append(self.last_steer)
        self.throttle_sequence.append(self.last_throttle)
        self.brake_sequence.append(self.last_brake)
        self.speed_sequence.append(self.last_speed)
        self.light_sequence.append(self.last_light)

        # Анализ опасных машин вокруг
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
        angle_to_near_waypoint_deg = math.degrees(normalized_angle_near_rad)

        vec_to_far_wp_x = log_far_node_coords[0] - vehicle_location.x
        vec_to_far_wp_y = log_far_node_coords[1] - vehicle_location.y
        angle_to_far_wp_rad = math.atan2(vec_to_far_wp_y, vec_to_far_wp_x)
        angle_diff_far_rad = angle_to_far_wp_rad - vehicle_heading_rad
        normalized_angle_far_rad = math.atan2(math.sin(angle_diff_far_rad), math.cos(angle_diff_far_rad))
        angle_to_far_waypoint_deg = math.degrees(normalized_angle_far_rad)


        depth_front_data = np.array(input_data['depth_front'])
        inst_seg_raw = np.array(input_data['instance_segmentation_front'])
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
            'angle_near': angle_to_near_waypoint_deg,
            'angle_far': angle_to_far_waypoint_deg,
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

    def destroy(self):
        print("AutopilotAgent: Destroying agent...")

        if self._agent:
            pass

        if self.collision_sensor and self.collision_sensor.is_alive:
            self.collision_sensor.stop()
            self.collision_sensor.destroy()
            self.collision_sensor = None
            print("AutopilotAgent: Collision sensor destroyed.")

        if self.data_saver and self.data_saver._worker_thread and self.data_saver._worker_thread.is_alive():
            self.data_saver.save_queue.put(None)
            self.data_saver.save_queue.join()
            self.data_saver._worker_thread.join(timeout=5.0)
            if self.data_saver._worker_thread.is_alive():
                print("AutopilotAgent: Warning - Save worker thread did not terminate cleanly.")
            else:
                print("AutopilotAgent: Save worker thread terminated.")

        print("AutopilotAgent: Destroyed.")


agent = AutopilotAgentMicroservice()


@app.post("/run_step/", response_model=ControlOutput)
async def run_step_api(sensor_input: SensorInput):
    try:
        # Здесь нужно декодировать из входных форматов (например, base64) в numpy, если нужно.
        # В твоём оригинале — numpy arrays приходят из CARLA напрямую, здесь нужно адаптировать по входу.
        # Для примера: считаем, что данные уже numpy arrays (требует модификации клиента).
        logger.info(f"API called with sensor_input: {sensor_input}")
        control = agent.run_step(sensor_input.dict(), datetime.now().timestamp())
        return ControlOutput(
            steer=control.steer,
            throttle=control.throttle,
            brake=control.brake,
            hand_brake=control.hand_brake,
            reverse=control.reverse
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("shutdown")
def shutdown_event():
    agent.destroy()


if __name__ == "__main__":
    # Параметры подключения, можно из конфига
    carla_host = "localhost"
    carla_port = 2000

    agent = AutopilotAgentMicroservice(carla_host, carla_port)

    uvicorn.run(app, host="0.0.0.0", port=8002)