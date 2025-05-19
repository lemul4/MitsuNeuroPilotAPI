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
        self.last_light = 0
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
        self.light_sequence = deque(np.zeros(20), maxlen=20)
        self.angle_far_unnorm = 0.0
        self.is_red_light_present_log = 0
        self.is_stops_present_log = 0
        self.hero_vehicle = None
        self._map = None

        self.behavior_name = "cautious"
        self.opt_dict = {}

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
        """Initializes components needed for data saving that depend on the world/vehicle being ready."""
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

            # Получаем имя карты
            map_name = self._map.name.split('/')[-1]  # Убираем путь, оставляя только название карты

            # Формируем путь: dataset/autopilot_behavior_data/<map_name>/<timestamp>/
            self.dataset_save_path = os.path.join("dataset", "autopilot_behavior_data", map_name, time_info)

            # Запускаем поток сохранения и создаем нужные папки
            self._worker_thread = threading.Thread(target=self._save_worker, daemon=True)
            self._worker_thread.start()
            self.init_dataset_folders(self.dataset_save_path)
            self.initialized_data_saving = True

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
        self.is_collision = False
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
            {'type':'sensor.speedometer','reading_frequency': 10,'id':'speed'},
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
            distanse = self.is_vehicle_hazard(vehicle_list, max_dist)
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
            self.save_data_async(
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

    def save_data_async(self, image_depth, image_seg, data):
        """
        Save data asynchronously using worker thread.
        """
        idx = self.data_count
        paths = self.subfolder_paths
        task = (image_depth, image_seg, data, paths, idx)
        self._save_queue.put(task)
        self.data_count += 1

    def is_vehicle_hazard(self, vehicle_list=None, max_distance=25.0):
        """
        Проверяет наличие ближайшего транспортного средства впереди в текущей полосе
        в пределах заданного максимального расстояния.

        :param vehicle_list (list of carla.Vehicle, optional): Список объектов транспортных средств для проверки.
            Если None, используются все транспортные средства в сцене.
        :param max_distance (float, optional): Максимальное расстояние для проверки препятствий (в метрах).
            По умолчанию 20.0 метров.
        :return: Расстояние до ближайшего обнаруженного транспортного средства впереди в текущей полосе,
            или -1.0, если такое транспортное средство не найдено.
        """
        # Если список транспортных средств не предоставлен, получаем все транспортные средства из мира
        if vehicle_list is None:
            # Предполагается, что у объекта self есть доступ к миру CARLA через self.world
            if not hasattr(self, 'world') or self.world is None:
                print("Ошибка: Объект 'self' не имеет доступа к миру CARLA (world).")
                return -1.0
            vehicle_list = self.world.get_actors().filter("*vehicle*")

        # Устанавливаем максимальное расстояние, если оно не задано (хотя в сигнатуре функции есть значение по умолчанию)
        # Эта проверка может быть полезна, если функция вызывается без аргумента max_distance,
        # но значение по умолчанию в сигнатуре уже обеспечивает это. Оставляем для гибкости.

        # Предполагается, что у объекта self есть доступ к объекту hero_vehicle (управляемому агенту)
        if not hasattr(self, 'hero_vehicle') or self.hero_vehicle is None:
            print("Ошибка: Объект 'self' не имеет объекта hero_vehicle (управляемого агента).")
            return -1.0

        # Получаем информацию о текущем агенте
        ego_transform = self.hero_vehicle.get_transform()
        ego_location = ego_transform.location
        ego_forward_vector = ego_transform.get_forward_vector()  # Вектор направления агента
        # Предполагается, что у объекта self есть доступ к объекту карты CARLA через self._map
        if not hasattr(self, '_map') or self._map is None:
            print("Ошибка: Объект 'self' не имеет доступа к карте CARLA (_map).")
            return -1.0
        ego_wpt = self._map.get_waypoint(ego_location)  # Путевая точка агента

        # Инициализируем переменные для отслеживания ближайшего препятствия
        min_distance = float('inf')  # Изначально ставим бесконечность как минимальное расстояние
        nearest_vehicle_distance = -1.0  # Значение по умолчанию для возврата, если препятствие не найдено

        # Перебираем все транспортные средства в списке
        for target_vehicle in vehicle_list:
            # Пропускаем само транспортное средство агента
            if target_vehicle.id == self.hero_vehicle.id:
                continue

            target_transform = target_vehicle.get_transform()
            target_location = target_transform.location

            # Пропускаем транспортные средства, которые заведомо дальше максимального расстояния
            # Это предварительная проверка для оптимизации
            distance_to_target_raw = ego_location.distance(target_location)
            if distance_to_target_raw > max_distance:
                continue

            # Получаем путевую точку целевого транспортного средства.
            # Указываем lane_type=carla.LaneType.Any, чтобы получить ближайшую точку на дороге,
            # а затем уже проверяем совпадение полос.
            target_wpt = self._map.get_waypoint(target_location, lane_type=carla.LaneType.Any)

            # --- Проверка, находится ли целевое транспортное средство в той же полосе ---
            # Сравниваем road_id и lane_id путевых точек агента и целевого транспортного средства.
            # Это строгая проверка нахождение в *точно* той же полосе.
            if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id:
                continue  # Пропускаем, если не в той же полосе

            # --- Проверка, находится ли целевое транспортное средство впереди агента ---
            # Рассчитываем вектор от местоположения агента до местоположения цели
            vec_ego_to_target = target_location - ego_location  # Объекты carla.Location поддерживают вычитание

            # Используем скалярное произведение векторов для проверки, находится ли цель впереди
            # Скалярное произведение вектора направления агента и вектора к цели должно быть положительным.
            # Если транспортные средства находятся очень близко друг к другу, может быть погрешность,
            # но проверка дистанции > 0 (или небольшого эпсилон) поможет избежать деления на ноль
            # при нормализации вектора, если это потребуется для более точного углового анализа.
            dot_product = ego_forward_vector.dot(vec_ego_to_target)

            # Если скалярное произведение отрицательно, цель находится позади агента. Пропускаем.
            if dot_product < 0:
                continue

            # Дополнительная проверка на угол: убедимся, что цель не сильно в стороне, а действительно "впереди"
            # Рассчитаем расстояние (еще раз, или используем distance_to_target_raw)
            distance = ego_location.distance(
                target_location)  # Можно использовать уже рассчитанное distance_to_target_raw

            # Избегаем деления на ноль, если транспортные средства находятся очень близко
            if distance > 1e-4:
                # Нормализуем вектор от агента к цели, чтобы получить только направление
                vec_ego_to_target_normalized = vec_ego_to_target / distance
                # Скалярное произведение двух единичных векторов равно косинусу угла между ними
                dot_product_normalized = ego_forward_vector.dot(vec_ego_to_target_normalized)

                # Ограничиваем значение dot_product_normalized диапазоном [-1, 1]
                # для предотвращения ошибок arccos из-за погрешностей вычислений с плавающей точкой
                dot_product_normalized = np.clip(dot_product_normalized, -1.0, 1.0)

                # Рассчитываем угол в радианах, затем переводим в градусы
                angle_rad = np.arccos(dot_product_normalized)
                angle_deg = np.degrees(angle_rad)

                # Определим допустимый угол для "впереди" (например, 45 градусов)
                # Это создает конус обзора перед агентом.
                ahead_angle_threshold = 45.0  # Можно настроить этот порог

                # Если угол больше порога, транспортное средство находится слишком в стороне. Пропускаем.
                if angle_deg > ahead_angle_threshold:
                    continue
            # Если distance <= 1e-4, транспортные средства очень близко или совпадают, считаем их "впереди"
            # и потенциальным препятствием, если они в той же полосе.

            # --- Проверка максимального расстояния (повторно, для ясности и после угловых проверок) ---
            # Транспортные средства дальше max_distance были пропущены на предыдущем шаге,
            # но явная проверка здесь делает логику более понятной после фильтрации по полосе и углу.
            if distance > max_distance:
                continue

            # Если все проверки пройдены (в той же полосе, впереди, в пределах max_distance)
            # и это транспортное средство ближе, чем ранее найденное ближайшее
            if distance < min_distance:
                min_distance = distance
                nearest_vehicle_distance = distance  # Обновляем возвращаемое расстояние

        # Возвращаем расстояние до ближайшего найденного транспортного средства, или -1.0, если никого не найдено
        return nearest_vehicle_distance

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
                depth_vis_uint16 = img_depth.astype(np.uint16)
                cv2.imwrite(os.path.join(paths[0], f"{idx:06d}.png"), depth_vis_uint16)

                cv2.imwrite(os.path.join(paths[1], f"{idx:06d}.png"),img_seg)

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
            self._agent.set_global_plan(route, False)
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