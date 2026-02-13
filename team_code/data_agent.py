"""
Child of the autopilot that additionally runs data collection and storage.
Adapted for 4 cameras setup + Dedicated High-Res Depth/Semantics.
No augmentation.
Restricted FOV (110 degrees) for BEV and Bounding Boxes.
FIXED: KeyError 'rendered' safely handled.
FIXED: BBox FOV Polygon alignment (Now looking Forward along X-axis).
FIXED: Visual Mask alignment (Now looking Right as requested).
"""

import cv2
import carla
import random
import torch
import numpy as np
import json
import os
import gzip
import math
from shapely.geometry import Polygon, Point
from shapely import affinity
from pathlib import Path

from autopilot import AutoPilot
import transfuser_utils as t_u

from birds_eye_view.chauffeurnet import ObsManager
from birds_eye_view.run_stop_sign import RunStopSign
from PIL import Image

from agents.tools.misc import (is_within_distance, get_trafficlight_trigger_location, compute_distance)

from agents.navigation.local_planner import LocalPlanner


def get_entry_point():
    return 'DataAgent'


class DataAgent(AutoPilot):
    """
    Child of the autopilot that additionally runs data collection and storage.
    """

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index, traffic_manager=None)
        self.weather_tmp = None
        self.step_tmp = 0

        self.tm = traffic_manager

        self.scenario_name = Path(path_to_conf_file).parent.name
        self.cutin_vehicle_starting_position = None
        
        # --- KONFIGURACJA KAMER (RGB) ---
        self.rgb_camera_ids = ['CAM_1', 'CAM_2', 'CAM_3', 'CAM_4']
        self.aux_id = 'CAM_AUX' # ID для глубины и сегментации

        # Настройки FOV фильтрации
        self.target_fov = 110.0
        self.half_fov_rad = np.deg2rad(self.target_fov / 2.0)

        if self.save_path is not None and self.datagen:
            # Создаем папки для 4-х RGB камер
            for cam_id in self.rgb_camera_ids:
                (self.save_path / 'rgb' / cam_id).mkdir(parents=True, exist_ok=True)
            
            # Semantics и Depth (отдельная камера высокого разрешения)
            (self.save_path / 'semantics' / self.aux_id).mkdir(parents=True, exist_ok=True)
            (self.save_path / 'depth' / self.aux_id).mkdir(parents=True, exist_ok=True)

            # BEV и Boxes общие
            (self.save_path / 'bev_semantics').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'boxes').mkdir(parents=True, exist_ok=True)

        self.tmp_visu = int(os.environ.get('TMP_VISU', 0))

        self._active_traffic_light = None
        self.last_ego_transform = None

        # Кэш для полигона FOV (относительно эго в 0,0)
        # Создаем большой треугольник, покрывающий радиус сохранения
        r = self.config.bb_save_radius
        
        # Точки для конуса. 
        # В CARLA: X - Вперед, Y - Вправо.
        # Строим треугольник вершиной в 0,0 и основанием на расстоянии r вдоль X.
        # Это создает сектор смотрящий ВПЕРЕД.
        y_max = r * np.tan(self.half_fov_rad)
        
        # Полигон смотрит ВПЕРЕД (вдоль оси X). Ротация не нужна.
        self.fov_polygon = Polygon([(0, 0), (r, y_max), (r, -y_max)])

    def _init(self, hd_map):
        super()._init(hd_map)
        if self.datagen:
            self.shuffle_weather()

        # Конфиг для BEV (карты)
        obs_config = {
            'width_in_pixels': self.config.lidar_resolution_width * 2,
            'pixels_ev_to_bottom': self.config.lidar_resolution_height,
            'pixels_per_meter': 4.0,
            'history_idx': [-1],
            'scale_bbox': True,
            'scale_mask_col': 1.0,
            'map_folder': 'maps_4ppm_cv'
        }

        self.stop_sign_criteria = RunStopSign(self._world)
        self.ss_bev_manager = ObsManager(obs_config, self.config)
        self.ss_bev_manager.attach_ego_vehicle(self._vehicle, criteria_stop=self.stop_sign_criteria)

        self._local_planner = LocalPlanner(self._vehicle, opt_dict={}, map_inst=self.world_map)

    def sensors(self):
        result = super().sensors()

        if self.save_path is not None and (self.datagen or self.tmp_visu):
            # --- 1. RGB CAMERAS CONFIGURATION (720p) ---
            # Базовые координаты плоскости
            base_x = 0.4
            base_z = 1.42
            
            # Характеристики
            cam_configs_data = [
                {"id": "CAM_1", "y": -0.075, "fov": 90.028}, # Left
                {"id": "CAM_2", "y": -0.025, "fov": 50.055}, # Mid-L
                {"id": "CAM_3", "y": +0.025, "fov": 50.055}, # Mid-R
                {"id": "CAM_4", "y": +0.075, "fov": 90.028}  # Right
            ]

            # Генерируем 4 RGB сенсора
            for conf in cam_configs_data:
                result.append({
                    'type': 'sensor.camera.rgb',
                    'x': base_x, 'y': conf['y'], 'z': base_z,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 1280, 'height': 720, 'fov': conf['fov'], # 720p
                    'id': conf['id']
                })
            
            # --- 2. DEPTH & SEGMENTATION CONFIGURATION (1080p, FOV 110) ---
            aux_x, aux_y, aux_z = 0.4, 0.0, 1.42
            aux_fov = 110
            aux_w, aux_h = 1920, 1080

            result.append({
                'type': 'sensor.camera.semantic_segmentation',
                'x': aux_x, 'y': aux_y, 'z': aux_z,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': aux_w, 'height': aux_h, 'fov': aux_fov,
                'id': self.aux_id + '_semantics'
            })
            
            result.append({
                'type': 'sensor.camera.depth',
                'x': aux_x, 'y': aux_y, 'z': aux_z,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': aux_w, 'height': aux_h, 'fov': aux_fov,
                'id': self.aux_id + '_depth'
            })

        return result

    def _apply_bev_mask(self, bev_image):
        """
        Applies a mask to the BEV image to keep only the 110-degree FOV.
        Rotated to look RIGHT (0 degrees in OpenCV).
        """
        # Гарантируем, что тип данных uint8
        if bev_image.dtype != np.uint8:
            bev_image = bev_image.astype(np.uint8)

        h, w = bev_image.shape[:2]
        
        # Определяем положение Ego на карте BEV (середина по ширине, внизу по высоте)
        ego_x = w // 2
        ego_y = h - self.config.lidar_resolution_height

        # Создаем маску (черный фон)
        mask = np.zeros((h, w), dtype=np.uint8)

        # Вычисляем координаты углов треугольника для FOV
        length = max(h, w) * 2 # Длина лучей
        
        # В OpenCV угол 0 градусов смотрит ВПРАВО.
        # Нам нужно смотреть НАПРАВО с FOV 110.
        # Диапазон от -55 до +55 градусов.
        angle_left_rad = np.deg2rad(-55)
        angle_right_rad = np.deg2rad(55)

        pt1 = (ego_x, ego_y) # Центр
        pt2_x = int(ego_x + length * np.cos(angle_left_rad))
        pt2_y = int(ego_y + length * np.sin(angle_left_rad))
        pt3_x = int(ego_x + length * np.cos(angle_right_rad))
        pt3_y = int(ego_y + length * np.sin(angle_right_rad))

        triangle_cnt = np.array([pt1, (pt2_x, pt2_y), (pt3_x, pt3_y)], np.int32)

        # Рисуем белый сектор на маске
        cv2.fillPoly(mask, [triangle_cnt], 255)

        # Применяем маску
        masked_img = cv2.bitwise_and(bev_image, bev_image, mask=mask)
            
        return masked_img

    def tick(self, input_data):
        result = {}

        if self.save_path is not None and (self.datagen or self.tmp_visu):
            # 1. Обработка RGB
            for cam_id in self.rgb_camera_ids:
                if cam_id in input_data:
                    result[cam_id] = input_data[cam_id][1][:, :, :3]
                else:
                    result[cam_id] = None

            # 2. Обработка Depth и Semantics
            if (self.aux_id + '_depth') in input_data:
                depth_raw = input_data[self.aux_id + '_depth'][1][:, :, :3]
                result[self.aux_id + '_depth'] = (t_u.convert_depth(depth_raw) * 255.0 + 0.5).astype(np.uint8)
            else:
                result[self.aux_id + '_depth'] = None

            if (self.aux_id + '_semantics') in input_data:
                result[self.aux_id + '_semantics'] = input_data[self.aux_id + '_semantics'][1][:, :, 2]
            else:
                result[self.aux_id + '_semantics'] = None

            rgb_visu = result['CAM_2'] if result['CAM_2'] is not None else np.zeros((720, 1280, 3), dtype=np.uint8)
        else:
            rgb_visu = None
            for cam_id in self.rgb_camera_ids:
                result[cam_id] = None
            result[self.aux_id + '_depth'] = None
            result[self.aux_id + '_semantics'] = None

        # Bounding Boxes (без лидара, но с геометрическим фильтром FOV)
        bounding_boxes = self.get_bounding_boxes(lidar=None)

        self.stop_sign_criteria.tick(self._vehicle)
        bev_data = self.ss_bev_manager.get_observation(self.close_traffic_lights)
        
        # --- ПРИМЕНЕНИЕ МАСКИ К BEV ---
        # 1. Основные семантические классы (нужны для сохранения)
        bev_semantics = bev_data['bev_semantic_classes']
        bev_semantics_masked = self._apply_bev_mask(bev_semantics)

        result.update({
            'bev_semantics': bev_semantics_masked,
            'bounding_boxes': bounding_boxes,
        })
        
        # 2. Визуализация (нужна только если включен режим)
        if self.tmp_visu:
            # ИСПРАВЛЕНИЕ: Проверяем наличие ключа 'rendered', прежде чем обращаться к нему
            if 'rendered' in bev_data:
                rendered_bev = bev_data['rendered']
                rendered_bev_masked = self._apply_bev_mask(rendered_bev)
                self.visualuize(rendered_bev_masked, rgb_visu)

        return result

    @torch.inference_mode()
    def run_step(self, input_data, timestamp, sensors=None, plant=False):
        self.step_tmp += 1
        control = super().run_step(input_data, timestamp, plant=plant)
        tick_data = self.tick(input_data)

        if self.step % self.config.data_save_freq == 0:
            if self.save_path is not None and self.datagen:
                self.save_sensors(tick_data)

        self.last_ego_transform = self._vehicle.get_transform()

        if plant:
            return {**tick_data, **control}
        else:
            return control

    def _get_night_mode(self, weather):
        """Check wheather or not the street lights need to be turned on"""
        SUN_ALTITUDE_THRESHOLD_1 = 15
        SUN_ALTITUDE_THRESHOLD_2 = 165
        CLOUDINESS_THRESHOLD = 80
        FOG_THRESHOLD = 40
        COMBINED_THRESHOLD = 10

        altitude_dist = weather.sun_altitude_angle - SUN_ALTITUDE_THRESHOLD_1
        altitude_dist = min(altitude_dist, SUN_ALTITUDE_THRESHOLD_2 - weather.sun_altitude_angle)
        cloudiness_dist = CLOUDINESS_THRESHOLD - weather.cloudiness
        fog_density_dist = FOG_THRESHOLD - weather.fog_density

        if altitude_dist < 0 or cloudiness_dist < 0 or fog_density_dist < 0:
            return True

        joined_threshold = int(altitude_dist < COMBINED_THRESHOLD)
        joined_threshold += int(cloudiness_dist < COMBINED_THRESHOLD)
        joined_threshold += int(fog_density_dist < COMBINED_THRESHOLD)

        if joined_threshold >= 2:
            return True

        return False

    def shuffle_weather(self):
        # change weather for visual diversity
        if self.weather_tmp is None:
            t = carla.WeatherParameters
            options = dir(t)[:22]
            chosen_preset = random.choice(options)
            self.chosen_preset = chosen_preset
            weather = t.__getattribute__(t, chosen_preset)
            self.weather_tmp = weather

        self._world.set_weather(self.weather_tmp)

        # night mode
        vehicles = self._world.get_actors().filter('*vehicle*')
        if self._get_night_mode(weather):
            for vehicle in vehicles:
                vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))
        else:
            for vehicle in vehicles:
                vehicle.set_light_state(carla.VehicleLightState.NONE)

    def save_sensors(self, tick_data):
        frame = self.step // self.config.data_save_freq

        for cam_id in self.rgb_camera_ids:
            if tick_data[cam_id] is not None:
                cv2.imwrite(str(self.save_path / 'rgb' / cam_id / (f'{frame:04}.jpg')), tick_data[cam_id])
            
        if tick_data[self.aux_id + '_semantics'] is not None:
            cv2.imwrite(str(self.save_path / 'semantics' / self.aux_id / (f'{frame:04}.png')), tick_data[self.aux_id + '_semantics'])
        
        if tick_data[self.aux_id + '_depth'] is not None:
            cv2.imwrite(str(self.save_path / 'depth' / self.aux_id / (f'{frame:04}.png')), tick_data[self.aux_id + '_depth'])

        if tick_data['bev_semantics'] is not None:
            cv2.imwrite(str(self.save_path / 'bev_semantics' / (f'{frame:04}.png')), tick_data['bev_semantics'])

        with gzip.open(self.save_path / 'boxes' / (f'{frame:04}.json.gz'), 'wt', encoding='utf-8') as f:
            json.dump(tick_data['bounding_boxes'], f, indent=4)

    def destroy(self, results=None):
        torch.cuda.empty_cache()

        if results is not None and self.save_path is not None:
            with gzip.open(os.path.join(self.save_path, 'results.json.gz'), 'wt', encoding='utf-8') as f:
                json.dump(results.__dict__, f, indent=2)

        super().destroy(results)

    def _is_in_fov(self, relative_pos, extent, relative_yaw):
        """
        Проверяет, попадает ли объект (даже частично) в сектор FOV 110 градусов.
        relative_pos: [x, y, z] относительно эго
        extent: [ex, ey, ez] (half-extents)
        relative_yaw: радианы, поворот объекта относительно эго
        """
        # 1. Создаем полигон объекта в координатах эго (2D: X, Y)
        # Углы прямоугольника до поворота (центр в relative_pos)
        dx = extent[0]
        dy = extent[1]
        
        # 4 угла относительно центра объекта
        corners = np.array([
            [dx, dy],
            [dx, -dy],
            [-dx, -dy],
            [-dx, dy]
        ])
        
        # Матрица поворота (по relative_yaw)
        c, s = np.cos(relative_yaw), np.sin(relative_yaw)
        R = np.array(((c, -s), (s, c)))
        
        # Поворачиваем углы и сдвигаем на позицию объекта
        rotated_corners = corners @ R.T
        final_corners = rotated_corners + np.array([relative_pos[0], relative_pos[1]])
        
        actor_poly = Polygon(final_corners)
        
        # 2. Проверяем пересечение с заранее созданным полигоном FOV
        return self.fov_polygon.intersects(actor_poly)

    def get_bounding_boxes(self, lidar=None):
        results = []

        ego_transform = self._vehicle.get_transform()
        ego_control = self._vehicle.get_control()
        ego_velocity = self._vehicle.get_velocity()
        ego_matrix = np.array(ego_transform.get_matrix())
        ego_rotation = ego_transform.rotation
        ego_extent = self._vehicle.bounding_box.extent
        ego_speed = self._get_forward_speed(transform=ego_transform, velocity=ego_velocity)
        ego_dx = np.array([ego_extent.x, ego_extent.y, ego_extent.z])
        ego_yaw = np.deg2rad(ego_rotation.yaw)
        ego_brake = ego_control.brake

        relative_yaw = 0.0
        relative_pos = t_u.get_relative_transform(ego_matrix, ego_matrix)

        # Retrieve all relevant actors
        self._actors = self._world.get_actors()
        vehicle_list = self._actors.filter('*vehicle*')

        # Ego сохраняем всегда
        result = {
            'class': 'ego_car',
            'extent': [ego_dx[0], ego_dx[1], ego_dx[2]],
            'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
            'yaw': relative_yaw,
            'num_points': -1,
            'distance': -1,
            'speed': ego_speed,
            'brake': ego_brake,
            'id': int(self._vehicle.id),
            'matrix': ego_transform.get_matrix()
        }
        results.append(result)

        for vehicle in vehicle_list:
            if vehicle.get_location().distance(self._vehicle.get_location()) < self.config.bb_save_radius:
                if vehicle.id != self._vehicle.id:
                    vehicle_transform = vehicle.get_transform()
                    vehicle_rotation = vehicle_transform.rotation
                    vehicle_matrix = np.array(vehicle_transform.get_matrix())
                    vehicle_extent = vehicle.bounding_box.extent
                    vehicle_extent_list = [vehicle_extent.x, vehicle_extent.y, vehicle_extent.z]
                    yaw = np.deg2rad(vehicle_rotation.yaw)

                    relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
                    relative_pos = t_u.get_relative_transform(ego_matrix, vehicle_matrix)
                    
                    # --- ПРОВЕРКА FOV ---
                    if not self._is_in_fov(relative_pos, vehicle_extent_list, relative_yaw):
                        continue
                    # --------------------

                    vehicle_control = vehicle.get_control()
                    vehicle_velocity = vehicle.get_velocity()
                    vehicle_speed = self._get_forward_speed(transform=vehicle_transform, velocity=vehicle_velocity)
                    vehicle_brake = vehicle_control.brake
                    vehicle_steer = vehicle_control.steer
                    vehicle_throttle = vehicle_control.throttle

                    if not lidar is None:
                        num_in_bbox_points = self.get_points_in_bbox(relative_pos, relative_yaw, vehicle_extent_list, lidar)
                    else:
                        num_in_bbox_points = -1

                    distance = np.linalg.norm(relative_pos)

                    result = {
                        'class': 'car',
                        'extent': vehicle_extent_list,
                        'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                        'yaw': relative_yaw,
                        'num_points': int(num_in_bbox_points),
                        'distance': distance,
                        'speed': vehicle_speed,
                        'brake': vehicle_brake,
                        'steer': vehicle_steer,
                        'throttle': vehicle_throttle,
                        'id': int(vehicle.id),
                        'role_name': vehicle.attributes['role_name'],
                        'type_id': vehicle.type_id,
                        'matrix': vehicle_transform.get_matrix()
                    }
                    results.append(result)

        walkers = self._actors.filter('*walker*')
        for walker in walkers:
            if walker.get_location().distance(self._vehicle.get_location()) < self.config.bb_save_radius:
                walker_transform = walker.get_transform()
                walker_rotation = walker.get_transform().rotation
                walker_matrix = np.array(walker_transform.get_matrix())
                walker_extent = walker.bounding_box.extent
                walker_extent_list = [walker_extent.x, walker_extent.y, walker_extent.z]
                yaw = np.deg2rad(walker_rotation.yaw)

                relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
                relative_pos = t_u.get_relative_transform(ego_matrix, walker_matrix)

                # --- ПРОВЕРКА FOV ---
                if not self._is_in_fov(relative_pos, walker_extent_list, relative_yaw):
                    continue
                # --------------------

                walker_velocity = walker.get_velocity()
                walker_speed = self._get_forward_speed(transform=walker_transform, velocity=walker_velocity)

                if not lidar is None:
                    num_in_bbox_points = self.get_points_in_bbox(relative_pos, relative_yaw, walker_extent_list, lidar)
                else:
                    num_in_bbox_points = -1

                distance = np.linalg.norm(relative_pos)

                result = {
                    'class': 'walker',
                    'extent': walker_extent_list,
                    'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                    'yaw': relative_yaw,
                    'num_points': int(num_in_bbox_points),
                    'distance': distance,
                    'speed': walker_speed,
                    'id': int(walker.id),
                    'matrix': walker_transform.get_matrix()
                }
                results.append(result)

        # Static objects (optional filter, but keeping consistent with request)
        static_list = self._actors.filter('*static*')
        for static in static_list:
            if static.get_location().distance(self._vehicle.get_location()) < self.config.bb_save_radius:
                static_transform = static.get_transform()
                static_rotation = static.get_transform().rotation
                static_matrix = np.array(static_transform.get_matrix())
                static_extent = static.bounding_box.extent
                static_extent_list = [static_extent.x, static_extent.y, static_extent.z]
                yaw = np.deg2rad(static_rotation.yaw)

                relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
                relative_pos = t_u.get_relative_transform(ego_matrix, static_matrix)

                # --- ПРОВЕРКА FOV ---
                # Для статики тоже применяем, чтобы не захламлять датасет
                if not self._is_in_fov(relative_pos, static_extent_list, relative_yaw):
                    continue
                # --------------------

                static_velocity = static.get_velocity()
                static_speed = self._get_forward_speed(transform=static_transform, velocity=static_velocity)

                if not lidar is None:
                    num_in_bbox_points = self.get_points_in_bbox(relative_pos, relative_yaw, static_extent_list, lidar)
                else:
                    num_in_bbox_points = -1

                distance = np.linalg.norm(relative_pos)

                result = {
                    'class': 'static',
                    'extent': static_extent_list,
                    'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                    'yaw': relative_yaw,
                    'num_points': int(num_in_bbox_points),
                    'distance': distance,
                    'speed': static_speed,
                    'id': int(static.id),
                    'matrix': static_transform.get_matrix(),
                    'type_id': static.type_id,
                    'mesh_path': static.attributes['mesh_path'] if 'mesh_path' in static.attributes else None
                }
                results.append(result)

        for traffic_light in self.close_traffic_lights:
            traffic_light_extent = [traffic_light[0].extent.x, traffic_light[0].extent.y, traffic_light[0].extent.z]
            traffic_light_transform = carla.Transform(traffic_light[0].location, traffic_light[0].rotation)
            traffic_light_rotation = traffic_light_transform.rotation
            traffic_light_matrix = np.array(traffic_light_transform.get_matrix())
            yaw = np.deg2rad(traffic_light_rotation.yaw)

            relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
            relative_pos = t_u.get_relative_transform(ego_matrix, traffic_light_matrix)
            
            if not self._is_in_fov(relative_pos, traffic_light_extent, relative_yaw):
                    continue
            
            distance = np.linalg.norm(relative_pos)

            result = {
                'class': 'traffic_light',
                'extent': traffic_light_extent,
                'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                'yaw': relative_yaw,
                'distance': distance,
                'state': str(traffic_light[1]),
                'id': int(traffic_light[2]),
                'affects_ego': traffic_light[3],
                'matrix': traffic_light_transform.get_matrix()
            }
            results.append(result)

        for stop_sign in self.close_stop_signs:
            stop_sign_extent = [stop_sign[0].extent.x, stop_sign[0].extent.y, stop_sign[0].extent.z]
            stop_sign_transform = carla.Transform(stop_sign[0].location, stop_sign[0].rotation)
            stop_sign_rotation = stop_sign_transform.rotation
            stop_sign_matrix = np.array(stop_sign_transform.get_matrix())
            yaw = np.deg2rad(stop_sign_rotation.yaw)

            relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
            relative_pos = t_u.get_relative_transform(ego_matrix, stop_sign_matrix)
            
            if not self._is_in_fov(relative_pos, stop_sign_extent, relative_yaw):
                    continue            

            distance = np.linalg.norm(relative_pos)

            result = {
                'class': 'stop_sign',
                'extent': stop_sign_extent,
                'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                'yaw': relative_yaw,
                'distance': distance,
                'id': int(stop_sign[1]),
                'affects_ego': stop_sign[2],
                'matrix': stop_sign_transform.get_matrix()
            }
            results.append(result)

        return results

    def get_points_in_bbox(self, vehicle_pos, vehicle_yaw, extent, lidar):
        """
        Checks for a given vehicle in ego coordinate system, how many LiDAR hit there are in its bounding box.
        """
        if lidar is None:
            return -1

        rotation_matrix = np.array([[np.cos(vehicle_yaw), -np.sin(vehicle_yaw), 0.0],
                                    [np.sin(vehicle_yaw), np.cos(vehicle_yaw), 0.0], [0.0, 0.0, 1.0]])

        # LiDAR in the with the vehicle as origin
        vehicle_lidar = (rotation_matrix.T @ (lidar - vehicle_pos).T).T

        # check points in bbox
        x, y, z = extent[0], extent[1], extent[2]
        num_points = ((vehicle_lidar[:, 0] < x) & (vehicle_lidar[:, 0] > -x) & (vehicle_lidar[:, 1] < y) &
                      (vehicle_lidar[:, 1] > -y) & (vehicle_lidar[:, 2] < z) & (vehicle_lidar[:, 2] > -z)).sum()
        return num_points

    def visualuize(self, rendered, visu_img):
        # Resize rendered map to match camera visualization (which is now 720p)
        rendered = cv2.resize(rendered, dsize=(visu_img.shape[1], visu_img.shape[1]), interpolation=cv2.INTER_LINEAR)
        visu_img = cv2.cvtColor(visu_img, cv2.COLOR_BGR2RGB)

        final = np.concatenate((visu_img, rendered), axis=0)

        Image.fromarray(final).save(self.save_path / (f'{self.step:04}.jpg'))

    def _vehicle_obstacle_detected(self,
                                   vehicle_list=None,
                                   max_distance=None,
                                   up_angle_th=90,
                                   low_angle_th=0,
                                   lane_offset=0):
        """
        Method to check if there is a vehicle in front of the agent blocking its path.
        """
        self._use_bbs_detection = False
        self._offset = 0

        def get_route_polygon():
            route_bb = []
            extent_y = self._vehicle.bounding_box.extent.y
            r_ext = extent_y + self._offset
            l_ext = -extent_y + self._offset
            r_vec = ego_transform.get_right_vector()
            p1 = ego_location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
            p2 = ego_location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
            route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            for wp, _ in self._local_planner.get_plan():
                if ego_location.distance(wp.transform.location) > max_distance:
                    break

                r_vec = wp.transform.get_right_vector()
                p1 = wp.transform.location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
                p2 = wp.transform.location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
                route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            # Two points don't create a polygon, nothing to check
            if len(route_bb) < 3:
                return None

            return Polygon(route_bb)

        if not vehicle_list:
            vehicle_list = self._world.get_actors().filter("*vehicle*")

        ego_transform = self._vehicle.get_transform()
        ego_location = ego_transform.location
        ego_wpt = self.world_map.get_waypoint(ego_location, lane_type=carla.libcarla.LaneType.Any)

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(self._vehicle.bounding_box.extent.x *
                                                       ego_transform.get_forward_vector())

        opposite_invasion = abs(self._offset) + self._vehicle.bounding_box.extent.y > ego_wpt.lane_width / 2
        use_bbs = self._use_bbs_detection or opposite_invasion or ego_wpt.is_junction

        # Get the route bounding box
        route_polygon = get_route_polygon()

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            target_transform = target_vehicle.get_transform()
            if target_transform.location.distance(ego_location) > max_distance:
                continue

            target_wpt = self.world_map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

            # General approach for junctions and vehicles invading other lanes due to the offset
            if (use_bbs or target_wpt.is_junction) and route_polygon:

                target_bb = target_vehicle.bounding_box
                target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                target_list = [[v.x, v.y, v.z] for v in target_vertices]
                target_polygon = Polygon(target_list)

                if route_polygon.intersects(target_polygon):
                    return (True, target_vehicle.id, compute_distance(target_vehicle.get_location(), ego_location))

            # Simplified approach, using only the plan waypoints (similar to TM)
            else:

                if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id + lane_offset:
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                    if not next_wpt:
                        continue
                    if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id + lane_offset:
                        continue

                target_forward_vector = target_transform.get_forward_vector()
                target_extent = target_vehicle.bounding_box.extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                )

                if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]):
                    return (True, target_vehicle.id, compute_distance(target_transform.location, ego_transform.location))

        return (False, None, -1)

    def _get_forward_speed(self, transform=None, velocity=None):
        """
        Calculate the forward speed of the vehicle based on its transform and velocity.
        """
        if not velocity:
            velocity = self._vehicle.get_velocity()

        if not transform:
            transform = self._vehicle.get_transform()

        # Convert the velocity vector to a NumPy array
        velocity_np = np.array([velocity.x, velocity.y, velocity.z])

        # Convert rotation angles from degrees to radians
        pitch_rad = np.deg2rad(transform.rotation.pitch)
        yaw_rad = np.deg2rad(transform.rotation.yaw)

        # Calculate the orientation vector based on pitch and yaw angles
        orientation_vector = np.array(
            [np.cos(pitch_rad) * np.cos(yaw_rad),
             np.cos(pitch_rad) * np.sin(yaw_rad),
             np.sin(pitch_rad)])

        # Calculate the forward speed by taking the dot product of velocity and orientation vectors
        forward_speed = np.dot(velocity_np, orientation_vector)

        return forward_speed