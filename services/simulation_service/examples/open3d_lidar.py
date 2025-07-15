#!/usr/bin/env python
"""Open3D Lidar визуализация с ручным управлением машины через клавиатуру,
   дополнительными камерами (RGB, Semantic segmentation, Depth) с выводом изображения в окно,
   а также датчиками: Obstacle, Collision, Lane invasion и GNSS, события которых выводятся в консоль.
   Документация: https://carla.readthedocs.io/en/0.9.15/ref_sensors/
"""

import glob
import os
import sys
import argparse
from datetime import datetime
import random
import numpy as np
import matplotlib
import open3d as o3d
import pygame
import threading

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# Параметры Lidar
SEM_LIDAR_UPPER_FOV = 15.0
SEM_LIDAR_LOWER_FOV = -25.0
SEM_LIDAR_CHANNELS = 64
SEM_LIDAR_RANGE = 100.0
SEM_LIDAR_POINTS_PER_SECOND = 500000
SEM_LIDAR_ROTATION_FREQUENCY = 1.0 / 0.05

# Цветовая палитра для семантической разметки
VIRIDIS = np.array(matplotlib.colormaps['plasma'].colors)
LABEL_COLORS = np.array([
    (0, 0, 0),       # Unlabeled
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (70, 70, 70),    # Building
    (102, 102, 156), # Wall
    (190, 153, 153), # Fence
    (153, 153, 153), # Pole
    (250, 170, 30),  # Traffic Light
    (220, 220, 0),   # Traffic Sign
    (107, 142, 35),  # Vegetation
    (152, 251, 152), # Terrain
    (70, 130, 180),  # Sky
    (220, 20, 60),   # Pedestrian
    (255, 0, 0),     # Rider
    (0, 0, 142),     # Car
    (0, 0, 70),      # Truck
    (0, 60, 100),    # Bus
    (0, 80, 100),    # Train
    (0, 0, 230),     # Motorcycle
    (119, 11, 32),   # Bicycle
    (110, 190, 160), # Static
    (170, 120, 50),  # Dynamic
    (55, 90, 80),    # Other
    (45, 60, 150),   # Water
    (157, 234, 50),  # Road Line
    (81, 0, 81),     # Ground
    (150, 100, 100), # Bridge
    (230, 150, 140), # Rail Track
    (180, 165, 180)  # Guard Rail
]) / 255.0

# Глобальные переменные и блокировки для кадров с камер
latest_rgb_frame = None
latest_seg_frame = None
latest_depth_frame = None

rgb_frame_lock = threading.Lock()
seg_frame_lock = threading.Lock()
depth_frame_lock = threading.Lock()


def semantic_lidar_callback(point_cloud, point_list):
    data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))
    # Меняем знак y для корректной визуализации в правосторонней системе Open3D
    points = np.array([data['x'], -data['y'], data['z']]).T
    labels = np.array(data['ObjTag'])
    colors = LABEL_COLORS[labels]
    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(colors)


def generate_lidar_bp(world):
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
    lidar_bp.set_attribute('upper_fov', str(SEM_LIDAR_UPPER_FOV))
    lidar_bp.set_attribute('lower_fov', str(SEM_LIDAR_LOWER_FOV))
    lidar_bp.set_attribute('channels', str(SEM_LIDAR_CHANNELS))
    lidar_bp.set_attribute('range', str(SEM_LIDAR_RANGE))
    lidar_bp.set_attribute('rotation_frequency', str(SEM_LIDAR_ROTATION_FREQUENCY))
    lidar_bp.set_attribute('points_per_second', str(SEM_LIDAR_POINTS_PER_SECOND))
    return lidar_bp


def add_open3d_axis(vis):
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)


def process_segmentation_image(image):
    """
    Обработка изображения с камеры semantic segmentation.
    Применяется палитра CityScapes для корректного отображения.
    """
    image.convert(carla.ColorConverter.CityScapesPalette)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    frame = array[:, :, :3]
    with seg_frame_lock:
        global latest_seg_frame
        latest_seg_frame = frame


def process_rgb_image(image):
    """
    Обработка изображения с RGB-камеры.
    """
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    frame = array[:, :, :3]
    with rgb_frame_lock:
        global latest_rgb_frame
        latest_rgb_frame = frame


def process_depth_image(image):
    """
    Обработка изображения с глубинной камеры.
    Применяется преобразователь Depth для корректного отображения.
    """
    image.convert(carla.ColorConverter.Depth)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    frame = array[:, :, :3]
    with depth_frame_lock:
        global latest_depth_frame
        latest_depth_frame = frame


def main():
    argparser = argparse.ArgumentParser(
        description="CARLA Manual Control с визуализацией Lidar, камер и дополнительными датчиками")
    argparser.add_argument('--host', default='127.0.0.1',
                           help='IP-адрес сервера симулятора (по умолчанию: 127.0.0.1)')
    argparser.add_argument('-p', '--port', type=int, default=2000,
                           help='TCP-порт подключения (по умолчанию: 2000)')
    args = argparser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)
    world = client.get_world()

    # Создаём окно pygame для отображения камер (ширина = 3*320, высота = 240)
    pygame.init()
    screen = pygame.display.set_mode((960, 240))
    pygame.display.set_caption("CARLA Camera Feeds и Keyboard Control")

    try:
        original_settings = world.get_settings()
        settings = world.get_settings()
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.*')[0]
        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_autopilot(False)

        # Лидар
        lidar_bp = generate_lidar_bp(world)
        lidar_transform = carla.Transform(carla.Location(x=-0.5, z=2.4))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
        point_list = o3d.geometry.PointCloud()
        lidar.listen(lambda data: semantic_lidar_callback(data, point_list))

        # Камеры (RGB, Semantic segmentation, Depth)
        rgb_camera_bp = blueprint_library.find('sensor.camera.rgb')
        rgb_camera_bp.set_attribute('image_size_x', '320')
        rgb_camera_bp.set_attribute('image_size_y', '240')
        rgb_camera_bp.set_attribute('fov', '110')
        rgb_transform = carla.Transform(carla.Location(x=1.5, y=-0.2, z=2.4))
        rgb_camera = world.spawn_actor(rgb_camera_bp, rgb_transform, attach_to=vehicle)
        rgb_camera.listen(process_rgb_image)

        seg_camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_camera_bp.set_attribute('image_size_x', '320')
        seg_camera_bp.set_attribute('image_size_y', '240')
        seg_camera_bp.set_attribute('fov', '110')
        seg_transform = carla.Transform(carla.Location(x=1.5, y=0.0, z=2.4))
        seg_camera = world.spawn_actor(seg_camera_bp, seg_transform, attach_to=vehicle)
        seg_camera.listen(process_segmentation_image)

        depth_camera_bp = blueprint_library.find('sensor.camera.depth')
        depth_camera_bp.set_attribute('image_size_x', '320')
        depth_camera_bp.set_attribute('image_size_y', '240')
        depth_camera_bp.set_attribute('fov', '110')
        depth_transform = carla.Transform(carla.Location(x=1.5, y=0.2, z=2.4))
        depth_camera = world.spawn_actor(depth_camera_bp, depth_transform, attach_to=vehicle)
        depth_camera.listen(process_depth_image)

        # Датчики, выводящие данные в консоль
        obstacle_bp = blueprint_library.find('sensor.other.obstacle')
        obstacle_transform = carla.Transform(carla.Location(x=2.0, z=1.0))
        obstacle_sensor = world.spawn_actor(obstacle_bp, obstacle_transform, attach_to=vehicle)
        obstacle_sensor.listen(lambda event: print("Obstacle detected:", event))

        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        collision_sensor.listen(lambda event: print("Collision:", event))

        lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')
        lane_invasion_sensor = world.spawn_actor(lane_invasion_bp, carla.Transform(), attach_to=vehicle)
        lane_invasion_sensor.listen(lambda event: print("Lane invasion:", event))

        gnss_bp = blueprint_library.find('sensor.other.gnss')
        gnss_transform = carla.Transform(carla.Location(x=0.0, z=2.8))
        gnss_sensor = world.spawn_actor(gnss_bp, gnss_transform, attach_to=vehicle)
        gnss_sensor.listen(lambda event: print("GNSS:", event))

        # Open3D визуализация для Lidar
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='Carla Lidar', width=500, height=500)
        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1
        vis.get_render_option().show_coordinate_frame = True
        add_open3d_axis(vis)

        pygame.joystick.init()
        clock = pygame.time.Clock()

        joystick = None
        if pygame.joystick.get_count() > 0:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            print("Используем устройство:", joystick.get_name())
        else:
            print("Руль не найден, управление будет только с клавиатуры.")

        control = carla.VehicleControl()
        control.throttle = 0.0
        control.steer = 0.0
        control.brake = 0.0
        control.reverse = False

        NITRO_BOOST = 3

        frame = 0
        dt0 = datetime.now()
        running = True

        while running:
            # Обновление Open3D визуализации для Lidar
            ctr = vis.get_view_control()
            ctr.set_front([0, 0, -1])
            ctr.set_lookat([0, 0, 0])
            ctr.set_up([0, -1, 0])
            ctr.set_zoom(0.3)

            # Обработка событий pygame (в том числе закрытие окна)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt

            keys = pygame.key.get_pressed()
            control.throttle = 0.0
            control.steer = 0.0
            control.brake = 0.0

            keyboard_used = False
            if keys[pygame.K_UP]:
                control.reverse = False
                control.throttle = 0.7
                keyboard_used = True
            elif keys[pygame.K_DOWN]:
                control.reverse = True
                control.throttle = 0.7
                keyboard_used = True

            if keys[pygame.K_LEFT]:
                control.steer = -0.3
                keyboard_used = True
            elif keys[pygame.K_RIGHT]:
                control.steer = 0.3
                keyboard_used = True

            if joystick is not None and not keyboard_used:
                steer_axis = joystick.get_axis(0)
                gas_axis = joystick.get_axis(1)
                brake_axis = joystick.get_axis(2)
                control.steer = steer_axis * 0.5
                throttle = max(0.0, min(1.0, (1 - gas_axis) / 2))
                brake = max(0.0, min(1.0, (1 - brake_axis) / 2))
                if brake < 0.0001 < throttle:
                    brake = 0.0
                if 0.0001 < brake:
                    throttle = 0.0
                control.brake = brake
                if joystick.get_button(6):
                    throttle *= NITRO_BOOST
                    throttle = min(throttle, 1.0)
                control.throttle = throttle
                control.reverse = joystick.get_button(0)

            vehicle.apply_control(control)

            # Обновление Open3D для Lidar
            if frame == 2:
                vis.add_geometry(point_list)
            vis.update_geometry(point_list)
            vis.poll_events()
            vis.update_renderer()

            # Отрисовка изображений с камер в окне pygame
            screen.fill((0, 0, 0))
            # RGB камера
            with rgb_frame_lock:
                rgb_frame = latest_rgb_frame.copy() if latest_rgb_frame is not None else None
            if rgb_frame is not None:
                rgb_surf = pygame.surfarray.make_surface(rgb_frame.swapaxes(0, 1))
                screen.blit(rgb_surf, (0, 0))
            # Semantic segmentation камера
            with seg_frame_lock:
                seg_frame = latest_seg_frame.copy() if latest_seg_frame is not None else None
            if seg_frame is not None:
                seg_surf = pygame.surfarray.make_surface(seg_frame.swapaxes(0, 1))
                screen.blit(seg_surf, (320, 0))
            # Depth камера
            with depth_frame_lock:
                depth_frame = latest_depth_frame.copy() if latest_depth_frame is not None else None
            if depth_frame is not None:
                depth_surf = pygame.surfarray.make_surface(depth_frame.swapaxes(0, 1))
                screen.blit(depth_surf, (640, 0))

            pygame.display.flip()

            world.tick()

            process_time = datetime.now() - dt0
            fps = 1.0 / process_time.total_seconds() if process_time.total_seconds() > 0 else 0
            sys.stdout.write(f'\rFPS: {fps:.2f} ')
            sys.stdout.flush()
            dt0 = datetime.now()
            frame += 1
            clock.tick(60)  # Ограничение до 60 FPS

    finally:
        world.apply_settings(original_settings)
        traffic_manager.set_synchronous_mode(False)
        vehicle.destroy()
        lidar.destroy()
        rgb_camera.destroy()
        seg_camera.destroy()
        depth_camera.destroy()
        obstacle_sensor.destroy()
        collision_sensor.destroy()
        lane_invasion_sensor.destroy()
        gnss_sensor.destroy()
        vis.destroy_window()
        pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
