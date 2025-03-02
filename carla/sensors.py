#!/usr/bin/env python
"""
CARLA Manual Control with RGB, Semantic Segmentation Cameras and Additional Sensors

Скрипт подключается к симулятору CARLA, спавнит транспортное средство,
прикрепляет к нему несколько сенсоров:
 - сенсор семантической сегментации,
 - RGB-камера,
 - семантический LIDAR,
 - GNSS-сенсор,
 - глубинная камера,
 - датчик препятствий,
 - датчик столкновений,
 - датчик выезда из полосы.
Пользователь может управлять автомобилем с клавиатуры и контроллером руля.
"""

import glob
import os
import sys
import argparse
import time
import threading
from datetime import datetime
import math

import pygame
import numpy as np
import cv2
import matplotlib
import open3d as o3d

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# Глобальные переменные для кадров сенсоров и блокировок
latest_seg_frame = None
seg_frame_lock = threading.Lock()

latest_rgb_frame = None
rgb_frame_lock = threading.Lock()

latest_depth_frame = None
depth_frame_lock = threading.Lock()

# Глобальные переменные для отображения данных водителя
latest_speed_kmh = 0.0
latest_rpm = 0.0
latest_game_time = "00:00:00"

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
    (0, 0, 0),           # Unlabeled
    (128, 64, 128),      # Road
    (244, 35, 232),      # Sidewalk
    (70, 70, 70),        # Building
    (102, 102, 156),     # Wall
    (190, 153, 153),     # Fence
    (153, 153, 153),     # Pole
    (250, 170, 30),      # Traffic Light
    (220, 220, 0),       # Traffic Sign
    (107, 142, 35),      # Vegetation
    (152, 251, 152),     # Terrain
    (70, 130, 180),      # Sky
    (220, 20, 60),       # Pedestrian
    (255, 0, 0),         # Rider
    (0, 0, 142),         # Car
    (0, 0, 70),          # Truck
    (0, 60, 100),        # Bus
    (0, 80, 100),        # Train
    (0, 0, 230),         # Motorcycle
    (119, 11, 32),       # Bicycle
    (110, 190, 160),     # Static
    (170, 120, 50),      # Dynamic
    (55, 90, 80),        # Other
    (45, 60, 150),       # Water
    (157, 234, 50),      # Road Line
    (81, 0, 81),         # Ground
    (150, 100, 100),     # Bridge
    (230, 150, 140),     # Rail Track
    (180, 165, 180)      # Guard Rail
]) / 255.0


# ---------------- Helper Functions ----------------

def get_frame_from_image(image, color_converter=None):
    """
    Универсальная функция для получения кадра из изображения сенсора.
    При наличии color_converter выполняется преобразование изображения.
    """
    if color_converter:
        image.convert(color_converter)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    return array[:, :, :3]


def semantic_lidar_callback(point_cloud, point_list):
    """
    Обработка данных Lidar с семантической разметкой.
    Инвертируется ось Y для корректной визуализации в Open3D.
    """
    data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))
    points = np.array([data['x'], -data['y'], data['z']]).T
    labels = np.array(data['ObjTag'])
    colors = LABEL_COLORS[labels]
    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(colors)


def generate_lidar_bp(world):
    """
    Генерация blueprint для Lidar с заданными параметрами.
    """
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
    lidar_bp.set_attribute('upper_fov', str(SEM_LIDAR_UPPER_FOV))
    lidar_bp.set_attribute('lower_fov', str(SEM_LIDAR_LOWER_FOV))
    lidar_bp.set_attribute('channels', str(SEM_LIDAR_CHANNELS))
    lidar_bp.set_attribute('range', str(SEM_LIDAR_RANGE))
    lidar_bp.set_attribute('rotation_frequency', str(SEM_LIDAR_ROTATION_FREQUENCY))
    lidar_bp.set_attribute('points_per_second', str(SEM_LIDAR_POINTS_PER_SECOND))
    return lidar_bp


def add_open3d_axis(vis):
    """
    Добавление осей координат в Open3D-визуализатор.
    Предварительный расчёт точек, линий и цветов для повышения производительности.
    """
    axis = o3d.geometry.LineSet()
    pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    axis.points = o3d.utility.Vector3dVector(pts)
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]
    ]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]))
    vis.add_geometry(axis)


# ---------------- Sensor Callback Functions ----------------

def process_segmentation_image(image):
    frame = get_frame_from_image(image, carla.ColorConverter.CityScapesPalette)
    with seg_frame_lock:
        global latest_seg_frame
        latest_seg_frame = frame


def process_rgb_image(image):
    frame = get_frame_from_image(image)
    with rgb_frame_lock:
        global latest_rgb_frame
        latest_rgb_frame = frame


def process_depth_image(image):
    frame = get_frame_from_image(image, carla.ColorConverter.Depth)
    with depth_frame_lock:
        global latest_depth_frame
        latest_depth_frame = frame


def process_gnss_data(data):
    print("GNSS: Latitude: {:.6f}, Longitude: {:.6f}, Altitude: {:.2f}".format(
        data.latitude, data.longitude, data.altitude))


def process_obstacle_data(event):
    # Добавляем эмодзи 🚧 для препятствий
    print("🚧 Obstacle event:", event)


def process_collision_data(event):
    # Добавляем эмодзи 💥 для столкновений
    print("💥 Collision with:", event.other_actor.type_id if event.other_actor is not None else "Unknown")


def process_lane_invasion_data(event):
    # Добавляем эмодзи 🚦 для выезда из полосы
    print("🚦 Lane invasion detected:", event.crossed_lane_markings)


# ---------------- Display Loop (cv2 окна) ----------------

def display_loop():
    """
    Отображение видеопотоков с камер для сенсоров, кроме RGB (он теперь в pygame).
    Создаются отдельные окна для семантической сегментации и глубины.
    """
    cv2.startWindowThread()
    cv2.namedWindow("Semantic Segmentation", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth Camera", cv2.WINDOW_NORMAL)

    last_seg_frame = None

    while True:
        with seg_frame_lock:
            seg_frame = latest_seg_frame.copy() if latest_seg_frame is not None else None
        with depth_frame_lock:
            depth_frame = latest_depth_frame.copy() if latest_depth_frame is not None else None

        if seg_frame is not None:
            last_seg_frame = seg_frame

        if last_seg_frame is not None:
            cv2.imshow("Semantic Segmentation", last_seg_frame)
        if depth_frame is not None:
            cv2.imshow("Depth Camera", depth_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.005)
    cv2.destroyAllWindows()


# ---------------- Cleanup Function ----------------

def cleanup(actors, vis):
    """
    Корректное завершение работы: остановка и уничтожение сенсоров и акторов,
    закрытие Open3D, cv2 и pygame окон.
    """
    for actor in actors:
        try:
            actor.stop()
        except Exception:
            pass
        try:
            actor.destroy()
        except Exception:
            pass
    try:
        vis.destroy_window()
    except Exception:
        pass
    cv2.destroyAllWindows()
    pygame.quit()


# ---------------- Main Function ----------------

def main():
    argparser = argparse.ArgumentParser(
        description="CARLA Manual Control with RGB, Semantic Segmentation Cameras and Additional Sensors")
    argparser.add_argument('--host', default='127.0.0.1',
                           help='IP-адрес сервера симулятора (по умолчанию: 127.0.0.1)')
    argparser.add_argument('-p', '--port', type=int, default=2000,
                           help='TCP-порт подключения (по умолчанию: 2000)')
    # Аргументы для отключения сенсоров
    argparser.add_argument('--disable-seg', action='store_true', help='Отключить сенсор семантической сегментации')
    argparser.add_argument('--disable-rgb', action='store_true', help='Отключить RGB камеру')
    argparser.add_argument('--disable-depth', action='store_true', help='Отключить датчик глубины')
    argparser.add_argument('--disable-lidar', action='store_true', help='Отключить Lidar с семантической разметкой')
    argparser.add_argument('--disable-gnss', action='store_true', help='Отключить GNSS сенсор')
    argparser.add_argument('--disable-obstacle', action='store_true', help='Отключить датчик препятствий')
    argparser.add_argument('--disable-collision', action='store_true', help='Отключить датчик столкновений')
    argparser.add_argument('--disable-laneinvasion', action='store_true', help='Отключить датчик выезда из полосы')
    args = argparser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)
    try:
        world = client.get_world()
    except RuntimeError:
        sys.exit("Ошибка: не удалось подключиться к симулятору. Проверьте, что CARLA запущен по адресу {}:{}".format(
            args.host, args.port))

    blueprint_library = world.get_blueprint_library()

    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        sys.exit("Ошибка: нет доступных точек спавна.")
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    print("Транспортное средство заспавнено.")

    # Список акторов для последующего удаления
    actors = []

    # ---------------- Настройка сенсоров ----------------

    # Сенсор семантической сегментации
    if not args.disable_seg:
        seg_cam_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_cam_bp.set_attribute('image_size_x', '800')
        seg_cam_bp.set_attribute('image_size_y', '600')
        seg_cam_bp.set_attribute('fov', '90')
        seg_cam_bp.set_attribute('sensor_tick', '0.05')
        seg_cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        seg_cam = world.spawn_actor(seg_cam_bp, seg_cam_transform, attach_to=vehicle)
        seg_cam.listen(process_segmentation_image)
        actors.append(seg_cam)
    else:
        print("Сенсор семантической сегментации отключен.")

    # Сенсор RGB камеры (отображается в окне pygame)
    if not args.disable_rgb:
        rgb_cam_bp = blueprint_library.find('sensor.camera.rgb')
        rgb_cam_bp.set_attribute('image_size_x', '1920')
        rgb_cam_bp.set_attribute('image_size_y', '1080')
        rgb_cam_bp.set_attribute('fov', '105')
        rgb_cam_bp.set_attribute('sensor_tick', '0.05')
        rgb_cam_transform = carla.Transform(
            carla.Location(x=0.25, y=-0.31, z=1.25),
            carla.Rotation(pitch=0, yaw=0, roll=0)
        )
        rgb_cam = world.spawn_actor(rgb_cam_bp, rgb_cam_transform, attach_to=vehicle)
        rgb_cam.listen(process_rgb_image)
        actors.append(rgb_cam)
    else:
        print("RGB камера отключена.")

    # Сенсор Lidar с семантической разметкой
    if not args.disable_lidar:
        lidar_bp = generate_lidar_bp(world)
        lidar_transform = carla.Transform(carla.Location(x=-0.5, z=2.4))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
        point_list = o3d.geometry.PointCloud()
        lidar.listen(lambda data: semantic_lidar_callback(data, point_list))
        actors.append(lidar)
    else:
        print("Lidar с семантической разметкой отключен.")

    # GNSS сенсор
    if not args.disable_gnss:
        gnss_bp = blueprint_library.find('sensor.other.gnss')
        gnss_bp.set_attribute('sensor_tick', '0.05')
        gnss_transform = carla.Transform(carla.Location(x=0, y=0, z=2.8))
        gnss_sensor = world.spawn_actor(gnss_bp, gnss_transform, attach_to=vehicle)
        gnss_sensor.listen(process_gnss_data)
        actors.append(gnss_sensor)
    else:
        print("GNSS сенсор отключен.")

    # Глубинная камера
    if not args.disable_depth:
        depth_cam_bp = blueprint_library.find('sensor.camera.depth')
        depth_cam_bp.set_attribute('image_size_x', '800')
        depth_cam_bp.set_attribute('image_size_y', '600')
        depth_cam_bp.set_attribute('fov', '90')
        depth_cam_bp.set_attribute('sensor_tick', '0.05')
        depth_cam_transform = carla.Transform(carla.Location(x=1.5, y=0.3, z=2.4))
        depth_cam = world.spawn_actor(depth_cam_bp, depth_cam_transform, attach_to=vehicle)
        depth_cam.listen(process_depth_image)
        actors.append(depth_cam)
    else:
        print("Датчик глубины отключен.")

    # Датчик препятствий
    if not args.disable_obstacle:
        obstacle_bp = blueprint_library.find('sensor.other.obstacle')
        obstacle_bp.set_attribute('sensor_tick', '0.05')
        obstacle_transform = carla.Transform(carla.Location(x=2.5, z=1.0))
        obstacle_detector = world.spawn_actor(obstacle_bp, obstacle_transform, attach_to=vehicle)
        obstacle_detector.listen(process_obstacle_data)
        actors.append(obstacle_detector)
    else:
        print("Датчик препятствий отключен.")

    # Датчик столкновений
    if not args.disable_collision:
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_transform = carla.Transform()
        collision_sensor = world.spawn_actor(collision_bp, collision_transform, attach_to=vehicle)
        collision_sensor.listen(process_collision_data)
        actors.append(collision_sensor)
    else:
        print("Датчик столкновений отключен.")

    # Датчик выезда из полосы
    if not args.disable_laneinvasion:
        lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')
        lane_invasion_transform = carla.Transform()
        lane_invasion_sensor = world.spawn_actor(lane_invasion_bp, lane_invasion_transform, attach_to=vehicle)
        lane_invasion_sensor.listen(process_lane_invasion_data)
        actors.append(lane_invasion_sensor)
    else:
        print("Датчик выезда из полосы отключен.")

    actors.append(vehicle)

    # Запуск отдельного потока для cv2 окон (семантическая сегментация и глубина)
    display_thread = threading.Thread(target=display_loop)
    display_thread.daemon = True
    display_thread.start()

    # ---------------- Настройка Open3D визуализации для Lidar ----------------
    # Открываем Open3D окно только если Lidar включен
    if not args.disable_lidar:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='Carla Lidar', width=500, height=500)
        render_opt = vis.get_render_option()
        render_opt.background_color = [0.05, 0.05, 0.05]
        render_opt.point_size = 1
        render_opt.show_coordinate_frame = True
        add_open3d_axis(vis)
    else:
        vis = o3d.visualization.Visualizer()  # Заглушка, чтобы cleanup не выдавал ошибку

    # ---------------- Инициализация pygame для управления и отображения RGB камеры ----------------

    pygame.init()
    pygame.joystick.init()
    # Начальные размеры окна
    display_width, display_height = 800, 600
    screen = pygame.display.set_mode((display_width, display_height), pygame.RESIZABLE)
    pygame.display.set_caption("CARLA Manual Control & RGB Camera")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)  # Шрифт для оверлея

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
    view_update_interval = 10  # Обновление настроек камеры Open3D не на каждом кадре

    try:
        while True:
            if view_update_interval:
                ctr = vis.get_view_control()
                ctr.set_front([0, 0, -1])
                ctr.set_lookat([0, 0, 0])
                ctr.set_up([0, -1, 0])
                ctr.set_zoom(0.3)
            # Обработка событий pygame, включая изменение размера окна
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt
                elif event.type == pygame.VIDEORESIZE:
                    display_width, display_height = event.size
                    screen = pygame.display.set_mode((display_width, display_height), pygame.RESIZABLE)

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
                if brake > 0.0001:
                    throttle = 0.0
                control.brake = brake
                if joystick.get_button(6):
                    throttle *= NITRO_BOOST
                    throttle = min(throttle, 1.0)
                control.throttle = throttle
                control.reverse = joystick.get_button(0)

            vehicle.apply_control(control)

            # Обновление Open3D визуализации для Lidar
            if frame == 2 and not args.disable_lidar:
                vis.add_geometry(point_list)
            if not args.disable_lidar:
                vis.update_geometry(point_list)
                vis.poll_events()
                vis.update_renderer()

            # Обновление данных о скорости и времени
            velocity = vehicle.get_velocity()
            speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6
            global latest_speed_kmh, latest_rpm, latest_game_time
            latest_speed_kmh = speed
            latest_rpm = control.throttle * 7000
            snapshot = world.get_snapshot()
            game_seconds = int(snapshot.timestamp.elapsed_seconds)
            hours = game_seconds // 3600
            minutes = (game_seconds % 3600) // 60
            seconds = game_seconds % 60
            latest_game_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            # Отображение RGB камеры и информационного оверлея в окне pygame
            if not args.disable_rgb:
                with rgb_frame_lock:
                    rgb_frame = latest_rgb_frame.copy() if latest_rgb_frame is not None else None

                if rgb_frame is not None:
                    # Преобразование BGR в RGB для отображения в pygame
                    rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
                    # Создание поверхности из буфера и масштабирование под размер окна
                    surface = pygame.image.frombuffer(rgb_frame.tobytes(), (rgb_frame.shape[1], rgb_frame.shape[0]), "RGB")
                    surface = pygame.transform.scale(surface, (display_width, display_height))
                    screen.blit(surface, (0, 0))
                else:
                    screen.fill((30, 30, 30))
            else:
                screen.fill((30, 30, 30))
                text_surface = font.render("RGB камера отключена", True, (255, 255, 255))
                screen.blit(text_surface, (10, 10))

            # Отрисовка текстовой информации (скорость, RPM, время)
            overlay_lines = [
                f"Speed: {latest_speed_kmh:.1f} km/h",
                f"RPM: {latest_rpm:.0f}",
                f"Time: {latest_game_time}"
            ]
            y = 10
            for line in overlay_lines:
                text_surface = font.render(line, True, (255, 255, 255))
                screen.blit(text_surface, (display_width - text_surface.get_width() - 10, y))
                y += text_surface.get_height() + 5

            pygame.display.flip()
            world.tick()

            process_time = datetime.now() - dt0
            fps = 1.0 / process_time.total_seconds() if process_time.total_seconds() > 0 else 0
            sys.stdout.write(f'\rFPS: {fps:.2f} ')
            sys.stdout.flush()
            dt0 = datetime.now()
            frame += 1
            clock.tick(60)
    except KeyboardInterrupt:
        print("\nЗавершение работы...")
    finally:
        cleanup(actors, vis)


if __name__ == '__main__':
    main()
