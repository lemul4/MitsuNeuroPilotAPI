#!/usr/bin/env python
"""
CARLA Manual Control with RGB, Semantic Segmentation Cameras and Additional Sensors

Скрипт подключается к симулятору CARLA, спавнит транспортное средство,
прикрепляет к нему несколько сенсоров:
 - сенсор семантической сегментации,
 - RGB-камера,
 - семантический LIDAR,
 - GNSS-сенсор,
 - датчик глубины,
 - датчик препятствий,
 - датчик столкновений,
 - датчик выезда из полосы.
Пользователь может управлять автомобилем с клавиатуры и контроллером руля.

При указании аргумента --record-data дополнительно осуществляется запись данных:
 - lidar_semantic, semantic_cam, depth_cam – кадры сохраняются с именем, содержащим номер кадра
 - gnss, obstacle_sensor, collision_sensor, lane_invasion_sensor – события записываются в CSV с первым столбцом, содержащим номер кадра
 - vehicle control и скорость – записываются в CSV с первым столбцом, содержащим номер кадра
"""

import glob
import os
import sys
import argparse
import time
import threading
from datetime import datetime
import math
import queue  # Используем очередь для асинхронной записи

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

# ---------------- Глобальные переменные и константы ----------------
CARLA_FPS = 20
SENSOR_TICK = '0.05'  # 20 fps

latest_seg_frame = None
seg_frame_lock = threading.Lock()

latest_rgb_frame = None
rgb_frame_lock = threading.Lock()

latest_depth_frame = None
depth_frame_lock = threading.Lock()

latest_speed_kmh = 0.0
latest_game_time = "00:00:00"

recording_enabled = False
recording_dirs = {}
control_csv_file = None
gnss_file = None
obstacle_file = None
collision_file = None
lane_invasion_file = None

# Очередь для задач записи в файлы и поток-работник
file_io_queue = queue.Queue()
file_io_worker_thread = None

def file_io_worker():
    """Поток-работник, выполняющий задачи записи в файлы из очереди."""
    while True:
        task = file_io_queue.get()
        if task is None:
            file_io_queue.task_done()
            break
        try:
            task_type = task["type"]
            if task_type == "image":
                cv2.imwrite(task["path"], task["data"])
            elif task_type == "np_save":
                np.save(task["path"], task["data"])
            elif task_type == "csv":
                file_obj = task["file"]
                file_obj.write(task["line"])
        except Exception as e:
            print("Ошибка записи в файл:", e)
        file_io_queue.task_done()

def start_file_io_worker():
    global file_io_worker_thread
    file_io_worker_thread = threading.Thread(target=file_io_worker, daemon=True)
    file_io_worker_thread.start()

def stop_file_io_worker():
    file_io_queue.put(None)
    file_io_worker_thread.join()

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

# ---------------- Вспомогательные функции ----------------

def get_frame_from_image(image, color_converter=None):
    """Универсальная функция для получения кадра из изображения сенсора.
    При наличии выполняет преобразование цвета."""
    if color_converter:
        image.convert(color_converter)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    return array[:, :, :3]

# ---------------- Callback для Lidar ----------------

def semantic_lidar_callback(point_cloud, point_list):
    """Обработка данных Lidar с семантической разметкой.
    Инвертируется ось Y для корректной визуализации в Open3D."""
    data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))
    points = np.array([data['y'], data['x'], data['z']]).T

    labels = np.array(data['ObjTag'])
    labels[labels == 24] = 1  # Переназначение разметки дороги в класс дороги
    mask = ~np.isin(labels, [11, 23])
    filtered_points = points[mask]
    filtered_labels = labels[mask]

    colors = LABEL_COLORS[filtered_labels]
    point_list.points = o3d.utility.Vector3dVector(filtered_points)
    point_list.colors = o3d.utility.Vector3dVector(colors)
    
    if recording_enabled:
        filename = os.path.join(recording_dirs['lidar_semantic'], f"lidar_{point_cloud.frame:06d}.npy")
        file_io_queue.put({"type": "np_save", "path": filename, "data": data})

def generate_lidar_bp(world):
    """Генерация blueprint для Lidar с заданными параметрами."""
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
    lidar_bp.set_attribute('upper_fov', '15.0')
    lidar_bp.set_attribute('lower_fov', '-25.0')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('range', '70')
    lidar_bp.set_attribute('rotation_frequency', '20.0')
    lidar_bp.set_attribute('points_per_second', '350000')
    lidar_bp.set_attribute('role_name', 'lidar_semantic')
    lidar_bp.set_attribute('sensor_tick', SENSOR_TICK)
    return lidar_bp

def add_open3d_axis(vis):
    """Добавление осей координат в Open3D-визуализатор."""
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

# ---------------- Callback функции для сенсоров ----------------

def process_segmentation_image(image):
    frame = get_frame_from_image(image, carla.ColorConverter.CityScapesPalette)
    with seg_frame_lock:
        global latest_seg_frame
        latest_seg_frame = frame
    if recording_enabled:
        filename = os.path.join(recording_dirs['semantic_cam'], f"semantic_{image.frame:06d}.png")
        file_io_queue.put({"type": "image", "path": filename, "data": frame})

def process_rgb_image(image):
    frame = get_frame_from_image(image)
    with rgb_frame_lock:
        global latest_rgb_frame
        latest_rgb_frame = frame

def process_depth_image(image):
    frame = get_frame_from_image(image, carla.ColorConverter.LogarithmicDepth)
    with depth_frame_lock:
        global latest_depth_frame
        latest_depth_frame = frame
    if recording_enabled:
        filename = os.path.join(recording_dirs['depth_cam'], f"depth_{image.frame:06d}.png")
        file_io_queue.put({"type": "image", "path": filename, "data": frame})

def process_gnss_data(data):
    msg = "GNSS: Latitude: {:.6f}, Longitude: {:.6f}, Altitude: {:.2f}".format(
        data.latitude, data.longitude, data.altitude)
    print(msg)
    if recording_enabled:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        line = f"{data.frame},{timestamp},{data.latitude},{data.longitude},{data.altitude}\n"
        file_io_queue.put({"type": "csv", "file": gnss_file, "line": line})

def process_obstacle_data(event):
    msg = "🚧 Obstacle event: " + str(event)
    print(msg)
    if recording_enabled:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        line = f"{event.frame},{timestamp},{event}\n"
        file_io_queue.put({"type": "csv", "file": obstacle_file, "line": line})

def process_collision_data(event):
    actor_type = event.other_actor.type_id if event.other_actor is not None else "Unknown"
    msg = "💥 Collision with: " + actor_type
    print(msg)
    if recording_enabled:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        line = f"{event.frame},{timestamp},{actor_type}\n"
        file_io_queue.put({"type": "csv", "file": collision_file, "line": line})

def process_lane_invasion_data(event):
    msg = "🚦 Lane invasion detected: " + str(event.crossed_lane_markings)
    print(msg)
    if recording_enabled:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        line = f"{event.frame},{timestamp},{event.crossed_lane_markings}\n"
        file_io_queue.put({"type": "csv", "file": lane_invasion_file, "line": line})

# ---------------- Display Loop (cv2 окна) ----------------

def display_loop():
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
    if recording_enabled:
        stop_file_io_worker()
    global control_csv_file, gnss_file, obstacle_file, collision_file, lane_invasion_file
    if recording_enabled:
        if control_csv_file: control_csv_file.close()
        if gnss_file: gnss_file.close()
        if obstacle_file: obstacle_file.close()
        if collision_file: collision_file.close()
        if lane_invasion_file: lane_invasion_file.close()

# ---------------- Main Function ----------------

def main():
    global recording_enabled, recording_dirs, control_csv_file, gnss_file, obstacle_file, collision_file, lane_invasion_file
    argparser = argparse.ArgumentParser(
        description="CARLA Manual Control with RGB, Semantic Segmentation Cameras and Additional Sensors")
    argparser.add_argument('--host', default='127.0.0.1',
                           help='IP-адрес сервера симулятора (по умолчанию: 127.0.0.1)')
    argparser.add_argument('-p', '--port', type=int, default=2000,
                           help='TCP-порт подключения (по умолчанию: 2000)')
    argparser.add_argument('--disable-seg', action='store_true', help='Отключить сенсор семантической сегментации')
    argparser.add_argument('--disable-rgb', action='store_true', help='Отключить RGB камеру')
    argparser.add_argument('--disable-depth', action='store_true', help='Отключить датчик глубины')
    argparser.add_argument('--disable-lidar', action='store_true', help='Отключить Lidar с семантической разметкой')
    argparser.add_argument('--disable-gnss', action='store_true', help='Отключить GNSS сенсор')
    argparser.add_argument('--disable-obstacle', action='store_true', help='Отключить датчик препятствий')
    argparser.add_argument('--disable-collision', action='store_true', help='Отключить датчик столкновений')
    argparser.add_argument('--disable-laneinvasion', action='store_true', help='Отключить датчик выезда из полосы')
    argparser.add_argument('--record-data', action='store_true', help='Включить запись и сохранение данных сенсоров и vehicle control')
    args = argparser.parse_args()

    # Настройка записи,
    recording_enabled = args.record_data
    if recording_enabled:
        base_record_dir = os.path.join("recordings", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(base_record_dir, exist_ok=True)
        recording_dirs = {
            'lidar_semantic': os.path.join(base_record_dir, "lidar_semantic"),
            'semantic_cam': os.path.join(base_record_dir, "semantic_cam"),
            'depth_cam': os.path.join(base_record_dir, "depth_cam"),
            'gnss': os.path.join(base_record_dir, "gnss"),
            'obstacle_sensor': os.path.join(base_record_dir, "obstacle_sensor"),
            'collision_sensor': os.path.join(base_record_dir, "collision_sensor"),
            'lane_invasion_sensor': os.path.join(base_record_dir, "lane_invasion_sensor"),
        }
        for d in recording_dirs.values():
            os.makedirs(d, exist_ok=True)
        control_csv_path = os.path.join(base_record_dir, "vehicle_control.csv")
        control_csv_file = open(control_csv_path, "w")
        control_csv_file.write("frame,timestamp,throttle,steer,brake,reverse,speed_kmh\n")
        gnss_csv_path = os.path.join(recording_dirs['gnss'], "gnss.csv")
        gnss_file = open(gnss_csv_path, "w")
        gnss_file.write("frame,timestamp,latitude,longitude,altitude\n")
        obstacle_csv_path = os.path.join(recording_dirs['obstacle_sensor'], "obstacle.csv")
        obstacle_file = open(obstacle_csv_path, "w")
        obstacle_file.write("frame,timestamp,event\n")
        collision_csv_path = os.path.join(recording_dirs['collision_sensor'], "collision.csv")
        collision_file = open(collision_csv_path, "w")
        collision_file.write("frame,timestamp,other_actor\n")
        lane_invasion_csv_path = os.path.join(recording_dirs['lane_invasion_sensor'], "lane_invasion.csv")
        lane_invasion_file = open(lane_invasion_csv_path, "w")
        lane_invasion_file.write("frame,timestamp,crossed_lane_markings\n")
        print(f"Запись данных включена. Данные сохраняются в {base_record_dir}")
        start_file_io_worker()  # Запуск потока для асинхронной записи

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.load_world('Town05_Opt', map_layers=carla.MapLayer.NONE)

    settings = world.get_settings()
    settings.no_rendering_mode = True
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / CARLA_FPS
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()

    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
    vehicle_bp.set_attribute('role_name', 'actor')
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        sys.exit("Ошибка: нет доступных точек спавна.")
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    print("Транспортное средство заспавнено.")

    actors = []

    if not args.disable_seg:
        seg_cam_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_cam_bp.set_attribute('image_size_x', '1280')
        seg_cam_bp.set_attribute('image_size_y', '720')
        seg_cam_bp.set_attribute('fov', '90')
        seg_cam_bp.set_attribute('sensor_tick', SENSOR_TICK)
        seg_cam_bp.set_attribute('lens_circle_falloff', '1.2')
        seg_cam_bp.set_attribute('lens_circle_multiplier', '1.0')
        seg_cam_bp.set_attribute('lens_k', '-0.2')
        seg_cam_bp.set_attribute('lens_kcube', '0.01')
        seg_cam_bp.set_attribute('role_name', 'semantic_cam')
        seg_cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        seg_cam = world.spawn_actor(seg_cam_bp, seg_cam_transform, attach_to=vehicle)
        seg_cam.listen(process_segmentation_image)
        actors.append(seg_cam)
    else:
        print("Сенсор семантической сегментации отключен.")

    if not args.disable_rgb:
        rgb_cam_bp = blueprint_library.find('sensor.camera.rgb')
        rgb_cam_bp.set_attribute('image_size_x', '1920')
        rgb_cam_bp.set_attribute('image_size_y', '1080')
        rgb_cam_bp.set_attribute('fov', '105')
        rgb_cam_bp.set_attribute('sensor_tick', SENSOR_TICK)
        rgb_cam_bp.set_attribute('lens_circle_falloff', '1.2')
        rgb_cam_bp.set_attribute('lens_circle_multiplier', '1.0')
        rgb_cam_bp.set_attribute('lens_k', '-0.2')
        rgb_cam_bp.set_attribute('lens_kcube', '0.01')
        rgb_cam_bp.set_attribute('role_name', 'rgb_camera')
        rgb_cam_transform = carla.Transform(
            carla.Location(x=0.25, y=-0.31, z=1.25),
            carla.Rotation(pitch=0, yaw=0, roll=0)
        )
        rgb_cam = world.spawn_actor(rgb_cam_bp, rgb_cam_transform, attach_to=vehicle)
        rgb_cam.listen(process_rgb_image)
        actors.append(rgb_cam)
    else:
        print("RGB камера отключена.")

    if not args.disable_lidar:
        lidar_bp = generate_lidar_bp(world)
        lidar_transform = carla.Transform(carla.Location(x=-0.5, z=2.4))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
        point_list = o3d.geometry.PointCloud()
        lidar.listen(lambda data: semantic_lidar_callback(data, point_list))
        actors.append(lidar)
    else:
        print("Lidar с семантической разметкой отключен.")

    if not args.disable_gnss:
        gnss_bp = blueprint_library.find('sensor.other.gnss')
        gnss_bp.set_attribute('sensor_tick', SENSOR_TICK)
        gnss_bp.set_attribute('role_name', 'gnss')
        gnss_transform = carla.Transform(carla.Location(x=0, y=0, z=2.8))
        gnss_sensor = world.spawn_actor(gnss_bp, gnss_transform, attach_to=vehicle)
        gnss_sensor.listen(process_gnss_data)
        actors.append(gnss_sensor)
    else:
        print("GNSS сенсор отключен.")

    if not args.disable_depth:
        depth_cam_bp = blueprint_library.find('sensor.camera.depth')
        depth_cam_bp.set_attribute('image_size_x', '1280')
        depth_cam_bp.set_attribute('image_size_y', '720')
        depth_cam_bp.set_attribute('fov', '90')
        depth_cam_bp.set_attribute('sensor_tick', SENSOR_TICK)
        depth_cam_bp.set_attribute('lens_circle_falloff', '1.2')
        depth_cam_bp.set_attribute('lens_circle_multiplier', '1.0')
        depth_cam_bp.set_attribute('lens_k', '-0.2')
        depth_cam_bp.set_attribute('lens_kcube', '0.01')
        depth_cam_bp.set_attribute('role_name', 'depth_cam')
        depth_cam_transform = carla.Transform(carla.Location(x=1.5, y=0.3, z=2.4))
        depth_cam = world.spawn_actor(depth_cam_bp, depth_cam_transform, attach_to=vehicle)
        depth_cam.listen(process_depth_image)
        actors.append(depth_cam)
    else:
        print("Датчик глубины отключен.")

    if not args.disable_obstacle:
        obstacle_bp = blueprint_library.find('sensor.other.obstacle')
        obstacle_bp.set_attribute('sensor_tick', SENSOR_TICK)
        obstacle_bp.set_attribute('debug_linetrace', 'False')
        obstacle_bp.set_attribute('distance', '2')
        obstacle_bp.set_attribute('only_dynamics', 'False')
        obstacle_bp.set_attribute('role_name', 'obstacle_sensor')
        obstacle_transform = carla.Transform(carla.Location(x=2.5, z=1.0))
        obstacle_detector = world.spawn_actor(obstacle_bp, obstacle_transform, attach_to=vehicle)
        obstacle_detector.listen(process_obstacle_data)
        actors.append(obstacle_detector)
    else:
        print("Датчик препятствий отключен.")

    if not args.disable_collision:
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_bp.set_attribute('role_name', 'collision_sensor')
        collision_transform = carla.Transform()
        collision_sensor = world.spawn_actor(collision_bp, collision_transform, attach_to=vehicle)
        collision_sensor.listen(process_collision_data)
        actors.append(collision_sensor)
    else:
        print("Датчик столкновений отключен.")

    if not args.disable_laneinvasion:
        lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')
        lane_invasion_bp.set_attribute('role_name', 'lane_invasion_sensor')
        lane_invasion_transform = carla.Transform()
        lane_invasion_sensor = world.spawn_actor(lane_invasion_bp, lane_invasion_transform, attach_to=vehicle)
        lane_invasion_sensor.listen(process_lane_invasion_data)
        actors.append(lane_invasion_sensor)
    else:
        print("Датчик выезда из полосы отключен.")

    actors.append(vehicle)

    display_thread = threading.Thread(target=display_loop)
    display_thread.daemon = True
    display_thread.start()

    if not args.disable_lidar:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='Carla Lidar', width=500, height=500)
        render_opt = vis.get_render_option()
        render_opt.background_color = [0.05, 0.05, 0.05]
        render_opt.point_size = 1
        render_opt.show_coordinate_frame = True
        add_open3d_axis(vis)
    else:
        vis = o3d.visualization.Visualizer()

    pygame.init()
    pygame.joystick.init()
    display_width, display_height = 800, 600
    screen = pygame.display.set_mode((display_width, display_height), pygame.RESIZABLE)
    pygame.display.set_caption("CARLA Manual Control & RGB Camera")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

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
    
    try:
        while True:
            if not args.disable_lidar:
                ctr = vis.get_view_control()
                ctr.set_lookat([0, 0, 0])
                ctr.set_up([0, 1, 0])
                ctr.set_zoom(0.3)
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
                control.steer = steer_axis * 0.25
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

            if frame == 2 and not args.disable_lidar:
                vis.add_geometry(point_list)
            if not args.disable_lidar:
                vis.update_geometry(point_list)
                vis.poll_events()
                vis.update_renderer()

            velocity = vehicle.get_velocity()
            speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6
            global latest_speed_kmh
            latest_speed_kmh = speed
            snapshot = world.get_snapshot()
            game_seconds = int(snapshot.timestamp.elapsed_seconds)
            hours = game_seconds // 3600
            minutes = (game_seconds % 3600) // 60
            seconds = game_seconds % 60
            global latest_game_time
            latest_game_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            # Запись данных vehicle control
            if recording_enabled:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                line = f"{snapshot.frame},{timestamp},{control.throttle},{control.steer},{control.brake},{control.reverse},{latest_speed_kmh}\n"
                file_io_queue.put({"type": "csv", "file": control_csv_file, "line": line})

            if not args.disable_rgb:
                with rgb_frame_lock:
                    rgb_frame = latest_rgb_frame.copy() if latest_rgb_frame is not None else None

                if rgb_frame is not None:
                    rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
                    surface = pygame.image.frombuffer(rgb_frame.tobytes(), (rgb_frame.shape[1], rgb_frame.shape[0]), "RGB")
                    surface = pygame.transform.scale(surface, (display_width, display_height))
                    screen.blit(surface, (0, 0))
                else:
                    screen.fill((30, 30, 30))
            else:
                screen.fill((30, 30, 30))
                text_surface = font.render("RGB камера отключена", True, (255, 255, 255))
                screen.blit(text_surface, (10, 10))

            overlay_lines = [
                f"Speed: {latest_speed_kmh:.1f} km/h",
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
            clock.tick(CARLA_FPS)
    except KeyboardInterrupt:
        print("\nЗавершение работы...")
    finally:
        cleanup(actors, vis)

if __name__ == '__main__':
    main()
