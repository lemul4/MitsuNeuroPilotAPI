#!/usr/bin/env python
"""
CARLA Manual Control with RGB & Semantic Segmentation Cameras

Скрипт подключается к симулятору CARLA, спавнит транспортное средство,
прикрепляет к нему два сенсора:
 сенсор семантической сегментации,
 обычную (RGB) камеру.
Пользователь может управлять автомобилем с клавиатуры.
"""

import glob
import os
import sys
import argparse
import time
import threading
import pygame
import numpy as np
import cv2
import math

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# Глобальные переменные для кадров с сенсоров и соответствующие блокировки
latest_seg_frame = None
seg_frame_lock = threading.Lock()

latest_rgb_frame = None
rgb_frame_lock = threading.Lock()

# Глобальные переменные для отображения данных водителя
latest_speed_kmh = 0.0
latest_rpm = 0.0
latest_game_time = "00:00:00"


def process_segmentation_image(image):
    image.convert(carla.ColorConverter.CityScapesPalette)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    frame = array[:, :, :3]
    with seg_frame_lock:
        global latest_seg_frame
        latest_seg_frame = frame


def process_rgb_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    frame = array[:, :, :3]
    with rgb_frame_lock:
        global latest_rgb_frame
        latest_rgb_frame = frame


def display_loop():
    """Отображение видеопотоков.
       Если для сегментации новый кадр не получен, используется последний.
       Для RGB-камеры поверх изображения в правом нижнем углу выводится
       оверлей с данными спидометра, оборотов двигателя, времени симуляции и др.
    """
    cv2.startWindowThread()
    cv2.namedWindow("Semantic Segmentation", cv2.WINDOW_NORMAL)
    cv2.namedWindow("RGB Camera", cv2.WINDOW_NORMAL)
    last_seg_frame = None  # Хранит предыдущий кадр сегментации

    while True:
        with seg_frame_lock:
            seg_frame = latest_seg_frame.copy() if latest_seg_frame is not None else None
        with rgb_frame_lock:
            rgb_frame = latest_rgb_frame.copy() if latest_rgb_frame is not None else None

        if seg_frame is not None:
            last_seg_frame = seg_frame

        if last_seg_frame is not None:
            cv2.imshow("Semantic Segmentation", last_seg_frame)

        if rgb_frame is not None:
            # Создаем копию для наложения данных
            overlay_frame = rgb_frame.copy()
            # Формируем строки с данными для оверлея
            lines = [
                f"Speed: {latest_speed_kmh:.1f} km/h",
                f"RPM: {latest_rpm:.0f}",
                f"Time: {latest_game_time}"
            ]
            # Настройки текста
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            color = (255, 255, 255)  # белый
            thickness = 2
            margin = 10
            y0 = overlay_frame.shape[0] - margin
            # Рисуем строки начиная с нижней части кадра
            for line in reversed(lines):
                (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
                y0 = y0 - text_height - baseline - 5
                x0 = overlay_frame.shape[1] - text_width - margin
                cv2.putText(overlay_frame, line, (x0, y0), font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.imshow("RGB Camera", overlay_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.01)
    cv2.destroyAllWindows()


def main():
    argparser = argparse.ArgumentParser(
        description="CARLA Manual Control with RGB & Semantic Segmentation Cameras")
    argparser.add_argument('--host', default='127.0.0.1',
                           help='IP-адрес сервера симулятора (по умолчанию: 127.0.0.1)')
    argparser.add_argument('-p', '--port', type=int, default=2000,
                           help='TCP-порт подключения (по умолчанию: 2000)')
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

    # Сенсор семантической сегментации
    seg_cam_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    seg_cam_bp.set_attribute('image_size_x', '800')
    seg_cam_bp.set_attribute('image_size_y', '600')
    seg_cam_bp.set_attribute('fov', '90')
    seg_cam_bp.set_attribute('sensor_tick', '0.05')
    seg_cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    seg_cam = world.spawn_actor(seg_cam_bp, seg_cam_transform, attach_to=vehicle)
    seg_cam.listen(lambda image: process_segmentation_image(image))

    # Сенсор RGB камеры, установленной внутри салона на уровне головы водителя
    rgb_cam_bp = blueprint_library.find('sensor.camera.rgb')
    rgb_cam_bp.set_attribute('image_size_x', '1920')
    rgb_cam_bp.set_attribute('image_size_y', '1080')
    rgb_cam_bp.set_attribute('fov', '90')
    rgb_cam_bp.set_attribute('sensor_tick', '0.05')
    rgb_cam_transform = carla.Transform(
        carla.Location(x=0.11, y=-0.34, z=1.3),
        carla.Rotation(pitch=0, yaw=0, roll=0)
    )
    rgb_cam = world.spawn_actor(rgb_cam_bp, rgb_cam_transform, attach_to=vehicle)
    rgb_cam.listen(lambda image: process_rgb_image(image))

    # Запуск потока отображения видеопотоков
    display_thread = threading.Thread(target=display_loop)
    display_thread.daemon = True
    display_thread.start()

    pygame.init()
    pygame.joystick.init()
    display_width, display_height = 800, 600
    screen = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption("CARLA Manual Control")
    clock = pygame.time.Clock()

    # Если руль (joystick) подключён – инициализируем его
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

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt

            keys = pygame.key.get_pressed()
            # Сброс управления
            control.throttle = 0.0
            control.steer = 0.0
            control.brake = 0.0

            # Если нажаты клавиши-стрелки, используем клавиатурное управление
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

            # Если клавиатура не используется и руль подключён – читаем данные с руля
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

            # Обновление данных для оверлея
            velocity = vehicle.get_velocity()
            speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6
            global latest_speed_kmh, latest_rpm, latest_game_time
            latest_speed_kmh = speed
            latest_rpm = control.throttle * 7000  # Примерная симуляция оборотов
            snapshot = world.get_snapshot()
            game_seconds = int(snapshot.timestamp.elapsed_seconds)
            hours = game_seconds // 3600
            minutes = (game_seconds % 3600) // 60
            seconds = game_seconds % 60
            latest_game_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            screen.fill((0, 0, 0))
            pygame.display.flip()
            clock.tick(60)
    except KeyboardInterrupt:
        print("Завершение работы...")
    finally:
        seg_cam.stop()
        seg_cam.destroy()
        rgb_cam.stop()
        rgb_cam.destroy()
        vehicle.destroy()
        pygame.quit()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
