#!/usr/bin/env python
"""
CARLA Manual Control with RGB & Semantic Segmentation Cameras

Скрипт подключается к симулятору CARLA, спавнит транспортное средство,
прикрепляет к нему два сенсора:
  - сенсор семантической сегментации,
  - обычную (RGB) камеру.
Пользователь может управлять автомобилем с клавиатуры. При нажатии
стрелки Вверх автомобиль едет вперёд, при нажатии стрелки Вниз — назад.
Изображения с сенсоров выводятся в отдельных окнах (OpenCV).
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

# Добавляем путь к egg-файлу CARLA
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


def process_segmentation_image(image):
    """
    Обработка изображения семантической сегментации:
      - Преобразование с использованием палитры CityScapes
      - Преобразование raw данных в numpy-массив
      - Обновление глобальной переменной latest_seg_frame
    """
    image.convert(carla.ColorConverter.CityScapesPalette)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    frame = array[:, :, :3]  # Используем первые 3 канала (BGR)
    with seg_frame_lock:
        global latest_seg_frame
        latest_seg_frame = frame


def process_rgb_image(image):
    """
    Обработка изображения RGB:
      - Преобразование raw данных в numpy-массив
      - Обновление глобальной переменной latest_rgb_frame
    """
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    frame = array[:, :, :3]  # Избавляемся от альфа-канала
    with rgb_frame_lock:
        global latest_rgb_frame
        latest_rgb_frame = frame


def display_segmentation_loop():
    """Отдельный цикл отображения для сенсора семантической сегментации."""
    cv2.namedWindow("Semantic Segmentation", cv2.WINDOW_NORMAL)
    while True:
        with seg_frame_lock:
            frame = latest_seg_frame.copy() if latest_seg_frame is not None else None
        if frame is not None:
            cv2.imshow("Semantic Segmentation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.01)
    cv2.destroyWindow("Semantic Segmentation")


def display_rgb_loop():
    """Отдельный цикл отображения для обычного RGB видеопотока."""
    cv2.namedWindow("RGB Camera", cv2.WINDOW_NORMAL)
    while True:
        with rgb_frame_lock:
            frame = latest_rgb_frame.copy() if latest_rgb_frame is not None else None
        if frame is not None:
            cv2.imshow("RGB Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.01)
    cv2.destroyWindow("RGB Camera")


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
        sys.exit("Ошибка: не удалось подключиться к симулятору. Проверьте, что CARLA запущен по адресу {}:{}".format(args.host, args.port))

    blueprint_library = world.get_blueprint_library()

    # Спавним транспортное средство (первый найденный blueprint из категории vehicle)
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        sys.exit("Ошибка: нет доступных точек спавна.")
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    print("Транспортное средство заспавнено.")

    # Прикрепляем сенсор семантической сегментации
    seg_cam_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    seg_cam_bp.set_attribute('image_size_x', '800')
    seg_cam_bp.set_attribute('image_size_y', '600')
    seg_cam_bp.set_attribute('fov', '90')
    seg_cam_bp.set_attribute('sensor_tick', '0.05')
    seg_cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    seg_cam = world.spawn_actor(seg_cam_bp, seg_cam_transform, attach_to=vehicle)
    seg_cam.listen(lambda image: process_segmentation_image(image))

    # Прикрепляем обычную RGB камеру
    rgb_cam_bp = blueprint_library.find('sensor.camera.rgb')
    rgb_cam_bp.set_attribute('image_size_x', '1920')
    rgb_cam_bp.set_attribute('image_size_y', '1080')
    rgb_cam_bp.set_attribute('fov', '90')
    rgb_cam_bp.set_attribute('sensor_tick', '0.05')
    rgb_cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    rgb_cam = world.spawn_actor(rgb_cam_bp, rgb_cam_transform, attach_to=vehicle)
    rgb_cam.listen(lambda image: process_rgb_image(image))

    # Запускаем потоки для отображения видеопотоков
    seg_display_thread = threading.Thread(target=display_segmentation_loop)
    seg_display_thread.daemon = True
    seg_display_thread.start()

    rgb_display_thread = threading.Thread(target=display_rgb_loop)
    rgb_display_thread.daemon = True
    rgb_display_thread.start()

    # Инициализируем pygame для управления автомобилем
    pygame.init()
    display_width, display_height = 800, 600
    screen = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption("CARLA Manual Control")
    clock = pygame.time.Clock()

    control = carla.VehicleControl()
    control.throttle = 0.0
    control.steer = 0.0
    control.brake = 0.0
    control.reverse = False

    try:
        while True:
            # Обработка событий pygame
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

            # Логика управления:
            # Если нажата стрелка Вверх - движение вперёд (reverse=False)
            # Если нажата стрелка Вниз - движение назад (reverse=True)
            if keys[pygame.K_UP]:
                control.reverse = False
                control.throttle = 0.7
            elif keys[pygame.K_DOWN]:
                control.reverse = True
                control.throttle = 0.7

            if keys[pygame.K_LEFT]:
                control.steer = -0.3
            elif keys[pygame.K_RIGHT]:
                control.steer = 0.3

            vehicle.apply_control(control)

            # Обновление окна pygame
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
