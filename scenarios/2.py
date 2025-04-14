#!/usr/bin/env python3
"""
Сценарий 1: Прямолинейное движение в идеальных условиях (управление через клавиатуру)
Описание:
  Эго-автомобиль управляется пользователем с клавиатуры (WASD) и должен добраться от точки А до точки Б по прямой дороге.
  В окружении нет других участников движения.
"""

import carla
import pygame
import math
import sys

def get_keyboard_control():
    """
    Функция считывает нажатые клавиши и формирует объект управления.
    """
    control = carla.VehicleControl()
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        control.throttle = 1.0
    if keys[pygame.K_s]:
        control.brake = 1.0
    if keys[pygame.K_a]:
        control.steer = -0.5
    elif keys[pygame.K_d]:
        control.steer = 0.5
    return control

def main():
    pygame.init()
    pygame.font.init()
    display = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Сценарий 1: Прямолинейное движение - Управление через клавиатуру")

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    blueprint_library = world.get_blueprint_library()
    ego_vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

    spawn_points = world.get_map().get_spawn_points()
    if len(spawn_points) < 2:
        print("Недостаточно точек спауна для сценария.")
        return

    start_point = spawn_points[0]
    destination_point = spawn_points[10]

    # Спавним эго-автомобиль (управление оставляем пользователю)
    ego_vehicle = world.spawn_actor(ego_vehicle_bp, start_point)
    print("Эго-автомобиль создан в точке:", start_point.location)

    clock = pygame.time.Clock()

    try:
        while True:
            # Обработка событий pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

            # Получаем управление от пользователя
            control = get_keyboard_control()
            ego_vehicle.apply_control(control)

            # Расчёт расстояния до пункта назначения
            current_location = ego_vehicle.get_location()
            dx = destination_point.location.x - current_location.x
            dy = destination_point.location.y - current_location.y
            distance = math.sqrt(dx**2 + dy**2)
            if distance < 2.0:
                print("Достигнута точка назначения!")
                break

            # Обновляем окно (можно добавить более подробную визуализацию)
            display.fill((0, 0, 0))
            font = pygame.font.SysFont("Arial", 24)
            text_surface = font.render(f"Расстояние до цели: {round(distance,1)} м", True, (255, 255, 255))
            display.blit(text_surface, (20, 20))
            pygame.display.flip()

            clock.tick(30)
    finally:
        ego_vehicle.destroy()
        pygame.quit()
        print("Эго-автомобиль удалён.")

if __name__ == '__main__':
    main()


