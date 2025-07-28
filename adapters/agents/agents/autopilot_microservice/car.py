"""Configuration for car Mitsubishi i-MiEV"""

"""
Длина: 3 475 мм
Ширина: 1 475 мм
Высота: 1 610 мм
Колёсная база: 2 550 мм
Мощность электродвигателя: 67 л.с.
Максимальная скорость: 130 км/ч
Запас хода: около 150 км
Масса: 1080 кг
"""

try:
    import carla
except ImportError as e:
    raise ModuleNotFoundError('CARLA scenarios require the "carla" Python package') from e

import time


def spawn_hero_vehicle(client: carla.Client, blueprint_filter="vehicle.audi.a2") -> carla.Actor:
    """
    Спавнит "геройскую" машину в свободной точке спавна.
    Возвращает carla.Actor или выбрасывает ошибку, если не удалось.
    """
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find(blueprint_filter)

    # Устанавливаем роль на blueprint перед спавном
    if vehicle_bp.has_attribute('role_name'):
        vehicle_bp.set_attribute('role_name', 'hero')

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available in the map")

    for spawn_point in spawn_points:
        # Проверяем, нет ли в радиусе 3 м уже машины
        overlapping_actors = world.get_actors().filter('vehicle.*')
        is_occupied = False
        for actor in overlapping_actors:
            dist = actor.get_location().distance(spawn_point.location)
            if dist < 3.0:
                is_occupied = True
                break
        if not is_occupied:
            try:
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                print(f"Spawned hero vehicle at {spawn_point.location}")

                # Настройка физики (пример)
                physics_control = vehicle.get_physics_control()
                physics_control.gear_switch_time = 0.5
                physics_control.mass = 1080  # Масса до 1080 кг
                vehicle.apply_physics_control(physics_control)

                # Включаем автопилот
                vehicle.set_autopilot(True)

                # Устанавливаем камеру наблюдателя сверху
                spectator = world.get_spectator()
                transform = vehicle.get_transform()
                spectator.set_transform(carla.Transform(transform.location + carla.Location(z=30),
                                                        carla.Rotation(pitch=-90)))

                return vehicle
            except RuntimeError:
                # Если спавн не удался (коллизия и т.п.) — пробуем другую точку
                continue

    raise RuntimeError("Failed to spawn hero vehicle: no free spawn points")


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    vehicle = spawn_hero_vehicle(client)

    print("Vehicle spawned. Running 10 seconds...")

    time.sleep(10)

    print("Destroying vehicle...")
    vehicle.destroy()
    print("Vehicle destroyed.")


if __name__ == "__main__":
    main()
