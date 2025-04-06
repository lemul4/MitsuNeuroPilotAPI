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

# Подключение к серверу CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Получаем мир
world = client.get_world()

# Получаем список моделей машин
blueprint_library = world.get_blueprint_library()

# Выбираем определенную машину
vehicle_bp = blueprint_library.find('vehicle.audi.a2')

# Настраиваем цвет машины
vehicle_bp.set_attribute('color', '255,0,0')  # Красный цвет

# Получаем список доступных точек спавна
spawn_points = world.get_map().get_spawn_points()

# Спавним машину в первой доступной точке
if spawn_points:
    spawn_point = spawn_points[0]
    
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print("Машина заспавнена!")
    
    physics_control = vehicle.get_physics_control()
    physics_control.gear_switch_time = 0.5
    vehicle.apply_physics_control(physics_control)

    # Изменяем физические параметры
    physics_control.mass = 1080  # Масса до 1080 кг

    # Включаем автопилот
    vehicle.set_autopilot(True)
    
    # Применяем настройки к автомобилю
    vehicle.apply_physics_control(physics_control)

    spectator = world.get_spectator()
    transform = vehicle.get_transform()
    spectator.set_transform(carla.Transform(transform.location + carla.Location(z=30),
    carla.Rotation(pitch=-90)))

    # Ждем 10 секунд
    time.sleep(10)
    
    # Удаляем машину
    vehicle.destroy()
    print("Машина удалена!")
else:
    print("Нет доступных точек спавна!")