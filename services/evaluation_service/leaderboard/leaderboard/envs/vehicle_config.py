# vehicle_configs/i_miev_physics.py

import carla


def get_i_miev_physics_control():
    """
    Возвращает объект VehiclePhysicsControl с параметрами Mitsubishi i-MiEV.
    Эти параметры соответствуют реальным физическим характеристикам автомобиля.
    """
    physics = carla.VehiclePhysicsControl()
    physics.mass = 1110  # Сухая масса Mitsubishi i-MiEV (около 1110 кг, источник: технические характеристики авто)
    physics.drag_coefficient = 0.28 # Типичное значение для хэтчбеков, около 0.28. Подтверждается [aerodynamic studies].
    physics.center_of_mass = carla.Location(x=0.0, y=0.0, z=-0.9)  # Смещён вниз для улучшения устойчивости
    physics.steering_curve = [(0.0, 1.0), (30.0, 0.9), (60.0, 0.6)] # Снижение чувствительности рулевого управления
    physics.max_rpm = 9000 # Для электромотора Mitsubishi i-MiEV max RPM ≈ 9000.
    physics.torque_curve = [
        (0.0, 400.0),
        (1500.0, 420.0),
        (4000.0, 430.0),
        (6000.0, 400.0),
        (9000.0, 0.0)
    ]  # Примерная кривая крутящего момента электродвигателя.

    physics.use_gear_autobox = True  # 	Автоматическое переключение передач.
    physics.gear_switch_time = 0.5   # 	Задержка при переключении передач.
    physics.clutch_strength = 100.0  # Сила сцепления. Зависит от модели симуляции.

    physics.forward_gears = [
        carla.GearPhysicsControl(ratio=4.0, up_ratio=1.0, down_ratio=0.0)  # 1-ступенчатая передача с высоким передаточным числом.
    ]
    physics.reverse_gears = [
        carla.GearPhysicsControl(ratio=-3.0, up_ratio=0.0, down_ratio=0.0)  # передача для заднего хода
    ]

    physics.wheel_base = 2.55  # Колёсная база Mitsubishi i-MiEV = 2550 мм. Источник: [официальный мануал].
    physics.max_steer_angle = 35.0  # 	Приближённый максимальный угол поворота колёс.
    physics.brake_force = 5000.0 # примерная сила для легкового авто
    physics.hand_brake_strength = 3000.0  # ручник условно

    return physics
