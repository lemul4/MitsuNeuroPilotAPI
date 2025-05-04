# vehicle_configs/i_miev_physics.py

from carla import VehiclePhysicsControl

def get_i_miev_physics_control() -> VehiclePhysicsControl:
    """
    Возвращает объект VehiclePhysicsControl с параметрами Mitsubishi i-MiEV.
    Эти параметры соответствуют реальным физическим характеристикам автомобиля.
    """
    physics = VehiclePhysicsControl()

    physics.mass = 1110  # кг
    physics.drag_coefficient = 0.28
    physics.center_of_mass = [0.0, 0.0, -0.9]  # центр масс, немного занижен
    physics.steering_curve = [(0.0, 1.0), (30.0, 0.9), (60.0, 0.6)]

    physics.max_rpm = 9000
    physics.torque_curve = [
        (0.0, 400.0),
        (1500.0, 420.0),
        (4000.0, 430.0),
        (6000.0, 400.0),
        (9000.0, 0.0)
    ]

    physics.use_gear_autobox = True
    physics.gear_switch_time = 0.5
    physics.clutch_strength = 100.0

    physics.forward_gears = [
        carla.GearPhysicsControl(ratio=4.0, up_ratio=1.0, down_ratio=0.0)
    ]
    physics.reverse_gears = [
        carla.GearPhysicsControl(ratio=-3.0, up_ratio=0.0, down_ratio=0.0)
    ]

    physics.wheel_base = 2.55  # метра
    physics.max_steer_angle = 35.0  # градусы
    physics.brake_force = 5000.0
    physics.hand_brake_strength = 3000.0

    return physics
