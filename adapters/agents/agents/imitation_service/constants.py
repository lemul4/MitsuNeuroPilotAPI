import math

DEBUG = False

SENSOR_CONFIG = {
    'width': 592,
    'height': 333,
    'fov': 135
}

stats = {
    'x': [-10000.0, 10000.0],
    'y': [-10000.0, 10000.0],
    'theta': [0.0, 2 * math.pi],
    'speed': [-5.0, 15.0],
    'near_node_x': [-10000.0, 10000.0],
    'near_node_y': [-10000.0, 10000.0],
    'far_node_x': [-10000.0, 10000.0],
    'far_node_y': [-10000.0, 10000.0],
    'angle_near': [-180, 180],
    'angle_far': [-180, 180],
    'distanse': [0.0, 20.0],
    'steer_sequence': [-1.0, 1.0],
    'throttle_sequence': [0.0, 1.0],
    'brake_sequence': [0.0, 1.0],
    'light_sequence': [0.0, 1.0],
    'steer': [-1.0, 1.0],
    'throttle': [0.0, 1.0],
    'brake': [0.0, 1.0],
    'light': [0.0, 1.0],
}