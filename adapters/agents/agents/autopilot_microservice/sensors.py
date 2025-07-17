import carla


class Sensors:
    @staticmethod
    def get_sensor_config():
        return [
            {'type': 'sensor.camera.depth', 'x': 1.3, 'y': 0.0, 'z': 2.3,
             'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 592, 'height': 333, 'fov': 135,
             'sensor_tick': 0.1,
             'id': 'depth_front'},
            {'type': 'sensor.camera.instance_segmentation', 'x': 1.3, 'y': 0.0, 'z': 2.3,
             'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 592, 'height': 333, 'fov': 135,
             'sensor_tick': 0.1,
             'id': 'instance_segmentation_front'},
            {'type': 'sensor.other.imu', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'sensor_tick': 0.1, 'id': 'imu'},
            {'type': 'sensor.speedometer', 'reading_frequency': 10, 'id': 'speed'},
        ]

    @staticmethod
    def init_collision_sensor(world, hero_vehicle, callback):
        bp_lib = world.get_blueprint_library()
        bp_collision = bp_lib.find('sensor.other.collision')
        collision_sensor = world.spawn_actor(bp_collision, carla.Transform(), attach_to=hero_vehicle)
        collision_sensor.listen(callback)
        print("Sensors: Collision sensor initialized.")
        return collision_sensor
