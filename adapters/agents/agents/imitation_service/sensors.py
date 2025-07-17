def sensors(sensor_config):
    return [
        {
            'type': 'sensor.camera.depth',
            'x': 1.3,
            'y': 0.0,
            'z': 2.3,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'width': sensor_config['width'],
            'height': sensor_config['height'],
            'fov': sensor_config['fov'],
            'sensor_tick': 0.1,
            'id': 'depth_front'
        },
        {
            'type': 'sensor.camera.instance_segmentation',
            'x': 1.3,
            'y': 0.0,
            'z': 2.3,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'width': sensor_config['width'],
            'height': sensor_config['height'],
            'fov': sensor_config['fov'],
            'sensor_tick': 0.1,
            'id': 'instance_segmentation_front'
        },
        {
            'type': 'sensor.other.imu',
            'x': 0.0,
            'y': 0.0,
            'z': 0.0,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'sensor_tick': 0.1,
            'id': 'imu'
        },
        {
            'type': 'sensor.speedometer',
            'reading_frequency': 20,
            'id': 'speed'
        }
    ]
