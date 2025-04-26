import copy
import logging
import numpy as np
import os
import time
from threading import Thread

from queue import Queue
from queue import Empty

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.setDaemon(True)
        thread.start()

        return thread
    return wrapper


class SensorConfigurationInvalid(Exception):
    """
    Exceptions thrown when the sensors used by the agent are not allowed for that specific submissions
    """

    def __init__(self, message):
        super(SensorConfigurationInvalid, self).__init__(message)


class SensorReceivedNoData(Exception):
    """
    Exceptions thrown when the sensors used by the agent take too long to receive data
    """

    def __init__(self, message):
        super(SensorReceivedNoData, self).__init__(message)


class GenericMeasurement(object):
    def __init__(self, data, frame):
        self.data = data
        self.frame = frame


class BaseReader(object):
    def __init__(self, vehicle, reading_frequency=1.0):
        self._vehicle = vehicle
        self._reading_frequency = reading_frequency
        self._callback = None
        self._run_ps = True
        self.run()

    def __call__(self):
        pass

    @threaded
    def run(self):
        first_time = True
        latest_time = GameTime.get_time()
        while self._run_ps:
            if self._callback is not None:
                current_time = GameTime.get_time()

                # Second part forces the sensors to send data at the first tick, regardless of frequency
                if current_time - latest_time > (1 / self._reading_frequency) \
                        or (first_time and GameTime.get_frame() != 0):
                    self._callback(GenericMeasurement(self.__call__(), GameTime.get_frame()))
                    latest_time = GameTime.get_time()
                    first_time = False

                else:
                    time.sleep(0.001)

    def listen(self, callback):
        # Tell that this function receives what the producer does.
        self._callback = callback

    def stop(self):
        self._run_ps = False

    def destroy(self):
        self._run_ps = False


class SpeedometerReader(BaseReader):
    """
    Sensor to measure the speed of the vehicle.
    """
    MAX_CONNECTION_ATTEMPTS = 10

    def _get_forward_speed(self, transform=None, velocity=None):
        """ Convert the vehicle transform directly to forward speed """
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed

    def __call__(self):
        """ We convert the vehicle physics information into a convenient dictionary """

        # protect this access against timeout
        attempts = 0
        while attempts < self.MAX_CONNECTION_ATTEMPTS:
            try:
                velocity = self._vehicle.get_velocity()
                transform = self._vehicle.get_transform()
                break
            except Exception:
                attempts += 1
                time.sleep(0.2)
                continue

        return {'speed': self._get_forward_speed(transform=transform, velocity=velocity)}


class OpenDriveMapReader(BaseReader):
    def __call__(self):
        return {'opendrive': CarlaDataProvider.get_map().to_opendrive()}


class CallBack(object):
    def __init__(self, tag, sensor_type, sensor, data_provider):
        self._tag = tag
        self._data_provider = data_provider

        self._data_provider.register_sensor(tag, sensor_type, sensor)

    def __call__(self, data):
        if isinstance(data, carla.libcarla.Image):
            self._parse_image_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.LidarMeasurement):
            self._parse_lidar_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.RadarMeasurement):
            self._parse_radar_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.GnssMeasurement):
            self._parse_gnss_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.IMUMeasurement):
            self._parse_imu_cb(data, self._tag)
        elif isinstance(data, GenericMeasurement):
            self._parse_pseudosensor(data, self._tag)
        else:
            logging.error('No callback method for this sensor.')

    # Parsing CARLA physical Sensors
    def _parse_image_cb(self, image, tag):
        if 'depth' in tag:
            # Преобразуем изображение в логарифмическую глубину
            image.convert(carla.ColorConverter.LogarithmicDepth)

            # Преобразуем данные изображения в массив NumPy
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))  # Формат BGRA

            # Извлекаем каналы B, G и R
            B = array[:, :, 0].astype(np.float32)
            G = array[:, :, 1].astype(np.float32)
            R = array[:, :, 2].astype(np.float32)

            # Вычисляем нормализованную глубину
            normalized_depth = (R + G * 256 + B * 256 * 256) / (256 ** 3 - 1)

            # Преобразуем нормализованную глубину в метры (максимальная глубина 1000 м)
            depth_meters = normalized_depth * 1000.0

            # Обновляем данные сенсора
            self._data_provider.update_sensor(tag, depth_meters, image.frame)

        elif 'instance' in tag:
            LABEL_COLORS = np.array([
                (0, 0, 0),  # Unlabeled
                (128, 64, 128),  # Road
                (244, 35, 232),  # Sidewalk
                (70, 70, 70),  # Building
                (102, 102, 156),  # Wall
                (190, 153, 153),  # Fence
                (153, 153, 153),  # Pole
                (250, 170, 30),  # Traffic Light
                (220, 220, 0),  # Traffic Sign
                (107, 142, 35),  # Vegetation
                (152, 251, 152),  # Terrain
                (70, 130, 180),  # Sky
                (220, 20, 60),  # Pedestrian
                (255, 0, 0),  # Rider
                (0, 0, 142),  # Car
                (0, 0, 70),  # Truck
                (0, 60, 100),  # Bus
                (0, 80, 100),  # Train
                (0, 0, 230),  # Motorcycle
                (119, 11, 32),  # Bicycle
                (110, 190, 160),  # Static
                (170, 120, 50),  # Dynamic
                (55, 90, 80),  # Other
                (45, 60, 150),  # Water
                (157, 234, 50),  # Road Line
                (81, 0, 81),  # Ground
                (150, 100, 100),  # Bridge
                (230, 150, 140),  # Rail Track
                (180, 165, 180)  # Guard Rail
            ]) / 255.0

            PALETTE = (LABEL_COLORS * 255).astype(np.uint8)
            DESIRED_IDS = {12, 13, 14, 15, 16, 17, 18, 19, }
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))

            # split
            B = array[:, :, 0].astype(np.uint16)
            G = array[:, :, 1].astype(np.uint16)
            sem = array[:, :, 2].astype(np.uint8)  # semantic id
            inst_id = B + (G << 8)  # уникальный instance id

            # подготовим выходное BGRA-поле
            out = np.zeros_like(array)

            # 1) для НЕ-интересных семантических классов раскрашиваем по палитре
            mask_semantic = ~np.isin(sem, list(DESIRED_IDS))
            out[mask_semantic, :3] = PALETTE[sem[mask_semantic]]
            out[mask_semantic, 3] = 255

            # 2) для интересных (истинных инстансов) закодируем inst_id в цвет
            #    — можно любой способ, здесь просто разложим обратно в два канала + sem
            mask_inst = ~mask_semantic
            out[mask_inst, 0] = (inst_id[mask_inst] & 0xFF).astype(np.uint8)  # B
            out[mask_inst, 1] = ((inst_id[mask_inst] >> 8) & 0xFF).astype(np.uint8)  # G
            out[mask_inst, 2] = sem[mask_inst]  # R = semantic id
            out[mask_inst, 3] = 255

            # и обновляем сенсор
            self._data_provider.update_sensor(tag, out, image.frame)

        elif 'semantic' in tag:
            image.convert(carla.ColorConverter.CityScapesPalette)
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))

            self._data_provider.update_sensor(tag, array, image.frame)

        else:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            self._data_provider.update_sensor(tag, array, image.frame)

    def _parse_lidar_cb(self, lidar_data, tag):
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        self._data_provider.update_sensor(tag, points, lidar_data.frame)

    def _parse_radar_cb(self, radar_data, tag):
        # [depth, azimuth, altitute, velocity]
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        points = np.flip(points, 1)
        self._data_provider.update_sensor(tag, points, radar_data.frame)

    def _parse_gnss_cb(self, gnss_data, tag):
        array = np.array([gnss_data.latitude,
                          gnss_data.longitude,
                          gnss_data.altitude], dtype=np.float64)
        self._data_provider.update_sensor(tag, array, gnss_data.frame)

    def _parse_imu_cb(self, imu_data, tag):
        array = np.array([imu_data.accelerometer.x,
                          imu_data.accelerometer.y,
                          imu_data.accelerometer.z,
                          imu_data.gyroscope.x,
                          imu_data.gyroscope.y,
                          imu_data.gyroscope.z,
                          imu_data.compass,
                         ], dtype=np.float64)
        self._data_provider.update_sensor(tag, array, imu_data.frame)

    def _parse_pseudosensor(self, package, tag):
        self._data_provider.update_sensor(tag, package.data, package.frame)


class SensorInterface(object):
    def __init__(self):
        self._sensors_objects = {}
        self._data_buffers = Queue()
        self._queue_timeout = 10

        # Only sensor that doesn't get the data on tick, needs special treatment
        self._opendrive_tag = None

    def register_sensor(self, tag, sensor_type, sensor):
        if tag in self._sensors_objects:
            raise SensorConfigurationInvalid("Duplicated sensor tag [{}]".format(tag))

        self._sensors_objects[tag] = sensor

        if sensor_type == 'sensor.opendrive_map': 
            self._opendrive_tag = tag

    def update_sensor(self, tag, data, frame):
        if tag not in self._sensors_objects:
            raise SensorConfigurationInvalid("The sensor with tag [{}] has not been created!".format(tag))

        self._data_buffers.put((tag, frame, data))

    def get_data(self, frame):
        """Read the queue to get the sensors data"""
        try:
            data_dict = {}
            while len(data_dict.keys()) < len(self._sensors_objects.keys()):
                # Don't wait for the opendrive sensor
                if self._opendrive_tag and self._opendrive_tag not in data_dict.keys() \
                        and len(self._sensors_objects.keys()) == len(data_dict.keys()) + 1:
                    break

                sensor_data = self._data_buffers.get(True, self._queue_timeout)
                if sensor_data[1] != frame:
                    continue
                data_dict[sensor_data[0]] = ((sensor_data[1], sensor_data[2]))

        except Empty:
            raise SensorReceivedNoData("A sensor took too long to send their data")

        return data_dict
