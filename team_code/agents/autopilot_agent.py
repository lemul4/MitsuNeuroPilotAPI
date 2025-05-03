import os
import weakref
import numpy as np
import carla
import random
import cv2
import json
import threading
import queue

from collections import deque
from datetime import datetime
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.autoagents import autonomous_agent

from agent_utils import base_utils
from agent_utils.pid_controller import PIDController
from agent_utils.planner import RoutePlanner


DEBUG = False
SENSOR_CONFIG = {
    'width': 592,
    'height': 333,
    'fov': 135
}

WEATHERS = {
        'ClearNoon': carla.WeatherParameters.ClearNoon,
}
WEATHERS_IDS = list(WEATHERS)


def get_entry_point():
    return 'AutopilotAgent'


class AutopilotAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self._sensor_data = SENSOR_CONFIG
        self.config_path = path_to_conf_file
        self.debug = DEBUG
        self.step = -1
        self.stop_step = 0
        self.not_brake_step = 0
        self.data_count = 0
        self.initialized = False
        self.weather_id = WEATHERS_IDS[0]
        self.last_brake = 0.0
        self.last_throttle = 0.0
        self.last_steer = 0.0
        self.last_speed = 0.0

        today = datetime.today()
        now = datetime.now()
        current_date = today.strftime("%b_%d_%Y")
        current_time = now.strftime("%H_%M_%S")
        time_info = f"/{current_date}-{current_time}/"

        self.dataset_save_path = os.path.join("dataset/imitation" + time_info)

        # Threading setup for async saving
        self._save_queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._save_worker, daemon=True)
        self._worker_thread.start()

    def _save_worker(self):
        while True:
            task = self._save_queue.get()
            if task is None:
                break
            img_depth, img_seg, data, paths, idx = task
            try:
                # Depth: normalize or save raw depth
                # Here img_depth is a float array (meters), convert to 16-bit
                depth_vis = (img_depth / np.nanmax(img_depth) * 65535).astype(np.uint16)
                cv2.imwrite(os.path.join(paths[0], f"{idx:04d}.png"), depth_vis)
                # Instance segmentation: assume img_seg is uint8 RGBA with class IDs
                seg_vis = img_seg
                cv2.imwrite(os.path.join(paths[1], f"{idx:04d}.png"), seg_vis)
                with open(os.path.join(paths[2], f"{idx:04d}.json"), 'w+', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"Error saving task {idx}: {e}")
            finally:
                self._save_queue.task_done()

    def tear_down(self):
        # Signal worker to exit and wait
        self._save_queue.put(None)
        self._worker_thread.join()

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        self._plan_HACK = global_plan_world_coord
        self._plan_gps_HACK = global_plan_gps

    def _init(self):
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._command_planner = RoutePlanner(7.5, 25.0, 257)
        self._route_planner.set_route(self._plan_gps_HACK, True)
        self._command_planner.set_route(self._global_plan, True)
        self.init_dataset(self.dataset_save_path)
        self.init_auto_pilot()
        self.init_privileged_agent()
        self.initialized = True
        self.count_vehicle_stop = 0
        self.count_is_seen = 0
        self.smoothed_steer = 0.0
        self.speed_sequence = deque(np.zeros(10), maxlen=10)
        self.brake_sequence = deque(np.zeros(10), maxlen=10)
        self.throttle_sequence = deque(np.zeros(10), maxlen=10)
        self.steer_sequence = deque(np.zeros(10), maxlen=10)
        if self.debug:
            cv2.namedWindow("rgb-front")

    def init_dataset(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.subfolder_paths = []
        """"rgb_front", "rgb_front_60", "rgb_rear","""
        subfolders = [
            "depth_front", "instance_segmentation_front", "measurements"
        ]
        for sub in subfolders:
            path = os.path.join(output_dir, sub)
            os.makedirs(path, exist_ok=True)
            self.subfolder_paths.append(path)

    def init_auto_pilot(self):
        self._turn_controller = PIDController(K_P=1.20, K_I=0.70, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

    def init_privileged_agent(self):
        self.hero_vehicle = CarlaDataProvider.get_hero_actor()
        self.world = self.hero_vehicle.get_world()
        self.privileged_sensors()

    def privileged_sensors(self):
        bp_lib = self.world.get_blueprint_library()
        bp_collision = bp_lib.find('sensor.other.collision')
        self.is_collision = False
        self.collision_sensor = self.world.spawn_actor(bp_collision, carla.Transform(), attach_to=self.hero_vehicle)
        self.collision_sensor.listen(lambda event: AutopilotAgent._on_collision(weakref.ref(self), event))

    def sensors(self):
        """{'type': 'sensor.camera.rgb', 'x': 1.3, 'y': 0.0, 'z': 2.3,
         'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
         'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
         'sensor_tick': 0.1,
         'id': 'rgb_front'},
        # RGB front 60
        {'type': 'sensor.camera.rgb', 'x': 1.3, 'y': 0.0, 'z': 2.3,
         'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
         'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': 60,
         'sensor_tick': 0.1,
         'id': 'rgb_front_60'},
        # RGB rear
        {'type': 'sensor.camera.rgb', 'x': -1.3, 'y': 0.0, 'z': 2.3,
         'roll': 0.0, 'pitch': 0.0, 'yaw': -180.0,
         'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
         'sensor_tick': 0.1,
         'id': 'rgb_rear'},"""
        return [
            # Depth front
            {'type': 'sensor.camera.depth', 'x':1.3,'y':0.0,'z':2.3,
             'roll':0.0,'pitch':0.0,'yaw':0.0,
             'width':self._sensor_data['width'],'height':self._sensor_data['height'],'fov':self._sensor_data['fov'],
             'sensor_tick': 0.1,
             'id':'depth_front'},
            # segmentation front
            {'type': 'sensor.camera.instance_segmentation', 'x':1.3,'y':0.0,'z':2.3,
             'roll':0.0,'pitch':0.0,'yaw':0.0,
             'width':self._sensor_data['width'],'height':self._sensor_data['height'],'fov':self._sensor_data['fov'],
             'sensor_tick': 0.1,
             'id':'instance_segmentation_front'},
            # GNSS
            {'type':'sensor.other.gnss','x':0.0,'y':0.0,'z':0.0,'roll':0.0,'pitch':0.0,'yaw':0.0,
             'sensor_tick':0.1,'id':'gps'},
            # IMU
            {'type':'sensor.other.imu','x':0.0,'y':0.0,'z':0.0,'roll':0.0,'pitch':0.0,'yaw':0.0,
             'sensor_tick':0.1,'id':'imu'},
            # Speedometer
            {'type':'sensor.speedometer','reading_frequency':10,'id':'speed'}
        ]

    def stop_function(self, is_stop):
        if self.stop_step < 200 and is_stop is not None:
            self.stop_step += 1
            self.not_brake_step = 0
            return True

        else:
            if self.not_brake_step < 300:
                self.not_brake_step += 1
            else:
                self.stop_step = 0
            return None

    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        data = self.tick(input_data)
        gps = self.get_position(data)
        speed = data['speed']
        compass = data['compass']



        # fix for theta=nan in some measurements
        if np.isnan(data['compass']):
            ego_theta = 0.0
        else:
            ego_theta = compass

        near_node, near_command = self._route_planner.run_step(gps)
        far_node, far_command = self._command_planner.run_step(gps)



        fused_inputs = np.zeros(3, dtype=np.float32)

        fused_inputs[0] = speed
        fused_inputs[1] = near_node[0] - gps[0]
        fused_inputs[2] = near_node[1] - gps[1]


        # if any of the following is not None, then the agent should brake
        is_light, is_walker, is_vehicle, is_stop = self.traffic_data()

        # by using priviledged information determine braking
        is_brake = any(x is not None for x in [is_light, is_walker, is_vehicle, is_stop])

        # apply pid controllers
        steer, throttle, brake, target_speed, angle = self.get_control(target=near_node, far_target=far_node, tick_data=data, brake=is_brake)


        self.is_collision = False

        applied_control = carla.VehicleControl()
        applied_control.throttle = throttle
        applied_control.steer = steer
        applied_control.brake = brake

        if self.step % 10 == 0:
            self.steer_sequence.append(self.last_steer)
            self.throttle_sequence.append(self.last_throttle)
            self.brake_sequence.append(self.last_brake)
            self.speed_sequence.append(self.last_speed)
            self.last_brake = float(brake)
            self.last_throttle = float(throttle)
            self.last_steer = float(steer)
            self.last_speed = float(speed)

            """rgb_front = input_data['rgb_front'][1][:, :, :3]
                    rgb_front_60 = input_data['rgb_front_60'][1][:, :, :3]
                    rgb_rear = input_data['rgb_rear'][1][:, :, :3]"""
            depth_front = input_data['depth_front'][1]  # float32 depth in meters
            inst_seg = input_data['instance_segmentation_front'][1][:, :, :-1]


            measurement_data = {
                'x': gps[0],
                'y': gps[1],

                'speed': speed,
                'theta': ego_theta,

                'x_command': far_node[0],
                'y_command': far_node[1],
                'far_command': far_command.value,

                'near_node_x': near_node[0],
                'near_node_y': near_node[1],
                'near_command': near_command.value,

                'steer': applied_control.steer,
                'throttle': applied_control.throttle,
                'brake': applied_control.brake,

                'angle': self.angle,
                'angle_unnorm': self.angle_unnorm,
                'angle_far_unnorm': self.angle_far_unnorm,

                'is_red_light_present': self.is_red_light_present,
                'is_stops_present': self.is_stops_present,

                'steer_sequence': np.array(self.steer_sequence, dtype=np.float32).tolist(),
                'throttle_sequence': np.array(self.throttle_sequence, dtype=np.float32).tolist(),
                'brake_sequence': np.array(self.brake_sequence, dtype=np.float32).tolist(),
                'speed_sequence': np.array(self.speed_sequence, dtype=np.float32).tolist(),

            }

            """image_front=data['rgb_front'],
                            image_front_60=data['rgb_front_60'],
                            image_rear=data['rgb_rear'],"""
            self.save_data(
                image_depth=depth_front,
                image_seg=inst_seg,
                data=measurement_data
            )

        return applied_control

    def tick(self, input_data):
        self.step += 1

        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]

        """'rgb_front': rgb_front,
        'rgb_front_60': rgb_front_60,
        'rgb_rear': rgb_rear,"""
        return {
            'gps': gps,
            'speed': speed,
            'compass': compass,

        }

    def get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._route_planner.mean) * self._route_planner.scale

        return gps

    def get_control(self, target, far_target, tick_data, brake):
        pos = self.get_position(tick_data)
        theta = tick_data['compass']
        speed = tick_data['speed']

        # steering
        angle_unnorm = base_utils.get_angle_to(pos, theta, target)
        angle = angle_unnorm / 90

        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        self.steer_alpha = 0.4  # степень сглаживания: 0.1–0.4 обычно ок

        self.smoothed_steer = (self.steer_alpha * steer +
                               (1 - self.steer_alpha) * self.smoothed_steer)
        steer = self.smoothed_steer

        # acceleration
        angle_far_unnorm = base_utils.get_angle_to(pos, theta, far_target)
        should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0
        target_speed = 5.0 if should_slow else 20

        if brake:
            target_speed = 0.0

        self.should_slow = int(should_slow)
        self.should_brake = int(brake)
        self.angle = angle
        self.angle_unnorm = angle_unnorm
        self.angle_far_unnorm = angle_far_unnorm

        delta = np.clip(target_speed - speed, 0.0, 0.3)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.8)

        if self.should_brake > 0.5:
            throttle = 0.0

        return steer, throttle, brake, target_speed, angle

    def destroy(self):
        if self.collision_sensor is not None:
            self.collision_sensor.stop()


    def traffic_data(self):
        all_actors = self.world.get_actors()

        lights_list = all_actors.filter('*traffic_light*')
        walkers_list = all_actors.filter('*walker*')
        vehicle_list = all_actors.filter('*vehicle*')
        stop_list = all_actors.filter('*stop*')

        traffic_lights = base_utils.get_nearby_lights(self.hero_vehicle, lights_list)
        stops = base_utils.get_nearby_lights(self.hero_vehicle, stop_list)

        if len(stops) == 0:
            stop = None
        else:
            stop = stops

        light = self.is_light_red(traffic_lights)
        walker = self.is_walker_hazard(walkers_list)
        vehicle = self.is_vehicle_hazard(vehicle_list)
        stop = self.stop_function(stop)

        self.is_vehicle_present = 1 if vehicle is not None else 0
        self.is_red_light_present = 1 if light is not None else 0
        self.is_pedestrian_present = 1 if walker is not None else 0
        self.is_stops_present = 1 if stop is not None else 0

        return light, walker, vehicle, stop

    def calculate_reward(self, throttle, ego_speed, angle, is_light, is_vehicle, is_walker, is_stop):
        reward = 0.0
        done = 0

        # give penalty if ego vehicle is not braking where it should brake
        if any(x is not None for x in [is_light, is_walker, is_vehicle, is_stop]):
            self.count_is_seen += 1

            # throttle desired after too much waiting around vehicle or walker
            if self.count_is_seen > 1200:
                print("[Penalty]: too much stopping when there is a vehicle or walker or stop sign or traffic light around !")
                reward -= 100

            # braking desired
            else:
                # accelerating while it should brake
                if throttle > 0.2:
                    print("[Penalty]: not braking !")
                    reward -= 50
                else:
                    print("[Reward]: correctly braking !")
                    reward += 50

            self.count_vehicle_stop = 0

        # terminate if vehicle is not moving for too long steps
        else:
            self.count_is_seen = 0

            if ego_speed <= 0.5:
                self.count_vehicle_stop += 1
            else:
                self.count_vehicle_stop = 0

            if self.count_vehicle_stop > 100:
                print("[Penalty]: too long stopping !")
                reward -= 20
                done = 1
            else:
                reward += ego_speed
                done = 0

        # negative reward for collision
        if self.is_collision:
            print("[Penalty]: collision !")
            reward -= 1000
            done = 1
        else:
            done = 0

        return reward, done

    def is_light_red(self, traffic_lights):
        for light in traffic_lights:
            if light.get_state() == carla.TrafficLightState.Red:
                return True
            elif light.get_state() == carla.TrafficLightState.Yellow:
                return True
        return None

    def is_walker_hazard(self, walkers_list):
        p1 = base_utils._numpy(self.hero_vehicle.get_location())
        v1 = 7.0 * base_utils._orientation(self.hero_vehicle.get_transform().rotation.yaw)
        for walker in walkers_list:
            v2_hat = base_utils._orientation(walker.get_transform().rotation.yaw)
            s2 = np.linalg.norm(base_utils._numpy(walker.get_velocity()))
            if s2 < 0.05:
                v2_hat *= s2
            p2 = -3.0 * v2_hat + base_utils._numpy(walker.get_location())
            v2 = 8.0 * v2_hat
            collides, collision_point = base_utils.get_collision(p1, v1, p2, v2)
            if collides:
                return walker
        return None

    def is_vehicle_hazard(self, vehicle_list):
        o1 = base_utils._orientation(self.hero_vehicle.get_transform().rotation.yaw)
        p1 = base_utils._numpy(self.hero_vehicle.get_location())
        s1 = max(8.5, 3.0 * np.linalg.norm(base_utils._numpy(self.hero_vehicle.get_velocity()))) # increases the threshold distance
        v1_hat = o1
        v1 = s1 * v1_hat
        for target_vehicle in vehicle_list:
            if target_vehicle.id == self.hero_vehicle.id:
                continue
            o2 = base_utils._orientation(target_vehicle.get_transform().rotation.yaw)
            p2 = base_utils._numpy(target_vehicle.get_location())
            s2 = max(5.0, 2.0 * np.linalg.norm(base_utils._numpy(target_vehicle.get_velocity())))
            v2_hat = o2
            v2 = s2 * v2_hat
            p2_p1 = p2 - p1
            distance = np.linalg.norm(p2_p1)
            p2_p1_hat = p2_p1 / (distance + 1e-4)
            angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))
            angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))
            angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
            angle_between_heading = min(angle_between_heading, 360.0 - angle_between_heading)
            if angle_between_heading > 50.0 and not (angle_to_car < 15 and distance < s1):
                continue
            elif angle_to_car > 25.0:
                continue
            elif distance > s1 and distance < s2:
                self.target_vehicle_speed = target_vehicle.get_velocity()
                continue
            elif distance > s1:
                continue
            return target_vehicle
        return None

    def define_classifier_label(self, reward):
        label = 0
        if reward < 0.0:
            if self.is_red_light_present:
                label = 1
            elif self.is_stops_present:
                label = 2
            else:
                label = 3
        else:
            label = 0
        return label

    def save_data(self, image_depth, image_seg, data,):
        idx = self.data_count
        paths = self.subfolder_paths
        task = (image_depth, image_seg, data, paths, idx)
        self._save_queue.put(task)
        self.data_count += 1


    def change_weather(self):
        index = random.choice(range(len(WEATHERS)))
        self.weather_id = WEATHERS_IDS[index]
        self.world.set_weather(WEATHERS[WEATHERS_IDS[index]])

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.is_collision = True