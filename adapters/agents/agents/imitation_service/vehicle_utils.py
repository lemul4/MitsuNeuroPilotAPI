import carla
import numpy as np

class HazardDetector:
    def __init__(self, agent, max_distance=25.0):
        self.agent = agent
        self.max_distance = max_distance

    def detect_vehicle_hazard(self, vehicle_list):
        if not vehicle_list:
            return -1

        ego_transform = self.agent.hero_vehicle.get_transform()
        ego_location = ego_transform.location
        ego_forward = ego_transform.get_forward_vector()

        min_distance = self.max_distance + 1

        for vehicle in vehicle_list:
            if vehicle.id == self.agent.hero_vehicle.id:
                continue

            v_transform = vehicle.get_transform()
            v_location = v_transform.location
            vec_to_v = v_location - ego_location
            distance = vec_to_v.length()

            if distance >= self.max_distance:
                continue

            dot = vec_to_v.x * ego_forward.x + vec_to_v.y * ego_forward.y + vec_to_v.z * ego_forward.z

            if dot > 0 and distance < min_distance:
                min_distance = distance

        return min_distance if min_distance <= self.max_distance else -1
