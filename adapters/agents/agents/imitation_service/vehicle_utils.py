import math
import numpy as np
import carla

def is_vehicle_hazard(world, hero_vehicle, _map, max_distance=25.0, vehicle_list=None):
    """
    Проверяет наличие ближайшего транспортного средства впереди в текущей полосе
    в пределах заданного максимального расстояния.

    :param world: объект мира CARLA
    :param hero_vehicle: управляемое транспортное средство агента
    :param _map: карта CARLA
    :param max_distance: максимальное расстояние для поиска
    :param vehicle_list: список транспортных средств (если None — берется из мира)
    :return: расстояние до ближайшего препятствия или -1.0 если нет
    """
    if vehicle_list is None:
        vehicle_list = world.get_actors().filter("*vehicle*")

    if hero_vehicle is None:
        print("Error: hero_vehicle is None")
        return -1.0
    if _map is None:
        print("Error: _map is None")
        return -1.0

    ego_transform = hero_vehicle.get_transform()
    ego_location = ego_transform.location
    ego_forward_vector = ego_transform.get_forward_vector()
    ego_wpt = _map.get_waypoint(ego_location)

    min_distance = float('inf')
    nearest_vehicle_distance = -1.0

    for target_vehicle in vehicle_list:
        if target_vehicle.id == hero_vehicle.id:
            continue

        target_transform = target_vehicle.get_transform()
        target_location = target_transform.location
        distance_to_target_raw = ego_location.distance(target_location)
        if distance_to_target_raw > max_distance:
            continue

        target_wpt = _map.get_waypoint(target_location, lane_type=carla.LaneType.Any)

        if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id:
            continue

        vec_ego_to_target = target_location - ego_location
        dot_product = ego_forward_vector.dot(vec_ego_to_target)
        if dot_product < 0:
            continue

        distance = distance_to_target_raw
        if distance > 1e-4:
            vec_ego_to_target_normalized = vec_ego_to_target / distance
            dot_product_normalized = ego_forward_vector.dot(vec_ego_to_target_normalized)
            dot_product_normalized = np.clip(dot_product_normalized, -1.0, 1.0)
            angle_rad = np.arccos(dot_product_normalized)
            angle_deg = np.degrees(angle_rad)

            ahead_angle_threshold = 45.0
            if angle_deg > ahead_angle_threshold:
                continue

        if distance > max_distance:
            continue

        if distance < min_distance:
            min_distance = distance
            nearest_vehicle_distance = distance

    return nearest_vehicle_distance
