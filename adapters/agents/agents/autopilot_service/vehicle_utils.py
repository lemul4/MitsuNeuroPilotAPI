import numpy as np
import carla
def is_vehicle_hazard(agent, vehicle_list=None, max_distance=25.0):
    """
    Проверяет наличие ближайшего транспортного средства впереди в текущей полосе агента.

    agent - объект агента, должен иметь hero_vehicle и _map
    """
    if vehicle_list is None:
        if not hasattr(agent, 'world') or agent.world is None:
            print("Ошибка: Объект 'agent' не имеет доступа к миру CARLA (world).")
            return -1.0
        vehicle_list = agent.world.get_actors().filter("*vehicle*")

    if not hasattr(agent, 'hero_vehicle') or agent.hero_vehicle is None:
        print("Ошибка: Объект 'agent' не имеет объекта hero_vehicle.")
        return -1.0

    if not hasattr(agent, '_map') or agent._map is None:
        print("Ошибка: Объект 'agent' не имеет доступа к карте CARLA (_map).")
        return -1.0

    ego_vehicle = agent.hero_vehicle
    ego_transform = ego_vehicle.get_transform()
    ego_location = ego_transform.location
    ego_forward_vector = ego_transform.get_forward_vector()
    ego_wpt = agent._map.get_waypoint(ego_location)

    min_distance = float('inf')
    nearest_vehicle_distance = -1.0

    for target_vehicle in vehicle_list:
        if target_vehicle.id == ego_vehicle.id:
            continue

        target_transform = target_vehicle.get_transform()
        target_location = target_transform.location

        distance_to_target_raw = ego_location.distance(target_location)
        if distance_to_target_raw > max_distance:
            continue

        target_wpt = agent._map.get_waypoint(target_location, lane_type=carla.LaneType.Any)

        if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id:
            continue

        vec_ego_to_target = target_location - ego_location
        dot_product = ego_forward_vector.dot(vec_ego_to_target)
        if dot_product < 0:
            continue

        distance = ego_location.distance(target_location)
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
