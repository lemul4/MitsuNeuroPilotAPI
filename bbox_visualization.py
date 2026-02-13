import json
import numpy as np
import matplotlib.pyplot as plt

def get_bbox_corners(pos, extent, yaw):
    x, y_carla, z = pos
    dx, dy, dz = extent
    y = -y_carla  # Инверсия Y для перехода из LHS в RHS
    actual_yaw = -yaw
    
    corners = np.array([
        [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
        [-dx, -dy, dz], [dx, -dy, dz], [dx, dy, dz], [-dx, dy, dz]
    ])
    rot_z = np.array([
        [np.cos(actual_yaw), -np.sin(actual_yaw), 0],
        [np.sin(actual_yaw),  np.cos(actual_yaw), 0],
        [0, 0, 1]
    ])
    return np.dot(corners, rot_z.T) + np.array([x, y, z])

def visualize_bboxes(input_data):
    # 1. Загрузка данных
    if input_data.endswith('.json'):
        with open(input_data, 'r') as f:
            raw_content = f.read().strip()
    else:
        raw_content = input_data.strip()

    # Исправляем проблему "Extra data" (если в файле два массива ][ )
    if "][" in raw_content:
        # Берем только первый массив или склеиваем их правильно
        raw_content = raw_content.replace("][", ",") 
        # Если после замены возникли двойные запятые, чистим их
        raw_content = raw_content.replace(",,", ",")

    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as e:
        print(f"Ошибка парсинга JSON: {e}")
        return

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = {'ego_car': 'red', 'car': 'blue', 'traffic_light': 'green'}
    all_points = []

    for obj in data:
        corners = get_bbox_corners(obj['position'], obj['extent'], obj.get('yaw', 0))
        all_points.extend(corners)
        
        # Ребра (индексы: 0-3 низ, 4-7 верх)
        edges = [[0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4], [0,4], [1,5], [2,6], [3,7]]
        color = colors.get(obj.get('class'), 'gray')
        
        for edge in edges:
            ax.plot(corners[edge, 0], corners[edge, 1], corners[edge, 2], color=color, linewidth=2)

    # --- Коррекция пропорций (чтобы не было "высоких" боксов) ---
    all_points = np.array(all_points)
    if len(all_points) > 0:
        mid_x = (all_points[:,0].max() + all_points[:,0].min()) * 0.5
        mid_y = (all_points[:,1].max() + all_points[:,1].min()) * 0.5
        mid_z = (all_points[:,2].max() + all_points[:,2].min()) * 0.5
        
        # Вычисляем максимальный размах по всем осям
        max_range = np.array([
            all_points[:,0].max() - all_points[:,0].min(),
            all_points[:,1].max() - all_points[:,1].min(),
            all_points[:,2].max() - all_points[:,2].min()
        ]).max() / 2.0

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X (Forward)')
    ax.set_ylabel('Y (Left)')
    ax.set_zlabel('Z (Height)')
    ax.view_init(elev=30, azim=-135)
    
    plt.title("Corrected 3D Visualization")
    plt.show()

# Запуск
visualize_bboxes("0010.json")