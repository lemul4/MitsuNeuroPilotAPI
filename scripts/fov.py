import math
import matplotlib.pyplot as plt

# =========================
# ПАРАМЕТРЫ
# =========================
SIDE_FOV_DEG = 90      # FOV боковых камер
CENTER_FOV_DEG = 60    # FOV центральной камеры

baseline = 0.30        # расстояние между левой и правой камерой, метры
target_distance = 1.5  # расстояние до точки пересечения внутренних границ, метры
ray_len = 20.0         # длина лучей для визуализации, метры



yaw_deg = -40


# =========================
# ГЕОМЕТРИЯ ДЛЯ ВИЗУАЛИЗАЦИИ
# =========================
def ray_endpoint(x0, y0, angle_deg_from_forward, length=20.0):
    """
    Угол задается относительно направления +Y (вперед).
    Положительный угол = поворот вправо.
    """
    a = math.radians(angle_deg_from_forward)
    x1 = x0 + length * math.sin(a)
    y1 = y0 + length * math.cos(a)
    return x1, y1

def draw_camera_fov(ax, x, y, yaw_deg, fov_deg, color, label, ray_len=20.0):
    left_edge = yaw_deg - fov_deg / 2.0
    right_edge = yaw_deg + fov_deg / 2.0

    # Лучи границ FOV
    x1, y1 = ray_endpoint(x, y, left_edge, ray_len)
    x2, y2 = ray_endpoint(x, y, right_edge, ray_len)

    # Центральный луч
    xc, yc = ray_endpoint(x, y, yaw_deg, ray_len)

    ax.plot([x, x1], [y, y1], color=color, linewidth=2)
    ax.plot([x, x2], [y, y2], color=color, linewidth=2)
    ax.plot([x, xc], [y, yc], color=color, linestyle='--', alpha=0.9)

    ax.scatter([x], [y], color=color, s=60)
    ax.text(x, y - 0.3, label, ha='center', va='top', fontsize=10)

    # Заливка сектора
    ax.fill([x, x1, x2], [y, y1, y2], color=color, alpha=0.08)

# =========================
# ПОЗИЦИИ КАМЕР
# =========================
left_cam = (-baseline / 2.0, 0.0)
center_cam = (0.0, 0.0)
right_cam = (baseline / 2.0, 0.0)

left_yaw = yaw_deg
right_yaw = -yaw_deg
center_yaw = 0.0

# =========================
# ВИЗУАЛИЗАЦИЯ
# =========================
fig, ax = plt.subplots(figsize=(12, 8))

draw_camera_fov(ax, left_cam[0], left_cam[1], left_yaw, SIDE_FOV_DEG, 'tab:blue', 'Left camera', ray_len)
draw_camera_fov(ax, center_cam[0], center_cam[1], center_yaw, CENTER_FOV_DEG, 'tab:green', 'Center camera', ray_len)
draw_camera_fov(ax, right_cam[0], right_cam[1], right_yaw, SIDE_FOV_DEG, 'tab:red', 'Right camera', ray_len)

# Точка пересечения внутренних краев
ax.scatter([0], [target_distance], color='black', s=100, marker='x', label='Intersection point')
ax.text(0.15, target_distance + 0.2, f"(0, {target_distance:.2f} m)", fontsize=10)

# Центральная ось
ax.axvline(0, color='gray', linestyle=':', alpha=0.8)

ax.set_title("Top view: FOV of 3 cameras (rays up to 20 m)")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y forward (m)")
ax.set_aspect('equal', adjustable='box')
ax.grid(True, alpha=0.3)
ax.legend()

# Масштаб под 20 метров
max_x = ray_len + 1
ax.set_xlim(-max_x, max_x)
ax.set_ylim(-1, ray_len + 1)

plt.show()
