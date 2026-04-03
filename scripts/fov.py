import math
import matplotlib.pyplot as plt

# =========================
# ПАРАМЕТРЫ
# =========================
SIDE_FOV_DEG = 90.0       # FOV боковых камер
CENTER_FOV_DEG = 60.0     # FOV центральной камеры

baseline = 0.30           # расстояние между левой и правой камерами, м
target_distance = 10.0     # дистанция, на которой анализируем покрытие, м
ray_len = 20.0            # длина лучей для визуализации, м

# Если True, yaw боковых камер будет вычислен автоматически так,
# чтобы внутренние границы встретились в точке (0, target_distance).
AUTO_COMPUTE_SIDE_YAW = False

# Если False, можно руками задать yaw боковых камер:
LEFT_YAW_DEG_MANUAL = -40.0
RIGHT_YAW_DEG_MANUAL = 40.0
CENTER_YAW_DEG = 0.0


# =========================
# ВСПОМОГАТЕЛЬНАЯ ГЕОМЕТРИЯ
# =========================
def ray_endpoint(x0, y0, angle_deg_from_forward, length=20.0):
    """
    Угол задается относительно направления +Y (вперед).
    Положительный угол = вправо.
    """
    a = math.radians(angle_deg_from_forward)
    x1 = x0 + length * math.sin(a)
    y1 = y0 + length * math.cos(a)
    return x1, y1


def x_at_y_from_ray(x0, y0, angle_deg_from_forward, target_y):
    """
    Возвращает X координату луча в точке, где он пересекает горизонтальную линию y = target_y.
    Угол задается относительно +Y.
    Формула:
        x = x0 + t * sin(a)
        y = y0 + t * cos(a)
        t = (target_y - y0) / cos(a)
    """
    a = math.radians(angle_deg_from_forward)
    cos_a = math.cos(a)
    if abs(cos_a) < 1e-8:
        return None  # почти горизонтальный луч
    t = (target_y - y0) / cos_a
    if t < 0:
        return None  # пересечение "позади" камеры
    x = x0 + t * math.sin(a)
    return x


def draw_camera_fov(ax, x, y, yaw_deg, fov_deg, color, label, ray_len=20.0):
    left_edge = yaw_deg - fov_deg / 2.0
    right_edge = yaw_deg + fov_deg / 2.0

    x1, y1 = ray_endpoint(x, y, left_edge, ray_len)
    x2, y2 = ray_endpoint(x, y, right_edge, ray_len)
    xc, yc = ray_endpoint(x, y, yaw_deg, ray_len)

    ax.plot([x, x1], [y, y1], color=color, linewidth=2)
    ax.plot([x, x2], [y, y2], color=color, linewidth=2)
    ax.plot([x, xc], [y, yc], color=color, linestyle="--", alpha=0.9)

    ax.scatter([x], [y], color=color, s=60)
    ax.text(x, y - 0.25, label, ha="center", va="top", fontsize=10)

    ax.fill([x, x1, x2], [y, y1, y2], color=color, alpha=0.08)


def interval_on_target_line(cam_x, cam_y, yaw_deg, fov_deg, target_y):
    """
    Возвращает [x_min, x_max] покрытия камеры на линии y=target_y.
    """
    edge1 = yaw_deg - fov_deg / 2.0
    edge2 = yaw_deg + fov_deg / 2.0

    x1 = x_at_y_from_ray(cam_x, cam_y, edge1, target_y)
    x2 = x_at_y_from_ray(cam_x, cam_y, edge2, target_y)

    if x1 is None or x2 is None:
        return None

    return (min(x1, x2), max(x1, x2))


def angular_interval_from_origin(x_min, x_max, y):
    """
    Угловой интервал, под которым от центра машины (0,0)
    виден отрезок [x_min, x_max] на линии y.
    """
    a1 = math.degrees(math.atan2(x_min, y))
    a2 = math.degrees(math.atan2(x_max, y))
    return min(a1, a2), max(a1, a2)


def compute_side_yaw_for_inner_intersection(side_fov_deg, baseline, target_distance):
    """
    Вычисляет yaw боковых камер так, чтобы внутренние границы боковых камер
    пересеклись в точке (0, target_distance).

    Геометрия:
    - Для левой камеры внутренняя граница = правая граница FOV.
    - Для правой камеры внутренняя граница = левая граница FOV.

    alpha = угол от камеры до точки пересечения.
    Тогда:
      left_yaw  + side_fov/2 = +alpha
      right_yaw - side_fov/2 = -alpha
    но у нас угол положительный = вправо, отрицательный = влево,
    поэтому симметричное решение:
      left_yaw  = -(side_fov/2 - alpha)
      right_yaw = +(side_fov/2 - alpha)
    """
    alpha = math.degrees(math.atan((baseline / 2.0) / target_distance))
    yaw_mag = side_fov_deg / 2.0 - alpha
    left_yaw = -yaw_mag
    right_yaw = yaw_mag
    return left_yaw, right_yaw, alpha


def overlap_length(interval_a, interval_b):
    if interval_a is None or interval_b is None:
        return 0.0
    left = max(interval_a[0], interval_b[0])
    right = min(interval_a[1], interval_b[1])
    return max(0.0, right - left)


# =========================
# ПОЗИЦИИ КАМЕР
# =========================
left_cam = (-baseline / 2.0, 0.0)
center_cam = (0.0, 0.0)
right_cam = (baseline / 2.0, 0.0)

if AUTO_COMPUTE_SIDE_YAW:
    left_yaw, right_yaw, alpha_deg = compute_side_yaw_for_inner_intersection(
        side_fov_deg=SIDE_FOV_DEG,
        baseline=baseline,
        target_distance=target_distance,
    )
else:
    left_yaw = LEFT_YAW_DEG_MANUAL
    right_yaw = RIGHT_YAW_DEG_MANUAL
    alpha_deg = None

center_yaw = CENTER_YAW_DEG


# =========================
# ИНТЕРВАЛЫ ПОКРЫТИЯ НА target_distance
# =========================
left_interval = interval_on_target_line(
    left_cam[0], left_cam[1], left_yaw, SIDE_FOV_DEG, target_distance
)
center_interval = interval_on_target_line(
    center_cam[0], center_cam[1], center_yaw, CENTER_FOV_DEG, target_distance
)
right_interval = interval_on_target_line(
    right_cam[0], right_cam[1], right_yaw, SIDE_FOV_DEG, target_distance
)

all_x = []
for interval in [left_interval, center_interval, right_interval]:
    if interval is not None:
        all_x.extend(interval)

global_x_min = min(all_x)
global_x_max = max(all_x)
global_width = global_x_max - global_x_min

global_angle_min_deg, global_angle_max_deg = angular_interval_from_origin(
    global_x_min, global_x_max, target_distance
)
global_hfov_deg = global_angle_max_deg - global_angle_min_deg

left_center_overlap = overlap_length(left_interval, center_interval)
center_right_overlap = overlap_length(center_interval, right_interval)

# =========================
# ПЕЧАТЬ МЕТРИК
# =========================
print("==== Camera setup ====")
print(f"SIDE_FOV_DEG   = {SIDE_FOV_DEG:.2f}")
print(f"CENTER_FOV_DEG = {CENTER_FOV_DEG:.2f}")
print(f"baseline       = {baseline:.3f} m")
print(f"target_distance= {target_distance:.3f} m")

if AUTO_COMPUTE_SIDE_YAW:
    print("\nAuto-computed side yaws from inner-edge intersection:")
    print(f"alpha to intersection point = {alpha_deg:.4f} deg")
print(f"left_yaw   = {left_yaw:.4f} deg")
print(f"center_yaw = {center_yaw:.4f} deg")
print(f"right_yaw  = {right_yaw:.4f} deg")

print("\n==== Coverage on y = target_distance ====")
print(f"Left interval   : {left_interval}")
print(f"Center interval : {center_interval}")
print(f"Right interval  : {right_interval}")

print(f"\nGlobal coverage width at {target_distance:.2f} m = {global_width:.4f} m")
print(f"Global x range = [{global_x_min:.4f}, {global_x_max:.4f}] m")

print("\n==== Effective global horizontal FOV ====")
print(f"angle_min = {global_angle_min_deg:.4f} deg")
print(f"angle_max = {global_angle_max_deg:.4f} deg")
print(f"GLOBAL_HFOV = {global_hfov_deg:.4f} deg")

print("\n==== Overlap with center camera on target line ====")
print(f"Left-Center overlap  = {left_center_overlap:.4f} m")
print(f"Center-Right overlap = {center_right_overlap:.4f} m")


# =========================
# ВИЗУАЛИЗАЦИЯ
# =========================
fig, ax = plt.subplots(figsize=(12, 8))

draw_camera_fov(ax, left_cam[0], left_cam[1], left_yaw, SIDE_FOV_DEG, "tab:blue", "Left camera", ray_len)
draw_camera_fov(ax, center_cam[0], center_cam[1], center_yaw, CENTER_FOV_DEG, "tab:green", "Center camera", ray_len)
draw_camera_fov(ax, right_cam[0], right_cam[1], right_yaw, SIDE_FOV_DEG, "tab:red", "Right camera", ray_len)

# Линия target_distance
ax.axhline(target_distance, color="black", linestyle="--", alpha=0.5)
ax.text(0.05, target_distance + 0.12, f"y = {target_distance:.2f} m", fontsize=10)

# Точка пересечения внутренних краев
ax.scatter([0], [target_distance], color="black", s=100, marker="x", label="Target point")
ax.text(0.15, target_distance + 0.25, f"(0, {target_distance:.2f})", fontsize=10)

# Рисуем интервалы покрытия на линии y=target_distance
def draw_interval(interval, y, color, label):
    if interval is None:
        return
    ax.plot([interval[0], interval[1]], [y, y], color=color, linewidth=5, alpha=0.9, label=label)

draw_interval(left_interval, target_distance, "tab:blue", "Left coverage at target distance")
draw_interval(center_interval, target_distance, "tab:green", "Center coverage at target distance")
draw_interval(right_interval, target_distance, "tab:red", "Right coverage at target distance")

# Общий интервал
ax.plot(
    [global_x_min, global_x_max],
    [target_distance + 0.18, target_distance + 0.18],
    color="purple",
    linewidth=6,
    alpha=0.8,
    label=f"Global coverage ({global_hfov_deg:.2f} deg)",
)

# Центральная ось
ax.axvline(0, color="gray", linestyle=":", alpha=0.8)

title = (
    f"3-camera top view | "
    f"global HFOV={global_hfov_deg:.2f} deg | "
    f"width@{target_distance:.1f}m={global_width:.2f} m"
)
ax.set_title(title)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y forward (m)")
ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")

max_x = max(ray_len, abs(global_x_min), abs(global_x_max)) + 1.0
ax.set_xlim(-max_x, max_x)
ax.set_ylim(-1, ray_len + 1)

plt.tight_layout()
plt.show()
