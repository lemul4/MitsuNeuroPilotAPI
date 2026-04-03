#!/usr/bin/env python3
"""Скрипт для визуализации сохранённого облака точек LiDAR в формате .laz"""

import argparse
import sys
from pathlib import Path

import laspy
import numpy as np
from laspy.errors import LaspyException

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_laz(path: str) -> np.ndarray:
    """Загружает .laz файл и возвращает массив точек (N, 4): x, y, z, time."""
    try:
        with laspy.open(path) as f:
            las = f.read()
    except LaspyException as exc:
        if "No LazBackend selected" in str(exc):
            print("[ERROR] Невозможно прочитать .laz: отсутствует backend декомпрессии для laspy.")
            print("Установите один из пакетов и повторите запуск:")
            print("  pip install lazrs")
            print("  или pip install laszip")
            print("  (conda) conda install -c conda-forge lazrs")
            sys.exit(1)
        raise

    x = np.array(las.x, dtype=np.float32)
    y = np.array(las.y, dtype=np.float32)
    z = np.array(las.z, dtype=np.float32)

    # Поле time — дополнительное измерение, добавленное в expert_data.py
    try:
        t = np.array(las["time"], dtype=np.float32)
    except Exception:
        t = np.zeros_like(x)

    points = np.stack([x, y, z, t], axis=1)
    print(f"Загружено точек: {points.shape[0]}")
    print(f"  X: [{x.min():.2f}, {x.max():.2f}]")
    print(f"  Y: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  Z: [{z.min():.2f}, {z.max():.2f}]")
    print(f"  Time (scan IDs): {np.unique(t)}")
    return points


def colorize_by_height(z: np.ndarray) -> np.ndarray:
    """Раскрашивает точки по высоте (Z) в RGB [0..1]."""
    z_norm = (z - z.min()) / (z.ptp() + 1e-8)
    colors = plt.cm.turbo(z_norm)[:, :3]  # type: ignore[attr-defined]
    return colors.astype(np.float64)


def colorize_by_time(t: np.ndarray) -> np.ndarray:
    """Раскрашивает точки по полю time (sweep ID)."""
    t_norm = (t - t.min()) / (t.ptp() + 1e-8)
    colors = plt.cm.hsv(t_norm)[:, :3]  # type: ignore[attr-defined]
    return colors.astype(np.float64)


def visualize_open3d(points: np.ndarray, color_mode: str) -> None:
    """Визуализация через Open3D (интерактивная)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))

    if color_mode == "height":
        colors = colorize_by_height(points[:, 2])
    elif color_mode == "time":
        colors = colorize_by_time(points[:, 3])
    else:
        colors = np.ones((points.shape[0], 3)) * 0.7

    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("\nУправление Open3D:")
    print("  Мышь — вращение/зум/пан")
    print("  Q / Esc — выход")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="LiDAR Point Cloud",
        width=1280,
        height=720,
        point_show_normal=False,
    )


def visualize_matplotlib(points: np.ndarray, color_mode: str) -> None:
    """Упрощённая визуализация через Matplotlib (статичная)."""
    # Прореживаем для производительности
    max_points = 50_000
    if points.shape[0] > max_points:
        idx = np.random.choice(points.shape[0], max_points, replace=False)
        pts = points[idx]
        print(f"Прореживание до {max_points} точек для Matplotlib.")
    else:
        pts = points

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    if color_mode == "height":
        c = pts[:, 2]
        cmap = "turbo"
    elif color_mode == "time":
        c = pts[:, 3]
        cmap = "hsv"
    else:
        c = "steelblue"
        cmap = None

    sc = ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c=c, cmap=cmap, s=0.3, alpha=0.6,
    )

    if cmap is not None:
        fig.colorbar(sc, ax=ax, label=color_mode, shrink=0.5)

    ax.set_xlabel("X (м)")
    ax.set_ylabel("Y (м)")
    ax.set_zlabel("Z (м)")
    ax.set_title(f"LiDAR Point Cloud  |  {points.shape[0]} точек")
    plt.tight_layout()
    plt.show()


def visualize_bev(points: np.ndarray, resolution: float = 0.1) -> None:
    """Вид сверху (BEV) — быстро и информативно."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    w = int((x_max - x_min) / resolution) + 1
    h = int((y_max - y_min) / resolution) + 1

    bev = np.full((h, w), np.nan, dtype=np.float32)
    xi = ((x - x_min) / resolution).astype(int)
    yi = ((y - y_min) / resolution).astype(int)
    bev[yi, xi] = z

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(
        bev,
        origin="lower",
        cmap="turbo",
        aspect="equal",
        extent=[x_min, x_max, y_min, y_max],
    )
    fig.colorbar(im, ax=ax, label="Z (высота, м)")
    ax.set_xlabel("X (м)")
    ax.set_ylabel("Y (м)")
    ax.set_title(f"LiDAR BEV  |  разрешение {resolution} м/пиксель")

    # Отметим предположительную позицию эго-автомобиля
    ax.plot(0, 0, "r*", markersize=12, label="Ego (0,0)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Визуализация облака точек LiDAR (.laz)"
    )
    parser.add_argument(
        "path",
        help="Путь к .laz файлу или директории с .laz файлами",
    )
    parser.add_argument(
        "--backend",
        choices=["open3d", "matplotlib", "bev", "auto"],
        default="auto",
        help="Бэкенд визуализации (default: auto)",
    )
    parser.add_argument(
        "--color",
        choices=["height", "time", "flat"],
        default="height",
        help="Режим окраски точек (default: height)",
    )
    parser.add_argument(
        "--bev-resolution",
        type=float,
        default=0.1,
        help="Разрешение BEV-карты в м/пиксель (default: 0.1)",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Номер фрейма при передаче директории (например: --frame 42)",
    )
    args = parser.parse_args()

    # Выбор файла
    input_path = Path(args.path)
    if input_path.is_dir():
        laz_files = sorted(input_path.glob("*.laz"))
        if not laz_files:
            print(f"[ERROR] Нет .laz файлов в {input_path}")
            sys.exit(1)
        if args.frame is not None:
            target = input_path / f"{args.frame:04}.laz"
            if not target.exists():
                print(f"[ERROR] Файл {target} не найден")
                sys.exit(1)
            laz_file = target
        else:
            laz_file = laz_files[0]
            print(f"Найдено {len(laz_files)} файлов. Показываю первый: {laz_file.name}")
            print("Используйте --frame N для выбора конкретного фрейма.")
    elif input_path.is_file():
        laz_file = input_path
    else:
        print(f"[ERROR] Путь не найден: {input_path}")
        sys.exit(1)

    print(f"\nФайл: {laz_file}")
    points = load_laz(str(laz_file))

    if points.shape[0] == 0:
        print("[WARN] Облако точек пустое.")
        sys.exit(0)

    # Выбор бэкенда
    backend = args.backend
    if backend == "auto":
        if HAS_OPEN3D:
            backend = "open3d"
        elif HAS_MATPLOTLIB:
            backend = "matplotlib"
        else:
            print("[ERROR] Установите open3d или matplotlib:")
            print("  pip install open3d")
            print("  pip install matplotlib")
            sys.exit(1)

    print(f"Бэкенд: {backend}, цвет: {args.color}\n")

    if backend == "open3d":
        if not HAS_OPEN3D:
            print("[ERROR] open3d не установлен: pip install open3d")
            sys.exit(1)
        visualize_open3d(points, args.color)
    elif backend == "matplotlib":
        if not HAS_MATPLOTLIB:
            print("[ERROR] matplotlib не установлен: pip install matplotlib")
            sys.exit(1)
        visualize_matplotlib(points, args.color)
    elif backend == "bev":
        if not HAS_MATPLOTLIB:
            print("[ERROR] matplotlib не установлен: pip install matplotlib")
            sys.exit(1)
        visualize_bev(points, resolution=args.bev_resolution)


if __name__ == "__main__":
    main()
