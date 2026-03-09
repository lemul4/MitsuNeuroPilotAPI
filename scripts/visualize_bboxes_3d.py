#!/usr/bin/env python3
"""Visualize 3D bounding boxes saved as pickle files."""

from __future__ import annotations

import argparse
import lzma
import math
import pickle
from pathlib import Path

np = None
plt = None
Line3DCollection = None

CLASS_COLORS = {
    "ego_car": "black",
    "car": "tab:blue",
    "walker": "tab:orange",
    "static": "tab:green",
    "traffic_light": "tab:red",
    "stop_sign": "tab:brown",
}

BOX_EDGES = [
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),
    (4, 5),
    (5, 7),
    (7, 6),
    (6, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize 3D bboxes from a single .pkl file or from a directory with "
            "multiple frame files."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to .pkl file or directory containing frame .pkl files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.pkl",
        help="Glob pattern for files when --input is a directory.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Frame id (by filename stem) to open first, e.g. 12 for 0012.pkl.",
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated class filter, e.g. car,walker,static",
    )
    parser.add_argument(
        "--include-ego",
        action="store_true",
        help="Include ego_car in visualization (hidden by default).",
    )
    parser.add_argument(
        "--max-boxes",
        type=int,
        default=0,
        help="Maximum number of boxes to draw (0 = no limit).",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Draw class:id text labels near boxes.",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=1.4,
        help="Edge line width.",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=26.0,
        help="Camera elevation for 3D view.",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=-60.0,
        help="Camera azimuth for 3D view.",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disable keyboard navigation for multiple frames.",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help=(
            "Matplotlib backend (e.g. TkAgg, Qt5Agg, WebAgg, Agg). "
            "If not set, script auto-selects Agg when no display is detected."
        ),
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save the rendered frame to this image path (e.g. out.png).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive window (useful with --save).",
    )
    return parser.parse_args()


def frame_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.isdigit():
        return (0, f"{int(stem):012d}")
    return (1, stem)


def list_frame_files(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    files = sorted(input_path.glob(pattern), key=frame_sort_key)
    if not files:
        raise FileNotFoundError(
            f"No files found in {input_path} with pattern '{pattern}'."
        )
    return files


def resolve_start_index(files: list[Path], frame: int | None) -> int:
    if frame is None:
        return 0
    accepted_stems = {str(frame), f"{frame:04d}", f"{frame:05d}"}
    for idx, file_path in enumerate(files):
        if file_path.stem in accepted_stems:
            return idx
    raise ValueError(f"Frame '{frame}' not found among files.")


def load_boxes(path: Path) -> list[dict]:
    data = None
    plain_error = None
    compressed_error = None

    try:
        with path.open("rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        plain_error = exc

    if data is None:
        try:
            with lzma.open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as exc:
            compressed_error = exc

    if data is None:
        raise ValueError(
            f"Unable to read pickle from {path}. "
            f"plain={plain_error!r}, lzma={compressed_error!r}"
        )

    if isinstance(data, list):
        return [box for box in data if isinstance(box, dict)]
    if isinstance(data, dict):
        for key in ("boxes", "bboxes", "bounding_boxes"):
            if key in data and isinstance(data[key], list):
                return [box for box in data[key] if isinstance(box, dict)]
    raise ValueError(f"Unsupported pickle format in {path}")


def extract_pose_and_extent(box: dict) -> tuple[np.ndarray, np.ndarray, float]:
    position = box.get("position")
    extent = box.get("extent")
    yaw = box.get("yaw", None)
    matrix = box.get("matrix", None)

    if position is None and matrix is not None:
        m = np.array(matrix, dtype=np.float32)
        if m.shape == (4, 4):
            position = m[:3, 3]
    if position is None:
        position = np.zeros(3, dtype=np.float32)
    else:
        position = np.array(position, dtype=np.float32).reshape(-1)
        if position.size < 3:
            position = np.pad(position, (0, 3 - position.size), constant_values=0.0)
        position = position[:3]

    if extent is None:
        extent = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    else:
        extent = np.array(extent, dtype=np.float32).reshape(-1)
        if extent.size < 3:
            extent = np.pad(extent, (0, 3 - extent.size), constant_values=0.5)
        extent = np.abs(extent[:3])

    if yaw is None and matrix is not None:
        m = np.array(matrix, dtype=np.float32)
        if m.shape == (4, 4):
            yaw = math.atan2(float(m[1, 0]), float(m[0, 0]))
    if yaw is None:
        yaw = 0.0

    return position, extent, float(yaw)


def box_corners_3d(center: np.ndarray, extent: np.ndarray, yaw: float) -> np.ndarray:
    ex, ey, ez = extent.tolist()
    corners_local = np.array(
        [
            [-ex, -ey, -ez],
            [ex, -ey, -ez],
            [-ex, ey, -ez],
            [ex, ey, -ez],
            [-ex, -ey, ez],
            [ex, -ey, ez],
            [-ex, ey, ez],
            [ex, ey, ez],
        ],
        dtype=np.float32,
    )
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return corners_local @ rot_z.T + center.reshape(1, 3)


def set_axes_equal(ax) -> None:
    x_limits = np.array(ax.get_xlim3d())
    y_limits = np.array(ax.get_ylim3d())
    z_limits = np.array(ax.get_zlim3d())

    ranges = np.array(
        [x_limits[1] - x_limits[0], y_limits[1] - y_limits[0], z_limits[1] - z_limits[0]]
    )
    centers = np.array(
        [
            (x_limits[0] + x_limits[1]) / 2.0,
            (y_limits[0] + y_limits[1]) / 2.0,
            (z_limits[0] + z_limits[1]) / 2.0,
        ]
    )
    radius = 0.5 * max(ranges.max(), 1e-3)

    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def draw_frame(
    ax,
    file_path: Path,
    classes_filter: set[str] | None,
    include_ego: bool,
    max_boxes: int,
    show_labels: bool,
    line_width: float,
    elev: float,
    azim: float,
) -> int:
    boxes = load_boxes(file_path)

    filtered: list[dict] = []
    for box in boxes:
        cls = str(box.get("class", "unknown"))
        if not include_ego and cls == "ego_car":
            continue
        if classes_filter is not None and cls not in classes_filter:
            continue
        filtered.append(box)
        if max_boxes > 0 and len(filtered) >= max_boxes:
            break

    ax.clear()
    ax.set_xlabel("x (forward)")
    ax.set_ylabel("y (right)")
    ax.set_zlabel("z (up)")
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.3)

    segments: list[np.ndarray] = []
    colors: list[str] = []
    all_points = [np.zeros((1, 3), dtype=np.float32)]
    for box in filtered:
        center, extent, yaw = extract_pose_and_extent(box)
        corners = box_corners_3d(center, extent, yaw)
        all_points.append(corners)

        cls = str(box.get("class", "unknown"))
        color = CLASS_COLORS.get(cls, "tab:gray")
        for i0, i1 in BOX_EDGES:
            segments.append(np.stack([corners[i0], corners[i1]], axis=0))
            colors.append(color)

        if show_labels:
            actor_id = box.get("id", "?")
            ax.text(
                center[0],
                center[1],
                center[2] + extent[2] + 0.15,
                f"{cls}:{actor_id}",
                color=color,
                fontsize=7,
            )

    if segments:
        collection = Line3DCollection(
            segments, colors=colors, linewidths=line_width, alpha=0.95
        )
        ax.add_collection3d(collection)

    all_points_np = np.concatenate(all_points, axis=0)
    min_xyz = all_points_np.min(axis=0) - np.array([2.0, 2.0, 1.0])
    max_xyz = all_points_np.max(axis=0) + np.array([2.0, 2.0, 1.0])
    ax.set_xlim(float(min_xyz[0]), float(max_xyz[0]))
    ax.set_ylim(float(min_xyz[1]), float(max_xyz[1]))
    ax.set_zlim(float(min_xyz[2]), float(max_xyz[2]))
    set_axes_equal(ax)

    # Ego-centered coordinate frame helper.
    ax.quiver(0, 0, 0, 2, 0, 0, color="r", arrow_length_ratio=0.12)
    ax.quiver(0, 0, 0, 0, 2, 0, color="g", arrow_length_ratio=0.12)
    ax.quiver(0, 0, 0, 0, 0, 2, color="b", arrow_length_ratio=0.12)
    ax.text(0, 0, 0, "ego", color="black", fontsize=8)

    ax.set_title(f"{file_path.name} | boxes: {len(filtered)}")
    return len(filtered)


def main() -> None:
    args = parse_args()
    files = list_frame_files(args.input, args.pattern)
    start_index = resolve_start_index(files, args.frame)

    global np, plt, Line3DCollection
    try:
        import numpy as _np
    except ImportError as exc:
        raise SystemExit("numpy is required. Install it with: pip install numpy") from exc
    np = _np

    try:
        import matplotlib
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required. Install it with: pip install matplotlib"
        ) from exc

    if args.backend:
        matplotlib.use(args.backend, force=True)
    else:
        import os

        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as _plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection as _Line3DCollection

    plt = _plt
    Line3DCollection = _Line3DCollection

    classes_filter = None
    if args.classes:
        classes_filter = {x.strip() for x in args.classes.split(",") if x.strip()}
        if not classes_filter:
            classes_filter = None

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    state = {"idx": start_index}

    def redraw() -> None:
        draw_frame(
            ax=ax,
            file_path=files[state["idx"]],
            classes_filter=classes_filter,
            include_ego=args.include_ego,
            max_boxes=args.max_boxes,
            show_labels=args.show_labels,
            line_width=args.line_width,
            elev=args.elev,
            azim=args.azim,
        )
        fig.canvas.draw_idle()

    redraw()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=180, bbox_inches="tight")
        print(f"Saved visualization to: {args.save}")

    use_interactive = (
        (len(files) > 1)
        and (not args.no_interactive)
        and (not args.no_show)
    )
    if use_interactive:
        print("Keyboard: Right/n next frame, Left/p previous frame, Home/End, q to quit.")

        def on_key(event) -> None:
            if event.key in ("right", "n"):
                state["idx"] = min(state["idx"] + 1, len(files) - 1)
                redraw()
            elif event.key in ("left", "p"):
                state["idx"] = max(state["idx"] - 1, 0)
                redraw()
            elif event.key == "home":
                state["idx"] = 0
                redraw()
            elif event.key == "end":
                state["idx"] = len(files) - 1
                redraw()
            elif event.key == "q":
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)

    plt.tight_layout()
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
