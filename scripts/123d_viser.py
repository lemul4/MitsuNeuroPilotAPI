"""
Hotkey script to start Viser viewer for visualizing 123D scenes.
By default, you can open viewer via: http://localhost:8080
"""

import argparse
import importlib.util
import os
import sys
import time
from typing import Literal

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

from py123d.api.scene.arrow.arrow_scene_builder import ArrowSceneBuilder
from py123d.api.scene.scene_filter import SceneFilter
from py123d.common.execution.thread_pool_executor import ThreadPoolExecutor
from py123d.datatypes.sensors.lidar import LidarID
from py123d.visualization.viser.elements import box_detections_se3_element as box_el
from py123d.visualization.viser.viser_config import ViserConfig
from py123d.visualization.viser.viser_viewer import ViserViewer


def _default_render_format() -> Literal["gif", "mp4", "png"]:
    """Pick a safe default render format for the local environment."""
    if importlib.util.find_spec("imageio_ffmpeg") is not None:
        return "mp4"
    return "gif"


def _patch_empty_box_detections_guard() -> None:
    """Guard py123d box detection element against empty detections.

    Some py123d versions crash when the current iteration has zero boxes.
    We skip rendering this element for such iterations.
    """
    original_update = box_el.BoxDetectionsSE3Element.update

    def safe_update(self, iteration: int) -> None:  # type: ignore[no-redef]
        scene = self._context.scene
        box_detections = scene.get_box_detections_se3_at_iteration(iteration)
        has_boxes = (
            box_detections is not None
            and getattr(box_detections, "box_detections", None) is not None
            and len(box_detections.box_detections) > 0
        )

        if not has_boxes:
            self._current_iteration = iteration
            self._update_center_frames(iteration)
            for key in self._handles:
                handle = self._handles[key]
                if handle is not None:
                    handle.visible = False
            return

        original_update(self, iteration)

    box_el.BoxDetectionsSE3Element.update = safe_update


def parse_args():
    default_render_format = _default_render_format()

    parser = argparse.ArgumentParser(
        description="Start Viser viewer to visualize 123D scenes in 3D",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split-names",
        type=str,
        nargs="+",
        default=None,
        help="Dataset split names to filter (e.g., 'train', 'val', 'test')",
    )
    parser.add_argument(
        "--future-duration-s",
        type=float,
        default=None,
        help="Future duration in seconds for each scene (None = complete log)",
    )
    parser.add_argument(
        "--timestamp-threshold-s",
        type=float,
        default=0.0,
        help="Timestamp threshold in seconds",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle scenes (default: shuffle enabled)",
    )
    parser.add_argument(
        "--require-map",
        action="store_true",
        help="Only include scenes/logs with an available map",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for Viser viewer server",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for Viser viewer server",
    )
    parser.add_argument(
        "--render-format",
        type=str,
        choices=["gif", "mp4", "png"],
        default=default_render_format,
        help="Render export format in viewer UI",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    _patch_empty_box_detections_guard()

    if args.render_format == "mp4":
        if importlib.util.find_spec("imageio_ffmpeg") is None:
            print(
                "[Viser] imageio-ffmpeg is not installed, switching render format to gif."
            )
            args.render_format = "gif"

    # --- Auto-restart logic ---
    # Only activate if watchdog is available and PY123D_DATA_ROOT is set
    data_root = os.environ.get("PY123D_DATA_ROOT", None)
    if WATCHDOG_AVAILABLE and data_root:

        class SceneChangeHandler(FileSystemEventHandler):
            def __init__(self, restart_callback):
                super().__init__()
                self.restart_callback = restart_callback

            def on_created(self, event):
                # Only restart for new files (not directories)
                if not event.is_directory:
                    print(
                        f"\n[Viser] New scene file detected: {event.src_path}. Restarting viewer..."
                    )
                    self.restart_callback()

        def restart_script():
            # Flush output, then exec self
            sys.stdout.flush()
            sys.stderr.flush()
            os.execv(sys.executable, [sys.executable] + sys.argv)

        observer = Observer()
        event_handler = SceneChangeHandler(restart_script)
        observer.schedule(event_handler, data_root, recursive=True)
        observer.start()
        print(f"[Viser] Watching for new scenes in: {data_root}")
        # Give watchdog a moment to start
        time.sleep(0.5)

    # Create scene filter
    scene_filter = SceneFilter(
        split_names=args.split_names,
        future_duration_s=args.future_duration_s,
        timestamp_threshold_s=args.timestamp_threshold_s,
        shuffle=not args.no_shuffle,
        has_map=True if args.require_map else None,
    )

    # Build scenes
    print("Building scenes from dataset...")
    executor = ThreadPoolExecutor()
    scenes = ArrowSceneBuilder().get_scenes(scene_filter, executor)

    dataset_splits = set(scene.log_metadata.split for scene in scenes)
    print(f"\nFound {len(scenes)} scenes from {len(dataset_splits)} dataset splits:")
    for split in dataset_splits:
        print(f" - {split}")

    if len(scenes) == 0:
        print("\nWarning: No scenes found with the given filter criteria!")
        print("Try adjusting your filter parameters or check your PY123D_DATA_ROOT.")
        return

    # Start Viser viewer
    print(f"\nStarting Viser viewer on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the viewer.")
    print("-" * 60)

    viser_config = ViserConfig(
        server={"host": args.host, "port": args.port},
        lidar={"ids": [LidarID.LIDAR_TOP]},
        render={"format": args.render_format, "view": "3rd Person"},
    )

    try:
        ViserViewer(scenes, viser_config=viser_config)
    finally:
        # Stop watchdog observer on exit
        if WATCHDOG_AVAILABLE and data_root:
            observer.stop()
            observer.join()


if __name__ == "__main__":
    main()
