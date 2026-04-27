#!/usr/bin/env python3
from pathlib import Path

from PIL import Image


CONFIG = {
    "data_root": Path("data/carla_leaderboard2/data"),
    "output_dir": Path("data/carla_leaderboard2/gif"),
    "camera_subdirs": [Path("rgb/cam1"), Path("rgb/cam3"), Path("rgb/cam3")],
    "camera_size": 384,
    "fps": 20,
    "frame_step": 2,
    "max_frames": None,
    "colors": 96,
    "dither": False,
    "loop": 0,
    "optimize": True,
}


def frame_sort_key(path: Path):
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def collect_frames(frames_dir: Path):
    frames = list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.jpeg"))
    return sorted(frames, key=frame_sort_key)


def build_frame_index(frames_dir: Path):
    return {path.stem: path for path in collect_frames(frames_dir)}


def sorted_frame_stems(stems):
    return sorted(stems, key=lambda stem: (0, int(stem)) if stem.isdigit() else (1, stem))


def find_first_route_camera_dirs(scenario_dir: Path, camera_subdirs: list[Path]):
    route_dirs = sorted(path for path in scenario_dir.iterdir() if path.is_dir())

    for route_dir in route_dirs:
        camera_dirs = [route_dir / camera_subdir for camera_subdir in camera_subdirs]
        if not all(camera_dir.is_dir() for camera_dir in camera_dirs):
            continue

        frame_indices = [build_frame_index(camera_dir) for camera_dir in camera_dirs]
        if any(not frame_index for frame_index in frame_indices):
            continue

        common_stems = set(frame_indices[0])
        for frame_index in frame_indices[1:]:
            common_stems &= set(frame_index)

        if common_stems:
            return camera_dirs, route_dir

    return None, None


def select_frames(frame_paths, frame_step: int, max_frames: int | None):
    if frame_step <= 0:
        raise ValueError("frame_step must be > 0")

    selected = frame_paths[::frame_step]
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be > 0 when provided")
        selected = selected[:max_frames]
    return selected


def prepare_camera_frame(path: Path, camera_size: int):
    if camera_size <= 0:
        raise ValueError("camera_size must be > 0")

    with Image.open(path) as image:
        frame = image.convert("RGB")

    if frame.size != (camera_size, camera_size):
        frame = frame.resize((camera_size, camera_size), Image.Resampling.LANCZOS)

    return frame


def prepare_composed_frame(camera_paths: list[Path], camera_size: int, colors: int, dither: bool):
    if colors < 2 or colors > 256:
        raise ValueError("colors must be in range [2, 256]")
    if len(camera_paths) != 3:
        raise ValueError("Exactly 3 camera paths are required")

    canvas = Image.new("RGB", (camera_size * 2, camera_size * 2))
    positions = [
        (0, 0),
        (camera_size, 0),
        (camera_size // 2, camera_size),
    ]

    prepared_frames = []
    try:
        for camera_path, position in zip(camera_paths, positions):
            camera_frame = prepare_camera_frame(camera_path, camera_size)
            prepared_frames.append(camera_frame)
            canvas.paste(camera_frame, position)
    finally:
        for frame in prepared_frames:
            frame.close()

    dither_mode = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    quantized_frame = canvas.quantize(
        colors=colors,
        method=Image.Quantize.MEDIANCUT,
        dither=dither_mode,
    )
    canvas.close()
    return quantized_frame



def create_gif(
    camera_dirs: list[Path],
    output_gif: Path,
    fps: float,
    frame_step: int,
    max_frames: int | None,
    camera_size: int,
    colors: int,
    dither: bool,
    loop: int,
    optimize: bool,
):
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if len(camera_dirs) != 3:
        raise ValueError("Exactly 3 camera directories are required")

    frame_indices = [build_frame_index(camera_dir) for camera_dir in camera_dirs]
    if any(not frame_index for frame_index in frame_indices):
        raise FileNotFoundError(f"No jpg/jpeg frames found in one of camera dirs: {camera_dirs}")

    common_stems = set(frame_indices[0])
    for frame_index in frame_indices[1:]:
        common_stems &= set(frame_index)

    sorted_stems = sorted_frame_stems(common_stems)
    if not sorted_stems:
        raise FileNotFoundError(
            f"No synchronized jpg/jpeg frames found across cameras: {camera_dirs}"
        )

    selected_paths = select_frames(
        frame_paths=sorted_stems,
        frame_step=frame_step,
        max_frames=max_frames,
    )
    if not selected_paths:
        raise ValueError("No frames selected after applying frame_step/max_frames")

    selected_camera_paths = [
        [frame_index[stem] for frame_index in frame_indices] for stem in selected_paths
    ]

    output_gif.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000 / fps))

    first_frame = prepare_composed_frame(
        selected_camera_paths[0],
        camera_size,
        colors,
        dither,
    )
    extra_frames = [
        prepare_composed_frame(paths, camera_size, colors, dither)
        for paths in selected_camera_paths[1:]
    ]

    try:
        first_frame.save(
            output_gif,
            save_all=True,
            append_images=extra_frames,
            duration=duration_ms,
            loop=loop,
            optimize=optimize,
            disposal=2,
        )
    finally:
        first_frame.close()
        for frame in extra_frames:
            frame.close()

    print(f"GIF created: {output_gif}")
    print(f"Frames (source synced): {len(sorted_stems)}")
    print(f"Frames (used): {len(selected_camera_paths)}")
    print(f"FPS: {fps}")
    print(f"frame_step: {frame_step}")
    print(f"camera_size: {camera_size}")
    print(f"colors: {colors}")
    print(f"optimize: {optimize}")
    print(f"Frame duration: {duration_ms} ms")


def create_gifs_for_all_scenarios(
    data_root: Path,
    output_dir: Path,
    camera_subdirs: list[Path],
    fps: float,
    frame_step: int,
    max_frames: int | None,
    camera_size: int,
    colors: int,
    dither: bool,
    loop: int,
    optimize: bool,
):
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_dirs = sorted(path for path in data_root.iterdir() if path.is_dir())

    created = 0
    skipped = 0

    for scenario_dir in scenario_dirs:
        scenario_name = scenario_dir.name
        camera_dirs, route_dir = find_first_route_camera_dirs(scenario_dir, camera_subdirs)

        if camera_dirs is None:
            skipped += 1
            print(f"[SKIP] {scenario_name}: no route with synced frames in {camera_subdirs}")
            continue

        output_gif = output_dir / f"{scenario_name}.gif"
        print(f"\n[SCENARIO] {scenario_name}")
        print(f"Route: {route_dir.name}")
        print(f"Cameras: {[camera_dir.name for camera_dir in camera_dirs]}")

        create_gif(
            camera_dirs=camera_dirs,
            output_gif=output_gif,
            fps=fps,
            frame_step=frame_step,
            max_frames=max_frames,
            camera_size=camera_size,
            colors=colors,
            dither=dither,
            loop=loop,
            optimize=optimize,
        )
        created += 1

    print("\n=== Summary ===")
    print(f"Scenarios total: {len(scenario_dirs)}")
    print(f"GIF created: {created}")
    print(f"Skipped: {skipped}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    create_gifs_for_all_scenarios(
        data_root=Path(CONFIG["data_root"]),
        output_dir=Path(CONFIG["output_dir"]),
        camera_subdirs=[Path(path) for path in CONFIG["camera_subdirs"]],
        fps=float(CONFIG["fps"]),
        frame_step=int(CONFIG["frame_step"]),
        max_frames=CONFIG["max_frames"],
        camera_size=int(CONFIG["camera_size"]),
        colors=int(CONFIG["colors"]),
        dither=bool(CONFIG["dither"]),
        loop=int(CONFIG["loop"]),
        optimize=bool(CONFIG["optimize"]),
    )
