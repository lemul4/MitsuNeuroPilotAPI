from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path

from PIL import Image
from tqdm import tqdm


DEFAULT_INPUT_DIR = (
    "data/carla_leaderboard2_dual_cameras_val/data/DynamicObjectCrossing/"
    "Town13_Rep0_1355_0_route0_05_18_23_40_37/viz"
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact, readable GIF from a folder of visualization frames."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=DEFAULT_INPUT_DIR,
        help="Directory with image frames.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output GIF path. Defaults to <input_dir>/animation.gif.",
    )
    parser.add_argument("--fps", type=float, default=8.0, help="GIF playback FPS.")
    parser.add_argument(
        "--max-width",
        type=int,
        default=900,
        help="Initial output width before auto-compression.",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=640,
        help="Smallest width allowed during auto-compression.",
    )
    parser.add_argument(
        "--colors",
        type=int,
        default=128,
        choices=(32, 64, 96, 128, 192, 256),
        help="Initial GIF palette size.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Use every Nth frame before auto-compression.",
    )
    parser.add_argument(
        "--target-mb",
        type=float,
        default=25.0,
        help="Try to keep the GIF under this size. Use <=0 to disable.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Optional cap after frame-step filtering.",
    )
    return parser.parse_args()


def normalize_path(path: str) -> Path:
    return Path(path.replace("\\", os.sep)).expanduser()


def natural_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.stem)
    return [int(part) if part.isdigit() else part for part in parts]


def collect_frames(input_dir: Path, frame_step: int, max_frames: int) -> list[Path]:
    frames = sorted(
        [
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=natural_key,
    )
    frames = frames[:: max(1, frame_step)]
    if max_frames > 0:
        frames = frames[:max_frames]
    if not frames:
        raise FileNotFoundError(f"No image frames found in {input_dir}")
    return frames


def resize_frame(image: Image.Image, max_width: int) -> Image.Image:
    image = image.convert("RGB")
    if image.width <= max_width:
        return image
    scale = max_width / image.width
    size = (max_width, max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def load_quantized_frames(paths: list[Path], width: int, colors: int) -> list[Image.Image]:
    frames = []
    for path in tqdm(paths, desc=f"Preparing {width}px/{colors} colors"):
        with Image.open(path) as image:
            resized = resize_frame(image, width)
            frames.append(
                resized.quantize(
                    colors=colors,
                    method=Image.Quantize.MEDIANCUT,
                    dither=Image.Dither.NONE,
                )
            )
    return frames


def save_gif(
    paths: list[Path],
    output_path: Path,
    width: int,
    colors: int,
    fps: float,
) -> int:
    frames = load_quantized_frames(paths, width, colors)
    duration_ms = max(1, round(1000.0 / fps))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return output_path.stat().st_size


def compression_candidates(
    max_width: int,
    min_width: int,
    colors: int,
    frame_step: int,
) -> list[tuple[int, int, int]]:
    color_steps = [colors, 96, 64, 32]
    color_steps = list(dict.fromkeys(c for c in color_steps if c <= colors))
    widths = []
    width = max_width
    while width >= min_width:
        widths.append(width)
        width = int(width * 0.85)
    if widths[-1] != min_width:
        widths.append(min_width)

    candidates = []
    for step in (frame_step, frame_step * 2, frame_step * 3):
        for width in widths:
            for color_count in color_steps:
                candidates.append((width, color_count, max(1, step)))
    return list(dict.fromkeys(candidates))


def main() -> None:
    args = parse_args()
    input_dir = normalize_path(args.input_dir)
    output_path = normalize_path(args.output) if args.output else input_dir / "animation.gif"

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.min_width > args.max_width:
        raise ValueError("--min-width must be <= --max-width")

    target_bytes = args.target_mb * 1024 * 1024 if args.target_mb > 0 else None
    best_temp_path = None
    best_size = None
    best_params = None

    with tempfile.TemporaryDirectory(prefix="gif_build_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        for attempt, (width, colors, step) in enumerate(
            compression_candidates(
                args.max_width, args.min_width, args.colors, args.frame_step
            ),
            start=1,
        ):
            paths = collect_frames(input_dir, step, args.max_frames)
            temp_output = temp_dir_path / f"attempt_{attempt:03d}.gif"
            size = save_gif(paths, temp_output, width, colors, args.fps)
            size_mb = size / 1024 / 1024
            print(
                f"attempt={attempt} frames={len(paths)} width={width} "
                f"colors={colors} step={step} size={size_mb:.2f} MB"
            )

            if best_size is None or size < best_size:
                best_size = size
                best_params = (len(paths), width, colors, step)
                best_temp_path = temp_output

            if target_bytes is None or size <= target_bytes:
                best_temp_path = temp_output
                best_size = size
                best_params = (len(paths), width, colors, step)
                break

        if best_temp_path is None or best_size is None or best_params is None:
            raise RuntimeError("GIF was not created.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(best_temp_path.read_bytes())

    frames_count, width, colors, step = best_params
    print(
        f"Saved: {output_path}\n"
        f"Frames: {frames_count}, width: {width}px, colors: {colors}, "
        f"frame_step: {step}, fps: {args.fps:g}\n"
        f"Size: {best_size / 1024 / 1024:.2f} MB"
    )


if __name__ == "__main__":
    main()
