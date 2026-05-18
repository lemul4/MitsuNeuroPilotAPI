#!/usr/bin/env python3
"""Clean and crop the CARLA dual-camera dataset.

By default this script is a dry run. Pass --apply to delete folders and rewrite
PNG files in place.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
TARGET_SEMANTICS_DIRS = ("cam1", "cam2", "cam3")
TARGET_RGB_DIRS = ("cam2",)
TARGET_GIF_PATTERNS = ("*.gif", "*.giff")


@dataclass
class PngInfo:
    width: int
    height: int
    bit_depth: int
    color_type: int
    compression: int
    filter_method: int
    interlace: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete *.gif/*.giff files, semantics/cam1..cam3 and rgb/cam2 folders, "
            "then crop 44 pixels from the left and 52 pixels from the right "
            "of semantics/*.png files."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/carla_leaderboard2_dual_cameras"),
        help="Dataset root containing the inner data/ directory.",
    )
    parser.add_argument("--left", type=int, default=44, help="Pixels to crop from the left.")
    parser.add_argument("--right", type=int, default=52, help="Pixels to crop from the right.")
    parser.add_argument("--top", type=int, default=0, help="Pixels to crop from the top.")
    parser.add_argument("--bottom", type=int, default=0, help="Pixels to crop from the bottom.")
    parser.add_argument(
        "--target-width",
        type=int,
        default=704,
        help="Skip files already at this width; default matches 800 - 44 - 52.",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=416,
        help="Skip files already at this height; default keeps the original 416 px height.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Worker processes for PNG rewriting when --apply is used.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="Print progress every N processed PNG files.",
    )
    parser.add_argument(
        "--compress-level",
        type=int,
        default=1,
        choices=range(0, 10),
        metavar="0-9",
        help="PNG zlib compression level. Lower is faster; default is 1.",
    )
    parser.add_argument(
        "--apply",
        "--execute",
        action="store_true",
        help="Actually delete directories and rewrite PNG files.",
    )
    args = parser.parse_args()
    if min(args.left, args.right, args.top, args.bottom) < 0:
        parser.error("crop values must be non-negative")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    return args


def read_chunks(path: Path) -> tuple[PngInfo, list[tuple[bytes, bytes]]]:
    with path.open("rb") as file:
        signature = file.read(8)
        if signature != PNG_SIGNATURE:
            raise ValueError("not a PNG file")

        chunks = []
        info = None
        while True:
            length_bytes = file.read(4)
            if not length_bytes:
                break
            length = struct.unpack(">I", length_bytes)[0]
            chunk_type = file.read(4)
            data = file.read(length)
            crc = file.read(4)
            if len(chunk_type) != 4 or len(data) != length or len(crc) != 4:
                raise ValueError("truncated PNG chunk")

            if chunk_type == b"IHDR":
                fields = struct.unpack(">IIBBBBB", data)
                info = PngInfo(*fields)

            chunks.append((chunk_type, data))
            if chunk_type == b"IEND":
                break

        if info is None:
            raise ValueError("missing IHDR chunk")

    return info, chunks


def read_png_info(path: Path) -> PngInfo:
    with path.open("rb") as file:
        signature = file.read(8)
        if signature != PNG_SIGNATURE:
            raise ValueError("not a PNG file")

        length_bytes = file.read(4)
        if len(length_bytes) != 4:
            raise ValueError("truncated PNG")
        length = struct.unpack(">I", length_bytes)[0]
        chunk_type = file.read(4)
        data = file.read(length)
        if chunk_type != b"IHDR":
            raise ValueError("first PNG chunk is not IHDR")
        if len(data) != length:
            raise ValueError("truncated IHDR chunk")

    return PngInfo(*struct.unpack(">IIBBBBB", data))


def channels_for_color_type(color_type: int) -> int:
    channels = {
        0: 1,  # grayscale
        2: 3,  # RGB
        3: 1,  # indexed color
        4: 2,  # grayscale + alpha
        6: 4,  # RGBA
    }
    try:
        return channels[color_type]
    except KeyError as exc:
        raise ValueError(f"unsupported PNG color type: {color_type}") from exc


def paeth_predictor(left: int, up: int, up_left: int) -> int:
    p = left + up - up_left
    pa = abs(p - left)
    pb = abs(p - up)
    pc = abs(p - up_left)
    if pa <= pb and pa <= pc:
        return left
    if pb <= pc:
        return up
    return up_left


def unfilter_and_crop_scanlines(
    raw: bytes,
    info: PngInfo,
    row_bytes: int,
    bpp: int,
    left: int,
    top: int,
    target_width: int,
    target_height: int,
) -> bytes:
    stride = row_bytes + 1
    expected = stride * info.height
    if len(raw) != expected:
        raise ValueError(f"unexpected decompressed size: {len(raw)} != {expected}")

    cropped = bytearray((target_width * bpp + 1) * target_height)
    cropped_offset = 0
    left_byte = left * bpp
    right_byte = left_byte + target_width * bpp
    previous = bytes(row_bytes)
    for row_index in range(info.height):
        offset = row_index * stride
        filter_type = raw[offset]
        scanline = raw[offset + 1 : offset + stride]

        if filter_type == 0:
            row = scanline
        else:
            row = bytearray(scanline)
        if filter_type == 1:
            for i, value in enumerate(row):
                left_value = row[i - bpp] if i >= bpp else 0
                row[i] = (value + left_value) & 0xFF
        elif filter_type == 2:
            for i, value in enumerate(row):
                row[i] = (value + previous[i]) & 0xFF
        elif filter_type == 3:
            for i, value in enumerate(row):
                left_value = row[i - bpp] if i >= bpp else 0
                up = previous[i]
                row[i] = (value + ((left_value + up) // 2)) & 0xFF
        elif filter_type == 4:
            for i, value in enumerate(row):
                left_value = row[i - bpp] if i >= bpp else 0
                up = previous[i]
                up_left = previous[i - bpp] if i >= bpp else 0
                row[i] = (value + paeth_predictor(left_value, up, up_left)) & 0xFF
        elif filter_type != 0:
            raise ValueError(f"unsupported PNG filter type: {filter_type}")

        if top <= row_index < top + target_height:
            cropped[cropped_offset] = 0
            cropped_offset += 1
            cropped[cropped_offset : cropped_offset + target_width * bpp] = row[left_byte:right_byte]
            cropped_offset += target_width * bpp
        previous = row

    return bytes(cropped)


def write_chunk(file, chunk_type: bytes, data: bytes) -> None:
    file.write(struct.pack(">I", len(data)))
    file.write(chunk_type)
    file.write(data)
    file.write(struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF))


def crop_png(
    path: Path,
    left: int,
    right: int,
    top: int,
    bottom: int,
    target_skip_width: int,
    target_skip_height: int,
    compress_level: int = 1,
) -> bool:
    info, chunks = read_chunks(path)

    if info.width == target_skip_width and info.height == target_skip_height:
        return False

    target_width = info.width - left - right
    target_height = info.height - top - bottom
    if info.interlace != 0:
        raise ValueError("interlaced PNG is not supported")
    if info.compression != 0 or info.filter_method != 0:
        raise ValueError("unsupported PNG compression or filter method")
    if info.bit_depth % 8 != 0:
        raise ValueError(f"bit depth {info.bit_depth} is not byte-aligned")
    if target_width <= 0 or target_height <= 0:
        raise ValueError(
            f"crop left={left}, right={right}, top={top}, bottom={bottom} "
            f"does not fit {info.width}x{info.height}"
        )

    channels = channels_for_color_type(info.color_type)
    bytes_per_sample = info.bit_depth // 8
    pixel_bytes = channels * bytes_per_sample
    row_bytes = info.width * pixel_bytes
    idat_data = b"".join(data for chunk_type, data in chunks if chunk_type == b"IDAT")
    cropped_raw = unfilter_and_crop_scanlines(
        zlib.decompress(idat_data),
        info,
        row_bytes,
        pixel_bytes,
        left,
        top,
        target_width,
        target_height,
    )

    new_info = PngInfo(
        target_width,
        target_height,
        info.bit_depth,
        info.color_type,
        info.compression,
        info.filter_method,
        info.interlace,
    )
    ihdr = struct.pack(
        ">IIBBBBB",
        new_info.width,
        new_info.height,
        new_info.bit_depth,
        new_info.color_type,
        new_info.compression,
        new_info.filter_method,
        new_info.interlace,
    )
    compressed = zlib.compress(cropped_raw, level=compress_level)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("wb") as file:
        file.write(PNG_SIGNATURE)
        write_chunk(file, b"IHDR", ihdr)
        for chunk_type, data in chunks:
            if chunk_type in (b"IHDR", b"IDAT", b"IEND"):
                continue
            write_chunk(file, chunk_type, data)
        write_chunk(file, b"IDAT", compressed)
        write_chunk(file, b"IEND", b"")

    temp_path.replace(path)
    return True


def crop_dimensions(info: PngInfo, left: int, right: int, top: int, bottom: int) -> tuple[int, int]:
    return info.width - left - right, info.height - top - bottom


def plan_png_crop(
    path: Path,
    left: int,
    right: int,
    top: int,
    bottom: int,
    target_width: int,
    target_height: int,
) -> tuple[bool, str | None]:
    try:
        info = read_png_info(path)
        if info.width == target_width and info.height == target_height:
            return False, None

        cropped_width, cropped_height = crop_dimensions(info, left, right, top, bottom)
        if cropped_width <= 0 or cropped_height <= 0:
            return False, (
                f"crop left={left}, right={right}, top={top}, bottom={bottom} "
                f"does not fit {info.width}x{info.height}"
            )
        return True, None
    except Exception as exc:  # noqa: BLE001 - keep processing the dataset.
        return False, str(exc)


def crop_png_worker(task: tuple[str, int, int, int, int, int, int, int]) -> tuple[str, bool, str | None]:
    path_raw, left, right, top, bottom, target_width, target_height, compress_level = task
    path = Path(path_raw)
    try:
        return (
            path_raw,
            crop_png(
                path,
                left,
                right,
                top,
                bottom,
                target_width,
                target_height,
                compress_level,
            ),
            None,
        )
    except Exception as exc:  # noqa: BLE001 - keep processing the dataset.
        return path_raw, False, str(exc)


def route_dirs(data_root: Path) -> list[Path]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"data directory does not exist: {data_root}")

    routes = []
    for scenario_dir in data_root.iterdir():
        if not scenario_dir.is_dir():
            continue
        for route_dir in scenario_dir.iterdir():
            if route_dir.is_dir():
                routes.append(route_dir)
    return routes


def remove_dir(path: Path, apply: bool) -> bool:
    if not path.is_dir():
        return False
    if apply:
        shutil.rmtree(path)
    return True


def remove_file(path: Path, apply: bool) -> bool:
    if not path.is_file():
        return False
    if apply:
        path.unlink()
    return True


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    data_root = dataset_root / "data"
    routes = route_dirs(data_root)

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"Mode: {mode}")
    print(f"Data root: {data_root}")
    print(f"Routes found: {len(routes)}")

    gif_files = 0
    semantics_dirs = 0
    rgb_dirs = 0
    png_seen = 0
    png_to_crop = 0
    png_cropped = 0
    png_skipped = 0
    errors = []
    crop_tasks = []

    for pattern in TARGET_GIF_PATTERNS:
        for gif_path in data_root.rglob(pattern):
            if remove_file(gif_path, args.apply):
                gif_files += 1

    for route_dir in routes:
        semantics_dir = route_dir / "semantics"
        for name in TARGET_SEMANTICS_DIRS:
            if remove_dir(semantics_dir / name, args.apply):
                semantics_dirs += 1

        rgb_dir = route_dir / "rgb"
        for name in TARGET_RGB_DIRS:
            if remove_dir(rgb_dir / name, args.apply):
                rgb_dirs += 1

        if not semantics_dir.is_dir():
            continue

        for png_path in semantics_dir.glob("*.png"):
            png_seen += 1
            if args.apply:
                crop_tasks.append(
                    (
                        str(png_path),
                        args.left,
                        args.right,
                        args.top,
                        args.bottom,
                        args.target_width,
                        args.target_height,
                        args.compress_level,
                    )
                )
                continue

            should_crop, error = plan_png_crop(
                png_path,
                args.left,
                args.right,
                args.top,
                args.bottom,
                args.target_width,
                args.target_height,
            )
            if error:
                errors.append((png_path, error))
                continue
            if not should_crop:
                png_skipped += 1
                continue

            png_to_crop += 1

    if args.apply and crop_tasks:
        print(
            f"\nCropping {len(crop_tasks)} PNG files with {args.workers} worker"
            f"{'' if args.workers == 1 else 's'}..."
        )
        if args.workers == 1:
            results = map(crop_png_worker, crop_tasks)
            for index, (path, cropped, error) in enumerate(results, start=1):
                if cropped:
                    png_to_crop += 1
                    png_cropped += 1
                elif error:
                    errors.append((Path(path), error))
                else:
                    png_skipped += 1
                if args.progress_every and index % args.progress_every == 0:
                    print(f"processed PNG files: {index}/{len(crop_tasks)}")
        else:
            with mp.Pool(processes=args.workers) as pool:
                results = pool.imap_unordered(crop_png_worker, crop_tasks, chunksize=64)
                for index, (path, cropped, error) in enumerate(results, start=1):
                    if cropped:
                        png_to_crop += 1
                        png_cropped += 1
                    elif error:
                        errors.append((Path(path), error))
                    else:
                        png_skipped += 1
                    if args.progress_every and index % args.progress_every == 0:
                        print(f"processed PNG files: {index}/{len(crop_tasks)}")

    print("\nSummary:")
    print(f"crop: left={args.left}, right={args.right}, top={args.top}, bottom={args.bottom}")
    print(f"already-cropped target: {args.target_width}x{args.target_height}")
    print(f"*.gif/*.giff files matched: {gif_files}")
    print(f"semantics/cam1..cam3 dirs matched: {semantics_dirs}")
    print(f"rgb/cam2 dirs matched: {rgb_dirs}")
    print(f"semantics PNG files seen: {png_seen}")
    print(f"PNG files already target size: {png_skipped}")
    print(f"PNG files to crop: {png_to_crop}")
    if args.apply:
        print(f"PNG files cropped: {png_cropped}")
    else:
        print("Dry-run only. Re-run with --apply to modify files.")

    if errors:
        print("\nErrors:")
        for path, error in errors[:20]:
            print(f"{path}: {error}")
        if len(errors) > 20:
            print(f"... {len(errors) - 20} more errors")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
