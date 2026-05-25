#!/usr/bin/env python3
"""Move N route folders per scenario category from val_extra to val."""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC_ROOT = REPO_ROOT / "data" / "carla_leaderboard2_dual_cameras_val_extra"
DEFAULT_DST_ROOT = REPO_ROOT / "data" / "carla_leaderboard2_dual_cameras_val"

ROUTE_RE = re.compile(r"^Town\d+_Rep\d+_(?P<route_id>\d+_\d+)_route\d+_")
ZERO_PADDED_ROUTE_RE = re.compile(
    r"^Town\d+_Rep\d+_route_(?P<route_number>\d+)_route(?P<variant>\d+)_"
)
NATURAL_RE = re.compile(r"(\d+)")


@dataclass(frozen=True)
class Route:
    category: str
    route_id: str
    data_dir: Path
    result_file: Path


def natural_key(path: Path) -> tuple[object, ...]:
    parts = NATURAL_RE.split(path.name)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move X route directories for every scenario category from "
            "carla_leaderboard2_dual_cameras_val_extra to "
            "carla_leaderboard2_dual_cameras_val."
        )
    )
    parser.add_argument(
        "routes_per_category",
        type=int,
        help="How many route folders to move from each scenario category.",
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=DEFAULT_SRC_ROOT,
        help=f"Source dataset root. Default: {DEFAULT_SRC_ROOT}",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=DEFAULT_DST_ROOT,
        help=f"Destination dataset root. Default: {DEFAULT_DST_ROOT}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be moved without changing files.",
    )
    parser.add_argument(
        "--require-results",
        action="store_true",
        help="Skip route folders that do not have a matching result JSON.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.routes_per_category < 0:
        raise SystemExit("routes_per_category must be >= 0")

    src_data = args.src_root / "data"
    src_results = args.src_root / "results"
    if not src_data.is_dir():
        raise SystemExit(f"Source data directory does not exist: {src_data}")
    if not src_results.is_dir() and args.require_results:
        raise SystemExit(f"Source results directory does not exist: {src_results}")


def get_route_id(route_dir: Path) -> str | None:
    match = ROUTE_RE.match(route_dir.name)
    if match:
        return match.group("route_id")

    match = ZERO_PADDED_ROUTE_RE.match(route_dir.name)
    if match:
        route_number = int(match.group("route_number"))
        variant = int(match.group("variant"))
        return f"{route_number}_{variant}"

    return None


def collect_routes(src_root: Path, require_results: bool) -> dict[str, list[Route]]:
    routes_by_category: dict[str, list[Route]] = {}
    src_data = src_root / "data"
    src_results = src_root / "results"

    for category_dir in sorted(src_data.iterdir(), key=natural_key):
        if not category_dir.is_dir():
            continue

        category_routes: list[Route] = []
        for route_dir in sorted(category_dir.iterdir(), key=natural_key):
            if not route_dir.is_dir():
                continue

            route_id = get_route_id(route_dir)
            if route_id is None:
                print(f"Skip route with unexpected name: {route_dir}")
                continue

            result_file = src_results / category_dir.name / f"{route_id}_result.json"
            if not result_file.exists() and require_results:
                print(f"Skip route without result JSON: {route_dir}")
                continue

            category_routes.append(
                Route(
                    category=category_dir.name,
                    route_id=route_id,
                    data_dir=route_dir,
                    result_file=result_file,
                )
            )

        routes_by_category[category_dir.name] = category_routes

    return routes_by_category


def ensure_destination_is_free(route: Route, dst_root: Path) -> None:
    dst_data_dir = dst_root / "data" / route.category / route.data_dir.name
    dst_result_file = dst_root / "results" / route.category / route.result_file.name

    if dst_data_dir.exists():
        raise FileExistsError(f"Destination route directory already exists: {dst_data_dir}")
    if route.result_file.exists() and dst_result_file.exists():
        raise FileExistsError(f"Destination result file already exists: {dst_result_file}")


def move_route(route: Route, dst_root: Path, dry_run: bool) -> None:
    dst_data_category = dst_root / "data" / route.category
    dst_results_category = dst_root / "results" / route.category
    dst_data_dir = dst_data_category / route.data_dir.name
    dst_result_file = dst_results_category / route.result_file.name

    print(f"{'[dry-run] ' if dry_run else ''}{route.category}: {route.data_dir.name}")
    if dry_run:
        return

    dst_data_category.mkdir(parents=True, exist_ok=True)
    dst_results_category.mkdir(parents=True, exist_ok=True)

    shutil.move(str(route.data_dir), str(dst_data_dir))
    if route.result_file.exists():
        shutil.move(str(route.result_file), str(dst_result_file))


def main() -> None:
    args = parse_args()
    validate_args(args)

    routes_by_category = collect_routes(
        src_root=args.src_root,
        require_results=args.require_results,
    )

    selected_routes: list[Route] = []
    for category, routes in routes_by_category.items():
        selected = routes[: args.routes_per_category]
        selected_routes.extend(selected)
        print(f"{category}: selected {len(selected)} of {len(routes)} available routes")

    for route in selected_routes:
        ensure_destination_is_free(route, args.dst_root)

    for route in selected_routes:
        move_route(route, args.dst_root, args.dry_run)

    action = "Would move" if args.dry_run else "Moved"
    print(f"{action} {len(selected_routes)} route folders from {args.src_root} to {args.dst_root}")


if __name__ == "__main__":
    main()
