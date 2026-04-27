#!/usr/bin/env python3
"""Summarize CARLA route time and snapshots.

Scans leaderboard data folders for route directories that contain `results.json`.
For each route, extracts:
- scenario (top-level folder, e.g. Accident)
- town (from route dir name, e.g. Town03)
- duration_game / duration_system from results.json -> meta
- snapshots from metas/*.pkl:
    - files_count: number of .pkl files
    - last_index: max numeric filename stem (e.g. 0216.pkl -> 216)

Then builds totals:
- global
- by town
- by scenario

Usage:
  python scripts/summarize_route_time_and_snapshots.py \
    --root data/carla_leaderboard2/data \
    --output data/carla_leaderboard2/route_time_snapshot_summary.json
"""

import argparse
import json
import os
import re
from collections import defaultdict


TOWN_RE = re.compile(r"^(Town\d+)_")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize route time and snapshot stats from CARLA leaderboard data"
    )
    parser.add_argument(
        "--root",
        default=os.path.join("data", "carla_leaderboard2", "data"),
        help="Root folder to scan recursively",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            "data", "carla_leaderboard2", "route_time_snapshot_summary.json"
        ),
        help="Path to write summary JSON",
    )
    return parser.parse_args()


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_town(route_dir_name):
    match = TOWN_RE.match(route_dir_name or "")
    if match:
        return match.group(1)
    return "UNKNOWN"


def collect_snapshot_stats(metas_dir):
    files_count = 0
    max_index = None
    total_size_bytes = 0

    if not os.path.isdir(metas_dir):
        return files_count, max_index, total_size_bytes

    for filename in os.listdir(metas_dir):
        if not filename.lower().endswith(".pkl"):
            continue

        file_path = os.path.join(metas_dir, filename)
        files_count += 1
        try:
            total_size_bytes += os.path.getsize(file_path)
        except OSError:
            pass

        stem, _ = os.path.splitext(filename)
        if stem.isdigit():
            current = int(stem)
            if max_index is None or current > max_index:
                max_index = current

    return files_count, max_index, total_size_bytes


def collect_route_size_bytes(route_dir):
    total_size_bytes = 0

    if not os.path.isdir(route_dir):
        return total_size_bytes

    for dirpath, _, filenames in os.walk(route_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                total_size_bytes += os.path.getsize(file_path)
            except OSError:
                pass

    return total_size_bytes


def new_bucket():
    return {
        "routes_count": 0,
        "duration_game_total": 0.0,
        "duration_system_total": 0.0,
        "snapshots_files_total": 0,
        "snapshots_size_bytes_total": 0,
        "snapshots_last_index_sum": 0,
        "routes_with_last_index": 0,
    }


def add_to_bucket(
    bucket,
    duration_game,
    duration_system,
    snapshots_files,
    snapshots_size_bytes,
    last_index,
):
    bucket["routes_count"] += 1
    bucket["duration_game_total"] += duration_game
    bucket["duration_system_total"] += duration_system
    bucket["snapshots_files_total"] += snapshots_files
    bucket["snapshots_size_bytes_total"] += snapshots_size_bytes
    if last_index is not None:
        bucket["snapshots_last_index_sum"] += last_index
        bucket["routes_with_last_index"] += 1


def finalize_bucket(bucket):
    routes_with_last = bucket["routes_with_last_index"]
    bucket["avg_last_snapshot_index"] = (
        bucket["snapshots_last_index_sum"] / routes_with_last if routes_with_last else None
    )
    bucket["snapshots_size_mb_total"] = bucket["snapshots_size_bytes_total"] / (1024 * 1024)
    return bucket


def collect(root_dir):
    all_routes = []
    totals = new_bucket()
    by_town = defaultdict(new_bucket)
    by_scenario = defaultdict(new_bucket)
    checked_results_files = 0

    for dirpath, _, filenames in os.walk(root_dir):
        if "results.json" not in filenames:
            continue

        checked_results_files += 1
        results_path = os.path.join(dirpath, "results.json")

        try:
            with open(results_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue

        rel_dir = os.path.relpath(dirpath, root_dir)
        parts = rel_dir.split(os.sep)
        scenario = parts[0] if len(parts) >= 2 else "UNKNOWN"
        route_dir_name = parts[1] if len(parts) >= 2 else parts[0]
        town = extract_town(route_dir_name)

        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        duration_game = safe_float(meta.get("duration_game"), 0.0)
        duration_system = safe_float(meta.get("duration_system"), 0.0)

        metas_dir = os.path.join(dirpath, "metas")
        snapshots_files, last_snapshot_index, metas_size_bytes = collect_snapshot_stats(
            metas_dir
        )
        snapshots_size_bytes = collect_route_size_bytes(dirpath)

        route_item = {
            "route_path": rel_dir,
            "scenario": scenario,
            "town": town,
            "route_id": payload.get("route_id") if isinstance(payload, dict) else None,
            "status": payload.get("status") if isinstance(payload, dict) else None,
            "duration_game": duration_game,
            "duration_system": duration_system,
            "snapshots_files_count": snapshots_files,
            "metas_size_bytes": metas_size_bytes,
            "metas_size_mb": metas_size_bytes / (1024 * 1024),
            "snapshots_size_bytes": snapshots_size_bytes,
            "snapshots_size_mb": snapshots_size_bytes / (1024 * 1024),
            "last_snapshot_index": last_snapshot_index,
        }
        all_routes.append(route_item)

        add_to_bucket(
            totals,
            duration_game,
            duration_system,
            snapshots_files,
            snapshots_size_bytes,
            last_snapshot_index,
        )
        add_to_bucket(
            by_town[town],
            duration_game,
            duration_system,
            snapshots_files,
            snapshots_size_bytes,
            last_snapshot_index,
        )
        add_to_bucket(
            by_scenario[scenario],
            duration_game,
            duration_system,
            snapshots_files,
            snapshots_size_bytes,
            last_snapshot_index,
        )

    result = {
        "root": root_dir,
        "checked_results_files": checked_results_files,
        "totals": finalize_bucket(totals),
        "by_town": {k: finalize_bucket(v) for k, v in sorted(by_town.items())},
        "by_scenario": {k: finalize_bucket(v) for k, v in sorted(by_scenario.items())},
        "routes": sorted(all_routes, key=lambda x: x.get("route_path") or ""),
    }
    return result


def main():
    args = parse_args()

    if not os.path.isdir(args.root):
        print(f"Root folder not found: {args.root}")
        return 2

    summary = collect(args.root)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    totals = summary["totals"]
    print(f"Checked results.json files: {summary['checked_results_files']}")
    print(f"Routes: {totals['routes_count']}")
    print(f"Total duration_game: {totals['duration_game_total']:.3f}")
    print(f"Total duration_system: {totals['duration_system_total']:.3f}")
    print(f"Total snapshots (.pkl files): {totals['snapshots_files_total']}")
    print(f"Total snapshots size (MB): {totals['snapshots_size_mb_total']:.3f}")
    print(f"Summary written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
