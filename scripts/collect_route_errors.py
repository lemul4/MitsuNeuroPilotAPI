#!/usr/bin/env python3
"""Collect selected infractions from CARLA leaderboard results.

Scans a directory tree for `results.json` files, extracts only target infraction
categories, and writes a summary file with route paths and their errors.

Usage:
  python scripts/collect_route_errors.py \
    --root data/carla_leaderboard2/data \
    --output data/carla_leaderboard2/route_error_summary.json

  python scripts/collect_route_errors.py \
    --root data/carla_leaderboard2/data \
    --output data/carla_leaderboard2/route_error_summary.json \
        --build-delete-targets \
        --delete-targets-file data/carla_leaderboard2/delete_targets.json

    python scripts/collect_route_errors.py \
        --root data/carla_leaderboard2/data \
        --output data/carla_leaderboard2/route_error_summary.json \
    --delete-error-routes \
    --delete-targets-file data/carla_leaderboard2/delete_targets.json
"""

import argparse
import json
import os
import re
import shutil
from collections import Counter


TARGET_INFRACTIONS = [
    "scenario_timeouts",
    "vehicle_blocked",
    "route_timeout",
    "collisions_layout",
    "collisions_pedestrian",
    "collisions_vehicle",
    "route_dev",
]

STATUS_FAILED_KEY = "status_failed"
MISSING_RESULTS_KEY = "missing_results_json"


def make_unique_route_key(payload, route_rel_dir):
    route_id = payload.get("route_id") if isinstance(payload, dict) else None
    if isinstance(route_id, str) and route_id.strip():
        return route_id
    return route_rel_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect selected route errors from CARLA results.json files"
    )
    parser.add_argument(
        "--root",
        default=os.path.join("data", "carla_leaderboard2", "data"),
        help="Root folder to scan recursively",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("data", "carla_leaderboard2", "route_error_summary.json"),
        help="Path to output JSON summary",
    )
    parser.add_argument(
        "--build-delete-targets",
        action="store_true",
        help="Build and save delete targets file without deleting files",
    )
    parser.add_argument(
        "--delete-error-routes",
        action="store_true",
        help=(
            "Delete route folders with target errors from data/results and matching "
            "log files from stderr/stdout under carla_leaderboard2"
        ),
    )
    parser.add_argument(
        "--delete-targets-file",
        default=os.path.join("data", "carla_leaderboard2", "delete_targets.json"),
        help="Path to save discovered delete targets for verification",
    )
    return parser.parse_args()


def count_entries(value):
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        return len(value)
    if isinstance(value, str):
        return 1 if value.strip() else 0
    if value in (None, False):
        return 0
    return 1


def list_route_dirs(root_dir):
    route_dirs = []
    try:
        scenario_names = os.listdir(root_dir)
    except OSError:
        return route_dirs

    for scenario_name in scenario_names:
        scenario_path = os.path.join(root_dir, scenario_name)
        if not os.path.isdir(scenario_path):
            continue

        try:
            route_names = os.listdir(scenario_path)
        except OSError:
            continue

        for route_name in route_names:
            route_path = os.path.join(scenario_path, route_name)
            if not os.path.isdir(route_path):
                continue
            route_rel_dir = os.path.join(scenario_name, route_name)
            route_dirs.append((route_rel_dir, route_path))

    return sorted(route_dirs, key=lambda item: item[0])


def collect(root_dir):
    routes_with_errors = []
    infraction_occurrences = Counter()
    infraction_routes = Counter()
    unique_error_routes = set()
    checked_files = 0

    routes_missing_results_json = 0
    for route_rel_dir, route_abs_dir in list_route_dirs(root_dir):
        results_path = os.path.join(route_abs_dir, "results.json")
        if not os.path.isfile(results_path):
            unique_route_key = make_unique_route_key(None, route_rel_dir)
            unique_error_routes.add(unique_route_key)
            routes_missing_results_json += 1
            infraction_occurrences[MISSING_RESULTS_KEY] += 1
            infraction_routes[MISSING_RESULTS_KEY] += 1
            routes_with_errors.append(
                {
                    "route_path": route_rel_dir,
                    "results_file": os.path.relpath(results_path, root_dir),
                    "route_id": None,
                    "timestamp": None,
                    "status": None,
                    "unique_route_key": unique_route_key,
                    "errors": {MISSING_RESULTS_KEY: "results.json not found"},
                    "error_counts": {MISSING_RESULTS_KEY: 1},
                    "total_errors": 1,
                }
            )
            continue

        checked_files += 1

        try:
            with open(results_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue

        infractions = payload.get("infractions", {}) if isinstance(payload, dict) else {}
        if not isinstance(infractions, dict):
            continue

        route_errors = {}
        route_error_counts = {}

        for key in TARGET_INFRACTIONS:
            value = infractions.get(key, [])
            count = count_entries(value)
            if count == 0:
                continue

            route_errors[key] = value
            route_error_counts[key] = count
            infraction_occurrences[key] += count
            infraction_routes[key] += 1

        status_value = payload.get("status") if isinstance(payload, dict) else None
        if isinstance(status_value, str) and "failed" in status_value.strip().lower():
            route_errors[STATUS_FAILED_KEY] = status_value
            route_error_counts[STATUS_FAILED_KEY] = 1
            infraction_occurrences[STATUS_FAILED_KEY] += 1
            infraction_routes[STATUS_FAILED_KEY] += 1

        if not route_errors:
            continue

        unique_route_key = make_unique_route_key(payload, route_rel_dir)
        unique_error_routes.add(unique_route_key)
        routes_with_errors.append(
            {
                "route_path": route_rel_dir,
                "results_file": os.path.relpath(results_path, root_dir),
                "route_id": payload.get("route_id") if isinstance(payload, dict) else None,
                "timestamp": payload.get("timestamp") if isinstance(payload, dict) else None,
                "status": status_value,
                "unique_route_key": unique_route_key,
                "errors": route_errors,
                "error_counts": route_error_counts,
                "total_errors": sum(route_error_counts.values()),
            }
        )

    all_error_keys = TARGET_INFRACTIONS + [STATUS_FAILED_KEY, MISSING_RESULTS_KEY]
    return {
        "root": root_dir,
        "checked_results_files": checked_files,
        "routes_missing_results_json_count": routes_missing_results_json,
        "routes_with_errors_count": len(routes_with_errors),
        "unique_error_routes_count": len(unique_error_routes),
        "infraction_occurrences": {k: infraction_occurrences.get(k, 0) for k in all_error_keys},
        "infraction_routes": {k: infraction_routes.get(k, 0) for k in all_error_keys},
        "routes": routes_with_errors,
    }


def delete_path(path):
    if os.path.islink(path) or os.path.isfile(path):
        os.remove(path)
        return "file"
    if os.path.isdir(path):
        shutil.rmtree(path)
        return "dir"
    return None


def parse_route_components(route_path):
    if not isinstance(route_path, str) or "/" not in route_path:
        return None

    scenario_name, route_dir_name = route_path.split("/", 1)
    rep_match = re.search(r"_Rep(\d+)_", route_dir_name)
    if not rep_match:
        return None

    rep_number = rep_match.group(1)
    core = re.sub(r"^Town[^_]*_Rep\d+_", "", route_dir_name)
    core = re.sub(r"_route\d+_.*$", "", core)
    return {
        "scenario_name": scenario_name,
        "rep": rep_number,
        "core": core,
    }


def find_log_matches(log_root, route_path):
    if not os.path.isdir(log_root):
        return []

    parsed = parse_route_components(route_path)
    if not parsed:
        return []

    prefix = f"{parsed['scenario_name']}_{parsed['core']}_Rep{parsed['rep']}_"
    fallback_prefix = f"{parsed['scenario_name']}_{parsed['core']}_Rep{parsed['rep']}"

    exact_matches = []
    fallback_matches = []
    for name in os.listdir(log_root):
        if not name.endswith(".log"):
            continue
        full_path = os.path.join(log_root, name)
        if name.startswith(prefix):
            exact_matches.append(full_path)
        elif name.startswith(fallback_prefix):
            fallback_matches.append(full_path)

    return exact_matches if exact_matches else fallback_matches


def route_path_to_result_relpath(route_path):
    if not isinstance(route_path, str) or "/" not in route_path:
        return None

    scenario_name, route_dir_name = route_path.split("/", 1)
    core = re.sub(r"^Town[^_]*_Rep\d+_", "", route_dir_name)
    core = re.sub(r"_route\d+_.*$", "", core)
    if not core:
        return None

    return os.path.join(scenario_name, f"{core}_result.json")


def build_delete_plan(root_dir, routes):
    data_root = os.path.abspath(root_dir)
    leaderboard_root = os.path.dirname(data_root)
    results_root = os.path.join(leaderboard_root, "results")
    stderr_root = os.path.join(leaderboard_root, "stderr")
    stdout_root = os.path.join(leaderboard_root, "stdout")

    route_paths = {
        route.get("route_path")
        for route in routes
        if isinstance(route, dict) and isinstance(route.get("route_path"), str)
    }

    data_dirs = []
    results_files = []
    stderr_logs = []
    stdout_logs = []
    unresolved_results_routes = []
    unresolved_log_routes = []

    for route_path in sorted(route_paths):
        if not route_path or route_path == ".":
            continue

        data_dirs.append(os.path.join(data_root, route_path))
        result_relpath = route_path_to_result_relpath(route_path)
        if result_relpath:
            results_files.append(os.path.join(results_root, result_relpath))
        else:
            unresolved_results_routes.append(route_path)

        stderr_matches = find_log_matches(stderr_root, route_path)
        stdout_matches = find_log_matches(stdout_root, route_path)

        if stderr_matches:
            stderr_logs.extend(stderr_matches)
        if stdout_matches:
            stdout_logs.extend(stdout_matches)
        if not stderr_matches and not stdout_matches:
            unresolved_log_routes.append(route_path)

    return {
        "requested_route_paths": len(route_paths),
        "targets": {
            "data": sorted(set(data_dirs)),
            "results": sorted(set(results_files)),
            "stderr": sorted(set(stderr_logs)),
            "stdout": sorted(set(stdout_logs)),
        },
        "unresolved_results_routes": sorted(set(unresolved_results_routes)),
        "unresolved_log_routes": sorted(set(unresolved_log_routes)),
    }


def delete_error_routes(delete_plan):
    targets = delete_plan.get("targets", {})
    deleted = {name: 0 for name in ("data", "results", "stderr", "stdout")}
    missing = {name: 0 for name in ("data", "results", "stderr", "stdout")}
    failed = []

    for root_name in ("data", "results", "stderr", "stdout"):
        for candidate_path in targets.get(root_name, []):
            try:
                deleted_kind = delete_path(candidate_path)
            except OSError as error:
                failed.append(
                    {
                        "root": root_name,
                        "target": candidate_path,
                        "error": str(error),
                    }
                )
                continue

            if deleted_kind is None:
                missing[root_name] += 1
            else:
                deleted[root_name] += 1

    return {
        "requested_route_paths": delete_plan.get("requested_route_paths", 0),
        "targets_total": {
            root_name: len(targets.get(root_name, []))
            for root_name in ("data", "results", "stderr", "stdout")
        },
        "deleted": deleted,
        "missing": missing,
        "unresolved_results_routes_count": len(delete_plan.get("unresolved_results_routes", [])),
        "unresolved_log_routes_count": len(delete_plan.get("unresolved_log_routes", [])),
        "failed_count": len(failed),
        "failed": failed,
    }


def main():
    args = parse_args()

    if not os.path.isdir(args.root):
        print(f"Root folder not found: {args.root}")
        return 2

    summary = collect(args.root)

    if args.build_delete_targets or args.delete_error_routes:
        delete_plan = build_delete_plan(args.root, summary.get("routes", []))
        summary["delete_targets_file"] = args.delete_targets_file
        summary["delete_targets"] = delete_plan

        delete_targets_dir = os.path.dirname(args.delete_targets_file)
        if delete_targets_dir:
            os.makedirs(delete_targets_dir, exist_ok=True)
        with open(args.delete_targets_file, "w", encoding="utf-8") as file:
            json.dump(delete_plan, file, ensure_ascii=False, indent=2)

        if args.delete_error_routes:
            summary["deletion"] = delete_error_routes(delete_plan)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"Checked results files: {summary['checked_results_files']}")
    print(f"Routes missing results.json: {summary['routes_missing_results_json_count']}")
    print(f"Routes with target errors: {summary['routes_with_errors_count']}")
    print(f"Unique error routes: {summary['unique_error_routes_count']}")
    if args.build_delete_targets or args.delete_error_routes:
        print(f"Saved delete targets: {args.delete_targets_file}")

    if args.delete_error_routes:
        deletion = summary.get("deletion", {})
        print("Deleted error routes across roots:")
        for root_name in ("data", "results", "stderr", "stdout"):
            total = deletion.get("targets_total", {}).get(root_name, 0)
            deleted = deletion.get("deleted", {}).get(root_name, 0)
            missing = deletion.get("missing", {}).get(root_name, 0)
            print(f"  {root_name}: targets={total}, deleted={deleted}, missing={missing}")
        print(f"Unresolved results routes: {deletion.get('unresolved_results_routes_count', 0)}")
        print(f"Unresolved log routes: {deletion.get('unresolved_log_routes_count', 0)}")
        print(f"Delete failures: {deletion.get('failed_count', 0)}")
    print(f"Saved summary: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
