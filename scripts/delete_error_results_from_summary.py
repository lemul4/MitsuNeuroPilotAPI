#!/usr/bin/env python3
"""Delete CARLA results JSON files for routes with errors from route_error_summary.json.

The script reads `route_path` entries from summary and maps them to files in:
  data/carla_leaderboard2/results/<Scenario>/<core>_result.json

Examples:
  VehicleTurningRoute/Town03_Rep0_Town03_Scenario4_95_route0_04_22_19_14_51
    -> data/carla_leaderboard2/results/VehicleTurningRoute/Town03_Scenario4_95_result.json

  Accident/Town12_Rep0_645_0_route0_04_22_14_12_00
    -> data/carla_leaderboard2/results/Accident/645_0_result.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete results JSON files for routes with errors from route_error_summary.json"
    )
    parser.add_argument(
        "--summary",
        default=os.path.join("data", "carla_leaderboard2", "route_error_summary.json"),
        help="Path to route_error_summary.json",
    )
    parser.add_argument(
        "--results-root",
        default=os.path.join("data", "carla_leaderboard2", "results"),
        help="Root folder with scenario result JSON files",
    )
    parser.add_argument(
        "--targets-output",
        default=os.path.join("data", "carla_leaderboard2", "results_delete_targets.json"),
        help="Where to save resolved JSON targets",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build/save targets and show counts; do not delete files",
    )
    return parser.parse_args()


def load_summary(summary_path: str) -> Dict:
    with open(summary_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Summary root must be a JSON object")
    return payload


def route_path_to_result_relpath(route_path: str) -> Optional[str]:
    if not isinstance(route_path, str) or "/" not in route_path:
        return None

    scenario_name, route_dir_name = route_path.split("/", 1)

    # Remove prefix like Town03_Rep0_ / Town10HD_Rep0_ / Town12_Rep0_
    core = re.sub(r"^Town[^_]*_Rep\d+_", "", route_dir_name)
    # Remove suffix starting from _route<digit>_...
    core = re.sub(r"_route\d+_.*$", "", core)

    if not core:
        return None

    return os.path.join(scenario_name, f"{core}_result.json")


def build_targets(summary: Dict, results_root: str) -> Tuple[List[str], List[str], List[str]]:
    routes = summary.get("routes", [])
    if not isinstance(routes, list):
        raise ValueError("`routes` in summary must be a list")

    rel_targets: Set[str] = set()
    unresolved_routes: List[str] = []

    for route in routes:
        if not isinstance(route, dict):
            continue
        route_path = route.get("route_path")
        if not isinstance(route_path, str):
            continue

        rel_target = route_path_to_result_relpath(route_path)
        if not rel_target:
            unresolved_routes.append(route_path)
            continue
        rel_targets.add(rel_target)

    abs_targets = [os.path.join(results_root, rel_path) for rel_path in sorted(rel_targets)]
    existing = [path for path in abs_targets if os.path.isfile(path)]
    missing = [path for path in abs_targets if not os.path.isfile(path)]

    return existing, missing, sorted(unresolved_routes)


def save_targets_file(
    targets_output: str,
    summary_path: str,
    results_root: str,
    existing: List[str],
    missing: List[str],
    unresolved_routes: List[str],
) -> None:
    output_dir = os.path.dirname(targets_output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    payload = {
        "summary": summary_path,
        "results_root": results_root,
        "existing_targets_count": len(existing),
        "missing_targets_count": len(missing),
        "unresolved_routes_count": len(unresolved_routes),
        "existing_targets": existing,
        "missing_targets": missing,
        "unresolved_routes": unresolved_routes,
    }

    with open(targets_output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def delete_files(paths: List[str]) -> Tuple[int, List[Dict[str, str]]]:
    deleted = 0
    failed: List[Dict[str, str]] = []

    for path in paths:
        try:
            os.remove(path)
            deleted += 1
        except OSError as error:
            failed.append({"path": path, "error": str(error)})

    return deleted, failed


def main() -> int:
    args = parse_args()

    if not os.path.isfile(args.summary):
        print(f"Summary file not found: {args.summary}")
        return 2

    summary = load_summary(args.summary)
    existing, missing, unresolved_routes = build_targets(summary, args.results_root)

    save_targets_file(
        targets_output=args.targets_output,
        summary_path=args.summary,
        results_root=args.results_root,
        existing=existing,
        missing=missing,
        unresolved_routes=unresolved_routes,
    )

    print(f"Saved targets file: {args.targets_output}")
    print(f"Result JSON targets (existing): {len(existing)}")
    print(f"Result JSON targets (missing): {len(missing)}")
    print(f"Unresolved route paths: {len(unresolved_routes)}")

    if args.dry_run:
        print("Dry-run mode: no files deleted")
        return 0

    deleted, failed = delete_files(existing)
    print(f"Deleted result JSON files: {deleted}")
    print(f"Delete failures: {len(failed)}")

    if failed:
        failed_path = os.path.splitext(args.targets_output)[0] + "_delete_failures.json"
        with open(failed_path, "w", encoding="utf-8") as file:
            json.dump(failed, file, ensure_ascii=False, indent=2)
        print(f"Saved delete failures: {failed_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
