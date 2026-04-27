#!/usr/bin/env python3
"""Delete selected CARLA route folders and related artifacts.

Deletes, for each selected route folder:
- route folder under data/carla_leaderboard2/data/<Scenario>/<RouteDir>
- matching result file under data/carla_leaderboard2/results/<Scenario>/<core>_result.json
- matching logs under data/carla_leaderboard2/stderr and stdout

By default runs in dry-run mode. Use --execute to actually delete.
"""

import argparse
import os
import re
import shutil
from dataclasses import dataclass


DEFAULT_TARGET_ROUTE_DIRS = [
    "/home/lemul/MitsuNeuroPilotAPI/data/carla_leaderboard2/data/ParkedObstacle/Town05_Rep0_route_001866_route0_04_23_02_23_11/",
    "/home/lemul/MitsuNeuroPilotAPI/data/carla_leaderboard2/data/ControlLoss/Town05_Rep0_Town05_Scenario1_27_route0_04_23_03_43_04/",
    "/home/lemul/MitsuNeuroPilotAPI/data/carla_leaderboard2/data/ControlLoss/Town05_Rep0_Town05_Scenario1_26_route0_04_23_03_39_03/",
    "/home/lemul/MitsuNeuroPilotAPI/data/carla_leaderboard2/data/noScenarios/Town07_Rep0_Town07_58_route0_04_27_06_22_07/",
    "/home/lemul/MitsuNeuroPilotAPI/data/carla_leaderboard2/data/NonSignalizedJunctionLeftTurn/Town12_Rep0_route_000901_route0_04_26_21_20_41/",
    "/home/lemul/MitsuNeuroPilotAPI/data/carla_leaderboard2/data/NonSignalizedJunctionLeftTurn/Town12_Rep0_route_000892_route0_04_26_21_01_04/",
    "/home/lemul/MitsuNeuroPilotAPI/data/carla_leaderboard2/data/NonSignalizedJunctionLeftTurn/Town07_Rep0_route_000912_route0_04_26_04_47_23/",
    "/home/lemul/MitsuNeuroPilotAPI/data/carla_leaderboard2/data/DynamicObjectCrossing/Town05_Rep0_Town05_Scenario3_27_route0_04_23_04_34_44/",
    "/home/lemul/MitsuNeuroPilotAPI/data/carla_leaderboard2/data/ControlLoss/Town05_Rep0_Town05_Scenario1_17_route0_04_23_03_33_02/",
    "/home/lemul/MitsuNeuroPilotAPI/data/carla_leaderboard2/data/DynamicObjectCrossing/Town03_Rep0_Town03_Scenario3_12_route0_04_22_17_56_17/",
    "/home/lemul/MitsuNeuroPilotAPI/data/carla_leaderboard2/data/NonSignalizedJunctionLeftTurn/Town12_Rep0_358_0_route0_04_24_09_37_37/",
    "/home/lemul/MitsuNeuroPilotAPI/data/carla_leaderboard2/data/VehicleTurningRoutePedestrian/Town12_Rep0_4415_0_route0_04_26_18_03_35/",
    "/home/lemul/MitsuNeuroPilotAPI/data/carla_leaderboard2/data/VehicleTurningRoutePedestrian/Town12_Rep0_1754_1_route0_04_25_13_21_59",
    "/home/lemul/MitsuNeuroPilotAPI/data/carla_leaderboard2/data/DynamicObjectCrossing/Town05_Rep0_Town05_Scenario3_16_route0_04_23_04_23_10/",
    "/home/lemul/MitsuNeuroPilotAPI/data/carla_leaderboard2/data/InvadingTurn/Town12_Rep0_47_0_route0_04_24_07_47_35/",
    "/home/lemul/MitsuNeuroPilotAPI/data/carla_leaderboard2/data/ParkedObstacle/Town06_Rep0_route_001932_route0_04_23_07_01_05",
    "/home/lemul/MitsuNeuroPilotAPI/data/carla_leaderboard2/data/ParkedObstacle/Town06_Rep0_route_001933_route0_04_23_07_06_44/",
    "/home/lemul/MitsuNeuroPilotAPI/data/carla_leaderboard2/data/ParkedObstacle/Town10HD_Rep0_route_001917_route0_04_23_09_45_35/",
    "/home/lemul/MitsuNeuroPilotAPI/data/carla_leaderboard2/data/ControlLoss/Town03_Rep0_Town03_Scenario1_13_route0_04_22_17_45_27/",
    "/home/lemul/MitsuNeuroPilotAPI/data/carla_leaderboard2/data/VehicleTurningRoute/Town07_Rep0_Town07_Scenario4_106_route0_04_23_09_10_07/",
    







]   


@dataclass
class RouteInfo:
    route_abs: str
    scenario: str
    route_dir: str
    rep: str | None
    core: str | None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Delete selected route folders and related results/stderr/stdout artifacts"
    )
    parser.add_argument(
        "--workspace-root",
        default=os.path.abspath("."),
        help="Workspace root path (default: current directory)",
    )
    parser.add_argument(
        "--targets-file",
        default=None,
        help=(
            "Optional text file with route directory paths; each line can be either "
            "a plain path or '<duration>  <path>'"
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete targets (default is dry-run preview)",
    )
    return parser.parse_args()


def core_from_route_dir(route_dir: str) -> str | None:
    core = re.sub(r"^Town[^_]*_Rep\d+_", "", route_dir)
    core = re.sub(r"_route\d+_.*$", "", core)
    if not core:
        return None
    return core


def rep_from_route_dir(route_dir: str) -> str | None:
    match = re.search(r"_Rep(\d+)_", route_dir)
    return match.group(1) if match else None


def parse_route_line(line: str) -> str | None:
    value = line.strip()
    if not value or value.startswith("#"):
        return None

    if "/" in value:
        return value[value.find("/"):].strip()
    return None


def load_targets(targets_file: str | None):
    if not targets_file:
        return list(DEFAULT_TARGET_ROUTE_DIRS)

    targets = []
    with open(targets_file, "r", encoding="utf-8") as file:
        for line in file:
            route = parse_route_line(line)
            if route:
                targets.append(route)
    return targets


def build_route_infos(route_dirs, data_root):
    infos = []
    for route in route_dirs:
        route_abs = os.path.abspath(route)
        rel = os.path.relpath(route_abs, data_root)
        parts = rel.split(os.sep)
        if len(parts) < 2 or parts[0] == "..":
            continue

        scenario = parts[0]
        route_dir = parts[1]
        infos.append(
            RouteInfo(
                route_abs=route_abs,
                scenario=scenario,
                route_dir=route_dir,
                rep=rep_from_route_dir(route_dir),
                core=core_from_route_dir(route_dir),
            )
        )
    return infos


def matching_logs(log_root, scenario, core, rep):
    if not (core and rep) or not os.path.isdir(log_root):
        return []

    prefix = f"{scenario}_{core}_Rep{rep}"
    matches = []
    for name in os.listdir(log_root):
        if name.endswith(".log") and name.startswith(prefix):
            matches.append(os.path.join(log_root, name))
    return sorted(matches)


def delete_path(path):
    if os.path.islink(path) or os.path.isfile(path):
        os.remove(path)
        return "file"
    if os.path.isdir(path):
        shutil.rmtree(path)
        return "dir"
    return None


def main():
    args = parse_args()

    workspace_root = os.path.abspath(args.workspace_root)
    data_root = os.path.join(workspace_root, "data", "carla_leaderboard2", "data")
    results_root = os.path.join(workspace_root, "data", "carla_leaderboard2", "results")
    stderr_root = os.path.join(workspace_root, "data", "carla_leaderboard2", "stderr")
    stdout_root = os.path.join(workspace_root, "data", "carla_leaderboard2", "stdout")

    route_dirs = load_targets(args.targets_file)
    route_infos = build_route_infos(route_dirs, data_root)

    targets = {"data": [], "results": [], "stderr": [], "stdout": []}

    for info in route_infos:
        targets["data"].append(info.route_abs)

        if info.core:
            results_file = os.path.join(results_root, info.scenario, f"{info.core}_result.json")
            targets["results"].append(results_file)

        targets["stderr"].extend(matching_logs(stderr_root, info.scenario, info.core, info.rep))
        targets["stdout"].extend(matching_logs(stdout_root, info.scenario, info.core, info.rep))

    for key in targets:
        targets[key] = sorted(set(targets[key]))

    print("Mode:", "EXECUTE" if args.execute else "DRY-RUN")
    for key in ("data", "results", "stderr", "stdout"):
        print(f"{key}: {len(targets[key])} targets")

    if not args.execute:
        print("\nDry-run preview complete. Re-run with --execute to delete.")
        return 0

    deleted = {"data": 0, "results": 0, "stderr": 0, "stdout": 0}
    missing = {"data": 0, "results": 0, "stderr": 0, "stdout": 0}
    failed = []

    for key in ("data", "results", "stderr", "stdout"):
        for path in targets[key]:
            try:
                result = delete_path(path)
            except OSError as error:
                failed.append((key, path, str(error)))
                continue

            if result is None:
                missing[key] += 1
            else:
                deleted[key] += 1

    print("\nDeletion summary:")
    for key in ("data", "results", "stderr", "stdout"):
        print(
            f"{key}: targets={len(targets[key])}, deleted={deleted[key]}, missing={missing[key]}"
        )

    print(f"failures: {len(failed)}")
    for key, path, error in failed:
        print(f"  {key}: {path} -> {error}")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
