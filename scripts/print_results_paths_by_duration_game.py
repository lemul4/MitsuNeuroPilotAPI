#!/usr/bin/env python3
"""Print full paths to results.json sorted by duration_game ascending."""

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find results.json files, sort by meta.duration_game ascending, "
            "and print full file paths"
        )
    )
    parser.add_argument(
        "--root",
        default=os.path.join("data", "carla_leaderboard2", "data"),
        help="Root folder to scan recursively",
    )
    return parser.parse_args()


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_sorted_paths(root_dir):
    rows = []

    for dirpath, _, filenames in os.walk(root_dir):
        if "results.json" not in filenames:
            continue

        results_path = os.path.join(dirpath, "results.json")
        try:
            with open(results_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue

        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        duration_game = safe_float(meta.get("duration_game"))
        if duration_game is None:
            continue

        rows.append((duration_game, os.path.abspath(results_path)))

    rows.sort(key=lambda item: (item[0], item[1]))
    return rows


def main():
    args = parse_args()

    if not os.path.isdir(args.root):
        print(f"Root folder not found: {args.root}")
        return 2

    for duration_game, full_path in collect_sorted_paths(args.root):
        print(f"{duration_game:10.3f}  {full_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
