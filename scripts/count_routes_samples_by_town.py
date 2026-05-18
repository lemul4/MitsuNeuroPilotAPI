#!/usr/bin/env python3
"""Count routes and samples per town in CARLA dual camera dataset."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Tuple


@dataclass
class TownStats:
    routes: int = 0
    samples: int = 0
    scenario_counts: Dict[str, int] = field(default_factory=dict)


def parse_town_name(route_dir_name: str) -> str:
    """Extract town name from route directory name."""
    if "_Rep" in route_dir_name:
        return route_dir_name.split("_Rep", 1)[0]
    # Fallback: use leading token before first underscore
    if "_" in route_dir_name:
        return route_dir_name.split("_", 1)[0]
    return route_dir_name


def iter_route_dirs(data_root: Path) -> Iterable[Tuple[str, str, Path]]:
    """Yield (scenario_name, town_name, route_path) for each route directory."""
    for scenario_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        scenario_name = scenario_dir.name
        for route_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
            town_name = parse_town_name(route_dir.name)
            yield scenario_name, town_name, route_dir


def count_samples(route_dir: Path, sample_dir_name: str) -> int:
    """Count samples inside a route directory using the sample subfolder."""
    sample_dir = route_dir / sample_dir_name
    if not sample_dir.exists() or not sample_dir.is_dir():
        return 0
    return sum(1 for p in sample_dir.iterdir() if p.is_file())


def collect_stats(data_root: Path, sample_dir_name: str) -> Dict[str, TownStats]:
    stats: Dict[str, TownStats] = defaultdict(TownStats)
    for scenario_name, town_name, route_dir in iter_route_dirs(data_root):
        stats[town_name].routes += 1
        stats[town_name].samples += count_samples(route_dir, sample_dir_name)
        stats[town_name].scenario_counts[scenario_name] = (
            stats[town_name].scenario_counts.get(scenario_name, 0) + 1
        )
    return stats


def format_table(stats: Dict[str, TownStats]) -> str:
    if not stats:
        return "No routes found."

    scenario_names = sorted({
        scenario
        for town_stats in stats.values()
        for scenario in town_stats.scenario_counts.keys()
    })

    rows = []
    for town, s in stats.items():
        scenario_values = [s.scenario_counts.get(name, 0) for name in scenario_names]
        rows.append((town, s.routes, s.samples, scenario_values))
    rows.sort(key=lambda r: r[0])

    town_w = max(len("Town"), max(len(r[0]) for r in rows))
    routes_w = max(len("Routes"), max(len(str(r[1])) for r in rows))
    samples_w = max(len("Samples"), max(len(str(r[2])) for r in rows))
    scenario_ws = [
        max(len(name), max(len(str(r[3][idx])) for r in rows))
        for idx, name in enumerate(scenario_names)
    ]

    lines = []
    header_parts = [
        f"{'Town':<{town_w}}",
        f"{'Routes':>{routes_w}}",
        f"{'Samples':>{samples_w}}",
    ]
    header_parts.extend(
        f"{name:<{scenario_ws[idx]}}" for idx, name in enumerate(scenario_names)
    )
    header = "  ".join(header_parts)
    lines.append(header)
    lines.append("-" * len(header))

    total_routes = 0
    total_samples = 0
    scenario_totals = [0 for _ in scenario_names]
    for town, routes, samples, scenario_values in rows:
        total_routes += routes
        total_samples += samples
        for idx, value in enumerate(scenario_values):
            scenario_totals[idx] += value
        row_parts = [
            f"{town:<{town_w}}",
            f"{routes:>{routes_w}}",
            f"{samples:>{samples_w}}",
        ]
        row_parts.extend(
            f"{scenario_values[idx]:>{scenario_ws[idx]}}" for idx in range(len(scenario_names))
        )
        lines.append("  ".join(row_parts))

    lines.append("-" * len(header))
    total_parts = [
        f"{'TOTAL':<{town_w}}",
        f"{total_routes:>{routes_w}}",
        f"{total_samples:>{samples_w}}",
    ]
    total_parts.extend(
        f"{scenario_totals[idx]:>{scenario_ws[idx]}}" for idx in range(len(scenario_names))
    )
    lines.append("  ".join(total_parts))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count routes and samples per town in the CARLA dataset.",
    )
    parser.add_argument(
        "--data-root",
        default="data/carla_leaderboard2_dual_cameras/data",
        help="Path to dataset root containing scenario folders.",
    )
    parser.add_argument(
        "--sample-dir",
        default="metas",
        help="Subfolder inside each route used to count samples.",
    )
    parser.add_argument(
        "--output",
        default="routes_samples_by_town.txt",
        help="Output file to save the table (use '-' for stdout only).",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.exists():
        raise SystemExit(f"Data root not found: {data_root}")

    stats = collect_stats(data_root, args.sample_dir)
    table = format_table(stats)
    print(table)
    if args.output != "-":
        output_path = Path(args.output).expanduser().resolve()
        output_path.write_text(table + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
