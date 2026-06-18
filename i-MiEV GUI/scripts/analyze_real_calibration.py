from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def _get(record, path, default=0.0):
    current = record
    for key in path.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _f(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _sign(value: float, threshold: float = 1e-6) -> int:
    value = _f(value)
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def load_records(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize real i-MiEV calibration JSONL.")
    parser.add_argument("jsonl", type=Path)
    args = parser.parse_args()

    records = load_records(args.jsonl)
    if not records:
        print("No records.")
        return 1

    real_records = [
        r for r in records
        if str(_get(r, "vehicle.descriptor_kind", "")).lower() in {"real_serial", "devicekind.real_serial"}
        or str(_get(r, "telemetry.pose_source", "")).lower() not in {"mock_odometry", "mission_start_mock"}
    ]
    if not real_records:
        print(f"Records: {len(records)}")
        print("No real-vehicle records detected.")
        print("This file looks like mock/test data; it is not useful for steering calibration.")
        return 0

    rows = []
    for record in real_records:
        target_raw = _f(_get(record, "arbiter.last_target_steering_raw"))
        raw = _f(_get(record, "command.steering_raw"))
        actual_angle = _f(_get(record, "telemetry.angle_deg"))
        target_angle = _f(_get(record, "telemetry.target_angle_deg"))
        wheel_angle = _f(_get(record, "intent_metadata.real_steer_front_wheel_angle_deg"))
        steer_norm = _f(_get(record, "intent.steer_norm"))
        speed = _f(_get(record, "telemetry.speed_kmh"))
        frame = int(_f(_get(record, "intent.frame_id")))
        rows.append({
            "record": record,
            "frame": frame,
            "target_raw": target_raw,
            "raw": raw,
            "target_angle": target_angle,
            "actual_angle": actual_angle,
            "wheel_angle": wheel_angle,
            "steer_norm": steer_norm,
            "speed": speed,
            "t": _f(_get(record, "created_at_monotonic")),
            "xtrack": _f(_get(record, "goal.cross_track_error_m")),
            "heading": _f(_get(record, "goal.heading_error_deg")),
            "progress": _f(_get(record, "goal.route_progress")),
            "wp": int(_f(_get(record, "goal.waypoint_index"))),
            "control_wp": int(_f(_get(record, "goal.control_waypoint_index"))),
            "distance": _f(_get(record, "goal.distance_to_target_m")),
            "rate_limited": bool(_get(record, "arbiter.last_steering_rate_limited", False)),
            "metadata_present": bool(_get(record, "intent_metadata", {}) or _get(record, "intent.metadata", {})),
        })

    t0 = rows[0]["t"]
    duration = max(1e-9, rows[-1]["t"] - rows[0]["t"])
    abs_errors = [abs(row["target_angle"] - row["actual_angle"]) for row in rows]
    saturated = [row for row in rows if abs(row["raw"]) >= 98.0 or abs(row["target_raw"]) >= 98.0]
    strong = [row for row in rows if abs(row["raw"]) >= 50.0 or abs(row["target_raw"]) >= 50.0]
    min_effective_hits = [row for row in rows if 69.0 <= abs(row["raw"]) <= 75.0]
    sign_conflicts = [
        row for row in rows
        if abs(row["raw"]) >= 20.0
        and abs(row["xtrack"]) >= 0.5
        and _sign(row["raw"], 20.0) == _sign(row["xtrack"], 0.5)
    ]

    print(f"Records: {len(records)}")
    print(f"Real records: {len(real_records)}")
    print(f"Duration: {duration:.1f}s rate={len(rows) / duration:.1f}Hz")
    print(f"Frame ids: unique={len(set(row['frame'] for row in rows))} first={rows[0]['frame']} last={rows[-1]['frame']}")
    print(f"Model metadata rows: {sum(1 for row in rows if row['metadata_present'])}")
    print(f"Speed: min={min(r['speed'] for r in rows):.2f}km/h max={max(r['speed'] for r in rows):.2f}km/h avg={mean(r['speed'] for r in rows):.2f}km/h")
    print(f"Route progress: first={rows[0]['progress']:.1f}% last={rows[-1]['progress']:.1f}%")
    print(f"Cross-track: min={min(r['xtrack'] for r in rows):.2f}m max={max(r['xtrack'] for r in rows):.2f}m avg_abs={mean(abs(r['xtrack']) for r in rows):.2f}m")
    print(f"Heading error: min={min(r['heading'] for r in rows):.1f}deg max={max(r['heading'] for r in rows):.1f}deg avg_abs={mean(abs(r['heading']) for r in rows):.1f}deg")
    print(f"Steering raw: min={min(r['raw'] for r in rows):.0f} max={max(r['raw'] for r in rows):.0f} avg_abs={mean(abs(r['raw']) for r in rows):.1f}")
    print(f"Steering angle error abs: avg={mean(abs_errors):.1f}deg max={max(abs_errors):.1f}deg")
    print(f"Rate-limited commands: {sum(1 for row in rows if row['rate_limited'])}")
    print(f"Strong steering commands: {len(strong)}")
    print(f"Saturated steering commands: {len(saturated)}")
    print(f"Min-effective steering jumps: {len(min_effective_hits)}")
    print(f"Raw/cross-track sign conflicts: {len(sign_conflicts)}")

    print("Waypoint transitions:")
    last_wp = None
    transitions = 0
    for row in rows:
        if row["wp"] == last_wp:
            continue
        transitions += 1
        last_wp = row["wp"]
        if transitions <= 40:
            print(
                f"  t={row['t'] - t0:.1f}s wp={row['wp']} control_wp={row['control_wp']} "
                f"progress={row['progress']:.1f}% dist={row['distance']:.2f}m "
                f"xtrack={row['xtrack']:.2f}m heading={row['heading']:.1f}deg "
                f"raw={row['raw']:.0f} angle={row['actual_angle']:.1f}deg"
            )
    print(f"Waypoint transitions total: {transitions}")

    print("Top steering angle errors:")
    for row in sorted(
        rows,
        key=lambda item: abs(item["target_angle"] - item["actual_angle"]),
        reverse=True,
    )[:10]:
        print(
            f"  t={row['t'] - t0:.1f}s frame={row['frame']} speed={row['speed']:.2f}km/h "
            f"steer_norm={row['steer_norm']:.3f} target_raw={row['target_raw']:.0f} raw={row['raw']:.0f} "
            f"target_angle={row['target_angle']:.1f} actual_angle={row['actual_angle']:.1f} "
            f"err={row['target_angle'] - row['actual_angle']:.1f} "
            f"xtrack={row['xtrack']:.2f}m heading={row['heading']:.1f}deg"
        )

    print("Top cross-track errors:")
    for row in sorted(rows, key=lambda item: abs(item["xtrack"]), reverse=True)[:10]:
        print(
            f"  t={row['t'] - t0:.1f}s progress={row['progress']:.1f}% wp={row['wp']} "
            f"xtrack={row['xtrack']:.2f}m heading={row['heading']:.1f}deg "
            f"raw={row['raw']:.0f} angle={row['actual_angle']:.1f}deg"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
