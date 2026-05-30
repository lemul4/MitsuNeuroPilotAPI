from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vehicle_control.preflight import RealVehiclePreflightChecker


def main() -> int:
    parser = argparse.ArgumentParser(description="Check real i-MiEV test readiness without actuating the vehicle.")
    parser.add_argument("--root", default=str(ROOT), help="i-MIEV GUI directory")
    args = parser.parse_args()
    report = RealVehiclePreflightChecker(args.root).check()
    print(report.to_text())
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
