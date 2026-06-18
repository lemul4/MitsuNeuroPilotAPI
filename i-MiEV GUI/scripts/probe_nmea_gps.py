from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import serial
import serial.tools.list_ports

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware.pose_provider import Nmea0183Parser, NmeaFixFilter, discover_nmea_serial_port


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe a GPS/GNSS NMEA-0183 serial port.")
    parser.add_argument("--port", default="", help="COM port, e.g. COM7. Empty = auto-detect.")
    parser.add_argument("--baud", type=int, default=9600, help="Serial baudrate.")
    parser.add_argument("--seconds", type=float, default=15.0, help="Probe duration.")
    parser.add_argument("--raw", action="store_true", help="Print raw NMEA sentences.")
    args = parser.parse_args()

    ports = list(serial.tools.list_ports.comports())
    print("Available serial ports:")
    for item in ports:
        print(f"  {item.device}: {item.description} [{item.hwid}]")

    port = args.port.strip() or discover_nmea_serial_port(ports)
    if not port:
        print("No likely NMEA/GPS serial port found.")
        return 2

    print(f"Using port={port} baud={args.baud}")
    parser_nmea = Nmea0183Parser()
    fix_filter = NmeaFixFilter()
    deadline = time.monotonic() + float(args.seconds)
    rx = parsed = accepted = bad = filtered = 0

    try:
        with serial.Serial(port, args.baud, timeout=0.5) as ser:
            while time.monotonic() < deadline:
                raw = ser.readline()
                if not raw:
                    continue
                rx += 1
                sentence = raw.decode("ascii", errors="ignore").strip()
                if args.raw:
                    print(f"RAW: {sentence}")
                fix = parser_nmea.parse(sentence)
                if fix is None:
                    bad += 1
                    continue
                parsed += 1
                out = fix_filter.accept(fix)
                if out is None:
                    filtered += 1
                    continue
                accepted += 1
                print(
                    "FIX: "
                    f"lat={out.lat:.8f} lon={out.lon:.8f} "
                    f"sats={out.satellites} hdop={out.hdop:.1f} "
                    f"speed={out.speed_mps:.2f}m/s course={out.course_deg}"
                )
    except Exception as exc:
        print(f"ERROR opening/reading {port}: {exc}")
        if "PermissionError" in repr(exc) or "Отказано в доступе" in str(exc):
            print("Port is probably already opened by another program or by the GUI.")
        return 1

    print(f"Summary: rx={rx} parsed={parsed} accepted={accepted} filtered={filtered} bad={bad}")
    return 0 if accepted else 3


if __name__ == "__main__":
    raise SystemExit(main())
