from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import serial
import serial.tools.list_ports

GUI_ROOT = Path(__file__).resolve().parents[1]
if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))

import utils
from vehicle_control.mcu_protocol import McuTelemetryParser
from vehicle_control.models import VehicleTelemetry


def _auto_port() -> str:
    ports = list(serial.tools.list_ports.comports())
    usb = [p for p in ports if "USB" in str(p.hwid).upper() or "USB" in str(p.description).upper()]
    preferred = usb or ports
    if not preferred:
        raise RuntimeError("Serial ports not found")
    return str(preferred[0].device)


def _hex(data) -> str:
    return " ".join(f"{int(x) & 0xFF:02X}" for x in data)


def sniff(port: str, baudrate: int, seconds: float, output_dir: Path) -> dict:
    parser = McuTelemetryParser.load()
    telemetry = VehicleTelemetry()
    buffer = utils.CircularBuffer(utils.PACKET_SIZE * 8)
    can_counts: Counter[int] = Counter()
    mapped_counts: Counter[int] = Counter()
    raw_samples: list[str] = []
    packet_samples: list[dict] = []
    byte_count = 0
    invalid_windows = 0
    started = time.monotonic()
    deadline = started + float(seconds)

    with serial.Serial(port=port, baudrate=baudrate, timeout=0.05) as ser:
        while time.monotonic() < deadline:
            chunk = ser.read(4096)
            if not chunk:
                continue
            byte_count += len(chunk)
            if len(raw_samples) < 8:
                raw_samples.append(_hex(chunk[:64]))
            for b in chunk:
                buffer.add(b)
                if buffer.count < utils.PACKET_SIZE:
                    continue
                frame = buffer.check_buffer()
                if frame:
                    try:
                        pkt = utils.Serial_Data(frame)
                        can_id = int(pkt.CAN_ID)
                        data = list(pkt.CAN_DATA.DATA)
                        can_counts[can_id] += 1
                        if parser.apply_packet(can_id, data, telemetry):
                            mapped_counts[can_id] += 1
                        if len(packet_samples) < 20:
                            packet_samples.append({
                                "can_id": f"0x{can_id:04X}",
                                "data": _hex(data),
                            })
                    except Exception as exc:
                        invalid_windows += 1
                        if len(packet_samples) < 20:
                            packet_samples.append({"error": str(exc), "raw": _hex(frame)})
                    finally:
                        buffer.remove(utils.PACKET_SIZE)
                elif buffer.count == buffer.size:
                    invalid_windows += 1
                    buffer.remove(1)

    elapsed = max(0.001, time.monotonic() - started)
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "port": port,
        "baudrate": baudrate,
        "duration_sec": round(elapsed, 3),
        "bytes_received": byte_count,
        "byte_rate_Bps": round(byte_count / elapsed, 1),
        "valid_packet_count": int(sum(can_counts.values())),
        "valid_packet_rate_Hz": round(sum(can_counts.values()) / elapsed, 2),
        "mapped_packet_count": int(sum(mapped_counts.values())),
        "mapped_packet_rate_Hz": round(sum(mapped_counts.values()) / elapsed, 2),
        "invalid_windows": invalid_windows,
        "can_ids": [
            {"can_id": f"0x{can_id:04X}", "count": int(count)}
            for can_id, count in can_counts.most_common()
        ],
        "mapped_can_ids": [
            {"can_id": f"0x{can_id:04X}", "count": int(count)}
            for can_id, count in mapped_counts.most_common()
        ],
        "packet_samples": packet_samples,
        "raw_samples": raw_samples,
        "telemetry": {
            "heartbeat_ok": bool(sum(can_counts.values()) > 0),
            "speed_kmh": getattr(telemetry, "speed_kmh", None),
            "angle_deg": getattr(telemetry, "angle_deg", None),
            "gear": str(getattr(telemetry, "gear", "")),
            "accel_pct": getattr(telemetry, "accel_pct", None),
            "brake_pct": getattr(telemetry, "brake_pct", None),
            "pose_valid": getattr(telemetry, "pose_valid", None),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"can_sniff_{port}_{stamp}.json"
    txt_path = output_dir / f"can_sniff_{port}_{stamp}.txt"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path.write_text(format_report(report), encoding="utf-8")
    report["json_path"] = str(json_path)
    report["txt_path"] = str(txt_path)
    return report


def format_report(report: dict) -> str:
    lines = [
        "CAN/MCU passive serial sniff report",
        f"Port: {report['port']} @ {report['baudrate']}",
        f"Duration: {report['duration_sec']} s",
        f"Bytes: {report['bytes_received']} ({report['byte_rate_Bps']} B/s)",
        f"Valid packets: {report['valid_packet_count']} ({report['valid_packet_rate_Hz']} Hz)",
        f"Mapped packets: {report['mapped_packet_count']} ({report['mapped_packet_rate_Hz']} Hz)",
        f"Invalid parser windows: {report['invalid_windows']}",
        "",
        "CAN IDs:",
    ]
    for item in report["can_ids"][:30]:
        lines.append(f"  {item['can_id']}: {item['count']}")
    if not report["can_ids"]:
        lines.append("  none")
    lines.append("")
    lines.append("Mapped CAN IDs:")
    for item in report["mapped_can_ids"][:30]:
        lines.append(f"  {item['can_id']}: {item['count']}")
    if not report["mapped_can_ids"]:
        lines.append("  none")
    lines.append("")
    lines.append("Packet samples:")
    for item in report["packet_samples"][:10]:
        if "can_id" in item:
            lines.append(f"  {item['can_id']}  {item['data']}")
        else:
            lines.append(f"  ERROR {item}")
    if not report["packet_samples"]:
        lines.append("  none")
    if report["raw_samples"] and not report["packet_samples"]:
        lines.append("")
        lines.append("Raw byte samples:")
        for sample in report["raw_samples"][:4]:
            lines.append(f"  {sample}")
    return "\n".join(lines) + "\n"


def main() -> int:
    argp = argparse.ArgumentParser(description="Passive read-only CAN/MCU serial sniffer")
    argp.add_argument("--port", default="auto", help="COM port, for example COM6; default: first USB serial port")
    argp.add_argument("--baudrate", type=int, default=1000000)
    argp.add_argument("--seconds", type=float, default=15.0)
    argp.add_argument("--output-dir", default=str(Path("outputs") / "can_diagnostics"))
    args = argp.parse_args()

    port = _auto_port() if str(args.port).lower() == "auto" else str(args.port)
    print(f"READ-ONLY sniff: opening {port} @ {args.baudrate} for {args.seconds}s")
    report = sniff(port, args.baudrate, args.seconds, Path(args.output_dir))
    print(format_report(report))
    print(f"Saved: {report['txt_path']}")
    print(f"Saved: {report['json_path']}")
    return 0 if report["valid_packet_count"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
