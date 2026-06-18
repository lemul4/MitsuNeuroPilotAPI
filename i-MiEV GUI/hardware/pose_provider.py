from __future__ import annotations

import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

try:
    from PySide6.QtCore import QThread, Signal
except Exception:  # pragma: no cover
    QThread = object
    def Signal(*_args, **_kwargs):
        return None


class JsonPoseProviderThread(QThread):
    """Polls a small JSON file and emits real vehicle pose.

    This gives a deterministic integration point for RTK/GNSS+IMU, MCU odometry,
    visual odometry or any external localization process before the MCU protocol
    is finalized. Expected JSON:
      {"x_m": 0.0, "y_m": 0.0, "yaw_deg": 0.0, "valid": true, "source": "rtk"}
    Optional fields: lat, lon, timestamp_ms.
    """

    pose_received = Signal(dict)
    status_changed = Signal(bool, str)

    def __init__(self, path: str | None = None, poll_interval_ms: int = 50, stale_after_ms: int = 750, parent=None):
        super().__init__(parent)
        self.path = Path(path or os.environ.get("MITSU_REAL_POSE_JSON", "config/current_pose.json"))
        self.poll_interval_ms = int(poll_interval_ms)
        self.stale_after_ms = int(stale_after_ms)
        self.running = True
        self._last_payload = None
        self._last_status_at = 0.0

    def run(self):
        while self.running:
            now = time.monotonic()
            try:
                if not self.path.exists():
                    if now - self._last_status_at > 1.0:
                        self._last_status_at = now
                        self.status_changed.emit(False, f"pose file not found: {self.path}")
                    self.msleep(self.poll_interval_ms)
                    continue
                with open(self.path, "r", encoding="utf-8") as file:
                    payload = json.load(file)
                payload = dict(payload or {})
                payload.setdefault("valid", True)
                payload.setdefault("source", "json_pose_provider")
                payload.setdefault("timestamp_monotonic", now)
                self._last_payload = payload
                self.pose_received.emit(payload)
                if now - self._last_status_at > 1.0:
                    self._last_status_at = now
                    self.status_changed.emit(bool(payload.get("valid", True)), f"pose OK: {payload.get('source')}")
            except Exception as exc:
                if now - self._last_status_at > 1.0:
                    self._last_status_at = now
                    self.status_changed.emit(False, f"pose error: {exc}")
            self.msleep(self.poll_interval_ms)

    def stop(self):
        self.running = False
        try:
            self.wait(1200)
        except Exception:
            pass


def _nmea_checksum_ok(sentence: str) -> bool:
    text = str(sentence or "").strip()
    if not text.startswith("$") or "*" not in text:
        return False
    body, expected = text[1:].split("*", 1)
    checksum = 0
    for char in body:
        checksum ^= ord(char)
    try:
        return checksum == int(expected[:2], 16)
    except Exception:
        return False


def _nmea_coordinate(value: str, hemisphere: str) -> float:
    raw = float(value)
    degrees = int(raw // 100)
    minutes = raw - degrees * 100
    coordinate = degrees + minutes / 60.0
    if str(hemisphere or "").upper() in {"S", "W"}:
        coordinate = -coordinate
    return coordinate


@dataclass(frozen=True)
class NmeaFix:
    lat: float
    lon: float
    valid: bool
    fix_quality: int = 0
    satellites: int = 0
    hdop: float = 99.0
    speed_mps: float = 0.0
    course_deg: Optional[float] = None


class Nmea0183Parser:
    def __init__(self):
        self.fix_quality = 0
        self.satellites = 0
        self.hdop = 99.0
        self.speed_mps = 0.0
        self.course_deg = None

    def parse(self, sentence: str) -> Optional[NmeaFix]:
        text = str(sentence or "").strip()
        if not _nmea_checksum_ok(text):
            return None
        fields = text[1:text.index("*")].split(",")
        kind = fields[0][-3:].upper()
        try:
            if kind == "GGA":
                fix_quality = int(fields[6] or 0)
                satellites = int(fields[7] or 0)
                hdop = float(fields[8] or 99.0)
                if fix_quality <= 0 or not fields[2] or not fields[4]:
                    self.fix_quality = fix_quality
                    self.satellites = satellites
                    self.hdop = hdop
                    return NmeaFix(
                        lat=0.0,
                        lon=0.0,
                        valid=False,
                        fix_quality=fix_quality,
                        satellites=satellites,
                        hdop=hdop,
                        speed_mps=self.speed_mps,
                        course_deg=self.course_deg,
                    )
                lat = _nmea_coordinate(fields[2], fields[3])
                lon = _nmea_coordinate(fields[4], fields[5])
                self.fix_quality = fix_quality
                self.satellites = satellites
                self.hdop = hdop
                return NmeaFix(
                    lat=lat,
                    lon=lon,
                    valid=self.fix_quality > 0,
                    fix_quality=self.fix_quality,
                    satellites=self.satellites,
                    hdop=self.hdop,
                    speed_mps=self.speed_mps,
                    course_deg=self.course_deg,
                )
            if kind == "RMC":
                if str(fields[2]).upper() != "A" or not fields[3] or not fields[5]:
                    return NmeaFix(
                        lat=0.0,
                        lon=0.0,
                        valid=False,
                        fix_quality=self.fix_quality,
                        satellites=self.satellites,
                        hdop=self.hdop,
                        speed_mps=self.speed_mps,
                        course_deg=self.course_deg,
                    )
                lat = _nmea_coordinate(fields[3], fields[4])
                lon = _nmea_coordinate(fields[5], fields[6])
                self.speed_mps = float(fields[7] or 0.0) * 0.514444
                self.course_deg = None if not fields[8] else float(fields[8]) % 360.0
                fix_quality = self.fix_quality or 1
                return NmeaFix(
                    lat=lat,
                    lon=lon,
                    valid=str(fields[2]).upper() == "A",
                    fix_quality=fix_quality,
                    satellites=self.satellites,
                    hdop=self.hdop,
                    speed_mps=self.speed_mps,
                    course_deg=self.course_deg,
                )
            if kind == "GLL":
                if str(fields[6]).upper() != "A" or not fields[1] or not fields[3]:
                    return NmeaFix(
                        lat=0.0,
                        lon=0.0,
                        valid=False,
                        fix_quality=self.fix_quality,
                        satellites=self.satellites,
                        hdop=self.hdop,
                        speed_mps=self.speed_mps,
                        course_deg=self.course_deg,
                    )
                lat = _nmea_coordinate(fields[1], fields[2])
                lon = _nmea_coordinate(fields[3], fields[4])
                fix_quality = self.fix_quality or 1
                return NmeaFix(
                    lat=lat,
                    lon=lon,
                    valid=True,
                    fix_quality=fix_quality,
                    satellites=self.satellites,
                    hdop=self.hdop,
                    speed_mps=self.speed_mps,
                    course_deg=self.course_deg,
                )
        except (ValueError, IndexError):
            return None
        return None


def _geo_distance_m(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    radius = 6378137.0
    lat1 = math.radians(float(a_lat))
    lat2 = math.radians(float(b_lat))
    dlat = lat2 - lat1
    dlon = math.radians(float(b_lon) - float(a_lon))
    value = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return radius * 2.0 * math.atan2(math.sqrt(value), math.sqrt(max(0.0, 1.0 - value)))


class NmeaFixFilter:
    def __init__(
        self,
        min_satellites: int = 6,
        max_hdop: float = 3.0,
        lock_samples: int = 3,
        lock_radius_m: float = 20.0,
        max_jump_m: float = 35.0,
        window_size: int = 5,
    ):
        self.min_satellites = int(min_satellites)
        self.max_hdop = float(max_hdop)
        self.lock_samples = max(1, int(lock_samples))
        self.lock_radius_m = float(lock_radius_m)
        self.max_jump_m = float(max_jump_m)
        self.window_size = max(1, int(window_size))
        self._candidates = []
        self._accepted = []
        self._locked = False
        self.allow_quality_unknown = os.environ.get("MITSU_GPS_ALLOW_RMC_GLL_ONLY", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def accept(self, fix: NmeaFix) -> Optional[NmeaFix]:
        if not fix.valid:
            return None
        if not (-90.0 <= fix.lat <= 90.0 and -180.0 <= fix.lon <= 180.0):
            return None
        if fix.fix_quality <= 0:
            return None
        quality_unknown = fix.satellites <= 0 and fix.hdop >= 99.0
        if quality_unknown and not self.allow_quality_unknown:
            return None
        if not quality_unknown and (fix.satellites < self.min_satellites or fix.hdop > self.max_hdop):
            return None

        if not self._locked:
            self._candidates.append(fix)
            self._candidates = self._candidates[-self.lock_samples:]
            if len(self._candidates) < self.lock_samples:
                return None
            center_lat = statistics.median(item.lat for item in self._candidates)
            center_lon = statistics.median(item.lon for item in self._candidates)
            if any(_geo_distance_m(center_lat, center_lon, item.lat, item.lon) > self.lock_radius_m for item in self._candidates):
                self._candidates = [fix]
                return None
            self._locked = True

        if self._accepted:
            previous = self._accepted[-1]
            if _geo_distance_m(previous.lat, previous.lon, fix.lat, fix.lon) > self.max_jump_m:
                return None

        self._accepted.append(fix)
        self._accepted = self._accepted[-self.window_size:]
        return NmeaFix(
            lat=statistics.median(item.lat for item in self._accepted),
            lon=statistics.median(item.lon for item in self._accepted),
            valid=True,
            fix_quality=fix.fix_quality,
            satellites=fix.satellites,
            hdop=fix.hdop,
            speed_mps=fix.speed_mps,
            course_deg=fix.course_deg,
        )


def discover_nmea_serial_port(comports: Optional[Sequence[object]] = None) -> str:
    """Pick the most likely USB GNSS/NMEA serial port.

    Bluetooth COM ports are deliberately avoided because the vehicle control
    port can also be exposed as a serial device and must not be grabbed here.
    """
    if comports is None:
        try:
            import serial.tools.list_ports

            comports = list(serial.tools.list_ports.comports())
        except Exception:
            return ""

    scored = []
    for port in list(comports or []):
        device = str(getattr(port, "device", "") or "")
        description = str(getattr(port, "description", "") or "")
        hwid = str(getattr(port, "hwid", "") or "")
        text = f"{device} {description} {hwid}".lower()
        if not device:
            continue
        if "bluetooth" in text or "bthenum" in text:
            continue
        score = 0
        for token in ("gps", "gnss", "nmea", "ublox", "u-blox", "cp210", "silicon labs", "usb to uart"):
            if token in text:
                score += 10
        if "usb" in text:
            score += 2
        if score > 0:
            scored.append((score, device, description))

    if not scored:
        return ""
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


class NmeaSerialPoseProviderThread(QThread):
    """Read a dedicated GNSS COM port carrying NMEA-0183 sentences."""

    pose_received = Signal(dict)
    status_changed = Signal(bool, str)

    def __init__(self, port: str, baudrate: int = 9600, stale_after_ms: int = 1500, parent=None):
        super().__init__(parent)
        self.port = str(port)
        self.baudrate = int(baudrate)
        self.stale_after_ms = int(stale_after_ms)
        self.running = True
        self._serial = None
        self._last_valid_at = 0.0
        self._stale_emitted = False
        self._last_status_at = 0.0
        self._last_diag_at = 0.0
        self._rx_lines = 0
        self._parsed_fixes = 0
        self._accepted_fixes = 0
        self._bad_lines = 0
        self._filtered_fixes = 0
        self.log_raw = os.environ.get("MITSU_GPS_LOG_RAW", "").strip().lower() in {"1", "true", "yes", "on"}
        self.parser = Nmea0183Parser()
        self.filter = NmeaFixFilter(
            min_satellites=int(os.environ.get("MITSU_GPS_MIN_SATELLITES", "6")),
            max_hdop=float(os.environ.get("MITSU_GPS_MAX_HDOP", "3.0")),
            lock_samples=int(os.environ.get("MITSU_GPS_LOCK_SAMPLES", "3")),
            lock_radius_m=float(os.environ.get("MITSU_GPS_LOCK_RADIUS_M", "20.0")),
            max_jump_m=float(os.environ.get("MITSU_GPS_MAX_JUMP_M", "35.0")),
            window_size=int(os.environ.get("MITSU_GPS_FILTER_WINDOW", "5")),
        )

    def run(self):
        try:
            import serial

            self._serial = serial.Serial(self.port, self.baudrate, timeout=0.2)
            self.status_changed.emit(
                False,
                f"NMEA GPS connected: port={self.port} baud={self.baudrate}; waiting for stable fix",
            )
            while self.running:
                raw = self._serial.readline()
                now = time.monotonic()
                if raw:
                    self._rx_lines += 1
                    sentence = raw.decode("ascii", errors="ignore").strip()
                    if self.log_raw:
                        print(f"NMEA RX {self.port}: {sentence}")
                    fix = self.parser.parse(sentence)
                    if fix is None:
                        self._bad_lines += 1
                    else:
                        self._parsed_fixes += 1
                    accepted = self.filter.accept(fix) if fix is not None else None
                    if accepted is not None:
                        self._accepted_fixes += 1
                        self._last_valid_at = now
                        self._stale_emitted = False
                        self.pose_received.emit({
                            "lat": accepted.lat,
                            "lon": accepted.lon,
                            "yaw_deg": (
                                accepted.course_deg
                                if accepted.speed_mps >= float(os.environ.get("MITSU_GPS_COURSE_MIN_SPEED_MPS", "0.5"))
                                else None
                            ),
                            "speed_mps": accepted.speed_mps,
                            "valid": True,
                            "source": "nmea0183_gnss",
                            "fix_quality": accepted.fix_quality,
                            "satellites": accepted.satellites,
                            "hdop": accepted.hdop,
                            "timestamp_monotonic": now,
                        })
                        if now - self._last_status_at >= 1.0:
                            self._last_status_at = now
                            self.status_changed.emit(
                                True,
                                f"NMEA GPS fix: sats={accepted.satellites} hdop={accepted.hdop:.1f}",
                            )
                    elif fix is not None:
                        self._filtered_fixes += 1
                if now - self._last_diag_at >= 2.0:
                    self._last_diag_at = now
                    self.status_changed.emit(
                        self._last_valid_at > 0.0 and not self._stale_emitted,
                        "NMEA GPS stats: "
                        f"port={self.port} baud={self.baudrate} "
                        f"rx={self._rx_lines} parsed={self._parsed_fixes} "
                        f"accepted={self._accepted_fixes} filtered={self._filtered_fixes} "
                        f"bad={self._bad_lines}"
                    )
                if (
                    self._last_valid_at > 0.0
                    and (now - self._last_valid_at) * 1000.0 > self.stale_after_ms
                    and not self._stale_emitted
                ):
                    self._stale_emitted = True
                    self.pose_received.emit({"valid": False, "source": "nmea0183_stale"})
                    self.status_changed.emit(False, "NMEA GPS pose stale")
        except Exception as exc:
            hint = ""
            if "PermissionError" in repr(exc) or "Access is denied" in str(exc) or "Отказано в доступе" in str(exc):
                hint = "; port is busy or locked by another app"
            self.status_changed.emit(False, f"NMEA GPS error on {self.port}: {exc}{hint}")
            self.pose_received.emit({"valid": False, "source": "nmea0183_error"})
        finally:
            try:
                if self._serial is not None:
                    self._serial.close()
            except Exception:
                pass
            self._serial = None

    def stop(self):
        self.running = False
        try:
            if self._serial is not None:
                self._serial.close()
        except Exception:
            pass
        try:
            self.wait(1500)
        except Exception:
            pass
