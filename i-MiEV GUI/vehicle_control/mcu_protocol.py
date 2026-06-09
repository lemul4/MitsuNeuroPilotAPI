from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .models import Gear, VehicleTelemetry


@dataclass(frozen=True)
class TelemetryFieldSpec:
    can_id: int
    field: str
    offset: int = 0
    length: int = 1
    signed: bool = False
    scale: float = 1.0
    bias: float = 0.0
    enum: Optional[str] = None
    valid_value: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TelemetryFieldSpec":
        d = dict(data or {})
        can_id = d.get("can_id", d.get("id", 0))
        if isinstance(can_id, str):
            can_id = int(can_id, 0)
        return cls(
            can_id=int(can_id),
            field=str(d.get("field", "")),
            offset=int(d.get("offset", 0)),
            length=int(d.get("length", 1)),
            signed=bool(d.get("signed", False)),
            scale=float(d.get("scale", 1.0)),
            bias=float(d.get("bias", 0.0)),
            enum=d.get("enum"),
            valid_value=d.get("valid_value"),
        )


@dataclass
class McuTelemetryParser:
    fields: list[TelemetryFieldSpec] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "McuTelemetryParser":
        candidates = []
        if path:
            candidates.append(Path(path))
        candidates.append(Path("config/mcu_telemetry_map.json"))
        candidates.append(Path("config/mcu_telemetry_map.example.json"))
        for candidate in candidates:
            if candidate.exists():
                with open(candidate, "r", encoding="utf-8") as file:
                    payload = json.load(file)
                specs = [TelemetryFieldSpec.from_dict(item) for item in payload.get("fields", [])]
                return cls(specs)
        return cls.default()

    @classmethod
    def default(cls) -> "McuTelemetryParser":
        return cls([
            TelemetryFieldSpec(0x0001, "angle_deg", 0, 2, True, 630.0 / 0x500),
            TelemetryFieldSpec(0x0003, "speed_kmh", 0, 1, False, 1.0),
            TelemetryFieldSpec(0x0017, "brake_pct", 0, 1, False, 1.0),
            TelemetryFieldSpec(0x0018, "accel_pct", 0, 1, False, 1.0),
            TelemetryFieldSpec(0x0004, "gear", 0, 1, False, 1.0, enum="gear_1_6"),
            TelemetryFieldSpec(0x0100, "x_m", 0, 2, True, 0.01),
            TelemetryFieldSpec(0x0100, "y_m", 2, 2, True, 0.01),
            TelemetryFieldSpec(0x0101, "yaw_deg", 0, 2, True, 0.01),
            TelemetryFieldSpec(0x0102, "pose_valid", 0, 1, False, 1.0, valid_value=1),
        ])

    def apply_packet(self, can_id: int, data: bytes | bytearray | list[int], telemetry: VehicleTelemetry) -> bool:
        updated = False
        raw = bytes(int(x) & 0xFF for x in data)
        for spec in self.fields:
            if int(spec.can_id) != int(can_id):
                continue
            end = spec.offset + spec.length
            if spec.offset < 0 or end > len(raw):
                continue
            value = int.from_bytes(raw[spec.offset:end], "little", signed=spec.signed)
            if spec.enum == "gear_1_6":
                telemetry.gear = Gear.from_value(value, telemetry.gear)
            elif spec.field == "pose_valid":
                telemetry.pose_valid = (value == spec.valid_value) if spec.valid_value is not None else bool(value)
            elif spec.field in {"x_m", "y_m", "yaw_deg"}:
                setattr(telemetry, spec.field, float(value) * spec.scale + spec.bias)
                telemetry.pose_valid = True if spec.field in {"x_m", "y_m", "yaw_deg"} else telemetry.pose_valid
            elif spec.field:
                setattr(telemetry, spec.field, float(value) * spec.scale + spec.bias)
            updated = True
        if updated and any(spec.can_id == can_id and spec.field in {"x_m", "y_m", "yaw_deg", "pose_valid"} for spec in self.fields):
            telemetry.pose_source = telemetry.pose_source or "mcu_telemetry_map"
        return updated
