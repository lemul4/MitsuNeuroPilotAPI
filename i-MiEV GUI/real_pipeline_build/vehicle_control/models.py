from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Optional, Tuple, Dict, Any, List


class DeviceKind(str, Enum):
    VIRTUAL_DEMO = "virtual_demo"
    REAL_SERIAL = "real_serial"
    MOCK_VEHICLE = "mock_vehicle"
    REPLAY_LOG = "replay_log"
    SERIAL_LOOPBACK = "serial_loopback"
    UNKNOWN_SERIAL = "unknown_serial"


class Gear(Enum):
    P = 1
    R = 2
    N = 3
    D = 4
    E = 5
    B = 6

    @classmethod
    def from_value(cls, value: object, default: "Gear" = None) -> "Gear":
        if default is None:
            default = cls.P
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            name = value.strip().upper()
            if name in cls.__members__:
                return cls[name]
            try:
                return cls(int(name))
            except Exception:
                return default
        try:
            return cls(int(value))
        except Exception:
            return default


class DriveState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTED_MANUAL = "CONNECTED_MANUAL"
    AI_PREVIEW = "AI_PREVIEW"
    READY_TO_ARM = "READY_TO_ARM"
    ARMING = "ARMING"
    AI_ACTIVE = "AI_ACTIVE"
    DISENGAGING = "DISENGAGING"
    FAULT = "FAULT"


class NavCommand(str, Enum):
    START = "start"
    STRAIGHT = "straight"
    LANE_FOLLOW = "lane_follow"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    INTERSECTION = "intersection"
    SLOW = "slow"
    STOP = "stop"
    GOAL = "goal"

    @classmethod
    def normalize(cls, value: object, default: "NavCommand" = None) -> "NavCommand":
        if default is None:
            default = cls.LANE_FOLLOW
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower().replace(" ", "_")
        aliases = {
            "go_straight": cls.STRAIGHT,
            "go": cls.STRAIGHT,
            "forward": cls.STRAIGHT,
            "left": cls.TURN_LEFT,
            "right": cls.TURN_RIGHT,
            "junction": cls.INTERSECTION,
            "crossroad": cls.INTERSECTION,
            "finish": cls.GOAL,
        }
        if text in aliases:
            return aliases[text]
        for item in cls:
            if item.value == text:
                return item
        return default


@dataclass(frozen=True)
class DeviceDescriptor:
    id: str
    label: str
    kind: DeviceKind
    port: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_combo_text(cls, text: str) -> "DeviceDescriptor":
        raw = str(text or "").strip()
        upper = raw.upper()
        if upper == "VIRTUAL_DEMO_MODE":
            return cls(id="virtual-demo", label="VIRTUAL_DEMO_MODE", kind=DeviceKind.VIRTUAL_DEMO)
        if upper == "TEST_MOCK_VEHICLE":
            return cls(id="test-mock-vehicle", label="TEST_MOCK_VEHICLE", kind=DeviceKind.MOCK_VEHICLE)
        if upper == "TEST_REPLAY_LOG":
            return cls(id="test-replay-log", label="TEST_REPLAY_LOG", kind=DeviceKind.REPLAY_LOG)
        if upper == "TEST_SERIAL_LOOPBACK":
            return cls(id="test-serial-loopback", label="TEST_SERIAL_LOOPBACK", kind=DeviceKind.SERIAL_LOOPBACK)

        port = raw
        for sep in (" - ", " -- ", " — ", "—"):
            if sep in raw:
                candidate = raw.split(sep)[-1].strip()
                if candidate:
                    port = candidate
        kind = DeviceKind.REAL_SERIAL if port else DeviceKind.UNKNOWN_SERIAL
        label = raw if raw else "Unknown serial device"
        return cls(id=f"serial:{port}", label=label, kind=kind, port=port)

    @property
    def is_real_control_like(self) -> bool:
        return self.kind in {
            DeviceKind.REAL_SERIAL,
            DeviceKind.MOCK_VEHICLE,
            DeviceKind.REPLAY_LOG,
            DeviceKind.SERIAL_LOOPBACK,
        }


@dataclass(frozen=True)
class Pose2D:
    x_m: float = 0.0
    y_m: float = 0.0
    yaw_deg: float = 0.0
    valid: bool = True
    source: str = "unknown"
    timestamp_monotonic: float = field(default_factory=time.monotonic)

    def age_ms(self) -> float:
        return (time.monotonic() - self.timestamp_monotonic) * 1000.0


@dataclass(frozen=True)
class ControlIntent:
    seq: int = 0
    timestamp_monotonic: float = field(default_factory=time.monotonic)
    frame_id: int = 0
    steer_norm: float = 0.0
    throttle_norm: float = 0.0
    brake_norm: float = 0.0
    target_angle_deg: float = 0.0
    confidence: float = 1.0
    prediction_age_ms: float = 0.0
    desired_speed_kmh: float = 0.0
    speed_cap_kmh: float = 3.0
    nav_maneuver: str = "idle"
    nav_target_distance_m: float = 0.0
    valid_for_ms: int = 100

    def is_expired(self, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        return (now - self.timestamp_monotonic) * 1000.0 > float(self.valid_for_ms)


@dataclass(frozen=True)
class VehicleCommand:
    seq: int = 0
    timestamp_monotonic: float = field(default_factory=time.monotonic)
    active: bool = False
    gear_request: Optional[Gear] = None
    steering_raw: int = 0
    accel_pct: int = 0
    brake_pct: int = 0
    cruise_enabled: bool = False
    valid_for_ms: int = 100
    reason: str = ""

    def is_expired(self, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        return (now - self.timestamp_monotonic) * 1000.0 > float(self.valid_for_ms)

    @classmethod
    def safe_stop(cls, seq: int = 0, brake_pct: int = 35, reason: str = "safe_stop") -> "VehicleCommand":
        return cls(
            seq=seq,
            active=False,
            gear_request=None,
            steering_raw=0,
            accel_pct=0,
            brake_pct=max(0, min(100, int(brake_pct))),
            cruise_enabled=False,
            valid_for_ms=100,
            reason=reason,
        )


@dataclass
class VehicleTelemetry:
    connected: bool = False
    heartbeat_ok: bool = False
    gear: Gear = Gear.P
    requested_gear: Gear = Gear.P
    speed_kmh: float = 0.0
    angle_deg: float = 0.0
    target_angle_deg: float = 0.0
    accel_pct: float = 0.0
    brake_pct: float = 0.0
    x_m: float = 0.0
    y_m: float = 0.0
    yaw_deg: float = 0.0
    pose_valid: bool = False
    pose_source: str = "none"
    last_rx_monotonic: float = field(default_factory=time.monotonic)
    fault: Optional[str] = None

    def age_ms(self) -> float:
        return (time.monotonic() - self.last_rx_monotonic) * 1000.0

    def pose(self) -> Pose2D:
        return Pose2D(
            x_m=float(self.x_m),
            y_m=float(self.y_m),
            yaw_deg=float(self.yaw_deg),
            valid=bool(self.pose_valid),
            source=str(self.pose_source or "telemetry"),
            timestamp_monotonic=self.last_rx_monotonic,
        )


@dataclass(frozen=True)
class ReadinessStatus:
    connected: bool = False
    mission_ok: bool = False
    pose_ok: bool = False
    cameras_ok: bool = False
    ai_preview_ok: bool = False
    vehicle_ok: bool = False
    speed_zero: bool = True
    gear_known: bool = True
    driver_override: bool = False
    fault: Optional[str] = None

    def blocked_reasons(self) -> List[str]:
        reasons = []
        if not self.connected:
            reasons.append("Vehicle is not connected")
        if not self.mission_ok:
            reasons.append("No valid mission")
        if not self.pose_ok:
            reasons.append("Pose is not valid")
        if not self.cameras_ok:
            reasons.append("Camera stream is not ready")
        if not self.ai_preview_ok:
            reasons.append("AI Preview is not enabled")
        if not self.vehicle_ok:
            reasons.append("Vehicle controller is not ready")
        if not self.speed_zero:
            reasons.append("Vehicle speed is not zero")
        if not self.gear_known:
            reasons.append("Gear feedback is unknown")
        if self.driver_override:
            reasons.append("Driver override is active")
        if self.fault:
            reasons.append(str(self.fault))
        return reasons

    @property
    def all_ok(self) -> bool:
        return not self.blocked_reasons()


@dataclass(frozen=True)
class ActivationDecision:
    allowed: bool
    reason: str = ""

    @classmethod
    def allow(cls) -> "ActivationDecision":
        return cls(True, "")

    @classmethod
    def block(cls, reason: str) -> "ActivationDecision":
        return cls(False, reason)


@dataclass(frozen=True)
class Waypoint:
    x_m: float
    y_m: float
    yaw_deg: float = 0.0
    speed_limit_kmh: float = 3.0
    action: str = "lane_follow"
    command: str = "lane_follow"
    hold_sec: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def nav_command(self) -> NavCommand:
        return NavCommand.normalize(self.command or self.action)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], default_speed_kmh: float = 3.0) -> "Waypoint":
        d = dict(data or {})
        command = d.get("command", d.get("action", "lane_follow"))
        return cls(
            x_m=float(d.get("x_m", d.get("x", 0.0))),
            y_m=float(d.get("y_m", d.get("y", 0.0))),
            yaw_deg=float(d.get("yaw_deg", d.get("yaw", d.get("heading_deg", 0.0)))),
            speed_limit_kmh=float(d.get("speed_limit_kmh", d.get("speed_kmh", default_speed_kmh))),
            action=str(d.get("action", command)),
            command=NavCommand.normalize(command).value,
            hold_sec=float(d.get("hold_sec", 0.0)),
            metadata=dict(d.get("metadata") or {}),
        )


@dataclass(frozen=True)
class Mission:
    mission_id: str
    name: str
    goal_label: str = ""
    speed_cap_kmh: float = 3.0
    waypoints: Tuple[Waypoint, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def default_test_mission(cls) -> "Mission":
        return cls(
            mission_id="test_loop_01",
            name="Test Loop 01",
            goal_label="Local coordinate test loop",
            speed_cap_kmh=3.0,
            waypoints=(
                Waypoint(0.0, 0.0, 0.0, 1.0, "start", "start"),
                Waypoint(5.0, 0.0, 0.0, 2.0, "lane_follow", "straight"),
                Waypoint(10.0, 0.0, 0.0, 2.0, "lane_follow", "straight"),
                Waypoint(14.0, 2.0, 26.5, 1.2, "turn_left", "turn_left"),
                Waypoint(16.0, 5.0, 90.0, 1.0, "stop", "stop"),
            ),
            metadata={"coordinate_frame": "local_meters", "generated": False},
        )


@dataclass(frozen=True)
class LocalNavigationGoal:
    mission_id: str = ""
    route_progress: float = 0.0
    target_x_m: float = 0.0
    target_y_m: float = 0.0
    target_heading_deg: float = 0.0
    distance_to_target_m: float = 0.0
    distance_to_goal_m: float = 0.0
    cross_track_error_m: float = 0.0
    heading_error_deg: float = 0.0
    waypoint_index: int = 0
    maneuver: str = "idle"
    desired_speed_kmh: float = 0.0
    speed_cap_kmh: float = 0.0
    stop_required: bool = False
    valid_until_monotonic: float = field(default_factory=lambda: time.monotonic() + 0.25)

    def is_valid(self) -> bool:
        return time.monotonic() <= self.valid_until_monotonic
