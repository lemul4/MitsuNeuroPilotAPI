from __future__ import annotations

from dataclasses import dataclass
from .models import DriveState, ReadinessStatus, ActivationDecision, Gear


@dataclass
class DriveStateMachine:
    state: DriveState = DriveState.DISCONNECTED
    last_fault: str = ""

    def on_connected(self) -> None:
        self.last_fault = ""
        self.state = DriveState.CONNECTED_MANUAL

    def on_disconnected(self) -> None:
        self.state = DriveState.DISCONNECTED
        self.last_fault = ""

    def on_ai_preview(self, enabled: bool, readiness: ReadinessStatus) -> None:
        if self.state == DriveState.DISCONNECTED:
            return
        if enabled:
            self.state = DriveState.READY_TO_ARM if readiness.all_ok else DriveState.AI_PREVIEW
        else:
            if self.state not in (DriveState.ARMING, DriveState.AI_ACTIVE, DriveState.MANUAL_ACTIVE, DriveState.DISENGAGING):
                self.state = DriveState.CONNECTED_MANUAL

    def refresh_ready_state(self, readiness: ReadinessStatus) -> None:
        if self.state in (DriveState.AI_PREVIEW, DriveState.READY_TO_ARM):
            self.state = DriveState.READY_TO_ARM if readiness.all_ok else DriveState.AI_PREVIEW

    def can_activate(self, readiness: ReadinessStatus) -> ActivationDecision:
        if self.state == DriveState.DISCONNECTED:
            return ActivationDecision.block("Vehicle is not connected")
        if self.state in (DriveState.ARMING, DriveState.AI_ACTIVE, DriveState.MANUAL_ACTIVE, DriveState.DISENGAGING):
            return ActivationDecision.block(f"Invalid state for activation: {self.state.value}")
        reasons = readiness.blocked_reasons()
        if reasons:
            return ActivationDecision.block("; ".join(reasons))
        return ActivationDecision.allow()

    def on_activate_requested(self) -> None:
        self.state = DriveState.ARMING

    def on_gear_drive_confirmed(self) -> None:
        if self.state == DriveState.ARMING:
            self.state = DriveState.AI_ACTIVE

    def on_deactivate_requested(self) -> None:
        if self.state in (DriveState.AI_ACTIVE, DriveState.MANUAL_ACTIVE, DriveState.ARMING, DriveState.READY_TO_ARM, DriveState.AI_PREVIEW):
            self.state = DriveState.DISENGAGING

    def on_manual_ready(self) -> None:
        if self.state != DriveState.DISCONNECTED:
            self.state = DriveState.CONNECTED_MANUAL

    def on_manual_active(self) -> None:
        if self.state != DriveState.DISCONNECTED:
            self.state = DriveState.MANUAL_ACTIVE

    def on_fault(self, reason: str) -> None:
        self.last_fault = str(reason or "fault")
        self.state = DriveState.FAULT

    @property
    def ai_has_authority(self) -> bool:
        return self.state == DriveState.AI_ACTIVE
