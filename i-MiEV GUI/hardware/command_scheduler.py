from __future__ import annotations

import asyncio
import concurrent.futures
import time
from typing import Optional

from vehicle_control.models import VehicleCommand


class ControlCommandScheduler:
    """Fixed-rate latest-command sender.

    The scheduler is intentionally separate from SerialManager. SerialManager is
    still a byte transport; this class owns the control-loop semantics: latest
    command wins, stale commands become safe-stop commands.
    """

    def __init__(self, gateway, tick_hz: float = 50.0, stale_stop_brake_pct: int = 35):
        self.gateway = gateway
        self.tick_hz = float(tick_hz)
        self.stale_stop_brake_pct = int(stale_stop_brake_pct)
        self._latest_command: Optional[VehicleCommand] = None
        self._task = None
        self._running = False
        self._seq = 0

    def _schedule_coro_threadsafe(self, coro, label="control scheduler"):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                try:
                    coro.close()
                except Exception:
                    pass
                return None
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception:
            try:
                coro.close()
            except Exception:
                pass
            return None

        def _done(done_future):
            try:
                done_future.result()
            except (asyncio.CancelledError, concurrent.futures.CancelledError):
                pass
            except Exception as exc:
                print(f"ControlCommandScheduler error in {label}: {exc}")
        future.add_done_callback(_done)
        return future


    def update_latest(self, command: VehicleCommand) -> None:
        self._latest_command = command

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._running = True
            self._task = self._schedule_coro_threadsafe(self._run(), "control_scheduler_run")

    async def stop(self) -> None:
        self._running = False
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            if isinstance(task, asyncio.Task):
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            else:
                await asyncio.sleep(0)

    async def _run(self) -> None:
        period = 1.0 / max(1.0, self.tick_hz)
        while self._running:
            command = self._latest_command
            if command is None or command.is_expired():
                self._seq = (self._seq + 1) & 0x7FFFFFFF
                command = VehicleCommand.safe_stop(self._seq, self.stale_stop_brake_pct, reason="scheduler_stale")
            self.gateway.write_vehicle_command_now(command)
            await asyncio.sleep(period)
