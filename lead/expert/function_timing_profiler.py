"""Function-level timing profiler for lead.expert package."""

from __future__ import annotations

import atexit
import os
import pathlib
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class _FunctionStats:
    count: int = 0
    total_s: float = 0.0
    max_s: float = 0.0


class ExpertFunctionTimingProfiler:
    """Collect per-function durations and invocation counters for lead.expert."""

    def __init__(self, log_path: pathlib.Path, min_duration_s: float = 0.0):
        self._log_path = log_path
        self._min_duration_s = max(0.0, float(min_duration_s))

        expert_dir = pathlib.Path(__file__).resolve().parent
        self._target_prefix = f"{expert_dir}{os.sep}"
        self._self_file = os.path.abspath(__file__)

        self._stats: dict[tuple[str, str], _FunctionStats] = {}
        self._stats_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._thread_local = threading.local()
        self._file_filter_cache: dict[str, bool] = {}

        self._running = False
        self._log_handle = None
        self._previous_sys_profile = None
        self._previous_thread_profile = None

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return

        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self._log_path.open("w", encoding="utf-8", buffering=1)

        self._previous_sys_profile = sys.getprofile()
        self._previous_thread_profile = threading.getprofile()
        self._running = True

        sys.setprofile(self._profile_callback)
        threading.setprofile(self._profile_callback)

    def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        sys.setprofile(self._previous_sys_profile)
        threading.setprofile(self._previous_thread_profile)

        self._write_summary()
        if self._log_handle is not None:
            self._log_handle.flush()
            self._log_handle.close()
            self._log_handle = None

    def _profile_callback(self, frame, event, _arg) -> None:
        if not self._running:
            return

        if event == "call":
            filename = frame.f_code.co_filename
            if not filename or not self._should_profile(filename):
                return

            function_name, function_location = self._build_function_identity(frame)
            self._get_start_times()[id(frame)] = (
                time.perf_counter(),
                function_name,
                function_location,
            )
            return

        if event not in ("return", "exception"):
            return

        start_times = getattr(self._thread_local, "start_times", None)
        if not start_times:
            return

        entry = start_times.pop(id(frame), None)
        if entry is None:
            return

        started_at, function_name, function_location = entry
        duration_s = time.perf_counter() - started_at

        stats_key = (function_name, function_location)
        with self._stats_lock:
            stats = self._stats.get(stats_key)
            if stats is None:
                stats = _FunctionStats()
                self._stats[stats_key] = stats

            stats.count += 1
            stats.total_s += duration_s
            if duration_s > stats.max_s:
                stats.max_s = duration_s

    def _write_summary(self) -> None:
        with self._stats_lock:
            sorted_stats = sorted(
                self._stats.items(), key=lambda item: item[1].total_s, reverse=True
            )

        for (function_name, function_location), stats in sorted_stats:
            if stats.max_s < self._min_duration_s:
                continue
            avg_s = stats.total_s / stats.count if stats.count else 0.0
            self._write_line(
                f"function={function_name} | location={function_location}"
                f" | count={stats.count} | total_s={stats.total_s:.6f}"
                f" | avg_s={avg_s:.6f} | max_s={stats.max_s:.6f}"
            )

    def _should_profile(self, filename: str) -> bool:
        cached = self._file_filter_cache.get(filename)
        if cached is not None:
            return cached

        abs_path = os.path.abspath(filename)
        should_profile = (
            abs_path.startswith(self._target_prefix) and abs_path != self._self_file
        )
        self._file_filter_cache[filename] = should_profile
        return should_profile

    def _build_function_identity(self, frame) -> tuple[str, str]:
        module_name = frame.f_globals.get("__name__", "<unknown>")
        qualname = getattr(frame.f_code, "co_qualname", frame.f_code.co_name)
        function_name = f"{module_name}.{qualname}"
        function_location = (
            f"{os.path.abspath(frame.f_code.co_filename)}:{frame.f_code.co_firstlineno}"
        )
        return function_name, function_location

    def _get_start_times(self) -> dict[int, tuple[float, str, str]]:
        start_times = getattr(self._thread_local, "start_times", None)
        if start_times is None:
            start_times = {}
            self._thread_local.start_times = start_times
        return start_times

    def _write_line(self, message: str) -> None:
        if self._log_handle is None:
            return

        with self._write_lock:
            self._log_handle.write(f"{message}\n")


_PROFILER_LOCK = threading.Lock()
_GLOBAL_PROFILER: ExpertFunctionTimingProfiler | None = None


def maybe_start_expert_function_timing(
    config: Any,
) -> ExpertFunctionTimingProfiler | None:
    """Start the profiler if config toggle is enabled."""
    enabled = bool(getattr(config, "enable_function_timing_logging", False))
    if not enabled:
        return None

    configured_path = getattr(
        config,
        "function_timing_log_file",
        "outputs/logs/expert_function_timing.log",
    )
    min_duration_s = getattr(config, "function_timing_log_min_seconds", 0.0)

    log_path = pathlib.Path(str(configured_path)).expanduser()
    if not log_path.is_absolute():
        project_root = getattr(config, "lead_project_root", None)
        base_dir = pathlib.Path(project_root) if project_root else pathlib.Path.cwd()
        log_path = base_dir / log_path

    with _PROFILER_LOCK:
        global _GLOBAL_PROFILER
        if _GLOBAL_PROFILER is not None and _GLOBAL_PROFILER.is_running:
            return _GLOBAL_PROFILER

        profiler = ExpertFunctionTimingProfiler(
            log_path=log_path,
            min_duration_s=min_duration_s,
        )
        profiler.start()
        _GLOBAL_PROFILER = profiler
        return profiler


def stop_expert_function_timing() -> None:
    """Stop and flush the global profiler if it is running."""
    with _PROFILER_LOCK:
        global _GLOBAL_PROFILER
        if _GLOBAL_PROFILER is None:
            return
        _GLOBAL_PROFILER.stop()
        _GLOBAL_PROFILER = None


@atexit.register
def _stop_profiler_on_exit() -> None:
    stop_expert_function_timing()
