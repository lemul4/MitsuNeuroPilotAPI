import contextlib
import os
import sys
import types
import unittest
from pathlib import Path


GUI_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = GUI_ROOT.parent
if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))

import main as gui_main  # noqa: E402


ROUTE_1 = PROJECT_ROOT / "data" / "data_routes" / "lead" / "Accident" / "route_001816.xml"
ROUTE_2 = PROJECT_ROOT / "data" / "data_routes" / "lead" / "Accident" / "route_001817.xml"


class _Signal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def emit(self, *args):
        for slot in list(self.slots):
            slot(*args)


class _FakeLeadAgentThread:
    instances = []

    def __init__(self, config):
        self.config = dict(config)
        self.log_received = _Signal()
        self.status_changed = _Signal()
        self.error_occurred = _Signal()
        self.finished = _Signal()
        self.started = False
        self.stopped = False
        _FakeLeadAgentThread.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def isRunning(self):
        return self.started and not self.stopped

    def wait(self, _timeout=None):
        return True

    def terminate(self):
        self.stopped = True


class _FakeRawTelemetryReader:
    def __init__(self, path):
        self.path = path

    def get_latest_data(self):
        return None


class _ImmediateTimer:
    @staticmethod
    def singleShot(_ms, callback):
        callback()


class _Timer:
    def __init__(self):
        self.started = False

    def start(self, *_args):
        self.started = True

    def stop(self):
        self.started = False


class _StatusBar:
    def __init__(self):
        self.messages = []

    def showMessage(self, message, *_args):
        self.messages.append(str(message))


class _View:
    def __init__(self):
        self._status = _StatusBar()
        self.active_routes = []
        self.runtime_states = []
        self.queue_positions = []
        self.finished_messages = []
        self.ai_checkbox_values = []

    def statusBar(self):
        return self._status

    def set_route_runtime_state(self, state, message=None):
        self.runtime_states.append((state, str(message or "")))

    def set_active_route(self, route, index=0, total=None, state="loading", message=""):
        self.active_routes.append(
            {
                "route": route,
                "index": int(index),
                "total": total,
                "state": state,
                "message": str(message or ""),
            }
        )

    def set_route_queue_position(self, current_index, total=None):
        self.queue_positions.append((int(current_index), total))

    def set_route_queue_finished(self, message=""):
        self.finished_messages.append(str(message or ""))

    def set_ai_checkbox(self, checked):
        self.ai_checkbox_values.append(bool(checked))


class _Vehicle:
    def __init__(self):
        self.target_angle = 0
        self.angle = 0
        self.target_accel = 0
        self.target_brake = 0
        self.accel = 0
        self.brake = 0
        self.force_drive_called = False

    def force_drive_gear(self):
        self.force_drive_called = True

    def reset_motion(self):
        pass


@contextlib.contextmanager
def _patched_gui_runtime():
    original_thread = gui_main.LeadAgentThread
    original_reader = gui_main.RawTelemetryJsonlReader
    original_qtimer = gui_main.QTimer
    _FakeLeadAgentThread.instances = []
    gui_main.LeadAgentThread = _FakeLeadAgentThread
    gui_main.RawTelemetryJsonlReader = _FakeRawTelemetryReader
    gui_main.QTimer = _ImmediateTimer
    try:
        yield
    finally:
        gui_main.LeadAgentThread = original_thread
        gui_main.RawTelemetryJsonlReader = original_reader
        gui_main.QTimer = original_qtimer


def _route_record(path: Path):
    return {
        "id": path.stem,
        "path": str(path),
        "town": "Town10HD",
        "scenario": "Accident",
    }


def _controller():
    controller = gui_main.AppController.__new__(gui_main.AppController)
    controller.view = _View()
    controller.vehicle = _Vehicle()
    controller.runtime_mode = "carla"
    controller.is_virtual = True
    controller.is_connected = True
    controller.control_active = False
    controller.ai_control_requested = True
    controller.ai_agent_loading = False
    controller.ai_agent_running = False
    controller.agent_thread = None
    controller.raw_telemetry_timer = _Timer()
    controller.raw_telemetry_reader = None
    controller.pending_route_queue = []
    controller.current_route_index = -1
    controller.queue_mode = None
    controller.queue_stop_requested = False
    controller.route_plan_prepared = False
    controller.route_transition_delay_ms = 0
    controller._agent_stop_requested = False
    controller._route_skip_after_stop = False
    controller._route_skip_reason = ""
    controller.carla_watchdog_host = "127.0.0.1"
    controller.carla_watchdog_port = 2000
    controller._reset_route_watchdog = types.MethodType(lambda self: None, controller)
    controller._clear_route_watchdog = types.MethodType(lambda self: None, controller)
    controller._send_cruise_state = types.MethodType(lambda self, active: None, controller)
    controller._set_control_button_checked_later = types.MethodType(lambda self, checked: None, controller)
    return controller


class CarlaRouteInterfaceTests(unittest.TestCase):
    def test_single_town10_route_starts_and_finishes_through_appcontroller(self):
        self.assertTrue(ROUTE_1.exists())
        route = _route_record(ROUTE_1)

        with _patched_gui_runtime():
            controller = _controller()
            controller.handle_route_launch_requested([route])

            self.assertTrue(controller.route_plan_prepared)
            self.assertEqual(controller.queue_mode, "single")
            self.assertEqual(controller.pending_route_queue, [route])

            controller.handle_control_toggle(True)

            self.assertTrue(controller.control_active)
            self.assertEqual(controller.current_route_index, 0)
            self.assertEqual(len(_FakeLeadAgentThread.instances), 1)
            first_agent = _FakeLeadAgentThread.instances[0]
            self.assertTrue(first_agent.started)
            self.assertEqual(first_agent.config["routes"], str(ROUTE_1))
            self.assertEqual(first_agent.config["host"], "127.0.0.1")
            self.assertEqual(first_agent.config["port"], 2000)

            controller.handle_agent_finished()

            self.assertFalse(controller.control_active)
            self.assertFalse(controller.route_plan_prepared)
            self.assertEqual(controller.pending_route_queue, [])
            self.assertEqual(controller.current_route_index, -1)
            self.assertTrue(controller.view.finished_messages)

    def test_town10_route_queue_advances_from_first_to_second_route(self):
        self.assertTrue(ROUTE_1.exists())
        self.assertTrue(ROUTE_2.exists())
        route_1 = _route_record(ROUTE_1)
        route_2 = _route_record(ROUTE_2)

        with _patched_gui_runtime():
            controller = _controller()
            controller.handle_route_launch_requested([route_1, route_2])

            self.assertTrue(controller.route_plan_prepared)
            self.assertEqual(controller.queue_mode, "queue")
            self.assertEqual(len(controller.pending_route_queue), 2)

            controller.handle_control_toggle(True)

            self.assertEqual(controller.current_route_index, 0)
            self.assertEqual(len(_FakeLeadAgentThread.instances), 1)
            self.assertEqual(_FakeLeadAgentThread.instances[0].config["routes"], str(ROUTE_1))

            controller.handle_agent_finished()

            self.assertTrue(controller.control_active)
            self.assertTrue(controller.route_plan_prepared)
            self.assertEqual(controller.current_route_index, 1)
            self.assertEqual(len(_FakeLeadAgentThread.instances), 2)
            self.assertEqual(_FakeLeadAgentThread.instances[1].config["routes"], str(ROUTE_2))
            self.assertIn((2, 2), controller.view.queue_positions)

            controller.handle_agent_finished()

            self.assertFalse(controller.control_active)
            self.assertFalse(controller.route_plan_prepared)
            self.assertEqual(controller.pending_route_queue, [])
            self.assertEqual(controller.current_route_index, -1)
            self.assertTrue(controller.view.finished_messages)


if __name__ == "__main__":
    unittest.main()
