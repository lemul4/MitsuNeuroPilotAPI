# MitsuNeuroPilot real vehicle control patch

This patch keeps the current CARLA/VIRTUAL_DEMO_MODE flow intact and adds a separate real-vehicle control layer.

## New layers

- `vehicle_control.models` - pure domain models: device, gear, command, telemetry, mission.
- `vehicle_control.state_machine` - drive state machine, independent from Qt and serial.
- `vehicle_control.arbiter` - clamps AI/manual intents into safe `VehicleCommand` objects.
- `vehicle_control.vehicle_gateway` - the only layer that converts `VehicleCommand` to existing CAN packet templates.
- `vehicle_control.adapters` - real serial and mock vehicle adapters behind one interface.
- `vehicle_control.control_service` - application service used by the GUI controller.
- `hardware.command_scheduler` - optional latest-command fixed-rate sender.
- `hardware.real_camera_service` - real camera ZMQ preview service compatible with the existing GUI receiver.
- `ui.real_mission_panel` - compact Navigator/Mission panel for real/mock modes.

## Files to overwrite

- `main.py`
- `ui/main_window.py`
- `hardware/serial_comm.py`

## Files to add

- the whole `vehicle_control/` directory
- `ui/real_mission_panel.py`
- `hardware/command_scheduler.py`
- `hardware/real_camera_service.py`

## Compatibility

`VIRTUAL_DEMO_MODE` still uses the existing CARLA route launcher, queue, LeadAgentThread and watchdog logic. Real/mock modes branch before the CARLA route start path.

## First test

1. Run the GUI.
2. Select `TEST_MOCK_VEHICLE`.
3. Press `Connect`.
4. Press `Validate Mission` in Navigator / Mission.
5. Enable `AI Preview`.
6. Press `Activate Control`.
7. Press the button again to deactivate; the mock should request Park after stop.

## Notes

- The real vehicle path uses `VehicleControlService` and `VehicleAdapter`, not `MainWindow`.
- Existing `SerialManager.send_command()` remains for service commands.
- New control path can use `write_packet_immediate()` / latest packet helpers to avoid FIFO buildup of stale steering/throttle commands.
