# Real Vehicle Mode: Cameras, Agent, Gear State Machine

This patch keeps the existing CARLA / VIRTUAL_DEMO_MODE path isolated. CARLA route queues, LeadAgentThread, CARLA watchdog, trace_log.jsonl polling and legacy ZMQ video on port 5555 are used only when VIRTUAL_DEMO_MODE is selected.

For TEST_MOCK_VEHICLE, TEST_REPLAY_LOG, TEST_SERIAL_LOOPBACK or a real serial vehicle descriptor, the GUI uses the real vehicle path:

1. Connect selected vehicle adapter.
2. Start the two-camera service and two preview receivers.
3. Display the 2.8 mm / 90 degree camera and the 6 mm / 50 degree camera in separate UI cells.
4. AI Preview enables the real agent analyzer.
5. Activate Control performs brake hold, requests Drive, waits for gear feedback and starts the autonomy loop.
6. Autonomy loop uses the latest external camera-model intent if fresh; otherwise it falls back to the waypoint PID follower.
7. Deactivate Control sends safe-stop commands, waits until speed is near zero, then requests Park.

## Camera streams

The real two-camera service publishes:

- Mosaic preview on tcp://127.0.0.1:5555 for legacy compatibility.
- Wide 2.8 mm camera cell on tcp://127.0.0.1:5556.
- Narrow 6 mm camera cell on tcp://127.0.0.1:5557.

Configure `config/real_cameras.json` from `config/real_cameras.example.json`.

## Real neural model hook

Set:

```powershell
$env:MITSU_REAL_AGENT_FACTORY = "your_module:create_agent"
```

The factory must return an object with:

```python
predict(frames: dict) -> dict | None
```

`frames` contains BGR OpenCV frames:

```python
{
    "wide_90": np.ndarray,
    "narrow_50": np.ndarray,
}
```

The prediction dict may contain:

```python
{
    "steer": -1.0..1.0,
    "throttle": 0.0..1.0,
    "brake": 0.0..1.0,
    "target_angle_deg": float,
    "confidence": 0.0..1.0,
}
```

The GUI does not send this directly to CAN. It is converted into `ControlIntent`, checked by `VehicleControlService`, limited by `ControlArbiter`, encoded by `VehicleGateway`, and then sent through the USB/MCU path.

## Safety notes

- Park is not requested while the car is moving.
- Deactivate Control first sends safe stop/brake, waits for speed <= threshold, then requests Park.
- Activate Control holds brake before requesting Drive.
- Manual gear and manual throttle are accepted only after AI authority is removed.
- Real serial mode must provide real pose, gear, speed and heartbeat feedback. Mock/replay can seed pose from the selected mission.
