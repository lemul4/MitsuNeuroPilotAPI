# RoadOption bridge for real-mode navigator

This hotfix adds a CARLA/Leaderboard-compatible high-level command bridge for the real/mock navigator.

## Why

The fallback `WaypointPIDController` can work with string maneuvers such as `turn_left` and `lane_follow`, but the TFv6 / ClosedLoopInference model path expects the same semantic command format used by CARLA Leaderboard: `RoadOption.LEFT`, `RIGHT`, `STRAIGHT`, `LANEFOLLOW`, `CHANGELANELEFT`, `CHANGELANERIGHT`, usually converted to a 6-class one-hot tensor.

## Added

- `vehicle_control/road_option.py`
- `LocalNavigationGoal.road_option`
- `LocalNavigationGoal.next_road_option`
- `Waypoint.road_option`
- `RealAgentBridge.build_model_command_payload(goal)`
- tests in `tests/test_road_option_bridge.py`

## Real model adapter usage

```python
from vehicle_control.ai_bridge import RealAgentBridge

payload = RealAgentBridge.build_model_command_payload(goal)
command = torch.tensor(payload["command_one_hot"], dtype=torch.float32, device=device).view(1, 6)
next_command = torch.tensor(payload["next_command_one_hot"], dtype=torch.float32, device=device).view(1, 6)
```

These tensors should be passed to the real dual-camera adapter together with RGB, speed, target points and any other checkpoint-required inputs.

## Important

`CHANGELANELEFT` and `CHANGELANERIGHT` are supported by the bridge, but the current OSM/OSRM navigator should not generate them automatically without lane-level data. They are preserved for future HD-map or manual lane-change commands.
