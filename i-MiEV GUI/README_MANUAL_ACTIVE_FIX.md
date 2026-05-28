# MANUAL_ACTIVE DriveState hotfix

Fixes failing tests after RoadOption bridge: `DriveState.MANUAL_ACTIVE` was referenced by `state_machine.py` and control-service tests, but the enum value was missing from `vehicle_control/models.py`.

Apply from project root:

```powershell
Expand-Archive -Force "mitsu_drive_state_manual_active_fix.zip" "i-MIEV GUI"
cd "E:\основы программирования\MitsuNeuroPilotAPI\i-MIEV GUI"
python -m unittest discover -s tests -v
```

Expected: all tests pass, including manual takeover tests.
