# Mitsubishi i-MiEV Real Vehicle Testing Guide

This guide describes the real-car test path for the MitsuNeuroPilot GUI after the vehicle-control modules have been separated from `MainWindow`.

The simulation/CARLA flow is intentionally preserved. `VIRTUAL_DEMO_MODE` still uses the existing route launcher, route queue, CARLA agent thread, CARLA watchdog and ZMQ video receiver. Real vehicle testing uses the new `vehicle_control/` modules, the compact `Navigator / Mission` panel, and mock/HIL/serial adapters.

## 1. Safety boundary

Do not start with a real driving test. The required order is:

1. `TEST_MOCK_VEHICLE` in the GUI.
2. `TEST_SERIAL_LOOPBACK` or MCU bench mode with CAN output disabled.
3. Real MCU connected by USB, telemetry only.
4. Real car stationary, brake hold, throttle hard-clamped to zero.
5. P/D/P gear test only.
6. Steering/brake test on stands or with the drive path physically disabled.
7. Closed-area movement at 1-3 km/h.
8. Higher-speed experiments only after logs prove that watchdogs, manual override and physical emergency stop work.

For the first real-car tests set these limits in MCU and software:

- max speed: 1-3 km/h
- max throttle: 5-10%
- brake priority over throttle
- command timeout: 100-150 ms
- camera timeout: 100-200 ms
- Park request only after speed <= 0.25-0.5 km/h
- physical emergency stop available to a second person

## 2. Runtime modes in the port selector

Use the dropdown as follows:

- `VIRTUAL_DEMO_MODE` - existing CARLA/simulation path.
- `TEST_MOCK_VEHICLE` - full real-mode UI and state machine without hardware.
- `TEST_REPLAY_LOG` - reserved for recorded logs/cameras.
- `TEST_SERIAL_LOOPBACK` - serial/CAN protocol testing without the car.
- `COMx` - real serial device. Use only after MCU handshake/dry-run validation.

For normal development choose `TEST_MOCK_VEHICLE`.

## 3. New module layout

```text
vehicle_control/
  models.py              Domain models: Mission, Waypoint, Pose2D, LocalNavigationGoal, VehicleCommand.
  navigation.py          A/B coordinate route planner and local navigator.
  pid.py                 Waypoint PID controller for steering, speed, throttle and brake intent.
  ai_bridge.py           Converts neural predictions + navigator goal into ControlIntent.
  arbiter.py             Safety limiter from ControlIntent to VehicleCommand.
  control_service.py     State machine, arming/disengage, telemetry loop, autonomous loop.
  adapters.py            RealSerialVehicleAdapter, MockVehicleAdapter, loopback adapter.
  vehicle_gateway.py     VehicleCommand -> existing CAN Serial_Data packets.
  mission_store.py       JSON mission load/save, including A/B mission generation.

hardware/
  dual_camera_service.py Two real camera ZMQ preview service.
  real_camera_service.py Earlier generic real camera preview service.
  serial_comm.py         Serial transport. Service commands use FIFO; control commands use latest/immediate path.
  can_commands.py        Existing CAN packet factory and templates.

config/
  real_cameras.example.json

missions/
  ab_route.example.json

scripts/
  generate_ab_mission.py
```

The GUI continues to receive video through the existing ZMQ JPEG protocol on `tcp://127.0.0.1:5555`.

## 4. Coordinate navigation model

The real car is driven by coordinate points, not by CARLA XML routes.

A mission contains:

- point coordinates in local meters: `x_m`, `y_m`, `yaw_deg`
- navigation command: `straight`, `turn_left`, `turn_right`, `intersection`, `slow`, `stop`
- speed limit per waypoint

The usual route source is two points A and B. The planner builds subpoints every `spacing_m` meters and uses optional route hints for turns and intersections.

Example:

```json
{
  "mission_id": "yard_a_to_b_001",
  "name": "Yard A to B 001",
  "speed_cap_kmh": 3.0,
  "spacing_m": 2.0,
  "start": {"x_m": 0.0, "y_m": 0.0, "yaw_deg": 0.0},
  "goal": {"x_m": 22.0, "y_m": 8.0, "yaw_deg": 20.0},
  "hints": [
    {"x_m": 8.0, "y_m": 0.0, "command": "straight", "speed_limit_kmh": 3.0},
    {"x_m": 12.0, "y_m": 2.5, "command": "turn_left", "speed_limit_kmh": 1.2},
    {"x_m": 18.0, "y_m": 6.0, "command": "intersection", "speed_limit_kmh": 1.0}
  ]
}
```

Important limitation: with only A and B and no map, the software cannot infer real intersections. It can densify a line from A to B. Real turns/intersections must come from route hints, a recorded route, or a future map/graph module.

## 5. Navigator to PID to CAN flow

The real-mode control loop is:

```text
Mission / A-B route
  -> CoordinateRoutePlanner
  -> Waypoints with commands
  -> NavigatorService
  -> LocalNavigationGoal
  -> WaypointPIDController or neural agent
  -> ControlIntent
  -> ControlArbiter
  -> VehicleCommand
  -> VehicleGateway
  -> CANCommandFactory
  -> SerialManager
  -> USB -> MCU -> CAN -> car
```

`NavigatorService` outputs a local goal:

- current target waypoint index
- target coordinate
- heading error
- cross-track error
- maneuver command
- desired speed
- speed cap
- stop requirement

The deterministic `WaypointPIDController` converts that local goal to a `ControlIntent` for real bench testing. A neural model can later replace or augment this controller by sending predictions through `RealAgentBridge`.

## 6. How the agent should use the route

The agent should not receive raw CAN access. It should receive camera frames plus local navigation context:

- front wide camera: 2.8 mm, approximately 90 degree FOV
- front narrow camera: 6 mm, approximately 50 degree FOV
- current speed
- current gear
- lookahead target coordinate
- heading error
- cross-track error
- maneuver command
- desired speed
- speed cap

The agent may predict:

- `steer`
- `throttle`
- `brake`
- `target_angle_deg`
- `confidence`
- optionally a speed target

Those values must go through `RealAgentBridge -> ControlIntent -> ControlArbiter`. Do not send model output directly to serial/CAN.

## 7. Two real camera setup

The real car has two cameras:

- 2.8 mm lens, approximately 90 degree FOV, used as the wide/front context camera.
- 6 mm lens, approximately 50 degree FOV, used as the narrow/longer-range camera.

Connecting cameras to a gigabit switch is not enough by itself. The GUI does not scan the network for cameras. The GUI subscribes to one JPEG stream on `tcp://127.0.0.1:5555`. Start `hardware/dual_camera_service.py` to capture the cameras and publish the same ZMQ JPEG protocol used by the CARLA camera service.

Copy the example config:

```powershell
copy "i-MIEV GUI\config\real_cameras.example.json" "i-MIEV GUI\config\real_cameras.json"
```

Edit the sources:

```json
"front_wide_28mm": {
  "source": "rtsp://192.168.1.101:554/stream1"
},
"front_narrow_6mm": {
  "source": "rtsp://192.168.1.102:554/stream1"
}
```

For USB/OpenCV devices you can use:

```json
"source": 0
```

or:

```json
"source": 1
```

Start the service in a separate terminal:

```powershell
cd "E:\основы программирования\MitsuNeuroPilotAPI\i-MIEV GUI"
python hardware\dual_camera_service.py --config config\real_cameras.json
```

Then start the GUI:

```powershell
python main.py
```

The GUI should display the wide camera as the main image and the 6 mm camera as an inset. If the GUI is black, verify the camera service first. The switch only provides network connectivity; the camera service is what publishes the GUI video stream.

Recommended bandwidth policy:

- AI input: only required resolution/FPS.
- GUI preview: 10-15 FPS is enough.
- Avoid raw full-resolution streams from both cameras over a single 1 Gbit uplink.
- Drop old frames. Never let camera frames queue up for control.

## 8. Mock test procedure

1. Start the GUI.
2. Select `TEST_MOCK_VEHICLE`.
3. Press `Connect`.
4. In `Navigator / Mission`, select `A -> B Coordinate Route` or `Test Loop 01`.
5. Open `Details` and set A/B coordinates if using A/B route.
6. Press `Validate`.
7. Turn on `AI Preview`.
8. Press `Activate Control`.

Expected log sequence:

```text
REAL CONTROL: Vehicle connected: TEST_MOCK_VEHICLE
REAL CONTROL: Mission validated: ...
REAL CONTROL: AI Preview enabled
REAL CONTROL: ARMING: brake hold
REAL CONTROL: ARMING: requesting Drive
REAL CONTROL: AI_ACTIVE
REAL CONTROL: NAV: wp=... maneuver=...
```

Press `Activate Control` again to disengage:

```text
REAL CONTROL: DISENGAGING: user_requested
REAL CONTROL: Gear P confirmed after stop
```

## 9. Serial loopback / MCU bench test

Do this before connecting to the real car.

1. Connect MCU by USB.
2. Keep CAN output disabled on MCU.
3. Select the MCU COM port or `TEST_SERIAL_LOOPBACK` if using mock loopback.
4. Press `Connect`.
5. Verify heartbeat/ACK logs.
6. Validate a mission.
7. Enable AI Preview.
8. Press Activate only if MCU is in dry-run.

The expected behavior is command packet generation without vehicle movement.

## 10. Real car telemetry-only test

1. Physical emergency stop ready.
2. Car stationary.
3. MCU connected to USB.
4. CAN output disabled or control authority disabled.
5. Select the real COM port.
6. Press `Connect` only.

Verify:

- heartbeat
- speed
- gear feedback
- brake feedback
- steering feedback
- pose/localization feedback if available

Do not press Activate Control if pose is not valid. In the current implementation real activation requires a valid pose.

## 11. Real stationary gear test

Only after telemetry-only works.

1. Brake hold active.
2. Throttle hard-clamped to zero in MCU.
3. Validate a short A/B mission.
4. AI Preview on.
5. Activate Control.
6. Confirm `D` feedback.
7. Deactivate.
8. Confirm stop then `P` feedback.

If `P` is requested while speed is nonzero, treat it as a failure. The software intentionally waits for near-zero speed before Park.

## 12. First low-speed movement

Use a straight A/B route:

- A: `0,0,0`
- B: `5,0,0`
- spacing: `1.0-2.0 m`
- speed cap: `1 km/h`
- no hints

Run only on a closed area. The deterministic waypoint PID will generate steering/throttle/brake via the same `ControlIntent -> Arbiter -> VehicleCommand` path that the future model will use.

## 13. Unit tests

From the GUI root:

```powershell
python -m unittest discover -s tests -v
```

Expected:

```text
OK
```

## 14. Known integration points still requiring vehicle-specific data

Before a real driving test, confirm these MCU/CAN details:

- handshake response format
- real gear feedback CAN ID
- real speed CAN ID and scale
- steering feedback CAN ID and scale
- pose/localization source and scale
- watchdog timeout behavior on MCU
- throttle/brake/steering clamp limits on MCU

The adapter contains placeholder optional localization IDs `0x0100` and `0x0101`. Change only `RealSerialVehicleAdapter.handle_can_packet()` when the actual MCU telemetry IDs are finalized.

---

## Hotfix: Pose validity in TEST_MOCK_VEHICLE

If `Activate Control` prints:

```text
REAL CONTROL: Activation blocked: Pose is not valid
```

then the navigator has no fresh localization. In real serial mode this is the correct safety behavior: do not activate autonomous control without pose feedback from MCU/GNSS/RTK/odometry.

For mock/replay/loopback tests, this hotfix seeds the pose from the first mission waypoint and refreshes the mock heartbeat timestamp. This allows the complete A→B test to run without external localization:

```text
TEST_MOCK_VEHICLE → Connect → Validate → AI Preview → Activate Control
```

Real serial mode still does **not** fake pose. For a real car, `Pose OK` must come from the real localization source.

## Interactive real map A/B picker

The `Navigator / Mission → Details` menu now has:

```text
Open real map
```

It opens an embedded map where:

```text
first click  = point A
second click = point B
next clicks  = update point B
markers can be dragged
```

The map returns WGS84 coordinates:

```text
lat, lon
```

The mission panel converts them into the local-meter frame used by the navigation and PID layers:

```text
A lat/lon, B lat/lon
  ↓
local x/y meters, origin = A
  ↓
CoordinateRoutePlanner
  ↓
waypoints + navigator commands
```

### Dependency

The embedded map requires Qt WebEngine. For PySide6 this is provided by the `PySide6-Addons` wheel, not by a package named `PySide6-WebEngine`:

```powershell
python -m pip install --upgrade PySide6 PySide6-Addons PySide6-Essentials
```

Quick check:

```powershell
python -c "from PySide6.QtWebEngineWidgets import QWebEngineView; print('QtWebEngine OK')"
```

If QtWebEngine is unavailable, the dialog falls back to manual lat/lon fields and a button that opens the same OpenStreetMap picker in the external browser. In external-browser mode, copy A/B coordinates from the browser text box back into the GUI fields and press `Apply A/B`.

### Initial map center / current location

The map no longer hardcodes Moscow as the default. On first opening it tries browser geolocation through the `Locate me` button and also calls it automatically when no A/B points are set. If geolocation is denied or unavailable, configure your test-site fallback center:

```powershell
copy config\map_settings.example.json config\map_settings.json
notepad config\map_settings.json
```

Example — replace these values with your own test-site coordinates:

```json
{
  "default_center": {
    "lat": 0.0,
    "lon": 0.0,
    "zoom": 2
  }
}
```

Alternatively set environment variables before launch:

```powershell
$env:MITSU_MAP_CENTER_LAT = "<your_lat>"
$env:MITSU_MAP_CENTER_LON = "<your_lon>"
$env:MITSU_MAP_CENTER_ZOOM = "17"
python main.py
```

### Internet requirement

The default map uses online Leaflet/OpenStreetMap tiles. The GUI or the external browser needs internet access for map tiles. If the machine is offline, the map may open but the background tiles will not load.

### 2GIS note

The current implementation uses Leaflet/OpenStreetMap by default because it can work without a private map key. 2GIS-style integration should be added as a provider layer after an API key and usage limits are configured. The vehicle navigation layer is already provider-independent because it only receives `lat/lon` A/B points.

## Real map safety note

A real map only chooses the mission geometry. It is **not** a safety oracle. The car still needs:

```text
real pose feedback
fresh camera frames
mission validation
speed cap
MCU heartbeat
manual override
physical emergency stop
```

Do not activate a real car merely because A/B are selected on the map.

## Дополнение: камеры, ручной режим и маршруты по дорогам

### Камеры через гигабитный свитч

Картинка не появляется только от факта подключения камер к свитчу. GUI слушает один JPEG-поток по ZMQ на `tcp://127.0.0.1:5555`. Для реальных камер нужно отдельно запустить сервис:

```powershell
cd "E:\основы программирования\MitsuNeuroPilotAPI\i-MIEV GUI"
copy config\real_cameras.example.json config\real_cameras.json
notepad config\real_cameras.json
python hardware\dual_camera_service.py --config config\real_cameras.json
```

Во втором терминале запускается GUI:

```powershell
python main.py
```

В `config\real_cameras.json` указываются реальные RTSP/HTTP/OpenCV sources двух камер:

- `front_wide_28mm` — объектив 2.8 мм, FOV около 90 градусов;
- `front_narrow_6mm` — объектив 6 мм, FOV около 50 градусов.

Сервис собирает широкую камеру как основное изображение и узкую камеру как inset.

### Ручное управление после отключения Activate Control

После `Deactivate Control` состояние должно вернуться в `CONNECTED_MANUAL`. В этом состоянии доступны:

- `1..6` — ручной запрос передачи P/R/N/D/E/B с safety interlocks;
- стрелки `←/→` — руль;
- стрелка `↑` — газ;
- стрелка `↓` или `S` — тормоз.

Во время `AI_ACTIVE` ручные gear-команды блокируются. Сначала отключается `Activate Control`, затем можно вручную запросить передачу и управлять.

### Маршрут по дорогам вместо прямой A→B

В `Navigator / Mission → Детали` добавлен режим:

- `Прямая A→B` — старый локальный прямой маршрут;
- `По дорогам OpenStreetMap/OSRM` — построение маршрута по дорожному графу.

Для road-route нужны A/B точки с карты, потому что маршрутизатор работает с WGS84 `lat/lon`. Если OSRM недоступен или вернул ошибку, GUI откатится к прямому A→B маршруту и напечатает предупреждение в лог.

Публичный OSRM demo server подходит только для разработки и проверки. Для реального автономного теста нужно локально поднять OSRM/Valhalla/GraphHopper на нужный регион OSM или использовать 2GIS Directions API с ключом.

---

## Обновление: дорожный маршрут по нужной стороне дороги

Стандартный OSRM/OpenStreetMap маршрут строится по дорожной геометрии OSM. Для большинства дорог это **центральная линия дороги**, а не точная геометрия полосы. Поэтому в real/mock режиме добавлен слой контрольной траектории:

```text
синяя линия на карте  = дорожный центр OSM/OSRM
зеленая линия на карте = контрольная траектория со смещением вправо по ходу движения
```

По умолчанию используется правостороннее движение:

```json
"route_planning": {
  "traffic_side": "right",
  "lane_offset_m": 1.7
}
```

`lane_offset_m` — приближенное смещение от дорожного центра. Для низкоскоростного теста на закрытой площадке обычно достаточно 1.5–2.0 м. Для реальной эксплуатации это не заменяет HD-карту и lane-level localization: если нужна точная полоса, потребуется Lanelet2/HD-map, разметка полос, RTK/IMU и локализация относительно карты.

### Как проверить

1. `TEST_MOCK_VEHICLE` → `Connect`.
2. Нажать `Карта и маршрут`.
3. Убедиться, что A стоит на текущей позиции. Если нужно, перетащить A вручную.
4. Кликнуть B.
5. Нажать `Построить по дорогам`.
6. Проверить линии:
   - желтая пунктирная — прямая A→B;
   - синяя — OSRM/OSM центр дороги;
   - зеленая — контрольная траектория для движения по нужной стороне дороги.
7. Нажать `Применить маршрут`.
8. В `Настройки` убедиться, что `Построение = По дорогам OpenStreetMap/OSRM`, `Траектория = По нужной стороне дороги`.
9. Нажать `Проверить маршрут`.

В консоли должно быть:

```text
REAL CONTROL: road routing requested: provider=OSRM
REAL CONTROL: road routing OK: raw_points=..., waypoints=..., trajectory=right_side_offset_approximation, lane_policy=right_side, offset=1.7m
```

Если вместо этого видишь fallback на direct A→B, значит OSRM не построил маршрут или точки не попали на routable-дороги.

## Более чистый real-mode UX

В real/mock режиме основной экран теперь оставляет только операционные действия:

```text
Маршрут | Карта и маршрут | Скорость | Проверить маршрут | готовность | Настройки
```

Технические поля A/B, шаг waypoint, выбор дорожного провайдера и смещение траектории спрятаны в `Настройки`. Длинные статусы и названия маршрутов прокручиваются внутри своих полей, не растягивая layout.
