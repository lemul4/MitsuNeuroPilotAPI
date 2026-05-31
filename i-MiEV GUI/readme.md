# MitsuNeuroPilot GUI: режимы запуска, камеры, навигация, агент и реальное авто

Документ описывает текущую рабочую логику `i-MIEV GUI` после разделения системы на два независимых контура:

1. `VIRTUAL_DEMO_MODE` — существующая логика CARLA / Leaderboard / `LeadAgentThread`. Этот режим должен работать как раньше и не должен зависеть от real-vehicle модулей.
2. `TEST_MOCK_VEHICLE`, `TEST_SERIAL_LOOPBACK`, реальный COM-порт — контур реального автомобиля: две камеры, навигатор A → B, агентная модель или PID-fallback, `VehicleControlService`, `ControlArbiter`, `VehicleGateway`, USB / MCU / CAN.

Главное архитектурное правило: **CARLA-код не запускается в real/mock режимах, а real-vehicle-код не меняет CARLA flow**.

---

## 0. Текущий статус проекта

На текущем этапе подготовлена программная база для стендовой проверки real-mode контура.

Готово:

- сохранен отдельный `VIRTUAL_DEMO_MODE` для CARLA;
- добавлен `TEST_MOCK_VEHICLE` для проверки real-mode без автомобиля;
- добавлен `TEST_SERIAL_LOOPBACK` для проверки транспортного слоя;
- реализован `VehicleControlService` со state machine;
- добавлены `AI_PREVIEW`, `AI_ACTIVE`, `MANUAL_ACTIVE`, `DISENGAGING`;
- реализован ручной перехват из `AI_ACTIVE` без мгновенного Park;
- реализован Park-after-stop: сначала остановка и brake hold, затем запрос `P`;
- добавлен навигатор A → B, waypoint-ы, road-routing и PID-fallback;
- добавлен `RoadOption`-bridge для CARLA/Leaderboard-compatible команд;
- заложены две ячейки камер: 2.8 мм / 90° и 6 мм / 50°;
- подготовлен real-agent bridge для подключения модели;
- добавлены safety config, dry-run guard, MCU telemetry map и preflight-проверка;
- unit-тесты real/mock логики проходят.

Не готово без железа:

- фактический MCU handshake и feedback;
- реальные CAN/packet-поля для `speed`, `gear`, `steering`, `brake`, `pose`;
- подтверждение, что `GEAR/ACCEL/BRAKE/ANGLE` корректно исполняются на i-MiEV;
- реальный pose-provider;
- подтвержденные RTSP/HTTP/GigE потоки двух камер;
- checkpoint-specific `RealDualCameraAgentAdapter` под конкретную модель;
- валидация поведения модели на реальных кадрах.

Итоговая формулировка статуса: **система готова к mock, loopback, bench и telemetry-only этапам; к автономному движению реального автомобиля переходить можно только после аппаратной валидации.**

---

## 1. Быстрый запуск GUI

`main.py` находится внутри папки `i-MIEV GUI`, поэтому из корня проекта команда `python main.py` не сработает.

Запуск из папки GUI:

```powershell
cd "E:\основы программирования\MitsuNeuroPilotAPI\i-MIEV GUI"
python main.py
```

Запуск из корня репозитория:

```powershell
python "i-MIEV GUI/main.py"
```

---

## 2. Проверка кода и тестов

### 2.1. Компиляция Python-файлов на Windows

В PowerShell нельзя использовать Linux-style wildcard вида `vehicle_control/*.py` напрямую с `python -m py_compile`. Используйте PowerShell-вариант:

```powershell
cd "E:\основы программирования\MitsuNeuroPilotAPI\i-MIEV GUI"

Get-ChildItem vehicle_control,hardware,real_agent_adapters -Filter *.py | ForEach-Object {
    python -m py_compile $_.FullName
}

python -m py_compile scripts\real_vehicle_preflight.py main.py
```

### 2.2. Unit-тесты

Тесты нужно запускать только из папки `i-MIEV GUI`, потому что папка `tests` находится внутри нее:

```powershell
cd "E:\основы программирования\MitsuNeuroPilotAPI\i-MIEV GUI"
python -m unittest discover -s tests -v
```

Если запустить из корня проекта, возможна ошибка:

```text
ImportError: Start directory is not importable: 'tests'
```

Это не ошибка кода, а запуск из неверной директории.

---

## 3. Режимы в выпадающем списке портов

### 3.1. `VIRTUAL_DEMO_MODE`

Старый режим симуляции. Здесь должны работать:

- CARLA;
- route picker;
- подготовка маршрута или очереди;
- `LeadAgentThread`;
- чтение `trace_log.jsonl`;
- CARLA watchdog;
- CARLA camera service / ZMQ preview;
- старый route/scenario UI.

Типовой сценарий:

```text
1. Выбрать VIRTUAL_DEMO_MODE.
2. Нажать «Подключить».
3. Выбрать CARLA route.
4. Нажать «Подготовить маршрут» или «Подготовить очередь».
5. Включить «Предпросмотр ИИ» / AI Control.
6. Нажать «Активировать управление».
7. CARLA-сценарий запускается через LeadAgentThread.
```

В этом режиме **не должен использоваться** новый `VehicleControlService` реального автомобиля.

---

### 3.2. `TEST_MOCK_VEHICLE`

Главный режим для проверки real-vehicle логики без железа.

Имитирует:

- подключение автомобиля;
- gear `P/D`;
- задержку переключения передачи;
- скорость;
- движение по waypoint-ам;
- safe stop;
- ручной takeover;
- возврат в `P` после остановки.

Типовой тест:

```text
1. Выбрать TEST_MOCK_VEHICLE.
2. Нажать «Подключить».
3. В блоке «Навигатор / миссия» выбрать маршрут A → B.
4. Нажать «Проверить маршрут».
5. Включить «Предпросмотр ИИ».
6. Нажать «Активировать управление».
7. Дождаться AI_ACTIVE.
8. Повторно нажать «Активировать управление».
9. Система должна перейти в MANUAL_ACTIVE / ручной takeover.
10. Проверить ручные команды: газ, тормоз, руль.
11. Завершить управление и убедиться, что Park выполняется после остановки.
```

Ожидаемые логи:

```text
REAL CONTROL: Vehicle connected: TEST_MOCK_VEHICLE
REAL CONTROL: Mission validated: ...
REAL CONTROL: AI Preview enabled
REAL CONTROL: ARMING: brake hold
REAL CONTROL: ARMING: requesting Drive
REAL CONTROL: AI_ACTIVE
REAL CONTROL: NAV: ...
REAL CONTROL: MANUAL_ACTIVE: user_takeover
REAL CONTROL: Gear P confirmed after stop
```

---

### 3.3. `TEST_SERIAL_LOOPBACK`

Режим для проверки serial / USB / CAN-транспорта без физического исполнения команд машиной.

Использовать для:

- проверки формата пакетов;
- проверки CRC/CNC;
- проверки ACK/timeout;
- проверки command scheduler;
- проверки, что управляющие команды не копятся в FIFO;
- проверки dry-run поведения перед реальным MCU.

---

### 3.4. Реальный COM-порт

Реальный COM-порт нельзя использовать сразу как режим «ехать». Сначала обязательны bench, dry-run и telemetry-only.

Правильная лестница:

```text
1. MCU bench, CAN output disabled.
2. Real COM telemetry-only.
3. Проверка heartbeat / ACK / CRC / CNC.
4. Проверка actual gear / speed / steering feedback / brake feedback.
5. Gear test без газа: P → D → P.
6. Steering/brake test на месте.
7. Закрытая площадка, speed cap 1 км/ч.
8. Только потом AI/manual движение.
```

---

## 4. Логика кнопок и флагов

### 4.1. «Предпросмотр ИИ»

Флаг включает анализ данных агентом, но **не передает authority машине**.

В real-mode:

```text
Предпросмотр ИИ ON:
  - камеры читаются;
  - агент/адаптер может анализировать кадры;
  - ControlIntent может формироваться;
  - VehicleGateway не имеет authority, пока не нажата «Активировать управление».
```

---

### 4.2. «Ручное управление»

Флаг ручного режима нужен для движения по маршруту без автономного агента.

```text
Ручное управление ON:
  - пользователь сам управляет стрелками/клавишами;
  - маршрут и waypoint-ы остаются подсказкой;
  - AI не имеет authority;
  - Activate Control переводит автомобиль в MANUAL_ACTIVE через safe arming sequence.
```

Клавиши:

```text
↑        газ
↓ / S    тормоз
← / →    руль
1        P
2        R
3        N
4        D
5        E
6        B
Space    toggle Activate Control
```

---

### 4.3. «Активировать управление» в AI mode

Если включен «Предпросмотр ИИ», а ручной режим выключен:

```text
1. brake hold;
2. throttle = 0;
3. request Drive;
4. wait actual gear D;
5. stabilization delay;
6. AI_ACTIVE;
7. prediction → ControlIntent → Arbiter → VehicleGateway → USB/MCU/CAN.
```

---

### 4.4. Повторное нажатие «Активировать управление» во время `AI_ACTIVE`

Повторное нажатие **не отправляет Park**.

Правильная логика:

```text
AI_ACTIVE
  ↓ повторное нажатие
AI authority OFF
  ↓
Manual flag ON
  ↓
MANUAL_ACTIVE
```

Машина остается в `D`, пользователь сразу берет управление на себя.

---

### 4.5. Полное отключение управления / Park

Park отправляется только после остановки.

```text
1. throttle = 0;
2. brake hold / safe stop;
3. wait speed <= threshold;
4. brake hold before Park;
5. request P;
6. wait/check actual gear P;
7. CONNECTED_MANUAL.
```

Для реального i-MiEV это критично: нельзя отправлять `P` во время движения.

---

## 5. Safety profile и dry-run guard

Перед реальным COM-портом должен быть создан safety config:

```powershell
copy config\real_vehicle_safety.example.json config\real_vehicle_safety.json
notepad config\real_vehicle_safety.json
```

По умолчанию real-mode должен быть защищен dry-run ограничителем. Для настоящей отправки команд физическим исполнительным системам должны быть выполнены все условия:

```powershell
$env:MITSU_REAL_ENABLE_ACTUATION = "1"
$env:MITSU_REAL_DRY_RUN = "0"
```

и в `config/real_vehicle_safety.json`:

```json
{
  "dry_run": false,
  "allow_real_actuation": true
}
```

До bench-тестов и CAN-disabled проверки эти значения включать нельзя.

Рекомендуемые лимиты первого движения:

```text
speed cap:       1 км/ч
throttle cap:    5–10%
brake priority:  ON
command timeout: ON
operator inside: ON
physical E-stop: ON
```

---

## 6. MCU telemetry map

Перед реальной машиной нужно описать фактический формат feedback от MCU.

Скопировать пример:

```powershell
copy config\mcu_telemetry_map.example.json config\mcu_telemetry_map.json
notepad config\mcu_telemetry_map.json
```

В этом файле должны быть реальные поля для:

```text
speed_kmh
gear
steering_angle / steering_raw
brake_pct
accel_pct
x_m
y_m
yaw_deg
pose_valid
heartbeat
ACK / watchdog status
```

Пример записи:

```json
{
  "can_id": "0x0004",
  "field": "gear",
  "offset": 0,
  "length": 1,
  "enum": "gear_1_6"
}
```

До заполнения этой карты реальными данными нельзя считать, что GUI видит настоящий `actual gear`, `speed` и `pose`.

---

## 7. Preflight перед реальной машиной

Запуск:

```powershell
cd "E:\основы программирования\MitsuNeuroPilotAPI\i-MIEV GUI"
python scripts\real_vehicle_preflight.py
```

Preflight должен проверить:

```text
safety config;
camera config;
pose provider;
MCU telemetry map;
MITSU_REAL_AGENT_FACTORY;
MITSU_REAL_MODEL_FACTORY;
actuation guard.
```

Если проверка показывает `FAIL`, проблему нужно устранить до подключения к реальному автомобилю.

---

## 8. Камеры реального авто

На реальном авто используются две камеры:

```text
1. 2.8 мм / FOV около 90° — широкая камера.
2. 6 мм / FOV около 50° — узкая камера.
```

Физическая схема:

```text
Камера 2.8 мм / 90°  ┐
                     ├── Gigabit switch ── Ethernet ── ноутбук
Камера 6 мм / 50°   ┘
```

Просто подключить камеры к switch недостаточно. Нужно:

- настроить IP-адреса;
- проверить `ping`;
- прописать источники в `config/real_cameras.json`;
- запустить `dual_camera_service.py` или разрешить автозапуск real-mode.

Пример IP-сети:

```text
ноутбук:      192.168.1.10
камера 90°:   192.168.1.101
камера 50°:   192.168.1.102
mask:         255.255.255.0
```

Проверка:

```powershell
ping 192.168.1.101
ping 192.168.1.102
```

Пример `config/real_cameras.json` для RTSP:

```json
{
  "preview_fps": 20,
  "jpeg_quality": 85,
  "cameras": {
    "front_wide_28mm": {
      "label": "Камера 2.8 мм / 90°",
      "source": "rtsp://192.168.1.101:554/stream1",
      "role": "wide_90",
      "port": 5556
    },
    "front_narrow_6mm": {
      "label": "Камера 6 мм / 50°",
      "source": "rtsp://192.168.1.102:554/stream1",
      "role": "narrow_50",
      "port": 5557
    }
  }
}
```

Ручной запуск сервиса камер:

```powershell
cd "E:\основы программирования\MitsuNeuroPilotAPI\i-MIEV GUI"
python hardware\dual_camera_service.py --config config\real_cameras.json
```

Во втором терминале:

```powershell
python main.py
```

В real/mock режиме должны быть две ячейки:

```text
Камера 2.8 мм · широкий угол 90°
Камера 6 мм · узкий угол 50°
```

Проверка UI-патча:

```powershell
Select-String -Path "ui\main_window.py" -Pattern "Камера 2.8"
Select-String -Path "ui\main_window.py" -Pattern "real_camera_wide"
```

---

## 9. Навигатор и маршрут

В real-mode маршрут задается как A → B.

Варианты построения:

```text
1. Прямая A → B.
2. По дорогам OpenStreetMap/OSRM.
3. По нужной стороне дороги — аппроксимация смещением от centerline.
```

На карте:

```text
синий маркер     текущее местоположение
A                старт маршрута, по умолчанию текущее местоположение
B                цель
желтая линия     прямая A → B
синяя линия      centerline дороги OSRM/OSM
зеленая линия    контрольная траектория со смещением по нужной стороне дороги
```

Типовой сценарий:

```text
1. Открыть «Карта и маршрут».
2. Убедиться, что A стоит на текущем местоположении.
3. При необходимости перетащить A вручную.
4. Кликнуть B.
5. Нажать «Построить по дорогам».
6. Проверить, что появилась синяя/зеленая линия.
7. Нажать «Применить маршрут».
8. Нажать «Проверить маршрут».
```

Если road-routing не сработал, система должна явно логировать fallback на прямую A → B.

---

## 10. RoadOption bridge для модели

Для модели из CARLA/Leaderboard-style контура недостаточно строк `turn_left`, `straight`, `lane_follow`. Модель получает high-level command как one-hot вектор.

Поддержанные значения:

```text
RoadOption.LEFT              = 1
RoadOption.RIGHT             = 2
RoadOption.STRAIGHT          = 3
RoadOption.LANEFOLLOW        = 4
RoadOption.CHANGELANELEFT    = 5
RoadOption.CHANGELANERIGHT   = 6
RoadOption.VOID              = -1
```

Порядок one-hot:

```text
[LEFT, RIGHT, STRAIGHT, LANEFOLLOW, CHANGELANELEFT, CHANGELANERIGHT]
```

Real-mode навигатор должен формировать:

```text
road_option
road_option_name
next_road_option
next_road_option_name
command_one_hot
next_command_one_hot
```

Для текущего OSRM/OpenStreetMap routing безопасная логика такая:

```text
обычное движение между точками → LANEFOLLOW
поворот налево                → LEFT
поворот направо               → RIGHT
прямо через узел              → STRAIGHT
перестроение                  → не генерировать автоматически без lane-level карты
```

---

## 11. Текущая pose машины

Для реального автономного движения нужна валидная pose:

```text
x_m
y_m
yaw_deg
pose_valid = True
speed_kmh
actual gear
heartbeat
```

Возможные источники:

```text
1. RTK/GNSS + IMU — предпочтительно.
2. MCU odometry — wheel speed + steering angle + IMU yaw.
3. Visual odometry.
4. Teach-and-repeat.
5. Внешний pose-provider.
```

Временный JSON pose-provider:

```powershell
copy config\current_pose.example.json config\current_pose.json
$env:MITSU_REAL_POSE_JSON = "config\current_pose.json"
python main.py
```

Формат:

```json
{
  "x_m": 0.0,
  "y_m": 0.0,
  "yaw_deg": 0.0,
  "valid": true,
  "source": "external_pose_provider"
}
```

Для маршрута по карте обычно лучше:

```text
lat/lon машины
  ↓
ENU/local conversion от точки A
  ↓
x_m / y_m / yaw_deg
  ↓
VehicleTelemetry
```

Без `pose_valid=True` реальный `Activate Control` должен блокироваться.

---

## 12. Как подключается агент/модель к реальным камерам

В CARLA `SensorAgent` получает CARLA-style input и внутри вызывает `ClosedLoopInference.forward(...)`. Для реального авто нужен отдельный adapter.

Минимальный real-agent hook:

```powershell
$env:MITSU_REAL_AGENT_FACTORY = "real_agent_adapters.lead_real_adapter:create_agent"
python main.py
```

Если используется конкретная модель через отдельную factory:

```powershell
$env:MITSU_REAL_MODEL_FACTORY = "my_model_module:create_model"
```

Factory должна вернуть объект с одним из интерфейсов:

```python
predict(input_tensors)
forward(data=input_tensors)
__call__(input_tensors)
```

Вход агентного адаптера:

```python
{
    "wide_90": frame_from_2_8mm_camera,      # OpenCV BGR
    "narrow_50": frame_from_6mm_camera,     # OpenCV BGR
}
```

Контекст real-mode:

```text
speed_kmh
speed_mps
target_point
target_point_previous
target_point_next
command_one_hot
next_command_one_hot
road_option
next_road_option
```

Выход модели:

```python
{
    "steer": -1.0,          # -1.0 ... +1.0
    "throttle": 0.0,        #  0.0 ...  1.0
    "brake": 0.0,           #  0.0 ...  1.0
    "target_angle_deg": 0,
    "confidence": 1.0
}
```

Дальше:

```text
predict(...)
  ↓
ControlIntent
  ↓
VehicleControlService.submit_ai_intent(...)
  ↓
ControlArbiter
  ↓
VehicleGateway
  ↓
USB → MCU → CAN
```

Если `MITSU_REAL_AGENT_FACTORY` не задан, real-mode может ехать по PID-fallback от навигатора.

---

## 13. Как связаны waypoint-ы модели, скорость и PID

Модельная архитектура содержит planning-output:

```text
pred_future_waypoints
pred_route
pred_target_speed_scalar
```

В closed-loop режиме эти выходы преобразуются в управление через PID:

```text
pred_future_waypoints + current speed
  ↓
execute_waypoints(...)
  ↓
steer / throttle / brake

pred_route + pred_target_speed_scalar + current speed
  ↓
execute_route_and_target_speed(...)
  ↓
steer / throttle / brake
```

Для реального авто предпочтительный первый вариант интеграции:

```text
ClosedLoopInference уже посчитал steer/throttle/brake
  ↓
ControlIntent
  ↓
ControlArbiter
  ↓
VehicleCommand
```

Не нужно вторично прогонять `steer/throttle/brake` через маршрутный PID — иначе получится двойной PID.

Если нужно использовать именно сырые waypoint-ы модели, тогда нужен отдельный `RealVehicleWaypointFollower`, который берет:

```text
pred_future_waypoints
pred_target_speed_scalar
current_speed
steering_feedback
```

и уже сам считает `ControlIntent`. В этом случае встроенный closed-loop PID модели лучше не использовать.

---

## 14. Что нужно для реального теста модели

Для реального теста именно модели, а не PID-fallback, нужны следующие компоненты.

### 14.1. Checkpoint и training config

Нужен конкретный checkpoint и config, с которым он обучался:

```text
model_*.pth
config.json
TrainingConfig
ClosedLoopConfig
sensor configuration
```

Проверить checkpoint:

```python
import torch
ckpt = torch.load("model_0011.pth", map_location="cpu")
print(type(ckpt))
print(ckpt.keys() if isinstance(ckpt, dict) else "not dict")
```

Если внутри только `state_dict`, нужен отдельный `config.json` из training run.

### 14.2. Sensor layout

Нужно точно знать:

```text
порядок камер;
разрешение;
crop/resize;
FOV;
extrinsics относительно машины;
num_available_cameras;
num_used_cameras;
used_cameras;
horizontal_fov_reduction;
нужен ли rasterized_lidar;
нужен ли radar;
нужен ли town id.
```

Камеры реального авто:

```text
wide_90    = 2.8 мм / 90°
narrow_50  = 6 мм / 50°
```

Но модель должна быть обучена или адаптирована под такой порядок и такой sensor layout. Иначе кадры будут технически подаваться, но семантически модель может работать неправильно.

### 14.3. Vision-only или dummy lidar

Если checkpoint обучен с `rasterized_lidar`, а на машине lidar нет, есть три варианта:

```text
1. использовать vision-only checkpoint;
2. переобучить/дообучить модель под две камеры;
3. подать dummy zero-lidar и доказать в replay/dry-run, что поведение приемлемо.
```

Для первого реального теста лучше `vision-only` checkpoint.

### 14.4. RealSensorInputAdapter

Нужен файл уровня:

```text
lead/inference/real_dual_camera_agent_adapter.py
```

Он должен:

```text
1. получить два BGR кадра;
2. привести их к train-time resolution/crop/FOV;
3. собрать rgb tensor в ожидаемом порядке;
4. сформировать speed в m/s;
5. сформировать target_point_previous / target_point / target_point_next;
6. сформировать command / next_command;
7. сформировать rasterized_lidar, если checkpoint требует;
8. вызвать ClosedLoopInference.forward(...);
9. вернуть steer/throttle/brake.
```

### 14.5. Pose-provider

Нужна реальная pose. Без нее невозможно корректно вычислять:

```text
target_point
target_point_next
command
heading error
cross-track error
```

### 14.6. MCU feedback и safety

MCU должен возвращать:

```text
heartbeat
actual gear
actual speed
steering feedback
brake feedback
ACK / timeout
watchdog status
```

И должен сам выполнять safety:

```text
USB lost → throttle 0 / brake safe
command stale → reject
CRC bad → reject
P while moving → reject
brake priority over throttle
```

---

## 15. Последовательность реального теста

### Этап 1 — GUI и mock

```text
TEST_MOCK_VEHICLE
подключить
проверить маршрут
AI Preview ON
Activate Control
AI_ACTIVE
manual takeover
manual control
safe stop / Park
```

### Этап 2 — камеры

```text
подключить две камеры к switch
проверить ping
запустить dual_camera_service.py
запустить GUI
проверить две video-cell
проверить latency/fps
```

### Этап 3 — модель без машины

```text
задать MITSU_REAL_AGENT_FACTORY
TEST_MOCK_VEHICLE
AI Preview ON
проверить, что агент получает кадры
проверить predict(...) output
не отправлять в реальный COM
```

### Этап 4 — MCU dry-run

```text
реальный COM
CAN output disabled
проверить heartbeat/ACK/CRC/CNC
проверить, что VehicleGateway отправляет команды
машина не исполняет команды
```

### Этап 5 — telemetry-only на машине

```text
реальная машина подключена
управление не активировать
читать actual gear/speed/pose/heartbeat
проверить pose_valid
```

### Этап 6 — gear test без газа

```text
brake hold
throttle hard clamp = 0
request D
wait actual gear D
request P только после speed == 0
```

### Этап 7 — первое движение

```text
закрытая площадка
человек за рулем
физический emergency stop
speed cap = 1 км/ч
throttle cap = 5–10%
короткий маршрут
AI Preview ON
Activate Control
следить за NAV / speed / command age / heartbeat
```

---

## 16. Частые проблемы

### `python main.py` из корня не работает

Нужно запускать из `i-MiEV GUI`:

```powershell
cd "E:\основы программирования\MitsuNeuroPilotAPI\i-MIEV GUI"
python main.py
```

### `python -m py_compile vehicle_control/*.py` на Windows не работает

PowerShell не раскрывает `*.py` так, как bash. Используйте:

```powershell
Get-ChildItem vehicle_control,hardware,real_agent_adapters -Filter *.py | ForEach-Object {
    python -m py_compile $_.FullName
}
```

### `python -m unittest discover -s tests -v` не видит tests

Вы запустили команду из корня проекта. Нужно:

```powershell
cd "E:\основы программирования\MitsuNeuroPilotAPI\i-MIEV GUI"
python -m unittest discover -s tests -v
```

### Карта черная, `L is not defined`

Leaflet не загрузился. Проверьте доступ к CDN или используйте локальные `leaflet.js` / `leaflet.css`.

### Нет текущего местоположения на карте

Настройте:

```text
config/map_settings.json
```

или переменные:

```powershell
$env:MITSU_CURRENT_LAT = "55.000000"
$env:MITSU_CURRENT_LON = "37.000000"
```

### Нет картинок с камер

Проверить:

```powershell
ping 192.168.1.101
ping 192.168.1.102
python hardware\dual_camera_service.py --config config\real_cameras.json
```

### Real Activate заблокирован: `Pose is not valid`

Это корректное поведение. Нужен pose-provider.

### В real/mock режиме виден CARLA route UI

Значит, не применен последний UI-patch или запускается не та папка.

Проверить:

```powershell
Select-String -Path "ui\main_window.py" -Pattern "Навигатор / миссия"
Select-String -Path "ui\main_window.py" -Pattern "Камера 2.8"
```

### `AttributeError: MANUAL_ACTIVE`

В `DriveState` должно быть состояние:

```python
MANUAL_ACTIVE = "MANUAL_ACTIVE"
```

### `VehicleControlService` не имеет `transfer_ai_to_manual`

В `VehicleControlService` должен быть публичный метод:

```python
async def transfer_ai_to_manual(self, reason: str = "driver_takeover") -> bool:
    ...
```

Он используется для сценария:

```text
AI_ACTIVE → повторное Activate Control → MANUAL_ACTIVE
```

---

## 17. Минимальный checklist перед подключением реального авто

Перед подключением реального автомобиля должны быть выполнены пункты:

```text
[ ] Unit-тесты проходят.
[ ] TEST_MOCK_VEHICLE проходит AI_ACTIVE → MANUAL_ACTIVE → Park-after-stop.
[ ] TEST_SERIAL_LOOPBACK проверен.
[ ] real_vehicle_preflight.py не показывает критических FAIL.
[ ] real_vehicle_safety.json создан и dry_run включен.
[ ] mcu_telemetry_map.json заполнен фактическими feedback-полями.
[ ] current_pose.json или другой pose-provider работает.
[ ] real_cameras.json настроен под реальные камеры.
[ ] Две камеры доступны по ping.
[ ] dual_camera_service.py показывает оба потока.
[ ] MCU подключен сначала в CAN_DISABLED / bench режиме.
[ ] Есть heartbeat/ACK/watchdog feedback.
[ ] Gear feedback подтверждает actual P/D/P.
[ ] Throttle hard clamp = 0 на этапе gear-test.
[ ] Физический emergency stop доступен оператору.
[ ] Человек находится в салоне или рядом с органами управления.
```

---

## 18. Минимальный статус проекта

Сейчас готово:

```text
CARLA flow изолирован в VIRTUAL_DEMO_MODE
mock vehicle работает
serial loopback подготовлен
road/navigation PID-fallback работает
RoadOption bridge готов
AI Preview flag есть
manual mode flag есть
manual takeover после AI_ACTIVE есть
brake-before-Drive заложен
Park-after-stop заложен
две camera-cell заложены
real-agent factory hook заложен
safety config и dry-run guard заложены
MCU telemetry map заложен
pose-provider hook заложен
preflight script заложен
```

Не хватает для реального автономного теста модели:

```text
реальный pose-provider
реальный MCU handshake/feedback
подтвержденная CAN/gear реализация на i-MiEV
рабочий real_cameras.json под конкретные камеры
RealDualCameraAgentAdapter под checkpoint/config
выбранный vision-only или совместимый checkpoint
проверенный sensor layout
safety watchdog на MCU
закрытый полигон и физический emergency stop
```

---

## 19. Рекомендуемый следующий порядок работ

```text
1. Закрепить текущую версию как stable real-mode preparation.
2. Проверить TEST_MOCK_VEHICLE после каждого изменения.
3. Заполнить mcu_telemetry_map.json по фактической прошивке MCU.
4. Поднять камеры через switch и проверить задержку.
5. Подключить pose-provider.
6. Подготовить RealDualCameraAgentAdapter под конкретный checkpoint.
7. Проверить модель на TEST_MOCK_VEHICLE.
8. Перейти к MCU bench / CAN_DISABLED.
9. Выполнить telemetry-only на машине.
10. Выполнить gear-test без газа.
11. Выполнить steering/brake test на месте.
12. Выполнить low-speed test на закрытой площадке.
```

