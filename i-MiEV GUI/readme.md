# MitsuNeuroPilot GUI: режимы запуска, камеры, навигация и реальное авто

Этот README описывает текущую логику интерфейса MitsuNeuroPilot GUI после разделения на два независимых контура:

1. `VIRTUAL_DEMO_MODE` — существующая логика CARLA/Leaderboard/LeadAgentThread, которая должна работать как раньше.
2. `TEST_MOCK_VEHICLE`, `TEST_SERIAL_LOOPBACK`, реальный COM-порт — новый контур реального автомобиля: две камеры, навигатор, агент/модель или PID-fallback, VehicleControlService, VehicleGateway, USB/MCU/CAN.

Главное правило архитектуры: **CARLA-код не должен запускаться в real/mock режимах**, а real-vehicle-код не должен менять CARLA flow.

---

## 1. Быстрый запуск GUI

Из корня репозитория не запускайте `python main.py`, потому что `main.py` находится внутри папки GUI.

```powershell
cd "E:\основы программирования\MitsuNeuroPilotAPI\i-MIEV GUI"
python main.py
```

Если запускаете из корня проекта:

```powershell
python "i-MIEV GUI/main.py"
```

---

## 2. Режимы в выпадающем списке портов

### 2.1. `VIRTUAL_DEMO_MODE`

Это старый режим симуляции. Здесь должны работать:

- CARLA;
- route picker;
- подготовка маршрута/очереди;
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

Этот режим не должен использовать новый `VehicleControlService` для реального авто.

---

### 2.2. `TEST_MOCK_VEHICLE`

Главный режим для проверки real-vehicle логики без железа.

Он имитирует:

- подключение автомобиля;
- gear `P/D`;
- задержку переключения;
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
```

---

### 2.3. `TEST_SERIAL_LOOPBACK`

Режим для проверки serial/USB/CAN-транспорта без реальной машины.

Использовать для:

- проверки формата 16-байтных пакетов;
- проверки CRC/CNC;
- проверки ACK/timeout;
- проверки command scheduler;
- проверки, что control-команды не копятся в FIFO.

---

### 2.4. Реальный COM-порт

Реальный COM-порт нельзя использовать сразу как «ехать». Сначала нужны dry-run и telemetry-only.

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

## 3. Логика кнопок и флагов

### 3.1. «Предпросмотр ИИ»

Этот флаг включает анализ данных агентом, но **не передает authority машине**.

В real-mode:

```text
Предпросмотр ИИ ON:
  - камеры читаются;
  - агент/адаптер может анализировать кадры;
  - ControlIntent может формироваться;
  - VehicleGateway не имеет authority, пока не нажата «Активировать управление».
```

---

### 3.2. «Ручное управление»

Флаг ручного режима нужен для движения по маршруту без автономного агента.

```text
Ручное управление ON:
  - пользователь сам управляет стрелками/клавишами;
  - маршрут и waypoint-ы остаются подсказкой;
  - AI не имеет authority.
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

### 3.3. «Активировать управление» в AI mode

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

### 3.4. Повторное нажатие «Активировать управление» во время AI_ACTIVE

Повторное нажатие **не должно сразу отправлять Park**.

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

### 3.5. Полное отключение управления / Park

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

## 4. Камеры реального авто

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

Если виден только один большой блок `VIDEO STREAM`, проверь, что установлен последний патч:

```powershell
Select-String -Path "ui\main_window.py" -Pattern "Камера 2.8"
Select-String -Path "ui\main_window.py" -Pattern "real_camera_wide"
```

---

## 5. Навигатор и маршрут

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

## 6. Текущая pose машины

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

## 7. Как подключается агент/модель к реальным камерам

В CARLA `SensorAgent` получает CARLA-style input и внутри вызывает `ClosedLoopInference.forward(...)`. Для реального авто нужен отдельный adapter.

Переменная окружения:

```powershell
$env:MITSU_REAL_AGENT_FACTORY = "lead.inference.real_dual_camera_agent_adapter:create_agent"
python main.py
```

Factory должен вернуть объект с методом:

```python
class RealDualCameraAgent:
    def predict(self, frames: dict) -> dict | None:
        ...
```

На вход:

```python
{
    "wide_90": frame_from_2_8mm_camera,      # OpenCV BGR
    "narrow_50": frame_from_6mm_camera,     # OpenCV BGR
}
```

На выход:

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

## 8. Как связаны waypoint-ы модели, скорость и PID

Модельная архитектура уже содержит planning-output:

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

и уже сам считает `ControlIntent`. Но тогда встроенный closed-loop PID модели лучше не использовать.

---

## 9. Что нужно для реального теста модели

Для реального теста именно модели, а не PID-fallback, еще нужны следующие компоненты.

### 9.1. Checkpoint и training config

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
ckpt = torch.load("model_2.pth", map_location="cpu")
print(type(ckpt))
print(ckpt.keys() if isinstance(ckpt, dict) else "not dict")
```

Если внутри только `state_dict`, нужен отдельный `config.json` из training run.

### 9.2. Sensor layout

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

### 9.3. Vision-only или dummy lidar

Если checkpoint обучен с `rasterized_lidar`, а на машине lidar нет, есть три варианта:

```text
1. использовать vision-only checkpoint;
2. переобучить/дообучить модель под две камеры;
3. подать dummy zero-lidar и доказать в replay/dry-run, что поведение приемлемо.
```

Для первого реального теста лучше `vision-only` checkpoint.

### 9.4. RealSensorInputAdapter

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

### 9.5. Pose-provider

Нужна реальная pose. Без нее невозможно корректно вычислять:

```text
target_point
target_point_next
command
heading error
cross-track error
```

### 9.6. MCU feedback и safety

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

## 10. Последовательность реального теста

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

## 11. Частые проблемы

### `python main.py` из корня не работает

Нужно запускать из `i-MiEV GUI`:

```powershell
cd "E:\основы программирования\MitsuNeuroPilotAPI\i-MIEV GUI"
python main.py
```

### Карта черная, `L is not defined`

Leaflet не загрузился. Проверь доступ к CDN или используй локальные `leaflet.js` / `leaflet.css`.

### Нет текущего местоположения на карте

Настрой:

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

Это правильно. Нужен pose-provider.

### В real/mock режиме виден CARLA route UI

Значит, не применен последний UI-patch или запускается не та папка.

Проверить:

```powershell
Select-String -Path "ui\main_window.py" -Pattern "Навигатор / миссия"
Select-String -Path "ui\main_window.py" -Pattern "Камера 2.8"
```

---

## 12. Минимальный статус проекта

Сейчас готово:

```text
CARLA flow изолирован в VIRTUAL_DEMO_MODE
mock vehicle работает
road/navigation PID-fallback работает
AI Preview flag есть
manual mode flag есть
manual takeover после AI_ACTIVE есть
brake-before-Drive заложен
Park-after-stop заложен
две camera-cell заложены
real-agent factory hook заложен
```

Не хватает для реального теста модели:

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
