# Model 0011 Dual Front Camera Inference I/O Report

Этот отчет описывает входные и выходные данные модели из
`outputs/model_0011/config_dual_front_camera.json` при инференсе в симуляторе
CARLA/leaderboard.

## Кратко

Модель работает в режиме двух фронтальных камер:

```json
{
  "backbone_sensor_mode": "dual_front_camera",
  "rgb_camera_ids": [1, 2],
  "left_camera_key": "rgb_left",
  "right_camera_key": "rgb_right",
  "use_lidar": false,
  "use_radars": false
}
```

На вход модели идут:

- две отдельные RGB-картинки `rgb_left` и `rgb_right`;
- текущая скорость `speed`;
- три локальные маршрутные точки `target_point_previous`, `target_point`, `target_point_next`;
- текущая и следующая high-level команда маршрута `command`, `next_command`;
- поле `town`, которое в этой модели является метаданными и напрямую не используется в `TFv6`.

Модель предсказывает:

- `pred_route` - локальные route checkpoints;
- `pred_future_waypoints` - будущие точки движения ego-автомобиля;
- `pred_target_speed_scalar` - целевую скорость;
- `pred_target_speed_distribution` - распределение по классам скорости;
- дополнительные perception outputs: semantic segmentation, BEV semantic map, bounding boxes.

Финальный выход в симулятор - `carla.VehicleControl(steer, throttle, brake)`,
который получается из предсказанных route/target speed через PID-контроллер.

## Где это находится в коде

Основной путь инференса:

1. `lead/inference/sensor_agent.py`
   - `SensorAgent.tick()` обрабатывает сырые сенсоры.
   - `SensorAgent.run_step()` собирает `input_data_tensors`.
2. `lead/inference/open_loop_inference.py`
   - `OpenLoopInference.forward()` вызывает одну или несколько моделей.
   - `ensemble_planning_decoder()` усредняет planning outputs.
3. `lead/inference/closed_loop_inference.py`
   - `ClosedLoopInference.ensemble()` переводит high-level predictions в control.
4. `lead/tfv6/tfv6.py`
   - `TFv6.forward()` выполняет forward pass модели.

## Сырые данные от CARLA

До подготовки батча `SensorAgent` получает от leaderboard/CARLA словарь примерно
такого вида:

```python
raw_input_data = {
    "rgb_1": (12345, rgb_1_image),
    "rgb_2": (12345, rgb_2_image),
    "imu": (12345, imu_array),
    "gps": (12345, gps_array),
    "speed": (12345, {"speed": 3.2}),
}
```

Где:

- `rgb_1`, `rgb_2` - изображения с двух RGB-камер;
- `imu` - IMU, из него берется compass/yaw и ускорения;
- `gps` - GNSS-позиция;
- `speed` - скорость ego-автомобиля в м/с.

В этом конкретном конфиге LiDAR и radar не используются:

```json
{
  "use_lidar": false,
  "use_radars": false
}
```

## Финальный вход модели

После `SensorAgent.tick()` и сборки в `SensorAgent.run_step()` модель получает
словарь:

```python
input_data_tensors = {
    "rgb_left": torch.Tensor,              # shape: (1, 3, 384, 384)
    "rgb_right": torch.Tensor,             # shape: (1, 3, 384, 384)

    "target_point_previous": torch.Tensor, # shape: (1, 2)
    "target_point": torch.Tensor,          # shape: (1, 2)
    "target_point_next": torch.Tensor,     # shape: (1, 2)

    "speed": torch.Tensor,                 # shape: (1,)
    "command": torch.Tensor,               # shape: (1, 6)
    "next_command": torch.Tensor,          # shape: (1, 6)

    "town": np.ndarray,                    # shape: (1,)
}
```

Пример с конкретными значениями:

```python
import numpy as np
import torch

data = {
    "rgb_left": torch.rand((1, 3, 384, 384), dtype=torch.float32) * 255.0,
    "rgb_right": torch.rand((1, 3, 384, 384), dtype=torch.float32) * 255.0,

    "target_point_previous": torch.tensor([[2.0, 0.1]], dtype=torch.float32),
    "target_point": torch.tensor([[14.5, -1.2]], dtype=torch.float32),
    "target_point_next": torch.tensor([[28.0, -3.8]], dtype=torch.float32),

    "speed": torch.tensor([3.2], dtype=torch.float32),

    "command": torch.tensor([[0, 0, 1, 0, 0, 0]], dtype=torch.float32),
    "next_command": torch.tensor([[0, 0, 1, 0, 0, 0]], dtype=torch.float32),

    "town": np.array(["Town13"]),
}
```

## `rgb_left` и `rgb_right`

### Формат

```python
rgb_left.shape == (1, 3, 384, 384)
rgb_right.shape == (1, 3, 384, 384)
rgb_left.dtype == torch.float32
rgb_right.dtype == torch.float32
```

Расшифровка shape:

```text
1   - batch size
3   - RGB-каналы
384 - высота изображения
384 - ширина изображения
```

Изображения идут отдельно:

- `rgb_left` берется из CARLA-сенсора `rgb_1`;
- `rgb_right` берется из CARLA-сенсора `rgb_2`.

Они не склеиваются в один широкий RGB-тензор. Склейка используется только для
визуализации `original_rgb`.

### Геометрия камер

Из конфига:

```json
{
  "left_camera_fov_deg": 90.0,
  "right_camera_fov_deg": 50.03,
  "camera_baseline_m": 0.135,
  "left_camera_translation_m": [0.9, -0.0675, 1.55],
  "right_camera_translation_m": [0.9, 0.0675, 1.55],
  "left_camera_yaw_deg": 0.0,
  "right_camera_yaw_deg": 0.0,
  "left_camera_pitch_deg": 0.0,
  "right_camera_pitch_deg": 0.0,
  "left_camera_roll_deg": 0.0,
  "right_camera_roll_deg": 0.0
}
```

## `target_point_previous`, `target_point`, `target_point_next`

Эти поля - координаты маршрутных точек в локальной системе ego-автомобиля.

Формат:

```python
target_point_previous.shape == (1, 2)
target_point.shape == (1, 2)
target_point_next.shape == (1, 2)
```

Каждое поле содержит:

```python
[[x, y]]
```

Где:

- `x` - продольная координата в метрах относительно ego-автомобиля;
- `y` - поперечная координата в метрах относительно ego-автомобиля.

Пример:

```python
target_point_previous = torch.tensor([[2.0, 0.1]], dtype=torch.float32)
target_point = torch.tensor([[14.5, -1.2]], dtype=torch.float32)
target_point_next = torch.tensor([[28.0, -3.8]], dtype=torch.float32)
```

Интерпретация:

- `target_point_previous = [[2.0, 0.1]]` - предыдущая/ближайшая точка около
  2 м впереди и 0.1 м вбок;
- `target_point = [[14.5, -1.2]]` - текущая целевая точка около 14.5 м впереди
  и 1.2 м вбок;
- `target_point_next = [[28.0, -3.8]]` - следующая точка около 28 м впереди
  и 3.8 м вбок.



## `speed`

`speed` - текущая скорость ego-автомобиля в м/с.

Формат:

```python
speed = torch.tensor([3.2], dtype=torch.float32)
speed.shape == (1,)
```

Пример:

```python
speed_mps = 3.2
speed_kmh = speed_mps * 3.6
# 11.52 км/ч
```

В конфиге:

```json
{
  "max_speed": 5.55555556
}
```

Это около 20 км/ч.

## `command` и `next_command`

`command` и `next_command` - high-level навигационные команды CARLA route planner
в one-hot формате.

Формат:

```python
command.shape == (1, 6)
next_command.shape == (1, 6)
```

В коде используется функция `command_to_one_hot(command_int)`.
Она делает:

```python
index = command_int - 1
```

Примеры кодирования:

```python
command_int = 1 -> [1, 0, 0, 0, 0, 0]
command_int = 2 -> [0, 1, 0, 0, 0, 0]
command_int = 3 -> [0, 0, 1, 0, 0, 0]
command_int = 4 -> [0, 0, 0, 1, 0, 0]
command_int = 5 -> [0, 0, 0, 0, 1, 0]
command_int = 6 -> [0, 0, 0, 0, 0, 1]
```

Если команда некорректная, код подставляет дефолтный индекс:

```python
command_int < 0 -> command_int = 4
unknown command -> command_int = 4
```

Пример текущей команды "прямо" или соответствующего route option:

```python
command = torch.tensor([[0, 0, 1, 0, 0, 0]], dtype=torch.float32)
```

Пример, где сейчас движение прямо, а дальше будет другая команда:

```python
command = torch.tensor([[0, 0, 1, 0, 0, 0]], dtype=torch.float32)
next_command = torch.tensor([[1, 0, 0, 0, 0, 0]], dtype=torch.float32)
```

## `town`

`town` выглядит так:

```python
town = np.array(["Town13"])
```

Это имя текущей CARLA-карты:

```python
self._world.get_map().name.split("/")[-1]
```

Для этой модели `town` напрямую не влияет на prediction. По коду `TFv6` и
`PlanningDecoder` используют изображения, speed, command и target points, но не
читают `data["town"]` как входной признак.

Зачем поле есть:

- совместимость с форматом датасета;
- логирование и сохранение meta;
- аналитика по городам;
- train/eval инфраструктура, например holdout Town13.

Если заменить только:

```python
town = np.array(["Town01"])
```

на:

```python
town = np.array(["Town13"])
```

при тех же RGB, speed, target points и command, output этой модели должен
остаться тем же. На результат влияет не строка `town`, а сама карта через другие
изображения, маршрутные точки и команды.

## Основные выходы модели

В конфиге включено:

```json
{
  "predict_target_speed": true,
  "predict_spatial_path": true,
  "predict_temporal_spatial_waypoints": true,
  "use_planning_decoder": true
}
```

Основные planning outputs:

```python
pred_route: torch.Tensor                     # shape: (1, 10, 2)
pred_future_waypoints: torch.Tensor          # shape: (1, 8, 2)
pred_target_speed_scalar: torch.Tensor       # shape: (1, 1)
pred_target_speed_distribution: torch.Tensor # shape: (1, 8)
```

## `pred_route`

`pred_route` - 10 route checkpoints в локальной системе ego-автомобиля.

Формат:

```python
pred_route.shape == (1, 10, 2)
```

Пример:

```python
pred_route = torch.tensor([[
    [ 2.0,  0.0],
    [ 4.5, -0.1],
    [ 7.0, -0.3],
    [10.0, -0.6],
    [13.0, -1.0],
    [16.0, -1.4],
    [20.0, -1.8],
    [24.0, -2.0],
    [28.0, -2.1],
    [32.0, -2.0],
]], dtype=torch.float32)
```

Каждая точка - `[x, y]` в метрах относительно ego.

Этот output используется для руления, если `steer_modality == "route"`.

## `pred_future_waypoints`

`pred_future_waypoints` - 8 будущих точек движения ego-автомобиля.

Формат:

```python
pred_future_waypoints.shape == (1, 8, 2)
```

Пример:

```python
pred_future_waypoints = torch.tensor([[
    [0.8,  0.0],
    [1.7, -0.1],
    [2.7, -0.2],
    [3.8, -0.4],
    [5.0, -0.7],
    [6.2, -1.0],
    [7.5, -1.3],
    [8.9, -1.5],
]], dtype=torch.float32)
```

Это альтернативный planning output. Он может использоваться для управления,
если closed-loop config выбирает waypoint modality.

## `pred_target_speed_distribution`

`pred_target_speed_distribution` - вероятности по классам целевой скорости.

Формат:

```python
pred_target_speed_distribution.shape == (1, 8)
```

Классы скорости из конфига:

```python
target_speed_classes = [
    0.0,
    0.79365079,
    1.58730159,
    2.38095238,
    3.17460317,
    3.96825397,
    4.76190476,
    5.55555556,
]
```

Пример:

```python
pred_target_speed_distribution = torch.tensor([[
    0.02,
    0.05,
    0.10,
    0.15,
    0.25,
    0.28,
    0.12,
    0.03,
]], dtype=torch.float32)
```

Интерпретация:

- максимальная вероятность здесь у класса `3.96825397`;
- значит модель считает целевую скорость около `4 м/с` наиболее вероятной;
- `4 м/с` - это около `14.4 км/ч`.

## `pred_target_speed_scalar`

`pred_target_speed_scalar` - декодированная целевая скорость в м/с.

Формат:

```python
pred_target_speed_scalar.shape == (1, 1)
```

Пример:

```python
pred_target_speed_scalar = torch.tensor([[3.7]], dtype=torch.float32)
```

Интерпретация:

```text
3.7 м/с = 13.32 км/ч
```

Этот output используется для throttle/brake, если:

```python
throttle_modality == "target_speed"
brake_modality == "target_speed"
```

## Дополнительные perception outputs

В конфиге включены:

```json
{
  "detect_boxes": true,
  "use_semantic": true,
  "use_bev_semantic": true,
  "use_depth": false,
  "num_semantic_classes": 10,
  "num_bev_semantic_classes": 14
}
```

Поэтому модель может возвращать:

```python
pred_semantic: torch.Tensor | None
pred_bev_semantic: torch.Tensor | None
pred_bounding_box_vehicle_system: list[PredictedBoundingBox] | None
pred_bounding_box_image_system: list[PredictedBoundingBox] | None
pred_depth: None
```

### `pred_semantic`

Формат:

```python
pred_semantic.shape == (1, 10, 384, 384)
```

Это perspective semantic segmentation по изображению.

### `pred_bev_semantic`

Формат:

```python
pred_bev_semantic.shape == (1, 14, H_bev, W_bev)
```

Это BEV semantic map. Для этого конфига BEV область:

```json
{
  "min_x_meter": -32,
  "max_x_meter": 64,
  "min_y_meter": -40,
  "max_y_meter": 40,
  "pixels_per_meter": 4.0
}
```

### Bounding boxes

Пример одного detection:

```python
PredictedBoundingBox(
    x=18.4,
    y=-2.1,
    w=2.0,
    h=4.5,
    yaw=0.05,
    velocity=1.2,
    brake=0.0,
    clazz=1,
    score=0.87,
)
```

Поля:

- `x`, `y` - положение объекта в ego-системе;
- `w`, `h` - размеры box;
- `yaw` - ориентация;
- `velocity` - скорость объекта;
- `brake` - признак торможения;
- `clazz` - класс объекта;
- `score` - confidence.

Эти outputs нужны для perception/визуализации/post-processing, но основной
closed-loop control строится по route и target speed.

## Финальный output в CARLA

Модель не выдает напрямую `steer`, `throttle`, `brake`. Она выдает high-level
planning predictions. Затем `ClosedLoopInference` через PID-контроллер строит:

```python
control = carla.VehicleControl(
    steer=-0.08,
    throttle=0.22,
    brake=0.0,
)
```

Формат:

```text
steer    - руление, диапазон [-1, 1]
throttle - газ, диапазон [0, 1]
brake    - тормоз, диапазон [0, 1]
```

В текущей closed-loop логике:

- `steer` берется из `pred_route`;
- `throttle` берется из `pred_target_speed_scalar`;
- `brake` берется из `pred_target_speed_scalar`.

Пример полного цикла:

```python
# Вход
data = {
    "rgb_left": torch.rand((1, 3, 384, 384)) * 255.0,
    "rgb_right": torch.rand((1, 3, 384, 384)) * 255.0,
    "target_point_previous": torch.tensor([[2.0, 0.1]]),
    "target_point": torch.tensor([[14.5, -1.2]]),
    "target_point_next": torch.tensor([[28.0, -3.8]]),
    "speed": torch.tensor([3.2]),
    "command": torch.tensor([[0, 0, 1, 0, 0, 0]], dtype=torch.float32),
    "next_command": torch.tensor([[0, 0, 1, 0, 0, 0]], dtype=torch.float32),
    "town": np.array(["Town13"]),
}

# Модель предсказала
pred_route = torch.tensor([[
    [2.0, 0.0],
    [4.5, -0.1],
    [7.0, -0.3],
    [10.0, -0.6],
    [13.0, -1.0],
    [16.0, -1.4],
    [20.0, -1.8],
    [24.0, -2.0],
    [28.0, -2.1],
    [32.0, -2.0],
]])

pred_target_speed_scalar = torch.tensor([[3.7]])

# PID-контроллер сделал управление
control = carla.VehicleControl(
    steer=-0.08,
    throttle=0.22,
    brake=0.0,
)
```
