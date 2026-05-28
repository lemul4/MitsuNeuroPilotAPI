# MitsuNeuroPilot: подготовка к первому подключению реального Mitsubishi i-MiEV

Этот пакет закрывает программные пробелы перед подключением реальной машины. Он не разрешает немедленный автономный выезд: по умолчанию включен host-side dry-run guard, который не отправляет активные команды тяги/передачи на реальный serial-адаптер, пока это явно не разрешено через конфигурацию и переменную окружения.

## Что добавлено

- `RealVehicleSafetyConfig`: отдельный профиль ограничений для реального автомобиля.
- Host-side dry-run guard в `RealSerialVehicleAdapter`.
- Конфигурируемый `McuTelemetryParser` для speed/gear/steering/brake/pose feedback.
- `JsonPoseProviderThread`: временный источник pose из JSON-файла для RTK/GNSS+IMU, MCU odometry или внешнего pose-provider.
- Контекстный вызов real-agent: `predict(frames, context)`.
- `real_agent_adapters/lead_real_adapter.py`: builder входов модели в стиле `SensorAgent` (`rgb`, `speed`, `target_point`, `command`, `next_command`, `rasterized_lidar`, `town`).
- `scripts/real_vehicle_preflight.py`: проверка готовности конфигурации без физического исполнения команд.
- Исправление `DriveState.MANUAL_ACTIVE` и ручного режима.

## Минимальные конфиги

```powershell
cd "E:\основы программирования\MitsuNeuroPilotAPI\i-MIEV GUI"
copy config\real_vehicle_safety.example.json config\real_vehicle_safety.json
copy config\mcu_telemetry_map.example.json config\mcu_telemetry_map.json
copy config\current_pose.example.json config\current_pose.json
copy config\real_cameras.example.json config\real_cameras.json
```

Отредактировать:

- `real_cameras.json`: IP/RTSP/HTTP источники двух камер;
- `mcu_telemetry_map.json`: фактические CAN ID/offset/scale из прошивки MCU;
- `current_pose.json`: временная pose, если нет MCU/RTK pose;
- `real_vehicle_safety.json`: оставить `dry_run=true`, пока не пройдены bench-тесты.

## Проверка без автомобиля

```powershell
python -m unittest discover -s tests -v
python scripts\real_vehicle_preflight.py
python main.py
```

Сценарий GUI:

1. Выбрать `TEST_MOCK_VEHICLE`.
2. Connect.
3. Построить или проверить маршрут A-B.
4. AI Preview ON или Manual mode ON.
5. Activate Control.
6. Проверить `AI_ACTIVE` или `MANUAL_ACTIVE`.
7. Повторное Activate из AI_ACTIVE должно переводить в manual takeover, а не сразу в Park.
8. Полное отключение должно выполнить safe stop и Park-after-stop.

## Запуск двух камер

```powershell
python hardware\dual_camera_service.py --config config\real_cameras.json
```

GUI читает две ячейки:

- wide_90: `tcp://127.0.0.1:5556`;
- narrow_50: `tcp://127.0.0.1:5557`.

## Подключение real-agent adapter

По умолчанию агентный analyzer проверяет свежесть двух камер, а управление идет через waypoint PID fallback. Чтобы подключить builder входов модели:

```powershell
$env:MITSU_REAL_AGENT_FACTORY = "real_agent_adapters.lead_real_adapter:create_agent"
```

Если задан только `MITSU_REAL_AGENT_FACTORY`, adapter будет строить входы модели, но не делать inference, пока не задана конкретная модель:

```powershell
$env:MITSU_REAL_MODEL_FACTORY = "my_model_module:create_model"
```

`my_model_module:create_model(builder)` должен вернуть объект с одним из методов:

- `predict(input_tensors) -> dict | object`;
- `forward(data=input_tensors) -> object`;
- `__call__(input_tensors) -> object`.

Результат должен содержать:

```python
{"steer": -1.0, "throttle": 0.0, "brake": 0.0, "confidence": 1.0}
```

## Pose-provider

Временно можно передавать pose через файл:

```powershell
$env:MITSU_REAL_POSE_JSON = "config\current_pose.json"
python main.py
```

Ожидаемый JSON:

```json
{"x_m": 0.0, "y_m": 0.0, "yaw_deg": 0.0, "valid": true, "source": "rtk"}
```

Для реального движения этот файл должен обновляться внешним процессом RTK/GNSS+IMU, MCU odometry или visual odometry.

## Host-side dry-run guard

По умолчанию `RealSerialVehicleAdapter` не отправляет активные команды тяги/gear request, даже если GUI дошел до AI_ACTIVE. Это защита от случайного запуска.

Для настоящего actuation после bench-тестов нужно одновременно:

1. В `config/real_vehicle_safety.json` выставить:

```json
{"dry_run": false, "allow_real_actuation": true}
```

2. Перед запуском задать:

```powershell
$env:MITSU_REAL_ENABLE_ACTUATION = "1"
$env:MITSU_REAL_DRY_RUN = "0"
```

Без этих двух условий реальный serial path остается в безопасном host dry-run.

## Обязательный порядок реального тестирования

1. `TEST_MOCK_VEHICLE` — проверить state machine и маршрут.
2. `TEST_SERIAL_LOOPBACK` — проверить транспортный слой.
3. MCU bench, CAN output disabled — проверить получение пакетов.
4. Real COM telemetry-only — проверить heartbeat, speed, gear, steering, brake, pose.
5. Gear test без газа: brake hold -> D -> actual D -> stop -> brake hold -> P.
6. Steering/brake test на месте.
7. Первое движение на закрытой площадке: 1 км/ч, throttle clamp 5-8%, физический E-stop.

## Что нельзя считать закрытым программно

Нельзя автоматически подтвердить без железа:

- фактические CAN ID MCU;
- реальный feedback actual gear/speed/brake/steering;
- корректность переключения P/D/P;
- задержки исполнительных механизмов;
- соответствие sensor layout checkpoint-а двум физическим камерам;
- надежность RTK/GNSS/IMU pose.

Эти пункты должны быть подтверждены bench/dry-run тестами до первого движения.
