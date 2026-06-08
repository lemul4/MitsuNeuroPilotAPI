# Отчет о тестировании программной системы MitsuNeuroPilot в режимах программной имитации и проверки транспортного слоя без физического автомобиля

Дата выполнения: 05.06.2026.

## 1. Условия тестирования

| Параметр | Значение |
|---|---|
| Дата тестирования | 05.06.2026 |
| ФИО тестировщика | не указано, рабочая учетная запись `sb170` |
| ОС | Майкрософт Windows 11 Домашняя для одного языка, версия 10.0.26200, build 26200 |
| Python version | Python 3.10.0 |
| Путь запуска проекта | `E:\основы программирования\MitsuNeuroPilotAPI\i-MiEV GUI` |
| Ветка Git | `autopilot_v3` |
| Commit hash | `ffa1e039 доработал навигатор` |
| Режим PyTorch | CPU |
| PyTorch | `2.5.0+cpu`; `torch.version.cuda = None`; `torch.cuda.is_available() = False` |
| CARLA доступна | Да: процесс CARLA 0.9.15 запущен, проектный Python API импортируется, `Town10HD` загружен через RPC |
| Физический автомобиль подключался | Нет |
| Используемый режим COM | Mock / Replay / Loopback / Dry-run, без физического исполнения команд |
| Используемые тестовые команды | `compileall`, `unittest`, `pytest`, `test_nav_ego_points.py`, CARLA API smoke |

Вывод обязательных команд:

```text
git log -1 --oneline
ffa1e039 доработал навигатор

python --version
Python 3.10.0

python -c "import sys; print(sys.executable)"
E:\основы программирования\MitsuNeuroPilotAPI\venv\Scripts\python.exe

python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
2.5.0+cpu
None
False

python -c "import cv2, PySide6, serial, serial_asyncio, zmq; print('imports ok')"
imports ok
```

## 2. Методика

Проверка выполнялась как программное, имитационное и стендовое тестирование без физического автомобиля. Для mock-режимов использовался `MockVehicleAdapter`. Для COM-контура использовались имитированные serial-сигналы, MCU/CAN-пакеты и шлюз команд, чтобы проверить транспортную логику без передачи управляющих команд на физический исполнительный контур.

Тесты проверяли:

- готовность системы к активации (`ReadinessStatus`);
- переходы состояний `CONNECTED_MANUAL`, `AI_PREVIEW`, `READY_TO_ARM`, `AI_ACTIVE`, `MANUAL_ACTIVE`, `DISENGAGING`;
- ручной перехват и ручные команды;
- dry-run защиту реального COM-адаптера;
- heartbeat по фактическим MCU-пакетам;
- приоритет тормоза над ускорением;
- преобразование маршрутных точек в локальную систему автомобиля;
- отсутствие регрессий в существующих unit-тестах.

## 3. Таблица тестовых сценариев

| ID | Подсистема | Сценарий | Метод проверки | Ожидаемый результат | Фактический результат | Статус | Артефакты |
|---|---|---|---|---|---|---|---|
| M-01 | Mock | Подключить `TEST_MOCK_VEHICLE` | Автотест `test_all_test_modes_seed_pose_and_do_not_require_physical_cameras` | Подключение без физического авто, телеметрия инициализируется | Подключение выполнено, pose seeded от первой точки миссии | Пройден | `test_mock_mode_matrix.py` |
| M-02 | Replay | Подключить `TEST_REPLAY_LOG` | Автотест с subtest | Режим запускается без COM-порта и физического авто | Режим классифицирован как тестовый, readiness проходит после AI Preview | Пройден | `test_mock_mode_matrix.py` |
| M-03 | Loopback | Подключить `TEST_SERIAL_LOOPBACK` | Автотест с subtest | Loopback доступен, активные команды не уходят в реальный контур | Loopback использует mock-адаптер, физический COM не задействован | Пройден | `test_mock_mode_matrix.py` |
| M-04 | Навигация | Задать маршрут A -> B | `test_ab_route_densifies_and_marks_turns`, `test_nav_ego_points.py` | Создается mission и набор маршрутных точек | Mission создается, prev/target/next формируются | Пройден | `test_navigation_core.py`, `test_nav_ego_points.py` |
| M-05 | AI Preview | Включить AI Preview | `test_ai_preview_without_activate_does_not_send_vehicle_command` | Preview не получает полномочий управления | Состояние `READY_TO_ARM`, команд в адаптер нет | Пройден | `test_mock_mode_matrix.py` |
| M-06 | Активация | Нажать Activate Control в mock-режиме | `test_mock_activation_requires_mission_and_ai_preview` | Переход в активное состояние только при readiness OK | После mission + AI Preview включается `AI_ACTIVE`, gear `D` | Пройден | `test_vehicle_control_core.py` |
| M-07 | Ручной перехват | Выполнить AI -> manual | `test_ai_deactivate_transfers_to_manual_without_parking` | Переход в `MANUAL_ACTIVE` без резкого Park | Gear остается `D`, ручные команды принимаются после takeover | Пройден | `test_vehicle_control_core.py`, `test_mock_mode_matrix.py` |
| M-08 | Отключение | Выполнить Disengage | `test_deactivate_parks_after_stop`, `test_deactivate_from_moving_mock_does_not_raise_and_parks` | Тяга обнуляется, торможение удерживается, Park после остановки | После остановки gear `P`, скорость ниже порога | Пройден | `test_vehicle_control_core.py` |
| N-01 | Ego-local | Проверить поля `previous_x_m`, `target_x_m`, `next_x_m` | `scripts\tools\test_nav_ego_points.py` | Поля присутствуют и идут в порядке маршрута | `previous=[5,0]`, `target=[10,0]`, `next=[15,0]` | Пройден | `test_nav_ego_points.py` |
| N-02 | Ego-local | Проверить систему координат автомобиля | `test_nav_ego_points.py` yaw=90 | `x` вперед, `y` боковое смещение | Для yaw=90 точки преобразованы в `[5,0]`, `[10,0]`, `[15,0]` | Пройден | `test_nav_ego_points.py` |
| N-03 | Ego-local | Проверить явные `target_point_*_ego` | `test_nav_ego_points.py` | Явные ego-точки имеют приоритет над world-точками | Явные точки переданы без повторного преобразования | Пройден | `test_nav_ego_points.py` |
| A-01 | AI Preview | Включить AI Preview без Activate Control | Автотест | Прогноз/адаптер не отправляет активную команду | `mock.last_command is None` | Пройден | `test_mock_mode_matrix.py` |
| A-02 | AI Preview | Подать устаревший intent модели | Автотест `test_stale_external_intent_falls_back_to_route_pid_in_ai_loop` | Intent отбрасывается или заменяется безопасным поведением | Использован резервный route PID | Пройден | `test_mock_mode_matrix.py` |
| A-03 | AI Preview | Отключить AI Preview | Автотест `test_disabling_ai_preview_clears_external_agent_influence` | Агент перестает влиять на контур | `external_agent_enabled=False`, external intent очищен | Пройден | `test_mock_mode_matrix.py` |
| A-04 | AI Preview | Preview без маршрута | Автотест readiness | Активное управление невозможно | Активация блокируется причиной `No valid mission` | Пройден | `test_mock_mode_matrix.py` |
| A-05 | AI Preview | Preview без валидной pose | Автотест readiness | Активное управление невозможно | Активация блокируется причиной `Pose is not valid` | Пройден | `test_mock_mode_matrix.py` |
| R-01 | Readiness | Нет маршрута | Автотест | Активация блокируется | `No valid mission` | Пройден | `test_mock_mode_matrix.py` |
| R-02 | Readiness | Pose невалидна | Автотест | Активация блокируется | `Pose is not valid` | Пройден | `test_mock_mode_matrix.py` |
| R-03 | Readiness | Камеры не готовы в COM-режиме | Автотест `test_real_com_readiness_blocks_when_camera_stream_is_not_ready` | Активация блокируется | `Camera stream is not ready` | Пройден | `test_com_mode_matrix.py` |
| R-04 | Readiness | Нет MCU heartbeat | Автотест COM readiness | Активация COM-контура блокируется | `vehicle_ok=False` до MCU-пакета | Пройден | `test_com_mode_matrix.py` |
| R-05 | Readiness | Не подтверждена передача Drive | Автотест `test_activation_fails_when_drive_feedback_is_not_confirmed` | `AI_ACTIVE` не включается | Состояние `FAULT`, причина `Gear D not confirmed` | Пройден | `test_mock_mode_matrix.py` |
| R-06 | Readiness | Агент не готов / AI Preview выключен | Автотест | `AI_ACTIVE` не включается | `AI Preview is not enabled` | Пройден | `test_mock_mode_matrix.py` |
| R-07 | Dry-run | dry-run включен | Автотест COM dry-run | Активная команда заменяется safe-stop | Причина команды `host_dry_run_guard` | Пройден | `test_com_mode_matrix.py` |
| H-01 | Manual | Ручная команда при `AI_ACTIVE` | Автотест manual matrix | Команда игнорируется до `MANUAL_ACTIVE` | `mock.last_command` не меняется | Пройден | `test_mock_mode_matrix.py` |
| H-02 | Manual | Повторное Activate/перехват AI | Existing + matrix tests | Выполняется manual takeover, не Park | `MANUAL_ACTIVE`, gear остается `D` | Пройден | `test_vehicle_control_core.py` |
| H-03 | Manual | Ручное ускорение в Park | Автотест manual matrix | Ускорение подавляется до нуля | `accel_pct=0` | Пройден | `test_mock_mode_matrix.py` |
| H-04 | Manual | Запрос Park при движении | Автотест manual matrix | Park откладывается, safe-stop | `manual_park_requested`, gear остается `D` | Пройден | `test_mock_mode_matrix.py` |
| H-05 | Manual | Ручная команда после takeover | Автотест manual matrix | Команда принимается как manual intent | Причина команды `manual`, ускорение > 0 | Пройден | `test_mock_mode_matrix.py` |
| C-01 | COM | COM-порт открыт, MCU-пакетов нет | Автотест `test_open_com_port_is_not_mcu_heartbeat_until_packet_arrives` | `connected=True`, `heartbeat_ok=False` | Условие выполнено | Пройден | `test_com_mode_matrix.py` |
| C-02 | COM | Получен валидный MCU/CAN-пакет скорости | Автотест | `heartbeat_ok=True`, скорость обновлена | `speed_kmh=36.0`, counters updated | Пройден | `test_com_mode_matrix.py` |
| C-03 | COM | Получен unmapped/короткий пакет | Автотест `test_unmapped_or_short_mcu_packet_does_not_update_mapped_telemetry` | Пакет не обновляет mapped telemetry | `mapped_packet_count=0`, скорость не изменилась | Пройден | `test_com_mode_matrix.py` |
| C-04 | COM | Нет heartbeat | Автотест readiness | Активация блокируется | До MCU-пакета `vehicle_ok=False` | Пройден | `test_com_mode_matrix.py` |
| C-05 | COM | Потеря heartbeat в активном режиме | Автотест `test_real_com_heartbeat_loss_in_ai_loop_results_in_dry_run_safe_stop` | Отправляется safe-stop | В dry-run шлюз получил `host_dry_run_guard`, brake >= 35 | Пройден | `test_com_mode_matrix.py` |
| C-06 | COM | `dry_run=True` и активная команда | Автотест | Команда заменяется safe-stop | Gear/accel не передаются в gateway | Пройден | `test_com_mode_matrix.py` |
| C-07 | COM | `allow_real_actuation=False` | Автотест dry-run | Активная команда не передается | Команда заменена dry-run safe-stop | Пройден | `test_com_mode_matrix.py` |
| C-08 | COM | `allow_real_actuation=True`, `dry_run=False` в имитации | Автотест | Команда проходит только в имитированный gateway | Sink получил исходную команду | Пройден | `test_com_mode_matrix.py` |
| G-01 | Gateway | `brake_pct > 0` и `accel_pct > 0` | Автотест gateway | Тормоз имеет приоритет, accel подавляется | Encoded accel = 0 | Пройден | `test_com_mode_matrix.py` |
| G-02 | Gateway | Safe-stop | Автотест `test_gateway_encodes_safe_stop_without_gear_or_accel` | Тяга 0, тормоз активен | Values `[40, 0, 0, 0]` | Пройден | `test_com_mode_matrix.py` |
| G-03 | Gateway | Gear D | Автотест gateway | Формируется команда Drive | Encoded gear value = `Gear.D.value` | Пройден | `test_com_mode_matrix.py` |
| G-04 | Gateway | Gear P при движении | Автотест manual safety | Park не отправляется до остановки | `manual_park_requested` вместо gear P | Пройден | `test_mock_mode_matrix.py` |
| G-05 | Gateway | Batch-отправка | Автотест `test_gateway_uses_latest_batch_sender_when_available` | Используется последний набор пакетов | Вызван `send_control_packet_set_latest` | Пройден | `test_com_mode_matrix.py` |
| G-06 | Gateway | Просроченная команда | Автотест `test_arbiter_replaces_expired_intent_with_safe_stop` | Команда заменяется safe-stop | Причина `stale_prediction` | Пройден | `test_com_mode_matrix.py` |
| G-07 | Gateway | CRC / счетчик пакета | Анализ слоя | В данном уровне не применимо: adapter/gateway получают уже сформированные объекты | CRC raw serial-пакета не проверялся в этих тестах | Не применимо | Ограничение уровня тестирования |
| V-01 | CARLA | Запустить/проверить GUI в режиме CARLA | RPC smoke + интерфейсный lifecycle | GUI/CARLA контур не требует real-COM | CARLA загружена на `Town10HD`; `AppController` route lifecycle работает без real-COM | Пройден | `test_carla_route_interface.py` |
| V-02 | CARLA | Выбрать CARLA route | Автотест `test_single_town10_route_starts_and_finishes_through_appcontroller` | Route подготавливается | Town10HD `route_001816.xml` подготовлен, стартовал через `Activate Control`, завершился штатно | Пройден | `test_carla_route_interface.py` |
| V-03 | CARLA | Проверить очередь CARLA route | Автотест `test_town10_route_queue_advances_from_first_to_second_route` | Очередь переходит от первого маршрута ко второму и завершается | `route_001816.xml -> route_001817.xml`, финал `2/2` | Пройден | `test_carla_route_interface.py` |
| V-04 | CARLA | Проверить CARLA-видеопоток | Не выполнялось | Камеры отображаются | Не проверялось в этом прогоне | Не выполнялся | Ограничение |
| V-05 | CARLA | Проверить телеметрию | Не выполнялось | Скорость и статус отображаются | Не проверялось в этом прогоне | Не выполнялся | Ограничение |
| V-06 | CARLA | Ошибка маршрута | Не выполнялось | Ошибка логируется, GUI не падает | Не проверялось в этом прогоне | Не выполнялся | Ограничение |

Важно: AI Preview не равен активному управлению. В проверке A-01 включение предварительного просмотра переводит систему в состояние готовности, но не вызывает отправку `VehicleCommand`.

## 4. Результаты автоматического прогона

| Команда | Количество тестов | Результат | Время выполнения | Комментарий |
|---|---:|---|---:|---|
| `python -m compileall -q "i-MiEV GUI" lead scripts` | не применимо | OK | 0.9 с | Ошибки компиляции отсутствуют |
| `python -m unittest tests.test_mock_mode_matrix tests.test_com_mode_matrix -v` | 20 | OK | 2.084 с | Целевая матрица mock/COM |
| `python -m unittest discover -s tests -p "test_*.py" -v` | 43 | OK | 14.718 с | Полный набор тестов интерфейса |
| `python -m pytest -q tests` | 43 + 3 subtests | OK | 20.34 с | Полный pytest-прогон |
| `python -m unittest discover -s real_pipeline_build/tests -p "test_*.py" -v` | 9 | OK | 8.545 с | Legacy-набор real pipeline |
| `python scripts\tools\test_nav_ego_points.py` | smoke | OK | 6.3 с | Проверка prev/target/next -> ego-local |
| CARLA API smoke | smoke | OK | 99.2 с | `load_world('Town10HD')` выполнен, текущий мир `Carla/Maps/Town10HD`, 23 актора |
| CARLA route interface smoke | 2 | OK | 0.023 с | Через методы `AppController` проверены одиночный Town10HD-маршрут и очередь из двух Town10HD-маршрутов |

Вывод `test_nav_ego_points.py`:

```text
goal:
  previous_world: 5.0 0.0
  target_world:   10.0 0.0
  next_world:     15.0 0.0
model points:
  previous: [5.0, 0.0]
  target:   [10.0, 0.0]
  next:     [15.0, 0.0]
OK: navigator prev/target/next are converted to ego-local model points.
```

CARLA API smoke:

```text
CARLA process: CarlaUE4, CarlaUE4-Win64-Shipping detected.
netstat: 0.0.0.0:2000 LISTENING by CarlaUE4-Win64-Shipping.
Python API source: E:\основы программирования\MitsuNeuroPilotAPI\venv\lib\site-packages\carla\__init__.py
Probe command: carla.Client('127.0.0.1', 2000), timeout=120.0 s.
Probe result: loaded Carla/Maps/Town10HD in 96.27 s; actors=23.
```

CARLA route interface smoke:

```text
Single route:
  route_001816.xml, town=Town10HD, scenario=Accident
  AppController prepared queue_mode=single
  Activate Control started LeadAgentThread config routes=route_001816.xml
  handle_agent_finished completed route and cleared queue

Route queue:
  route_001816.xml -> route_001817.xml, both town=Town10HD, scenario=Accident
  AppController prepared queue_mode=queue
  first finish advanced current_route_index from 0 to 1
  second finish completed queue: 2/2 routes
```

## 5. Выявленные дефекты

| ID дефекта | Подсистема | Описание | Шаги воспроизведения | Ожидаемое поведение | Фактическое поведение | Критичность |
|---|---|---|---|---|---|---|
| D-01 | COM / RealSerialVehicleAdapter | Факт открытия COM-порта мог считаться heartbeat автомобиля | Открыть COM-порт без MCU-пакетов | `connected=True`, но `heartbeat_ok=False` до валидного MCU-пакета | До исправления `heartbeat_ok` выставлялся при открытии порта | Высокая |

Статус D-01: исправлено в `vehicle_control/adapters.py`. Добавлен тест `test_open_com_port_is_not_mcu_heartbeat_until_packet_arrives`. Также добавлены диагностические счетчики `rx_packet_count`, `mapped_packet_count`, `last_packet_can_id`.

## 6. Ограничения проверки

Тестирование выполнялось без подключения физического автомобиля Mitsubishi i-MiEV. Не проверялись фактическое движение, физическое переключение передач, физическое ускорение, физическое торможение, физический поворот рулевого механизма, реальная обратная связь от MCU/CAN автомобиля и безопасность автономного движения на закрытой площадке.

COM-порт проверялся только как программный транспортный контур через mock serial manager, loopback и dry-run. Передача активных управляющих команд на физический исполнительный контур не выполнялась.

Скриншоты GUI не формировались в данном автоматическом прогоне. Вместо них приложены артефакты автотестов и диагностические выводы команд. Для ручного дипломного приложения можно дополнительно снять состояния GUI: `TEST_MOCK_VEHICLE`, `TEST_REPLAY_LOG`, `TEST_SERIAL_LOOPBACK`, `AI Preview`, `READY_TO_ARM`, `AI_ACTIVE`, `MANUAL_ACTIVE`, `DISENGAGING`, `activation blocked`, `COM dry-run`.

Проверка полноценного прохождения CARLA-маршрута с тяжелым модельным агентом, видеопотоком и телеметрией не выполнялась. На машине был запущен процесс CARLA 0.9.15, порт `2000` находился в состоянии `LISTENING`, проектный Python API CARLA импортировался из виртуального окружения, а RPC-запрос `load_world('Town10HD')` с таймаутом 180 секунд успешно загрузил мир `Carla/Maps/Town10HD`. Через интерфейсный слой `AppController` проверены подготовка, запуск и завершение одного Town10HD-маршрута, а также переход очереди из двух Town10HD-маршрутов. Сценарии V-04..V-06 вынесены в ограничения данного прогона, так как запуск реального видеопотока, телеметрии и ошибочного маршрута CARLA не выполнялся.

## 7. Итоговый вывод

Проведенное тестирование подтвердило корректность программной логики режимов программной имитации, предварительного просмотра ИИ, безопасной активации, ручного перехвата, dry-run защиты, навигации между точками A и B, преобразования маршрутных точек в локальную систему координат автомобиля и кодирования управляющих команд в условиях отсутствия физического автомобиля. Система может быть допущена к следующему этапу стендовой проверки с MCU/COM при сохранении dry-run защиты и без физического исполнения команд. Результаты не являются подтверждением готовности к автономному движению на реальном автомобиле.
