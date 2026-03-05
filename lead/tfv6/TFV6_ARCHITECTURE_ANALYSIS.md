# Подробный разбор архитектуры TFv6 (lead/tfv6)

## 1. Область анализа и что было разобрано

В этом документе разобрана текущая реализация в:

- `lead/tfv6/tfv6.py`
- `lead/tfv6/transfuser_backbone.py`
- `lead/tfv6/transfuser_utils.py`
- `lead/tfv6/planning_decoder.py`
- `lead/tfv6/tfv5_planning_decoder.py`
- `lead/tfv6/bev_decoder.py`
- `lead/tfv6/center_net_decoder.py`
- `lead/tfv6/perspective_decoder.py`
- `lead/tfv6/radar_detector.py`

И источники конфигурации, которые реально меняют состав модели:

- `lead/training/config_training.py`
- `lead/common/config_base.py`
- `lead/common/constants.py`
- `lead/data_loader/carla_dataset.py`

Фокус:

- Реальный граф модулей и поток тензоров.
- Как архитектура меняется в зависимости от флагов конфига.
- Поддерживается ли режим только с изображениями (без радара и лидара).

---

## 2. Высокоуровневый граф модели

Точка входа: `TFv6` в `lead/tfv6/tfv6.py`.

Базовый граф:

1. `TransfuserBackbone` создается всегда (`tfv6.py:37`).
2. Опциональные головы подключаются по конфигу:
   - Perspective semantic (`tfv6.py:39-48`)
   - Perspective depth (`tfv6.py:50-59`)
   - BEV semantic (варианты CARLA/NavSim) (`tfv6.py:61-76`)
   - CenterNet boxes (варианты CARLA/NavSim) (`tfv6.py:78-92`)
   - Radar detector (`tfv6.py:94-99`)
   - Planning decoder (`tfv6.py:101-113`)
3. Во время `forward` всегда выполняется:
   - backbone -> `(bev_features, image_features)` (`tfv6.py:125`)
   - `top_down(bev_features)` -> `bev_feature_grid` (`tfv6.py:161`)
     Это выполняется даже если головы detection/BEV отключены.
4. Вычисление лоссов модульное и условное по каждой голове (`tfv6.py:204-263`).

Важный момент дизайна:

- TFv6 модульный на уровне голов.
- Backbone в текущем коде не модульный по модальностям: внутри всегда есть image + lidar ветки.

---

## 3. Внутреннее устройство Backbone (TransfuserBackbone)

Файл: `lead/tfv6/transfuser_backbone.py`

### 3.1 Создание двух энкодеров

- Image encoder: TIMM-модель с `features_only=True` (`transfuser_backbone.py:34-36`).
- Lidar encoder: отдельная TIMM-модель с `features_only=True` (`transfuser_backbone.py:48-53`).
- Число входных каналов в lidar-ветку:
  - `2`, если `LTF=True` (`x/y` позиционная сетка).
  - `1`, если `LTF=False` (растр лидара в BEV).
  - См. `transfuser_backbone.py:51`.

### 3.2 Четыре стадии фьюжна

На каждой из 4 стадий (`transfuser_backbone.py:245-255`):

1. Проходим один блок энкодера image-ветки.
2. Проходим один блок энкодера lidar-ветки.
3. Сжимаем обе карты признаков adaptive pooling до якорной сетки.
4. Согласуем каналы через 1x1 адаптеры:
   - lidar->image (`transfuser_backbone.py:60-73`)
   - image->lidar (`transfuser_backbone.py:74-87`)
5. Выполняем фьюжн GPT-подобным трансформером (`transfuser_backbone.py:92-103`, `326+`).
6. Апсемплим fused-резидуалы обратно в разрешение каждой ветки и добавляем их (`transfuser_backbone.py:307-321`).

То есть это не поздний фьюжн. Это многомасштабный двунаправленный фьюжн на каждом уровне.

### 3.3 Токенизация в fusion GPT

`GPT.forward()` (`transfuser_backbone.py:393-441`):

- Image и lidar карты признаков разворачиваются в последовательности токенов.
- Последовательности конкатенируются.
- Добавляется обучаемое позиционное embedding `pos_emb` (`345-352`).
- Последовательность проходит через блоки трансформера (`355-366`).
- После этого токены снова разделяются на image/lidar карты признаков.

### 3.4 BEV top-down neck

`top_down()` (`transfuser_backbone.py:141-158`):

- Принимает только финальные признаки lidar-ветки.
- Строит BEV feature pyramid через апсемплинг и свертки.
- Возвращает BEV-сетку для:
  - CenterNet decoder
  - BEV semantic decoder

Следствие:

- Все BEV-heads структурно завязаны на признаки lidar-ветки.

### 3.5 Контракт входа в backbone

`forward()` (`transfuser_backbone.py:160-201`):

- Всегда читает `data["rgb"]`.
- Если `LTF=True`: создает синтетический 2-канальный lidar-вход из координат (`181-197`).
- Иначе: читает `data["rasterized_lidar"]` (`198-200`).

Даже при `LTF=True` в архитектуре остается lidar-ветка. Меняется только тип входа (псевдо-лидар вместо реального).

---

## 4. Варианты planning-модуля

Есть две реализации планировщика:

- Новый planning decoder: `lead/tfv6/planning_decoder.py`
- Legacy TFv5 style: `lead/tfv6/tfv5_planning_decoder.py`

Выбор задается через:

- `use_planning_decoder`
- `use_tfv5_planning_decoder`
- См. `tfv6.py:101-113`.

### 4.1 Новый PlanningDecoder (гибкий по query)

`planning_decoder.py:32-40` динамически считает число query:

- +`num_route_points_prediction`, если `predict_spatial_path`
- +`num_way_points_prediction`, если `predict_temporal_spatial_waypoints`
- +1, если `predict_target_speed`

Выходы также условные (`planning_decoder.py:140-165`):

- `route`
- `waypoints`
- `target_speed_dist` и декодированное скалярное значение
- опционально `headings` для NavSim

### 4.2 Состав токенов в PlanningContextEncoder

Контекстные токены собираются из:

- BEV-признаков (всегда, через 1x1 conv, `planning_decoder.py:466-468`, `608-610`)
- Опциональных статус-токенов по конфигу:
  - velocity (`400-406`)
  - acceleration (`407-413`)
  - command (`414-421`)
  - target point(s) (`423-435`, `543-573`)
  - past positions/speeds (`436-444`)
  - radar tokens (`446-457`, `575-597`)

Radar-токены добавляются только если одновременно истинны:

- `use_radars`
- `radar_detection`
- `use_radar_detection`

См. `planning_decoder.py:446-450` и `575-579`.

### 4.3 TFv5PlanningDecoder

Legacy-вариант:

- Фиксированная раскладка query (route + waypoints + speed) (`tfv5_planning_decoder.py:27-35`).
- Добавляет GRU-авторегрессию для route и waypoints (`50-63`, `121-125`).
- Более простой context encoder (velocity + command; radar-пути здесь нет).

---

## 5. Perception-головы

### 5.1 PerspectiveDecoder (semantic/depth)

Файл: `lead/tfv6/perspective_decoder.py`

- Deconv-подобный апсемплинг-стек из image features (`45-87`, `157-165`).
- Выход приводится к ожидаемым `final_image_height/final_image_width` с проверкой (`168-179`).
- Для depth выход сжимается до `(B,H,W)` (`181-182`).

### 5.2 BEVDecoder

Файл: `lead/tfv6/bev_decoder.py`

- Conv -> ReLU -> 1x1 class conv -> bilinear upsample (`34-57`).
- Используется маска видимости фрустума `valid_bev_pixels`, чтобы игнорировать невидимые пиксели в лоссе (`59-79`, `146-160`).
- Лосс маскируется по source-dataset в смешанном обучении (`97-130`).

### 5.3 CenterNetDecoder

Файл: `lead/tfv6/center_net_decoder.py`

- Multi-head dense prediction:
  - heatmap, wh, offset, yaw_class, yaw_res, опционально velocity (`50-63`)
- Лосс маскируется по source-dataset (`138-144`) и нормализуется по `avg_factor` валидных боксов (`174-203` и дальше).
- В инференсе есть декодирование top-k пиков heatmap и восстановление боксов (`324+`).

### 5.4 RadarDetector

Файл: `lead/tfv6/radar_detector.py`

- Входы:
  - BEV-признаки
  - radar points `data["radar"]`
  - скорость ego
- Пайплайн:
  1. Проекция BEV -> radar token space (`25`).
  2. Токенизация каждой radar-точки через sampled BEV features + rel velocity + sensor one-hot (`159-199`).
  3. Transformer decoder с обучаемыми query (`49-63`, `124`).
  4. Декодирование `(x,y,v)` + logit валидности (`126-156`).
- Лосс использует batch Hungarian matching (`221-234`, `297-312`).

---

## 6. Матрица модульности по конфигу

Ниже ключевые переключатели, которые меняют состав/поведение TFv6.

| Флаг конфига | Эффект в архитектуре |
|---|---|
| `use_planning_decoder` | Создает planner head (`tfv6.py:101`) и planner-loss (`tfv6.py:258`) |
| `use_tfv5_planning_decoder` | Переключает реализацию planner (`tfv6.py:102-113`) |
| `predict_spatial_path` | Добавляет route queries/decoder (`planning_decoder.py:34-35`, `61-63`) |
| `predict_temporal_spatial_waypoints` | Добавляет waypoint queries/decoder (`36-37`, `63-67`) |
| `predict_target_speed` | Добавляет speed query/decoder (`38-39`, `67-78`) |
| `use_semantic` + `use_carla_data` | Включает perspective semantic head (`tfv6.py:39-48`) |
| `use_depth` + `use_carla_data` | Включает perspective depth head (`tfv6.py:50-59`) |
| `use_bev_semantic` | Включает BEV semantic head(ы) (`tfv6.py:61-76`) |
| `detect_boxes` | Включает CenterNet head(ы) (`tfv6.py:78-92`) |
| `radar_detection` + `use_carla_data` | Включает radar detector модуль и radar-loss (`tfv6.py:94-99`, `249-255`) |
| `use_radar_detection` | Управляет передачей radar-выходов в planner (`tfv6.py:136-137`) |
| `LTF` | Заменяет реальный rasterized lidar вход синтетической позиционной сеткой (`transfuser_backbone.py:181-197`) |
| `image_architecture` / `lidar_architecture` | Выбирает TIMM backbone для каждой ветки (`transfuser_backbone.py:34-35`, `49-52`) |
| `use_carla_data` / `use_navsim_data` | Создает dataset-specific decoder’ы и source-masked loss’ы (`tfv6.py:61-92`, decoder losses) |

---

## 7. Поведение с разными источниками данных

TFv6 поддерживает смешанные батчи из разных источников (CARLA/NavSim/Waymo в training pipeline), и каждый лосс головы маскируется по `source_dataset`:

- BEV decoder: `bev_decoder.py:97-130`
- CenterNet decoder: `center_net_decoder.py:138-144` и дальше
- Perspective decoder: `perspective_decoder.py:107-145`

Это позволяет держать несколько голов в одной модели, где каждый sample влияет только на релевантные лоссы.
