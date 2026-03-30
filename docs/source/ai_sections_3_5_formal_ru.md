# 3. Алгоритмы обучения моделей искусственного интеллекта

## 3.1. Процесс подготовки данных

### 3.1.1. Область применения и ограничения

В настоящем описании используется только классический контур сбора данных LEAD:

- агент-эксперт: `lead/expert/expert.py`;
- модуль сохранения и структуры датасета: `lead/expert/expert_data.py`.

### 3.1.2. Источник данных

Данные формируются в симуляторе CARLA при движении rule-based эксперта по заданным маршрутам.
Запуск сбора выполняется через wrapper:

1. запуск CARLA-сервера;
2. запуск `lead/leaderboard_wrapper.py` с флагом `--expert` без флага `--py123d`;
3. сохранение результатов в `outputs/expert_evaluation/...`.

### 3.1.3. Состав сохраняемых артефактов

В режиме expert/expert_data формируются:

- RGB-кадры (`rgb/`);
- LiDAR (`lidar/`);
- карты глубины (`depth/`) и семантики (`semantics/`) при включении соответствующих сенсоров;
- HDMap-представления (`hdmap/`);
- 3D боксы объектов (`bboxes/`);
- метаданные кадра (`metas/*.pkl`);
- итог маршрута (`results.json`).

### 3.1.4. Перенос в тренировочный корень

Для обучения TFv6 по CARLA используется корень `data/carla_leaderboard2/data`.
Рекомендуемый регламент:

1. собрать маршруты в `outputs/expert_evaluation`;
2. перенести/синхронизировать валидные маршруты в `data/carla_leaderboard2/data/<ScenarioType>/<RouteDir>`;
3. убедиться в наличии файлов `metas/0000.pkl`, `results.json`, `bboxes/*.json|*.pkl` (в зависимости от сохранения).

### 3.1.5. Фильтрация маршрутов (контроль качества)

При построении post-train buckets используется фильтрация `lead/data_buckets/route_filtering.py`.
Маршрут исключается из trainable-пула, если:

- отсутствует `results.json`;
- отсутствуют метаданные кадра;
- статус маршрута неуспешный (`Failed*`);
- есть существенные инфракции, кроме допустимых «мягких» случаев.

Допустимые случаи (маршрут оставляется):

- идеальный проезд (`score_composed=100`);
- только `min_speed_infractions`;
- completed-маршрут с инфракциями только из набора `min_speed_infractions` + `outside_route_lanes`.

Итого: в post-training обучается модель на подмножестве данных с более чистым поведенческим сигналом.

### 3.1.6. Построение buckets

Buckets строятся отдельно для фаз pretrain/posttrain:

- pretrain: `scripts/build_buckets_pretrain.py`;
- posttrain: `scripts/build_buckets_posttrain.py`.

Функции buckets:

- фиксируют trainable кадры и маршруты;
- задают схему выборки в DataLoader;
- исключают первые/последние кадры последовательности через `skip_first/skip_last`.

### 3.1.7. Построение кеша

Для ускорения I/O строится persistent cache:

- скрипт: `scripts/build_cache.py`;
- датасет с `build_cache=True`;
- кеш хранит предобработанные признаки сенсоров и снижает стоимость повторной загрузки.

Дополнительно при обучении может использоваться session cache (`diskcache`) на быстрых локальных дисках.

### 3.1.8. Формирование обучающего примера

Основной даталоадер CARLA: `lead/data_loader/carla_dataset.py`.

Для каждого индекса формируются:

1. метаданные сцены и ego-состояния;
2. целевые траектории:
   - `future_waypoints` (временная траектория),
   - `route` (пространственный путь),
   - `target_speed`;
3. тензоры сенсоров:
   - multi-camera RGB,
   - растеризованный LiDAR,
   - auxiliary GT для семантики/глубины/BEV/боксов;
4. статусные признаки:
   - скорость, команда, target points (current/previous/next),
   - дополнительные поля в зависимости от конфигурации.

### 3.1.9. Аугментации и perturbation

В обучении применяются:

- цветовые аугментации (`use_color_aug`, вероятностный режим);
- сенсорные perturbations (смещения/повороты) с вероятностью `use_sensor_perburtation_prob`;
- согласованное искажение меток траектории под perturbation.

Это повышает робастность к рассогласованию калибровки и шумам сенсоров.

---

## 3.2. Алгоритм обучения модели TFv6

### 3.2.1. Архитектурная схема

Модель `lead/tfv6/tfv6.py` состоит из:

1. Backbone: `TransfuserBackbone` (fusion image + LiDAR, опционально RADAR);
2. Perception heads (auxiliary):
   - semantic decoder,
   - depth decoder,
   - BEV semantic decoder,
   - CenterNet bbox decoder,
3. Planning head:
   - `PlanningDecoder` .

В post-training активируется `use_planning_decoder=true`.

### 3.2.2. Логика planning decoder (TFv6)

`PlanningDecoder` принимает:

- BEV признаки от backbone;
- статусные токены (speed, command, target points, etc.).

Контекст формируется `PlanningContextEncoder`, затем обрабатывается `TransformerDecoder`.
Декодируются:

- `route` (пространственные контрольные точки);
- `future_waypoints`;
- `target_speed_distribution` + декодированный `target_speed_scalar`.

Для route/waypoints используется накопление приращений через `cumsum`, что стабилизирует геометрию траектории.

Формально кодирование контекста на шаге $t$ задается как:

$$
z_t^{ctx}=C_{\psi}(s_t,q_t,h_t)
$$

где:

- $z_t^{ctx}\in\mathbb{R}^{D}$ (или стек токенов в $\mathbb{R}^{N\times D}$) — контекстное представление для planning decoder;
- $C_{\psi}(\cdot)$ — параметризованный энкодер контекста (набор линейных проекций + позиционные эмбеддинги + конкатенация токенов), параметры $\psi$ обучаются совместно с моделью;
- $s_t$ — текущее динамическое состояние ego-автомобиля;
- $q_t$ — навигационная команда в момент $t$;
- $h_t$ — история движения и дополнительные навигационные признаки.

Декомпозиция входов:

$$
s_t=[v_t,\ a_t],
\qquad
q_t=\text{onehot}(\text{command}_t),
\qquad
h_t=[tp_t,\ tp_{t-1},\ tp_{t+1},\ P_{t-K:t-1},\ V_{t-K:t-1}]
$$

где:

- $v_t$ — текущая скорость;
- $a_t$ — ускорение (если включено в конфиге);
- $tp_t, tp_{t-1}, tp_{t+1}\in\mathbb{R}^2$ — текущая, предыдущая и следующая target point;
- $P_{t-K:t-1}$ — набор прошлых позиций;
- $V_{t-K:t-1}$ — набор прошлых скоростей.

Нормализация выполняется до линейных проекций:

$$
	ilde{v}_t=\frac{v_t}{v_{max}},
\qquad
	ilde{a}_t=\frac{a_t}{a_{max}},
\qquad
\widetilde{tp}_t=tp_t\oslash c_{tp}
$$

где $c_{tp}$ — вектор констант нормализации target point, оператор $\oslash$ означает поэлементное деление.

Итоговый контекстный набор токенов формируется как объединение BEV-токенов и status-токенов:

$$
Z_t^{ctx}=\operatorname{Concat}\left(Z_t^{bev}+E_t^{bev},\ Z_t^{status}+E_t^{status}\right)
$$

где $E_t^{bev},E_t^{status}$ — позиционные эмбеддинги для BEV и статусных токенов соответственно.

### 3.2.3. Функции потерь

Итоговый loss формируется как взвешенная сумма задач:

$$
\mathcal{L}_{total}=\sum_{k\in\mathcal{K}}\tilde{w}_k\,\mathcal{L}_k,
\qquad
\sum_{k\in\mathcal{K}}\tilde{w}_k=1
$$

где:

- $\mathcal{K}$ — множество активных задач в текущем конфиге;
- $w_k$ — «сырые» веса из `detailed_loss_weights(...)`;
- $\tilde{w}_k$ — нормированные веса, используемые в backprop.

Нормализация выполняется в два шага:

$$
w_k^{(src)}=\text{detailed\_loss\_weights}(src),
\qquad
\hat{w}_k=\frac{w_k}{\sum_{j\in\mathcal{K}} w_j}
$$

и далее в trainer:

$$
\mathcal{L}_{total}=\sum_{k\in\mathcal{K}}\hat{w}_k\,\mathcal{L}_k
$$

### 3.2.3.1. Формулы планировочных потерь

Планировочные потери:

1. `loss_spatio_temporal_waypoints`:
   - L1 между предсказанными waypoint и GT;
   - при NavSim-режиме добавляется L1 по heading.

$$
\mathcal{L}_{wp}=
\frac{1}{BN}\sum_{b=1}^{B}\sum_{t=1}^{N}
\left\lVert \hat{\mathbf{p}}_{b,t}-\mathbf{p}_{b,t}\right\rVert_1
$$

Для NavSim:

$$
\mathcal{L}_{wp}^{navsim}=\mathcal{L}_{wp}+
\frac{1}{BN}\sum_{b=1}^{B}\sum_{t=1}^{N}
\left|\hat{\psi}_{b,t}-\psi_{b,t}\right|
$$

2. `loss_spatial_route`:
   - L1 (ADE-компонента) по route;
   - дополнительная L1 по последней точке (FDE-компонента).

$$
\mathcal{L}_{route}=
\underbrace{\frac{1}{BM}\sum_{b=1}^{B}\sum_{m=1}^{M}
\left\lVert \hat{\mathbf{r}}_{b,m}-\mathbf{r}_{b,m}\right\rVert_1}_{\text{ADE-like}}
+
\underbrace{\frac{1}{B}\sum_{b=1}^{B}
\left\lVert \hat{\mathbf{r}}_{b,M}-\mathbf{r}_{b,M}\right\rVert_1}_{\text{FDE-like}}
$$

3. `loss_target_speed`:
   - cross-entropy по two-hot распределению целевой скорости.

$$
\mathbf{y}_b=\text{two\_hot}(v_b),
\qquad
\mathbf{z}_b=\text{logits}_b,
\qquad
\mathcal{L}_{speed}=-\frac{1}{B}\sum_{b=1}^{B}\sum_{c=1}^{C} y_{b,c}\log\text{softmax}(\mathbf{z}_b)_c
$$

Декодирование скаляра скорости на инференсе:

$$
\hat{v}_b=\sum_{c=1}^{C} p_{b,c}\,s_c,
\qquad p_b=\text{softmax}(\mathbf{z}_b)
$$

где $s_c$ — значение speed-класса из `target_speed_classes`.

### 3.2.3.2. Perception-losses

Perception-losses (если включены):

- semantic/depth/BEV semantic;
- CenterNet-компоненты (`heatmap`, `wh`, `offset`, `yaw class/res`, `velocity`);
- `radar_loss`.

Итоговая мультитаск-цель:

$$
\mathcal{L}_{total}=
\widetilde{w}_{wp}\mathcal{L}_{wp}+
\widetilde{w}_{route}\mathcal{L}_{route}+
\widetilde{w}_{speed}\mathcal{L}_{speed}+
\sum_{q\in\mathcal{Q}_{perc}}\tilde{w}_q\mathcal{L}_q
$$

где $\mathcal{Q}_{perc}$ — множество активных perception-задач.

### 3.2.4. Двухфазный режим обучения

1. Pre-training (перцепция):
   - `use_planning_decoder=false`;
   - planning-losses обнуляются.

2. Post-training (end-to-end с planner):
   - загрузка checkpoint pretrain;
   - `use_planning_decoder=true`;
   - обучение всех нужных голов в едином графе.

### 3.2.5. Оптимизация

- Оптимизатор: AdamW (`amsgrad=True`, `fused=True`);
- регуляризация: `weight_decay`;
- scheduler:
  - CosineAnnealingWarmRestarts (по умолчанию для CARLA), либо
  - CosineAnnealingLR;
- mixed precision (BF16/FP16 по GPU);
- GradScaler при FP16;
- DDP при multi-GPU;
- optional ZeRO Redundancy Optimizer.

### 3.2.6. Псевдокод эпохи

1. Перемешать датасет и обновить sampler.
2. Пересчитать веса задач `detailed_loss_weights`.
3. Для каждого batch:
   - forward TFv6;
   - вычислить набор loss-компонент;
   - собрать взвешенный total loss;
   - backward (с AMP);
   - шаг optimizer/scaler/scheduler;
   - логирование loss/metrics/debug.
4. В конце эпохи:
   - optional оценка внешней метрики (например, RFM для Waymo);
   - сохранение checkpoint.

### 3.2.7. Детализация весов лоссов при обучении

Базовые «сырые» веса из `detailed_loss_weights` (для CARLA):

- `loss_semantic = 1.0`
- `loss_depth = 0.00001`
- `loss_bev_semantic = 1.0`
- `loss_center_net_heatmap = 1.0`
- `loss_center_net_wh = 1.0`
- `loss_center_net_offset = 1.0`
- `loss_center_net_yaw_class = 1.0`
- `loss_center_net_yaw_res = 1.0`
- `loss_center_net_velocity = 1.0` (обнуляется при `training_used_lidar_steps <= 1`)
- `radar_loss = 1.0` (обнуляется при `radar_detection=false`)
- `loss_spatio_temporal_waypoints = 1.0` (обнуляется в pretrain)
- `loss_target_speed = 1.0` (обнуляется в pretrain)
- `loss_spatial_route = 1.0` (обнуляется в pretrain)

Дополнительные правила обнуления в конфиге:

- при `use_semantic=false` обнуляется `loss_semantic`;
- при `use_depth=false` обнуляется `loss_depth`;
- при `use_bev_semantic=false` обнуляется `loss_bev_semantic`;
- при `detect_boxes=false` обнуляются все CenterNet-компоненты.

Для non-CARLA источников (NavSim/Waymo) вводится префикс ключей (`navsim_...`, `waymo_...`), а ряд CARLA-специфичных лоссов принудительно зануляется (например, `radar_loss`, `loss_depth`, `loss_semantic`, `loss_center_net_velocity`).

Нормированный вес каждой задачи вычисляется как:

$$
\widetilde{w}_k=\frac{w_k}{\sum_{j\in\mathcal{K}}w_j}
$$

#### Пример 1: CARLA pre-training (planner выключен)

При включенных semantic/BEV/CenterNet/radar и выключенном planner:

$$
\sum w = 9.00001
$$

Тогда для всех единичных задач:

$$
\widetilde{w}_{1.0}=\frac{1}{9.00001}\approx 0.111111
$$

Для depth:

$$
\widetilde{w}_{depth}=\frac{10^{-5}}{9.00001}\approx 1.111\times 10^{-6}
$$

#### Пример 2: CARLA post-training (planner включен)

При тех же условиях плюс 3 planning-loss:

$$
\sum w = 12.00001
$$

Единичные задачи получают:

$$
\widetilde{w}_{1.0}=\frac{1}{12.00001}\approx 0.083333
$$

А depth:

$$
\widetilde{w}_{depth}=\frac{10^{-5}}{12.00001}\approx 8.333\times 10^{-7}
$$

Практический вывод: depth выступает как слабый вспомогательный регуляризатор, а основной вклад в градиент дают единично-взвешенные perception/planning задачи.

---

## 3.3. Настраиваемые гиперпараметры

Ниже перечислены ключевые параметры (файл `lead/training/config_training.py`).

### 3.3.1. Оптимизация

- `lr` (базовый learning rate, по умолчанию 3e-4);
- `weight_decay` (по умолчанию 0.01);
- `epochs` (CARLA leaderboard режим: 31);
- `batch_size`;
- `use_cosine_annealing_with_restarts`;
- `grad_scaler_*` (init/growth/backoff/max).

### 3.3.2. Архитектура TFv6

- `image_architecture`, `lidar_architecture`;
- `transfuser_token_dim`;
- `transfuser_num_bev_cross_attention_layers`;
- `transfuser_num_bev_cross_attention_heads`;
- `n_layer`, `n_head` (внутренний transformer backbone-блок);
- dropout-параметры: `embd_pdrop`, `resid_pdrop`, `attn_pdrop`, `radar_dropout`.

### 3.3.3. Планировщик

- `use_planning_decoder`;
- `predict_temporal_spatial_waypoints`;
- `predict_spatial_path`;
- `predict_target_speed`;
- `num_way_points_prediction`;
- `num_route_points_prediction`;
- `target_speed_classes`;
- `smooth_route`, `num_route_points_smoothing`.

### 3.3.4. Данные и выборка

- `carla_root`, `carla_data`;
- `skip_first`, `skip_last`;
- `waypoints_spacing`;
- `use_persistent_cache`, `use_training_session_cache`;
- `force_rebuild_data_cache`, `force_rebuild_bucket`;
- `hold_out_town13_routes`;
- `randomize_route_order`.

### 3.3.5. Аугментации/робастность

- `use_color_aug`, `use_color_aug_prob`;
- `use_sensor_perburtation`, `use_sensor_perburtation_prob`;
- `training_used_lidar_steps`.

---

## 3.4. Методы валидации и регуляризации

### 3.4.1. Валидация

Для CARLA-контуров используются два уровня контроля:

1. Внутри тренировочного цикла:
   - контроль потерь и online-метрик planner:
     - `metric/route_ade`, `metric/route_fde`,
     - `metric/waypoints_ade`, `metric/waypoints_fde`,
     - `metric/target_speed_error`, `metric/target_speed_correlation`.

2. Внешняя closed-loop валидация:
   - Bench2Drive,
   - Longest6 v2,
   - Town13.

Метрики closed-loop берутся из JSON-отчетов эпизодов/маршрутов и агрегируются по набору маршрутов.

### 3.4.2. Регуляризация

В проекте реализованы:

- AdamW + weight decay;
- dropout в transformer/radar ветках;
- цветовые аугментации;
- perturbation сенсоров и согласованная коррекция GT;
- мультитаск-обучение (auxiliary losses), снижающее переобучение planner;
- curriculum через buckets и фильтрация некачественных экспертных маршрутов.

### 3.4.3. Практика контрольных срезов

Рекомендуется фиксировать:

- кривые train-loss и planner-метрик на каждой эпохе;
- промежуточную closed-loop проверку каждые 3-5 эпох post-training;
- отдельный контроль набора Town13 как стресс-теста обобщения.

---

## 3.5. Критерии остановки обучения

### 3.5.1. Реализованный критерий в коде

Основной критерий в текущем pipeline: достижение фиксированного числа эпох (`epochs`).

Для CARLA leaderboard режима: 31 эпоха.

### 3.5.2. Дополнительные регламентные критерии (рекомендуемые)

Для производственного регламента рекомендуется добавить раннюю остановку по внешним метрикам:

1. если улучшение Bench2Drive-score < 0.2 пункта за последние 3 проверки;
2. если route_FDE на тренировочных логах перестал снижаться статистически значимо;
3. если увеличивается доля инфракций при сопоставимом score;
4. если наблюдается дивергенция (рост loss, частые skipped steps по scaler).

Остановка фиксируется протоколом эксперимента с указанием выбранного чекпоинта.


# 4. Результаты обучения моделей искусственного интеллекта

## 4.1. Количественные результаты обучения

Ниже приведены опубликованные в репозитории результаты TFv6 (CARLA closed-loop):

| Конфигурация | Bench2Drive | Longest6 v2 | Town13 |
|---|---:|---:|---:|
| Full TFv6 (RegNetY032) | 95.2 | 62 | 5.24 |
| TFv6 ResNet34 | 94.7 | 57 | 5.01 |
| + Rear camera | 95.1 | 53 | TBD |
| No radar | 94.7 | 52 | TBD |
| Vision only | 91.6 | 43 | TBD |
| Town13 held-out training | 93.1 | 52 | 3.52 |

Источник: таблица в `README.md`.

## 4.2. Динамика обучения

Динамика контролируется в TensorBoard/WandB логах:

- unscaled/scaled losses по всем задачам;
- планировочные метрики (`route_ade/fde`, `waypoints_ade/fde`, `target_speed_error/correlation`);
- технические метрики (`lr`, `grad_scale`, `gradient_steps_skipped`, I/O time).

Обязательные графики в отчете:

1. `loss_spatio_temporal_waypoints` по эпохам;
2. `loss_spatial_route` по эпохам;
3. `loss_target_speed` по эпохам;
4. `metric/route_fde` и `metric/waypoints_fde`;
5. learning rate scheduler-кривая.

## 4.3. Сравнение конфигураций

Анализ таблицы показывает:

- переход к vision-only снижает качество closed-loop, подтверждая вклад LiDAR/RADAR;
- исключение RADAR умеренно ухудшает устойчивость (Longest6/Town13);
- добавление rear camera улучшает Bench2Drive, но влияет на Longest6 неоднозначно;
- hold-out Town13 меняет профиль обобщения: умеренное падение Bench2Drive, но улучшение специализированного стресс-теста.

## 4.4. Анализ сходимости и устойчивости

Признаки устойчивой сходимости:

1. монотонное/квазимонотонное снижение planner-loss на post-training;
2. отсутствие длительных плато с ростом FDE;
3. невысокая частота `gradient_steps_skipped`;
4. улучшение внешних closed-loop метрик без роста инфракций.

Признаки неустойчивости:

- сильные осцилляции route/waypoint FDE;
- деградация closed-loop score при уменьшении train-loss (переобучение на surrogate-задачи);
- рост доли торможений/излишне консервативного поведения при завышенном brake-бине target speed.


# 5. Алгоритмы применения моделей в программном средстве

## 5.1. Алгоритм использования обученной модели (инференс)

Используются компоненты:

- open-loop ансамблирование: `lead/inference/open_loop_inference.py`;
- closed-loop управление: `lead/inference/closed_loop_inference.py`;
- агент симулятора: `lead/inference/sensor_agent.py`.

### 5.1.1. Последовательность выполнения

1. Инициализация TFv6, загрузка одного или нескольких checkpoint файлов.
2. Получение сенсорного кадра из CARLA.
3. Препроцессинг входа в формат train-time:
   - формирование тензора multi-camera RGB,
   - растеризация LiDAR,
   - препроцессинг RADAR,
   - формирование target points/command/speed.
4. Прямой проход модели (или ансамбля моделей).
5. Постобработка предсказаний (см. 5.3).
6. Генерация управляющих команд (steer/throttle/brake).
7. Передача команды в vehicle control API.

## 5.2. Последовательность обработки входных данных

Входные данные приводятся к тем же каналам, что и в train:

- `rgb`: батч-сцепка фронтальных камер;
- `rasterized_lidar`: BEV-представление текущего и исторических LiDAR кадров;
- `radar` (опционально): объединенный массив радар-детекций;
- статусный вектор: `speed`, `command`, `target_point_previous/current/next`.

Ключевое требование: согласование препроцессинга с обучающим пайплайном (формат, нормализация, порядок камер, частота выборки).

## 5.3. Постобработка результатов

### 5.3.1. Ансамблирование

Open-loop ансамбль выполняет:

- усреднение logits target-speed и декодирование two-hot в скаляр;
- усреднение route и waypoints;
- NMS для боксов объектов;
- агрегирование semantic/depth/BEV карт.

### 5.3.2. Контроль торможения и масштаб скорости

После декодирования target-speed применяется:

- brake-threshold логика: если вероятность нулевого speed-бина выше порога, скорость принудительно 0;
- optional понижение target speed (`lower_target_speed_factor`).

### 5.3.3. Контуры безопасности

В `sensor_agent.py` используются post-processors:

- force-move (вывод из застревания);
- stop-sign post-processor;
- дополнительные эвристики для устойчивости управления.

## 5.4. Использование результатов модели в системе принятия решений

### 5.4.1. Два канала принятия решения

1. Route + target_speed канал:
   - латеральное управление по route (LateralPIDController),
   - продольное управление по target speed.

2. Waypoints канал:
   - вычисление desired speed из waypoint-геометрии,
   - PID для steer/throttle/brake.

Итоговые команды выбираются по конфигурации модальностей:

- `steer_modality` in {route, waypoint};
- `throttle_modality` in {target_speed, waypoint};
- `brake_modality` in {target_speed, waypoint}.

### 5.4.2. Финальная команда управления

На каждом такте формируется

$$
u_t = [\text{steer}_t,\text{throttle}_t,\text{brake}_t]
$$

с учетом:

- выбранной модальности,
- ограничений безопасности,
- условий низкой скорости и anti-stuck логики.


# Формализованные требования к оформлению (выполнение)

1. Однозначность терминов:
   - использовать фиксированные названия артефактов (`route`, `future_waypoints`, `target_speed`).
2. Трассируемость:
   - для каждого этапа указан исполняемый модуль/скрипт.
3. Воспроизводимость:
   - фиксировать конфиг обучения (`config.json`), seed, путь к checkpoint, версию датасета.
4. Проверяемость:
   - хранить TensorBoard/WandB логи и агрегированные closed-loop JSON-отчеты.
5. Сопоставимость экспериментов:
   - сравнение конфигураций проводить на одинаковом наборе маршрутов и одинаковом протоколе инференса.
