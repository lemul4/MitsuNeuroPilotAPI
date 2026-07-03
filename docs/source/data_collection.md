# Сбор данных

Это руководство описывает локальный сбор данных в CARLA с использованием LEAD.

```{note}
Этот материал предназначен для локальной разработки и отладки. Для крупномасштабного сбора данных на HPC-кластерах см. [руководство по SLURM](slurm_data_collection.md).
```

## Запуск локального сбора данных

1. Запустите сервер CARLA:

```bash
bash scripts/start_carla.sh
```

2. Запустите экспертного агента:

```bash
python -m lead --expert --routes data/data_routes/lead/noScenarios/short_route.xml
```

**Сбор данных в формате Py123D:**

Чтобы собрать данные в формате Py123D для совместимости между датасетами, задайте соответствующие переменные окружения перед запуском эксперта:

```bash
# Задать корень данных Py123D и конфигурацию эксперта
export PY123D_DATA_ROOT="data/carla_leaderboard2_py123d/"
export LEAD_EXPERT_CONFIG="target_dataset=6 py123d_data_format=true use_radars=false lidar_stack_size=2 save_only_non_ground_lidar=false save_lidar_only_inside_bev=false"

# Запустить эксперта с агентом py123d
python -m lead \
  --expert \
  --py123d \
  --routes data/data_routes/lead/noScenarios/short_route.xml
```

Формат Py123D предоставляет унифицированное представление данных, совместимое с другими крупными датасетами автономного вождения, что упрощает кросс-датасетное обучение и оценку.

## Проверка собранных данных

Проверяйте собранные данные с помощью ноутбука [notebooks/inspect_expert_output.ipynb](https://github.com/kesai-labs/lead/blob/main/notebooks/inspect_expert_output.ipynb).

## [Опционально] Обзор эксперта

```{note}
Следующие разделы необязательны и актуальны только если вы хотите понять или изменить поведение эксперта.
```

Эксперт сбора данных в [lead/expert/expert.py](https://github.com/kesai-labs/lead/blob/main/lead/expert/expert.py) работает в четыре шага:

1. Предлагает кратчайший путь, выполняя поиск по графу полос с помощью A\*
2. Дополняет этот путь, чтобы избегать столкновений со статическими препятствиями
3. Предлагает начальную целевую скорость с использованием Intelligent Driver Model (IDM)
4. Корректирует целевую скорость, чтобы избегать столкновений с динамическими опасностями

## [Опционально] Разбор реализации эксперта

```{note}
Эксперт не оптимален для вождения на уровне человека или для переноса политики в обученные модели. Мы рекомендуем экспериментировать с логикой и параметрами эксперта, если вы обнаружите неоптимальное поведение.
```

Основная логика вождения реализована в [lead/expert/expert.py](https://github.com/kesai-labs/lead/blob/main/lead/expert/expert.py),
вспомогательные функции и свойства находятся в [lead/expert/expert_base.py](https://github.com/kesai-labs/lead/blob/main/lead/expert/expert_base.py), а логика сбора данных — в [lead/expert/expert_data.py](https://github.com/kesai-labs/lead/blob/main/lead/expert/expert_data.py).

Точка входа — функция `run_step` в [lead/expert/expert.py](https://github.com/kesai-labs/lead/blob/main/lead/expert/expert.py), которая:

1. Собирает данные сенсоров, bounding boxes и метаданные
2. Выполняет логику вождения в `_get_control`

Эксперт обращается к внутренним состояниям симулятора CARLA для обработки различных сценариев. Центральный реестр состояний находится в [3rd_party/scenario_runner_autopilot/srunner/scenariomanager/carla_data_provider.py](https://github.com/kesai-labs/lead/blob/main/3rd_party/scenario_runner_autopilot/srunner/scenariomanager/carla_data_provider.py).

Чтобы понять регистрацию данных сценариев, изучите классы сценариев в [3rd_party/scenario_runner_autopilot/srunner/scenarios](https://github.com/kesai-labs/lead/tree/main/3rd_party/scenario_runner_autopilot/srunner/scenarios).

Мы поддерживаем пользовательские форки инструментов оценки CARLA с изменениями для доступа эксперта: [scenario_runner_autopilot](https://github.com/ln2697/scenario_runner_autopilot) и [leaderboard_autopilot](https://github.com/ln2697/leaderboard_autopilot).

## [Опционально] Оценка эксперта

Эксперта можно оценивать напрямую на бенчмарках Longest6 v2 и Town13.

Оценка Bench2Drive нетривиальна из-за значительных изменений кода для доступа к внутреннему состоянию. В нашей статье мы оценивали нижнюю границу качества эксперта, прогоняя маршруты Bench2Drive с помощью инструментов оценки эксперта. Маршрутам, где были только нарушения `MinSpeed`, присваивается оценка 100.
