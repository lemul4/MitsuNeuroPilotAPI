# Структура проекта

Эта страница дает подробный обзор структуры директорий и организации проекта LEAD.

## Основной Python-пакет (`lead/`)

### `lead/common/`
Базовые утилиты, используемые во всем проекте:
- Управление конфигурацией
- Утилиты логирования и визуализации
- Геометрические преобразования и системы координат
- Вспомогательные функции предобработки данных

### `lead/data_buckets/`
Управляет организацией данных в бакеты для эффективного обучения:
- Создание и валидация бакетов
- Стратегии сэмплирования данных
- Статистика и метаданные датасета

### `lead/data_loader/`
Pipeline загрузки данных PyTorch:
- Реализации датасетов для разных бенчмарков
- Стратегии аугментации данных
- Эффективная загрузка батчей и кэширование

### `lead/expert/`
Реализация экспертного водителя:
- Привилегированное планирование с ground truth
- Оркестрация сбора данных

### `lead/inference/`
Pipeline closed-loop оценки:
- Обертка для inference модели
- Предобработка данных сенсоров
- Вычисление метрик
- Визуализация и логирование

### `lead/webapp/`
Интерактивный веб-дашборд для анализа ошибок вождения:
- Воспроизведение видео проездов
- Подсветка и аннотирование нарушений

### `lead/tfv6/`
Архитектура модели TransFuser V6:
- Сети vision encoder (ResNet, RegNet)
- Извлечение признаков LiDAR/Radar
- Transformer для multi-modal fusion
- Головы предсказания траектории
- Слои выходного управления

### `lead/training/`
Pipeline обучения и утилиты:
- Обучающие модули PyTorch Lightning
- Loss functions и метрики
- Конфигурации optimizer и scheduler
- Checkpointing и логирование
- Поддержка распределенного обучения

## Верхнеуровневая структура

```
lead/                        # Основной Python-пакет
├── common/                  # Общие утилиты и helpers
├── data_buckets/            # Управление бакетами данных
├── data_loader/             # Загрузка данных PyTorch
├── expert/                  # Реализация экспертного водителя
├── inference/               # Closed-loop оценка
├── webapp/                  # Интерактивный просмотрщик нарушений
├── tfv6/                    # Архитектура модели TransFuser V6
└── training/                # Pipeline обучения

3rd_party/                   # Сторонние зависимости
├── alpamayo/                # Модель NVIDIA Alpamayo
├── Bench2Drive/             # Инструменты бенчмарка Bench2Drive
├── CARLA_0915/              # Симулятор CARLA
├── carla_route_generator/   # Инструменты генерации маршрутов
├── leaderboard/             # CARLA Leaderboard 2.0
├── leaderboard_autopilot/   # Leaderboard для autopilot
├── navsim_workspace/        # Оценка NAVSIM
├── scenario_runner/         # Выполнение сценариев CARLA
└── scenario_runner_autopilot/  # Scenario runner для autopilot

data/                        # Описания маршрутов и отладочные данные
├── benchmark_routes/        # Официальные маршруты бенчмарков
├── carla_leaderboard2/      # Данные сенсоров
├── data_routes/             # Маршруты для сбора обучающих данных
├── debug_routes/            # Отладочные маршруты
└── expert_debug/            # Отладочные данные экспертного водителя

docs/                        # Исходники документации
├── source/                  # Файлы документации Sphinx
├── build/                   # Собранная документация (генерируется)
└── assets/                  # Изображения и медиа

outputs/                     # Выходные данные обучения и оценки
├── checkpoints/             # Checkpoints моделей
├── evaluation/              # Результаты оценки через SLURM
├── local_evaluation/        # Результаты локальной оценки
├── local_training/          # Локальные результаты обучения
├── training/                # Результаты обучения через SLURM
├── training_viz/            # Визуализации обучения
└── visualization/           # Общие визуализации

scripts/                     # Утилитарные скрипты
├── generate_data.py            # Скрипт генерации данных
├── build_buckets_pretrain.py   # Бакеты данных для предобучения
├── build_buckets_posttrain.py  # Бакеты данных для постобучения
├── build_cache.py              # Генерация кэша данных
├── setup_carla.sh              # Установка CARLA
├── start_carla.sh              # Запуск CARLA
└── ...                         # Другие утилитарные скрипты

slurm/                       # Скрипты SLURM-задач
├── init.sh                  # Bash-конструктор SLURM-задачи
├── experiments/             # Описание задач
└── evaluation/              # Вспомогательные скрипты оценки

notebooks/                   # Jupyter-ноутбуки
├── carla_offline_inference.ipynb
├── data_format.ipynb
├── inspect_expert_output.ipynb
└── ...

tests/                       # Unit- и integration-тесты
```
