# Команды запуска

Эта страница дублирует основные команды запуска из `README.md` в формате короткой шпаргалки.
Выполняйте команды из корня репозитория.

## Подготовка окружения

Создайте и активируйте Conda-окружение:

```bash
pip install conda-lock
conda-lock install -n lead conda-lock.yml
conda activate lead
```

Установите Python-зависимости и пакет в editable-режиме:

```bash
pip install uv
uv pip install -r requirements.txt
uv pip install -e .
```

На Windows используйте:

```powershell
conda activate lead
pip install uv
uv pip install -r requirements-win.txt
uv pip install -e .
```

Опциональные инструменты разработки:

```bash
conda install -c conda-forge ffmpeg parallel tree gcc zip unzip
pre-commit install
```

## CARLA

Скопируйте `DefaultEngine.ini` в конфигурационный каталог CARLA:

```text
<CARLA_ROOT>/CarlaUE4/Config/DefaultEngine.ini
```

Типичные параметры запуска CARLA:

```text
-quality-level=Epic -world-port=2000 -nosound -carla-streaming-port=2001
```

## Оценка в CARLA

Запустите один closed-loop маршрут через wrapper:

```bash
python lead/leaderboard_wrapper.py \
  --checkpoint outputs/model_0011/config_dual_front_camera.json \
  --routes data/benchmark_routes/bench2drive/23687.xml \
  --bench2drive
```

Откройте dashboard нарушений:

```bash
python lead/infraction_webapp/app.py
```

Затем перейдите по адресу:

```text
http://localhost:5000/?dir=outputs%2Flocal_evaluation
```

## Обучение и подготовка данных

Постройте кэш данных:

```bash
python scripts/build_cache.py
```

Запустите pretraining:

```bash
python3 lead/training/train.py \
  logdir=outputs/local_training/pretrain
```

Запустите posttraining:

```bash
python3 lead/training/train.py \
  logdir=outputs/local_training/posttrain \
  load_file=outputs/local_training/pretrain/model.pth \
  use_planning_decoder=true
```

## GUI

Запустите GUI:

```bash
python "i-MiEV GUI/main.py"
```

