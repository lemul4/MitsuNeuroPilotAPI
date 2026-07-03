# Обучение через SLURM

(Возможно не работает)

```{note}
Это полностью опциональная функция. Интеграция со SLURM предназначена для пользователей с доступом к HPC-кластерам, которые хотят эффективно масштабировать эксперименты. Вся функциональность также может запускаться локально без SLURM.
```

После сбора датасета следующий шаг — обучение моделей. SLURM wrapper управляет предобучением, fine-tuning и multi-seed экспериментами с организованными выходными директориями и возобновлением из checkpoints.

## Полный training pipeline

Полный пример pipeline доступен в [slurm/experiments/001_example](https://github.com/kesai-labs/lead/tree/main/slurm/experiments/001_example) и демонстрирует workflow от начального обучения до моделей, готовых к оценке.

## Шаг 1: предобучение с нуля

Обучите perception backbone на собранных данных:

```bash
bash slurm/experiments/001_example/000_pretrain1_0.sh
```

Это запускает задачу предобучения с random seed `0` (на него указывает последняя цифра в имени скрипта). Выходные данные автоматически организуются:

```
outputs/training/001_example/000_pretrain1_0/<year><month><day>_<hour><minute><second>
```

Checkpoints, логи и метрики сохраняются в эту директорию для отслеживания прогресса обучения.

## Шаг 2: постобучение (fine-tuning)

После завершения предобучения выполните fine-tuning, добавив компоненты вроде planning decoder. Запустите постобучение с seed `2`:

```bash
bash slurm/experiments/001_example/012_postrain32_2.sh
```

Структура скрипта:

```{code-block} bash
:linenos:

#!/usr/bin/bash

source slurm/init.sh

export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG image_architecture=regnety_032 lidar_architecture=regnety_032"   # То же, что и в предобучении
export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG use_planning_decoder=true"                                       # Добавить planner сверху
posttrain outputs/training/001_example/000_pretrain1_0/251018_092144                                                # Загрузить предобученные веса

train --cpus-per-task=64 --partition=L40Sday --time=4-00:00:00 --gres=gpu:4
```

**Ключевые моменты**:
- **Строки 5–6**: конфигурация архитектуры модели, согласованная с предобучением плюс новые компоненты
- **Строка 7**: функция `posttrain` (определена в [slurm/init.sh](https://github.com/kesai-labs/lead/blob/main/slurm/init.sh)) помогает скрипту обучения загрузить предобученные веса
- **Строка 9**: параметры SLURM для задачи fine-tuning

## Обработка падений: автоматическое возобновление

Обучающие задачи падают из-за аппаратных сбоев, лимитов времени или OOM-ошибок. Для восстановления нужно добавить одну строку:

```{code-block} bash
:linenos:
:emphasize-lines: 8

#!/usr/bin/bash

source slurm/init.sh

export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG image_architecture=regnety_032 lidar_architecture=regnety_032"
export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG use_planning_decoder=true"
posttrain outputs/training/001_example/000_pretrain1_0/251018_092144
resume outputs/training/001_example/012_postrain32_2/251018_092144                                                  # Возобновить с этого checkpoint

train --cpus-per-task=64 --partition=L40Sday --time=4-00:00:00 --gres=gpu:4
```

Функция `resume` загружает последний checkpoint из указанной директории. Перезапустите обучение, выполнив тот же скрипт без изменений.

## Что дальше?

С обученными моделями переходите к:
- **[Оценке](slurm_evaluation.md)**: тестируйте модели на разных сценариях и измеряйте качество на бенчмарках
