# Оценка через SLURM

(Возможно не работает)

```{note}
Это полностью опциональная функция. Интеграция со SLURM предназначена для пользователей с доступом к HPC-кластерам, которые хотят эффективно масштабировать эксперименты. Вся функциональность также может запускаться локально без SLURM.
```

После обучения моделей следующий шаг — систематическая оценка на сценариях бенчмарков. SLURM wrapper выполняет batch-оценку с автоматическим распределением маршрутов и отслеживанием прогресса.

## Обзор

Полный пример pipeline доступен в [slurm/experiments/001_example](https://github.com/kesai-labs/lead/tree/main/slurm/experiments/001_example) и демонстрирует workflow обучения и оценки.

Для надежных результатов обучите три модели с разными random seed и оцените каждую один раз на Bench2Drive и Longest6 v2. Для большей уверенности оцените каждый seed три раза.

## Запуск оценки

Пример скрипта оценки [slurm/experiments/001_example/020_b2d_0.sh](https://github.com/kesai-labs/lead/blob/main/slurm/experiments/001_example/020_b2d_0.sh):

```{code-block} bash
:linenos:

#!/usr/bin/bash

source slurm/init.sh

export CHECKPOINT_DIR=outputs/training/001_example/010_postrain32_0/251025_182327  # Указывает на checkpoint обученной модели
export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG"                                # Переопределить параметры обучения во время теста (редко нужно)
export LEAD_CLOSED_LOOP_CONFIG="$LEAD_CLOSED_LOOP_CONFIG"                          # Переопределить параметры closed-loop оценки

evaluate_bench2drive220
```

**Разбор по строкам**:

- **Строка 5**: задает директорию checkpoint из обучения
- **Строка 6**: опциональное переопределение конфигурации обучения во время теста (используется редко)
- **Строка 7**: опциональное переопределение параметров closed-loop оценки (например, настроек симуляции)
- **Строка 9**: запускает оценку через функцию `evaluate_bench2drive220`, которая стартует сессию `screen` и распределяет SLURM-задачи для каждого маршрута

Runtime-параметры оценки настраиваются в [slurm/configs](https://github.com/kesai-labs/lead/tree/main/slurm/configs).

## Отслеживание результатов

Для метрик оценки поддерживается логирование WandB:

![](../assets/eval_wandb.png)

Это дает отслеживание завершения маршрутов, success rate и метрик качества в реальном времени на протяжении запуска оценки.
