# Обучающие Jupyter-ноутбуки

(возможно не работает)

Ноутбуки были протестированы в VSCode. Этот [файл настроек](https://github.com/kesai-labs/lead/blob/main/.vscode/settings.json) гарантирует, что ноутбуки запускаются с корнем проекта в качестве рабочей директории.

```{warning}
Если вы используете редактор, отличный от VSCode, убедитесь, что корневая директория Jupyter kernel совпадает с корнем проекта.
```

## Ноутбуки для проверки pipeline

### Проверка выхода эксперта

**Ноутбук:** [notebooks/inspect_expert_output.ipynb](https://github.com/kesai-labs/lead/blob/main/notebooks/inspect_expert_output.ipynb)

Запустите экспертного агента, создайте сэмплы данных и проверьте корректность работы.

### Загрузка предобученной модели и офлайн-инференс

**Ноутбук:** [notebooks/carla_offline_inference.ipynb](https://github.com/kesai-labs/lead/blob/main/notebooks/carla_offline_inference.ipynb)

Загрузите checkpoints модели, визуализируйте случайные сэмплы данных и запустите офлайн-инференс на собранных данных.

## Понимание формата данных

**Ноутбук:** [notebooks/data_format.ipynb](https://github.com/kesai-labs/lead/blob/main/notebooks/data_format.ipynb)

Изучите структуру выхода dataloader и узнайте, как адаптировать ее для пользовательских моделей.

## Отладка closed-loop оценки

**Ноутбук:** [notebooks/inspect_sensor_agent_io.ipynb](https://github.com/kesai-labs/lead/blob/main/notebooks/inspect_sensor_agent_io.ipynb)

Интерактивно отлаживайте входы и выходы модели во время closed-loop оценки, чтобы диагностировать проблемы.
