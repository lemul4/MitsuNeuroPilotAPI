# FAQ

Дополнительная документация:

- [scenario_runner](https://scenario-runner.readthedocs.io/en/latest/FAQ/)
- [carla](https://carla.readthedocs.io/en/latest/build_faq/)
- [leaderboard](https://leaderboard.carla.org/help/)

## При воспроизведении предоставленных checkpoints я получаю другие результаты

Результаты оценки CARLA различаются между запусками. Типичные вариации, которые мы наблюдаем: ~1–2 DS на Bench2Drive, ~5–7 DS на Longest6 v2, ~1.0 DS на Town13. Это эмпирические оценки; фактическая дисперсия зависит от конфигурации системы и случайности.

## Почему существует несколько версий `leaderboard` и `scenario_runner`?

Каждый бенчмарк требует собственного протокола оценки, а значит — отдельных форков этих репозиториев. Экспертный сборщик данных также использует собственный форк.

## Как создать больше маршрутов?

См. [carla_route_generator](https://github.com/autonomousvision/carla_route_generator) и [supplemental LEAD](https://ln2697.github.io/assets/pdf/Nguyen2026LEADSUPP.pdf).

## Где посмотреть изменения в `leaderboard` и `scenario_runner`?

Наши форки с изменениями:

- [scenario_runner_autopilot](https://github.com/ln2697/scenario_runner_autopilot)
- [leaderboard_autopilot](https://github.com/ln2697/leaderboard_autopilot)
- [Bench2Drive](https://github.com/ln2697/Bench2Drive)
- [scenario_runner](https://github.com/ln2697/scenario_runner)
- [leaderboard](https://github.com/ln2697/leaderboard)

## Какие версии TransFuser доступны?

См. [этот список](https://github.com/autonomousvision/carla_garage/blob/main/docs/history.md).

## Как часто CARLA не запускается?

Примерно 10% попыток запуска могут завершаться неудачно из-за зависаний при старте, конфликтов портов или проблем инициализации GPU. Это типичное поведение CARLA.

**Шаги восстановления:**

- Очистить zombie-процессы: `bash scripts/clean_carla.sh`
- Перезапустить: `bash scripts/start_carla.sh`
- Проверить, что порты 2000–2002 доступны
- Docker: `docker compose restart carla`

## Как добавить пользовательские сценарии?

См. [3rd_party/scenario_runner_autopilot/srunner/scenarios](https://github.com/kesai-labs/lead/tree/main/3rd_party/scenario_runner_autopilot/srunner/scenarios).

## Как эксперт получает доступ к данным, специфичным для сценария?

См. [3rd_party/scenario_runner_autopilot/srunner/scenariomanager/carla_data_provider.py](https://github.com/kesai-labs/lead/blob/main/3rd_party/scenario_runner_autopilot/srunner/scenariomanager/carla_data_provider.py).
