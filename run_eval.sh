#!/bin/bash

# --- Настройки ---

# Добавляем текущую директорию в PYTHONPATH.
# Это обязательно, чтобы Python нашел модули 'leaderboard' и 'team_code'.
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Переменные для удобного редактирования
HOST="172.24.80.1"
PORT="2000"
AGENT="team_code/sensor_agent.py"
AGENT_CONFIG="pretrained_models/1"
ROUTES="leaderboard/data/longest6.xml"
CHECKPOINT="./my_local_evaluation_results.json"

# --- Логирование ---
echo "=============================================="
echo "Запуск оценки агента (Leaderboard Local)"
echo "Host: $HOST:$PORT"
echo "Agent: $AGENT"
echo "Route: $ROUTES"
echo "=============================================="

# --- Запуск ---
# Используем "\" для переноса строк, чтобы код был читаемым
python leaderboard/leaderboard/leaderboard_evaluator_local.py \
    --host "$HOST" \
    --port "$PORT" \
    --agent "$AGENT" \
    --agent-config "$AGENT_CONFIG" \
    --routes "$ROUTES" \
    --checkpoint "$CHECKPOINT"