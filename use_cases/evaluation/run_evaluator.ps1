# Запуск ./run_evaluator.ps1

# Устанавливаем переменные окружения с учётом вложенной структуры
$env:LEADERBOARD_ROOT = "$PSScriptRoot\..\..\services\evaluation_service\leaderboard\leaderboard"
$env:PYTHONPATH = "$PSScriptRoot\..\..\.."
$env:TEAM_AGENT = "adapters.agents.agents.imitation_service.agent"
$env:ROUTES = "$env:LEADERBOARD_ROOT\..\data\merged_routes.xml"
$env:ROUTES_SUBSET = "88" # номер сценария
$env:REPETITIONS = "1"
$env:DEBUG_CHALLENGE = "0"
$env:CHALLENGE_TRACK_CODENAME = "SENSORS"
$env:CHECKPOINT_ENDPOINT = "$PSScriptRoot\..\..\data\autopilot_behavior_data\results2.json"
$env:RECORD_PATH = ""
$env:RESUME = ""

# Запускаем Python-скрипт
python "$env:LEADERBOARD_ROOT\leaderboard_evaluator.py" `
  --routes=$env:ROUTES `
  --track=$env:CHALLENGE_TRACK_CODENAME `
  --checkpoint=$env:CHECKPOINT_ENDPOINT `
  --agent=$env:TEAM_AGENT `
  --agent-config="" `
  --debug=$env:DEBUG_CHALLENGE `
  --resume=$env:RESUME
