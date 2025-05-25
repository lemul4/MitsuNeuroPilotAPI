# Запуск ./run_evaluator.ps1

# Устанавливаем переменные окружения
$env:LEADERBOARD_ROOT = "$PSScriptRoot\packages\leaderboard"
$env:TEAM_AGENT = "team_code\agents\imitation_agent.py"
$env:ROUTES = "$env:LEADERBOARD_ROOT\data\merged_routes.xml"
$env:ROUTES_SUBSET = "88" # номер сценария
$env:REPETITIONS = "1"
$env:DEBUG_CHALLENGE = "0"
$env:CHALLENGE_TRACK_CODENAME = "SENSORS"
$env:CHECKPOINT_ENDPOINT = "dataset\autopilot_behavior_data\results2.json"
$env:RECORD_PATH = ""
$env:RESUME = ""

  # --routes-subset=$env:ROUTES_SUBSET `
  # --repetitions=$env:REPETITIONS `
  # --record=$env:RECORD_PATH `
# Запускаем Python-скрипт
python "$env:LEADERBOARD_ROOT\leaderboard\leaderboard_evaluator.py" `
  --routes=$env:ROUTES `
  --track=$env:CHALLENGE_TRACK_CODENAME `
  --checkpoint=$env:CHECKPOINT_ENDPOINT `
  --agent=$env:TEAM_AGENT `
  --agent-config="" `
  --debug=$env:DEBUG_CHALLENGE `
  --resume=$env:RESUME
