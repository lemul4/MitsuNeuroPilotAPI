# Запуск ./run_evaluator.ps1

# Устанавливаем переменные окружения
$env:LEADERBOARD_ROOT = "$PSScriptRoot\packages\leaderboard"
$env:TEAM_AGENT = "$env:LEADERBOARD_ROOT\leaderboard\autoagents\human_agent.py"
$env:ROUTES = "$env:LEADERBOARD_ROOT\data\routes_devtest.xml"
$env:ROUTES_SUBSET = "0"
$env:REPETITIONS = "1"
$env:DEBUG_CHALLENGE = "1"
$env:CHALLENGE_TRACK_CODENAME = "SENSORS"
$env:CHECKPOINT_ENDPOINT = "$env:LEADERBOARD_ROOT\results.json"
$env:RECORD_PATH = ""
$env:RESUME = ""

# Запускаем Python-скрипт
python "$env:LEADERBOARD_ROOT\leaderboard\leaderboard_evaluator.py" `
  --routes=$env:ROUTES `
  --routes-subset=$env:ROUTES_SUBSET `
  --repetitions=$env:REPETITIONS `
  --track=$env:CHALLENGE_TRACK_CODENAME `
  --checkpoint=$env:CHECKPOINT_ENDPOINT `
  --agent=$env:TEAM_AGENT `
  --agent-config="" `
  --debug=$env:DEBUG_CHALLENGE `
  --record=$env:RECORD_PATH `
  --resume=$env:RESUME
