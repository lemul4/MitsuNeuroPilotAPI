# Enter data here
$env:ROUTES = "data/data_routes/leaderboard1/BlockedIntersection/Town06_13.xml"
# $env:LEAD_LOG_LEVEL = "DEBUG"

# Set standard environment variables
$env:SCENARIO_NAME = Split-Path (Split-Path $env:ROUTES -Parent) -Leaf
$env:ROUTE_NUMBER = [System.IO.Path]::GetFileNameWithoutExtension($env:ROUTES)

if ([string]::IsNullOrEmpty($env:PYTHONPATH)) {
    $env:PYTHONPATH = "3rd_party\CARLA_0915\PythonAPI\carla;3rd_party\leaderboard_autopilot;3rd_party\scenario_runner_autopilot"
} else {
    $env:PYTHONPATH = "3rd_party\CARLA_0915\PythonAPI\carla;3rd_party\leaderboard_autopilot;3rd_party\scenario_runner_autopilot;$env:PYTHONPATH"
}

if ([string]::IsNullOrEmpty($env:SCENARIO_RUNNER_ROOT)) {
    $env:SCENARIO_RUNNER_ROOT = "3rd_party\scenario_runner_autopilot"
}

$env:DEBUG_CHALLENGE = "0"
$env:TEAM_CONFIG = $env:ROUTES
$env:DATAGEN = "1"

# Expert config overrides:
# TargetDataset.CARLA_LEADERBOARD2_ONLY2CAMERAS == 3
# One-line toggles:
$env:SAVE_CAMERA_PC = "False"
$env:ENABLE_PERTURBATED_SENSORS = "False"
$env:SYNC_SENSOR_PROCESSING_WITH_SAVE_FREQ = "True"
$env:COMPUTE_CAMERA_PC = "False"
$env:COMPRESS_IMAGES = "True"
$env:CAMERA_LIDAR_SENSOR_TICK_FROM_DATA_SAVE_FREQ = "False"

$env:PY123D_DATA_FORMAT = "False"
$env:LEAD_EXPERT_CONFIG = "target_dataset=3 py123d_data_format=$($env:PY123D_DATA_FORMAT) save_legacy_outputs_with_py123d=$($env:PY123D_DATA_FORMAT) use_radars=false lidar_stack_size=2 save_only_non_ground_lidar=false save_lidar_only_inside_bev=false save_camera_pc=$($env:SAVE_CAMERA_PC) perturbate_sensors=$($env:ENABLE_PERTURBATED_SENSORS) camera_lidar_sensor_tick_from_data_save_freq=$($env:CAMERA_LIDAR_SENSOR_TICK_FROM_DATA_SAVE_FREQ) sync_sensor_processing_with_data_save_freq=$($env:SYNC_SENSOR_PROCESSING_WITH_SAVE_FREQ) compute_camera_pc=$($env:COMPUTE_CAMERA_PC) compress_images=$($env:COMPRESS_IMAGES)"

if ($env:PY123D_DATA_FORMAT.ToLower() -eq "true") {
    $env:AGENT_MODULE = "lead/expert/expert_py123d.py"
} else {
    $env:AGENT_MODULE = "lead/expert/expert.py"
}

# Set paths for saving data and results
$env:SAVE_PATH = "data/expert_debug/data/$($env:SCENARIO_NAME)"
$env:CHECKPOINT_ENDPOINT = "data/expert_debug/results/$($env:ROUTE_NUMBER)_result.json"

# Start the evaluation
python -u 3rd_party/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py `
    --host=localhost `
    --port=2000 `
    --traffic-manager-port=8000 `
    --traffic-manager-seed=0 `
    --routes=$env:ROUTES `
    --repetitions=1 `
    --track=MAP `
    --checkpoint=$env:CHECKPOINT_ENDPOINT `
    --agent=$env:AGENT_MODULE `
    --agent-config=$env:ROUTES `
    --debug=0 `
    --resume=0 `
    --timeout=600