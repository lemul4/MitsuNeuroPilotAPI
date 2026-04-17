#!/bin/bash

# Enter data here
export ROUTES=/home/lemul/lead/data/data_routes/lead/SignalizedJunctionLeftTurn/route_002524.xml
# export LEAD_LOG_LEVEL="DEBUG"

# Set standard environment variables
export SCENARIO_NAME=$(basename $(dirname $ROUTES))
export ROUTE_NUMBER=$(basename $ROUTES .xml)
export PYTHONPATH=3rd_party/CARLA_0915/PythonAPI/carla:$PYTHONPATH
export PYTHONPATH=3rd_party/leaderboard_autopilot:$PYTHONPATH
export PYTHONPATH=3rd_party/scenario_runner_autopilot:$PYTHONPATH
export SCENARIO_RUNNER_ROOT=${SCENARIO_RUNNER_ROOT:-3rd_party/scenario_runner_autopilot}
export DEBUG_CHALLENGE=0
export TEAM_CONFIG=$ROUTES
export DATAGEN=1

# Expert config overrides:
# TargetDataset.CARLA_LEADERBOARD2_ONLY3CAMERAS == 2
# One-line toggles:
export SAVE_CAMERA_PC=False
export ENABLE_PERTURBATED_SENSORS=False
export SYNC_SENSOR_PROCESSING_WITH_SAVE_FREQ=True
export COMPUTE_CAMERA_PC=False
export COMPRESS_IMAGES=True
export CAMERA_LIDAR_SENSOR_TICK_FROM_DATA_SAVE_FREQ=False
# If true, disables speed reduction caused by bad visibility (night/rain/fog)
# while preserving all other visibility-related behavior (e.g., maneuver timing).
export DISABLE_SPEED_REDUCTION_BAD_VISIBILITY=True
# If false, min_speed_infractions are logged but ignored in score_penalty/score_composed.
export USE_MIN_SPEED_INFRACTIONS_IN_SCORE=False

export PY123D_DATA_FORMAT=False
export LEAD_EXPERT_CONFIG="target_dataset=2 py123d_data_format=${PY123D_DATA_FORMAT} save_legacy_outputs_with_py123d=${PY123D_DATA_FORMAT} use_radars=false lidar_stack_size=2 save_only_non_ground_lidar=false save_lidar_only_inside_bev=false save_camera_pc=${SAVE_CAMERA_PC} perturbate_sensors=${ENABLE_PERTURBATED_SENSORS} camera_lidar_sensor_tick_from_data_save_freq=${CAMERA_LIDAR_SENSOR_TICK_FROM_DATA_SAVE_FREQ} sync_sensor_processing_with_data_save_freq=${SYNC_SENSOR_PROCESSING_WITH_SAVE_FREQ} compute_camera_pc=${COMPUTE_CAMERA_PC} compress_images=${COMPRESS_IMAGES} disable_speed_reduction_bad_visibility=${DISABLE_SPEED_REDUCTION_BAD_VISIBILITY}"

if [[ "${PY123D_DATA_FORMAT,,}" == "true" ]]; then
    AGENT_MODULE="lead/expert/expert_py123d.py"
else
    AGENT_MODULE="lead/expert/expert.py"
fi

# Set paths for saving data and results
export SAVE_PATH=data/expert_debug/data/$SCENARIO_NAME
export CHECKPOINT_ENDPOINT=data/expert_debug/results/${ROUTE_NUMBER}_result.json

# CARLA host override support for WSL<->Windows setups.
# Priority:
# 1) CARLA_HOST env var
# 2) WSL default gateway
# 3) /etc/resolv.conf nameserver
# 4) fallback to 127.0.0.1
if [[ -z "${CARLA_HOST:-}" ]]; then
    CARLA_HOST=$(ip route 2>/dev/null | awk '/^default/ {print $3; exit}')
fi

if [[ -z "${CARLA_HOST:-}" && -f /etc/resolv.conf ]]; then
    CARLA_HOST=$(awk '/^nameserver / {print $2; exit}' /etc/resolv.conf)
fi

export CARLA_HOST=${CARLA_HOST:-127.0.0.1}
echo "Using CARLA_HOST=${CARLA_HOST}"
# 172.30.96.1
# Start the evaluation
python -u 3rd_party/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py \
    --host=${CARLA_HOST} \
    --port=2000 \
    --traffic-manager-port=8000 \
    --traffic-manager-seed=0 \
    --routes=$ROUTES \
    --repetitions=1 \
    --track=MAP \
    --checkpoint=${CHECKPOINT_ENDPOINT} \
    --agent=${AGENT_MODULE} \
    --agent-config=$ROUTES \
    --debug=0 \
    --resume=0 \
    --timeout=60 &

PYTHON_PID=$!
wait $PYTHON_PID
