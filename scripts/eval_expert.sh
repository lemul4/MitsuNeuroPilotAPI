#!/bin/bash

# Enter data here
export ROUTES=data/data_routes/leaderboard1/BlockedIntersection/Town06_13.xml
# export LEAD_LOG_LEVEL="DEBUG"

# Set standard environment variables
export SCENARIO_NAME=$(basename $(dirname $ROUTES))
export ROUTE_NUMBER=$(basename $ROUTES .xml)
export PYTHONPATH=3rd_party/CARLA_0915/PythonAPI/carla:$PYTHONPATH
export PYTHONPATH=3rd_party/leaderboard_autopilot:$PYTHONPATH
export PYTHONPATH=3rd_party/scenario_runner_autopilot:$PYTHONPATH
export DEBUG_CHALLENGE=0
export TEAM_CONFIG=$ROUTES
export DATAGEN=0

# Expert config overrides:
# TargetDataset.CARLA_LEADERBOARD2_ONLY3CAMERAS == 2
# One-line toggles:
export SAVE_CAMERA_PC=False
export ENABLE_PERTURBATED_SENSORS=False
export SYNC_SENSOR_PROCESSING_WITH_SAVE_FREQ=True
export COMPUTE_CAMERA_PC=False
export COMPRESS_IMAGES=True
export LEAD_EXPERT_CONFIG="target_dataset=2 save_camera_pc=${SAVE_CAMERA_PC} perturbate_sensors=${ENABLE_PERTURBATED_SENSORS} sync_sensor_processing_with_data_save_freq=${SYNC_SENSOR_PROCESSING_WITH_SAVE_FREQ} compute_camera_pc=${COMPUTE_CAMERA_PC} compress_images=${COMPRESS_IMAGES}"

# Set paths for saving data and results
export SAVE_PATH=data/expert_debug/data/$SCENARIO_NAME
export CHECKPOINT_ENDPOINT=data/expert_debug/results/${ROUTE_NUMBER}_result.json



# Start the evaluation
python -u 3rd_party/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py \
    --host=172.30.96.1 \
    --port=2000 \
    --traffic-manager-port=8000 \
    --traffic-manager-seed=0 \
    --routes=$ROUTES \
    --repetitions=1 \
    --track=MAP \
    --checkpoint=${CHECKPOINT_ENDPOINT} \
    --agent=lead/expert/expert.py \
    --agent-config=$ROUTES \
    --debug=0 \
    --resume=1 \
    --timeout=60 &

PYTHON_PID=$!
wait $PYTHON_PID
