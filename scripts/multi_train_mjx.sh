#!/bin/bash
export PYTHONPATH=$(pwd)

# Define training script path
TRAIN_SCRIPT="src/scripts/mjx_train.py"

# Define task and environment parameters
TASK="locomotion"
ENV="humanoid_legs"
TERRAIN="flat"

# Video recording parameters
VIDEO_LENGTH=1000
VIDEO_INTERVAL=10000000

LOG_PROJECT_NAME="default_humanoid_legs_locomotion"

# Define hyperparameter arrays for sweeping
LEARNING_RATES=(1e-4 2e-4 3e-4)
ENTROPY_COSTS=(1e-2 2e-2 3e-2)
TRACKING_SIGMAS=(1.0 5.0 10.0 50.0 100.0)


# Convert arrays to comma-separated strings for Hydra
LEARNING_RATES_STR=$(IFS=,; echo "${LEARNING_RATES[*]}")
ENTROPY_COSTS_STR=$(IFS=,; echo "${ENTROPY_COSTS[*]}")
TRACKING_SIGMAS_STR=$(IFS=,; echo "${TRACKING_SIGMAS[*]}")

# Run the training script with Hydra parameters
python $TRAIN_SCRIPT \
    --video_length="$VIDEO_LENGTH" \
    --video_interval="$VIDEO_INTERVAL" \
    --task="$TASK" \
    --log_project_name="$LOG_PROJECT_NAME" \
    --video \
    -m \
    hydra.sweep.dir=multirun \
    hydra.sweep.subdir=\${agent.learning_rate}_\${agent.entropy_cost}_\${sim.rewards.tracking_sigma} \
    agent.learning_rate=$LEARNING_RATES_STR \
    agent.entropy_cost=$ENTROPY_COSTS_STR \
    sim.rewards.tracking_sigma=$TRACKING_SIGMAS_STR
