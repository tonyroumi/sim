#!/bin/bash
export PYTHONPATH=$(pwd)

# Define training script path
TRAIN_SCRIPT="src/scripts/mjx_train.py"

# Define task and environment parameters
TASK="locomotion"
ENV="humanoid_legs"
TERRAIN="flat"

SEED=0

# Video recording parameters
VIDEO=false
VIDEO_LENGTH=1000
VIDEO_INTERVAL=10000000

# Checkpoint loading parameters
RESUME=false
LOAD_RUN=""
CHECKPOINT=""
LOG_PROJECT_NAME=""

# Define hyperparameter arrays for sweeping
LEARNING_RATES=(1e-4 2e-4 3e-4)
ENTROPY_COSTS=(1e-2 2e-2 3e-2)

# Convert arrays to comma-separated strings for Hydra
LEARNING_RATES_STR=$(IFS=,; echo "${LEARNING_RATES[*]}")
ENTROPY_COSTS_STR=$(IFS=,; echo "${ENTROPY_COSTS[*]}")

# Run the training script with Hydra parameters
python $TRAIN_SCRIPT \
    --video_length="$VIDEO_LENGTH" \
    --video_interval="$VIDEO_INTERVAL" \
    --task="$TASK" \
    --seed="$SEED" \
    --load_run="$LOAD_RUN" \
    --checkpoint="$CHECKPOINT" \
    --log_project_name="$LOG_PROJECT_NAME" \
    --video \
    -m \
    hydra.sweep.dir=multirun \
    hydra.sweep.subdir=\${agent.learning_rate}_\${agent.entropy_cost} \
    agent.learning_rate=$LEARNING_RATES_STR \
    agent.entropy_cost=$ENTROPY_COSTS_STR
    # --resume \
