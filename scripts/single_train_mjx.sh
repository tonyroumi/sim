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
VIDEO=true
VIDEO_LENGTH=1000
VIDEO_INTERVAL=10000000

# Checkpoint loading parameters
RESUME=false
LOAD_RUN=""
CHECKPOINT=""
LOG_PROJECT_NAME=""

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
    # --resume \
