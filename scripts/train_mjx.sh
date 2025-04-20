#!/bin/bash
export PYTHONPATH=$(pwd)

# Define training script path
TRAIN_SCRIPT="src/scripts/mjx_train.py"

# Define task and environment parameters
TASK="locomotion"
ENV="humanoid_legs"
TERRAIN="flat"

# Define agent parameters
AGENT_TYPE="ppo"
NUM_EVALS=1000
EPISODE_LENGTH=1000
UNROLL_LENGTH=20
NUM_MINIBATCHES=4
NUM_UPDATES_PER_BATCH=4
DISCOUNTING=0.97
LEARNING_RATE=1e-4
ENTROPY_COST=5e-4
CLIPPING_EPSILON=0.2
NUM_ENVS=1024
BATCH_SIZE=256
SEED=42
RENDER_INTERVAL=50
NORMALIZE_OBS=true
ACTION_REPEAT=1.0
MAX_GRAD_NORM=1.0
POLICY_LAYERS="[512,256,128]"
VALUE_LAYERS="[512,256,128]"
NUM_RESETS_PER_EVAL=1

# Video recording parameters
VIDEO=true
VIDEO_LENGTH=10000000
VIDEO_INTERVAL=100000

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
