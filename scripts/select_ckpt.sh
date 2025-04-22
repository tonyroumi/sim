#!/bin/bash
export PYTHONPATH=$(pwd)

# Define training script path
TRAIN_SCRIPT="src/scripts/mjx_train.py"

# Video recording parameters
VIDEO_LENGTH=1000
VIDEO_INTERVAL=50000000

#Task and environment will be determined by the checkpoint
CHECKPOINT_PATH="/home/anthony-roumi/Desktop/sim/multirun/0.0001_0.03_50.0/logs/checkpoints/92160000"

# Path to the config file
CONFIG_DIR=$(echo "$CHECKPOINT_PATH" | sed 's|/logs/checkpoints/.*||')
CONFIG_PATH="$CONFIG_DIR/.hydra"

OVERRIDES=(
    #Reward scales
    "sim.reward_scales.survival=1.0"
    "sim.reward_scales.lin_vel=2.5"
    "sim.reward_scales.ang_vel=1.0"
    "sim.reward_scales.torques=-2.5e-5"
    "sim.reward_scales.action_rate=-0.01"
    "sim.reward_scales.energy=0.0"
    "sim.reward_scales.feet_slip=-0.25"
    "sim.reward_scales.feet_clearance=0.0"
    "sim.reward_scales.feet_height=0.0"
    "sim.reward_scales.feet_phase=1.0"
    "sim.reward_scales.stand_still=-1.0"
    "sim.reward_scales.distance_traveled=0.0"
    "sim.reward_scales.pose=-0.5"
    "sim.reward_scales.orientation=-0.5"

    "sim.rewards.tracking_sigma=5.0"

    # #Agent confnig
    # "agent.learning_rate=1e-4"
    # "agent.entropy_cost=1e-2"


)

# Run the training script with Hydra parameters
python $TRAIN_SCRIPT \
    --video \
    --video_length="$VIDEO_LENGTH" \
    --video_interval="$VIDEO_INTERVAL" \
    --resume \
    --config-path="$CONFIG_DIR" \
    --config-name="config" \
    --checkpoint="$CHECKPOINT_PATH" \
    "${OVERRIDES[@]}"
    
 
