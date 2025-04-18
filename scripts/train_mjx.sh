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

# Run the training script with Hydra parameters
python $TRAIN_SCRIPT \
    task=$TASK \
    +env=$ENV \
    env.terrain="$TERRAIN" \
    +agent=$AGENT_TYPE \
    agent.num_evals=$NUM_EVALS \
    agent.episode_length=$EPISODE_LENGTH \
    agent.unroll_length=$UNROLL_LENGTH \
    agent.num_minibatches=$NUM_MINIBATCHES \
    agent.num_updates_per_batch=$NUM_UPDATES_PER_BATCH \
    agent.discounting=$DISCOUNTING \
    agent.learning_rate=$LEARNING_RATE \
    agent.entropy_cost=$ENTROPY_COST \
    agent.clipping_epsilon=$CLIPPING_EPSILON \
    agent.num_envs=$NUM_ENVS \
    agent.batch_size=$BATCH_SIZE \
    agent.seed=$SEED \
    agent.render_interval=$RENDER_INTERVAL \
    agent.normalize_observations=$NORMALIZE_OBS \
    agent.action_repeat=$ACTION_REPEAT \
    agent.max_grad_norm=$MAX_GRAD_NORM \
    agent.policy_hidden_layer_sizes="$POLICY_LAYERS" \
    agent.value_hidden_layer_sizes="$VALUE_LAYERS" \
    agent.num_resets_per_eval=$NUM_RESETS_PER_EVAL

    # # Robot configuration. INCLUDE KP PARAMS AND OTHER THINGS.
    # robot=humanoid_legs \
    
    # # Simulation configuration (MJX)
    # sim=mjx \
    # # Random seed and other global settings
    # seed=42 \