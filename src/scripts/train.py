import argparse
from datetime import datetime
import time
import functools
import json
import os
from typing import Any
from etils import epath
from absl import app
from absl import flags
from absl import logging

import hydra
from hydra.core.config_store import ConfigStore

from src.locomotion.default_humanoid_legs.config.ppo_config import PPOConfig

cs = ConfigStore.instance()
cs.store(name="train_config", node=PPOConfig)

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")

import jax
import warnings
#For now, I'm using the brax library to load the environment
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

from omegaconf import OmegaConf
from orbax import checkpoint as ocp
from flax.training import orbax_utils

import wandb

env = envs.get_environment('humanoid')
eval_env = envs.get_environment('humanoid')
checkpoint = None
# robot = Robot(args.robot)

#train_cfg = ?
#env = 
#eval_env = 

now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")
run_name = f"humanoid-{timestamp}"

checkpoint_path = None

# Suppress warnings

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")

@hydra.main(config_path=None, config_name="train_config")
def train(
    train_cfg: PPOConfig,
):  
    """ Trains a reinforcement learning agent using Proximal Policy Optimization (PPO) 
    
    This function sets up the training environment, initializes configurations, and manages the training process.
    """
    

    #Change this to the hydra logging directory eventually
    logdir = epath.Path("logs").resolve() / run_name
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"Logs are being stored to {logdir}")

    wandb.init(
        project="MJX_RL",
        name=run_name,
        config=OmegaConf.to_container(train_cfg)
    )
    checkpoint = None

    if checkpoint is not None:
        checkpoint = epath.Path(checkpoint).resolve()
        print(f"Restoring from checkpoint: {checkpoint}")
    else:
        print("No checkpoint provided. Training from scratch.")


    ckpt_path = logdir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint path: {ckpt_path}")

    #Save environment configuration
    with open(ckpt_path / "config.json", "w") as f:
        json.dump(OmegaConf.to_container(train_cfg), f, indent=4)


    def policy_params_fn(current_step: int, make_policy: Any, params: Any):
        # save checkpoints
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        path = os.path.abspath(os.path.join(ckpt_path, f"{current_step}"))        
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)
        policy_path = os.path.join(path, "policy")

        model.save_params(policy_path, (params[0], params[1].policy))

    domain_randomize_fn = None
    #add domain randomization
    #can choose to randomize body mass, friction, sim and robot parameters
    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=train_cfg.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=train_cfg.value_hidden_layer_sizes,
    )

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=train_cfg.num_timesteps,
        num_evals=train_cfg.num_evals,
        episode_length=train_cfg.episode_length,
        unroll_length=train_cfg.unroll_length,
        num_minibatches=train_cfg.num_minibatches,
        num_updates_per_batch=train_cfg.num_updates_per_batch,
        discounting=train_cfg.discounting,
        learning_rate=train_cfg.learning_rate,
        entropy_cost=train_cfg.entropy_cost,
        clipping_epsilon=train_cfg.clipping_epsilon,
        num_envs=train_cfg.num_envs,
        batch_size=train_cfg.batch_size,
        seed=train_cfg.seed,
        network_factory=make_networks_factory,
        randomization_fn=domain_randomize_fn,
        policy_params_fn=policy_params_fn,
        restore_checkpoint_path=checkpoint_path,
    )

    times = [time.time()]
    
    last_ckpt_step = 0
    best_ckpt_step = 0
    best_episode_reward = -float("inf")

    def progress(num_steps, metrics):
        nonlocal best_episode_reward, best_ckpt_step, last_ckpt_step

        times.append(time.time())

        wandb.log(metrics, step=num_steps)

        last_ckpt_step = num_steps

        episode_reward = float(metrics.get("episode_reward", 0))
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            best_ckpt_step = num_steps
        
        print(f"{num_steps}: {metrics['eval/episode_reward']}")
    
    try:
        _, params, _ = train_fn(
            environment=env, eval_env=eval_env, progress_fn=progress
        )
    except KeyboardInterrupt:
        pass


    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
    print(f"best checkpoint step: {best_ckpt_step}")
    print(f"best episode reward: {best_episode_reward}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a policy using mjx")
    parser.add_argument(
        "--robot",
        type=str,
        help="The name of the robot. Must match the name in robots"
    )
    parser.add_argument(
        "--env",
        type=str,
        help="The name of the environment",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="The path to the checkpoint"
    )

    args = parser.parse_args()
    # env = envs.get_environment(args.env)
    # eval_env = envs.get_environment(args.env)
    # robot = Robot(args.robot)

    #train_cfg = ?
    #env = 
    #eval_env = 

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    experiment_name = f"{args.env}-{timestamp}"
    print(f"Experiment name: {experiment_name}")


    train(
    )