""" Script to train RL agent with Brax"""

import argparse
import sys
import cli_args
import jax

#add argparse argumets
parser = argparse.ArgumentParser(description="Train a RL agent with Brax")
parser.add_argument("--video", type=bool, default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--disable_jit", type=bool, default=False, help="Disable JIT compilation")

cli_args.add_rl_args(parser)

args_cli, hydra_args = parser.parse_known_args()

#clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

import os
from src.locomotion import get_env_class

import time
import json
import hydra
import warnings
import functools
from typing import Any
from etils import epath
from absl import logging

from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

from omegaconf import OmegaConf, DictConfig
from orbax import checkpoint as ocp
from flax.training import orbax_utils

from src.config.config import Config 
from src.tools.rollouts import save_rollout
from src.robots.robot import Robot

import wandb

#Let's actually understand these flags?
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

# Suppress warnings

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    robot = Robot(cfg.robot.name)

    EnvClass = get_env_class(cfg.env.name)
    env_cfg = cfg.sim
    train_cfg = cfg.agent

    
    env = EnvClass(
        cfg.robot.name,
        robot,
        cfg.env.terrain, 
        env_cfg)
    
    eval_env = EnvClass(
        cfg.robot.name,
        robot,
        cfg.env.terrain,
        env_cfg)
    
    test_env = EnvClass(
        cfg.robot.name,
        robot,
        cfg.env.terrain,
        env_cfg
    )

    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=train_cfg.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=train_cfg.value_hidden_layer_sizes,
    )

    time_str = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{robot.name}_{cfg.env.name}_{cfg.agent.name}_{time_str}"

    logdir = epath.Path("logs").resolve() 

    logdir.mkdir(parents=True, exist_ok=True)
    print(f"Logs are being stored to {logdir}")

    wandb.init(
        project=args_cli.log_project_name,
        name=run_name,
        config=OmegaConf.to_container(train_cfg)
    )
    
    checkpoint_path = None
    if False:
        checkpoint_path = epath.Path(args_cli.checkpoint).resolve()
        print(f"Restoring from checkpoint: {checkpoint_path}") 
    else:
        print("No checkpoint provided. Training from scratch.")


    ckpt_path = logdir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint path: {ckpt_path}")

    #Save environment configuration
    with open(logdir / "train_config.json", "w") as f:
        json.dump(OmegaConf.to_container(train_cfg), f, indent=4)


    #Save robot configuration
    with open(logdir / "robot_config.json", "w") as f:
        json.dump(robot.config, f, indent=4)


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
    last_video_step = 0

    def progress(num_steps, metrics):
        nonlocal best_episode_reward, best_ckpt_step, last_ckpt_step, last_video_step

        times.append(time.time())
        wandb.log(metrics, step=num_steps)

        if args_cli.video and last_ckpt_step != 0 and (num_steps - last_video_step >= args_cli.video_interval):
            print(f"Saving rollout at step {last_ckpt_step}")
            current_ckpt_path = os.path.join(logdir, "checkpoints")
            current_policy_path = os.path.join(current_ckpt_path, f"{last_ckpt_step}", "policy")
            save_path = os.path.join(current_ckpt_path, f"{last_ckpt_step}.mp4")
            save_rollout(save_path, current_policy_path, test_env, make_networks_factory, args_cli.video_length)
            last_video_step = num_steps

        last_ckpt_step = num_steps

        episode_reward = float(metrics.get("episode_reward", 0))
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            best_ckpt_step = num_steps
        
        print(f"{num_steps}: {metrics['eval/episode_reward']}")
    
    try:
        make_policy, params, _ = train_fn(
            environment=env, eval_env=eval_env, progress_fn=progress
        )
    except KeyboardInterrupt:
        pass

    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
    print(f"best checkpoint step: {best_ckpt_step}")
    print(f"best episode reward: {best_episode_reward}")

    print(f"Saving rollout for best checkpoint")
    best_ckpt_path = os.path.join(logdir, "checkpoints", best_ckpt_step)
    best_policy_path = os.path.join(best_ckpt_path, f"{best_ckpt_step}", "policy")
    save_rollout(best_ckpt_path, best_policy_path, test_env, make_networks_factory, 1000)

if __name__ == "__main__":
    main()
