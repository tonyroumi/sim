from typing import Any
import jax
import mujoco
import numpy as np
import scipy
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from jax import numpy as jp
from mujoco import mjx
from mujoco.mjx._src import support  # type: ignore
from src.utils.file_utils import find_robot_file_path
from src.locomotion.default_humanoid_legs.config.ppo_config import MJXConfig

class Joystick(PipelineEnv):
    def __init__(
            self,
            name: str,
            robot: base.Robot,
            scene: str,
            cfg: MJXConfig,
            **kwargs: Any,
    ):
        """ Initalizes the environment with the specified configuration and robot parameters"""
        self.name = name
        self.robot = robot
        self.cfg = cfg

        xml_path = find_robot_file_path(robot.name, scene, '.xml')


        sys = mjcf.load(xml_path)
        sys = sys.tree_replace(
            {
                "opt.timestep": cfg.sim.timestep,
                "opt.solver": cfg.sim.solver,
                "opt.iterations": cfg.sim.iterations,
                "opt.ls_iterations": cfg.sim.ls_iterations,
            }
        )
        kwargs["n_frames"] = cfg.action.n_frames
        kwargs["backend"] = "mjx"

        super().__init__(sys, **kwargs)

    def reset(self, rng: jp.ndarray) -> State:
        """ Resets the environment to the initial state"""
        pass

    def step(self, state: State, action: jp.ndarray) -> State:
        """ Performs a step in the environment"""
        pass

    def _get_obs(
            self, data: mjx.Data, action: jp.ndarray,
    ) -> jp.ndarray:
        """ Returns the observation"""
        pass
        


        