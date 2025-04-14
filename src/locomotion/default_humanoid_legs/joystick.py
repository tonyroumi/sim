from dataclasses import asdict
from typing import Any, Callable, List
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

        # self._init_env()
        self._init_reward()

    def _init_reward(self) -> None:
        """Initializes the reward system by filtering and scaling reward components.

        This method processes the reward scales configuration by removing any components with a scale of zero and scaling the remaining components by a time factor. It then prepares a list of reward function names and their corresponding scales, which are stored for later use in reward computation. Additionally, it sets parameters related to health and tracking rewards.
        """
        reward_scale_dict = asdict(self.cfg.reward_scales)
        # Remove zero scales and multiply non-zero ones by dt
        for key in list(reward_scale_dict.keys()):
            if reward_scale_dict[key] == 0:
                reward_scale_dict.pop(key)

        # prepare list of functions
        self.reward_names = list(reward_scale_dict.keys())
        self.reward_functions: List[Callable[..., jax.Array]] = []
        self.reward_scales = jp.zeros(len(reward_scale_dict))
        for i, (name, scale) in enumerate(reward_scale_dict.items()):
            self.reward_functions.append(getattr(self, "_reward_" + name))
            self.reward_scales = self.reward_scales.at[i].set(scale)

        self.healthy_z_range = self.cfg.rewards.healthy_z_range
        self.tracking_sigma = self.cfg.rewards.tracking_sigma


    def reset(self, rng: jp.ndarray) -> State:
        """ Resets the environment to the initial state"""

        qpos = self._init_q #need to store 
        qvel = jp.zeros(self.nv)

        #Here we can include some randomizations like joint positions, velocities of base, and position of base and orientation.
        #not rn

        pipeline_state = self.pipeline_init(qpos, qvel) 

        # We can take either a phase based walking approach or a trajectory based walking approach. (ZMP)
        # Phase, freq=U(1.25, 1.5) : Gate cycle
        rng, key = jax.random.split(rng)
        gait_freq = jax.random.uniform(key, (1,), minval=1.25, maxval=1.5)
        phase_dt = 2 * jp.pi * self.dt * gait_freq #Change in phase per time step (how gait phase evolves over time)
        phase = jp.array([0, jp.pi]) #One leg starts at beginning of gait cycle, other starts at mid-gait cycle (natural alternation between two legs)

        #Generate a random command
        rng, cmd_rng = jax.random.split(rng)
        cmd = self.sample_command(cmd_rng) #[lin_vel_x, lin_vel_y, ang_vel_yaw]

        #can add a push if desired

        state_info = {
            "rng": rng,
            "step": 0,
            "command": cmd,
            "last_act": jp.zeros(self.num_actions),
            "last_last_act": jp.zeros(self.num_actions),

            #Phase related.
            "phase_dt": phase_dt,
            "phase": phase,



        }

        reward, done, zero = jp.zeros(3)

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = zero
        
        obs = None
        privileged_obs = None
        

        return State(
            pipeline_state, obs, privileged_obs, reward, done, metrics, state_info
        )

    def step(self, state: State, action: jp.ndarray) -> State:
        """ Performs a step in the environment"""
        pass

    def _get_obs(
            self, data: mjx.Data, action: jp.ndarray,
    ) -> jp.ndarray:
        """ Returns the observation"""
        pass

    def sample_command(self, rng: jax.Array) -> jax.Array:
        """ 
        Sample random commands with 10% chance of setting everything to 0. 
        Used for exploration, robust learning, and generalization. 
        """
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

        #x,y velocity should be between [-1, 1]
        lin_vel_x = jax.random.uniform(
            rng1, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            rng3,
            minval=self._config.ang_vel_yaw[0],
            maxval=self._config.ang_vel_yaw[1],
        )

        # With 10% chance, set everything to zero.
        return jp.where(
            jax.random.bernoulli(rng4, p=0.1),
            jp.zeros(3),
            jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw]),
        )
        


        