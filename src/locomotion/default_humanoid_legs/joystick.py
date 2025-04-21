from dataclasses import asdict
from typing import Any, Callable, List
import jax
import mujoco
import numpy as np
from omegaconf import OmegaConf
import scipy
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from jax import numpy as jp
from mujoco import mjx
from mujoco.mjx._src import support  # type: ignore
from src.utils.file_utils import find_robot_file_path
from src.utils.math_utils import quat2euler, quat_inv, rotate_vec
from src.rewards import get_reward_function
from src.tools.collision import check_feet_contact
from src.locomotion.default_humanoid_legs.base import DefaultHumanoidEnv


class Joystick(DefaultHumanoidEnv):
    def __init__(
            self,
            name: str,
            robot: any,
            scene: str,
            cfg,
            **kwargs: Any,
    ):
        """ Initalizes the environment with the specified configuration and robot parameters"""

        super().__init__(name, robot, scene, cfg, **kwargs)

        self._init_env()
        self._init_reward()

    def _init_env(self) -> None:
        """Initializes the environment by setting up the system and its components."""

        self.init_q = jp.array(self.sys.mj_model.keyframe("home").qpos)
        self.default_pose = jp.array(self.sys.mj_model.keyframe("home").qpos)[7:]

        self.nu = self.sys.nu
        self.nq = self.sys.nq
        self.nv = self.sys.nv

        feet_link_mask = jp.array(
            np.char.find(self.sys.link_names, "foot") >= 0
        )
        self.feet_link_ids = jp.arange(self.sys.num_links())[feet_link_mask]

        self._weights = jp.array([
        1.0, 1.0, 0.01, 0.01, 1.0, 1.0,  # left leg.
        1.0, 1.0, 0.01, 0.01, 1.0, 1.0,  # right leg.
    ])

        foot_linvel_sensor_adr = []
        for site in ["l_foot", "r_foot"]:
            sensor_id = self.sys.mj_model.sensor(f"{site}_global_linvel").id
            sensor_adr = self.sys.mj_model.sensor_adr[sensor_id]
            sensor_dim = self.sys.mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(
                list(range(sensor_adr, sensor_adr + sensor_dim))
            )
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)
        
        self.feet_body_ids = jp.array(
            [
                support.name2id(self.sys, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in ["foot_left", "foot_right"]
            ]
        )
        self.torso_body_id = support.name2id(self.sys, mujoco.mjtObj.mjOBJ_BODY, "torso")

        self.torso_sensor_id = support.name2id(self.sys, mujoco.mjtObj.mjOBJ_SENSOR, "torso")

        self.motor_indices = jp.array(
            [
                support.name2id(self.sys, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.motor_ordering
            ]
        )
        
        self.obs_noise_scale = self.cfg.noise.obs_noise_scale * jp.concatenate(
            [
                jp.ones(3) * self.cfg.noise.lin_vel,
                jp.ones(3) * self.cfg.noise.ang_vel,
                jp.ones(3) * self.cfg.noise.euler,
                jp.zeros(self.action_size) * self.cfg.noise.motor_pos,
                jp.ones(self.action_size) * self.cfg.noise.motor_vel,
                jp.zeros(4),
                jp.zeros(3),
                jp.zeros(self.action_size)
            ]
        )

        self.lin_vel_x = self.cfg.domain_rand.lin_vel_x
        self.lin_vel_y = self.cfg.domain_rand.lin_vel_y
        self.ang_vel_yaw = self.cfg.domain_rand.ang_vel_yaw

        #observation
        self.num_obs_history = self.cfg.obs.frame_stack
        self.num_privileged_obs_history = self.cfg.obs.c_frame_stack
        self.obs_size = self.cfg.obs.num_single_obs
        self.privileged_obs_size = self.cfg.obs.num_single_privileged_obs
        self.obs_scales = self.cfg.obs_scales

        self.resample_time = self.cfg.commands.resample_time
        self.resample_steps = int(self.resample_time / self.dt)
        self.reset_time = self.cfg.commands.reset_time
        self.reset_steps = int(self.reset_time / self.dt)

    def _init_reward(self) -> None:
        """Initializes the reward system by filtering and scaling reward components.

        This method processes the reward scales configuration by removing any components with a scale of zero and scaling the remaining components by a time factor. It then prepares a list of reward function names and their corresponding scales, which are stored for later use in reward computation. Additionally, it sets parameters related to health and tracking rewards.
        """
        reward_scale_dict = OmegaConf.to_container(self.cfg.reward_scales)
        # Remove zero scales and multiply non-zero ones by dt
        for key in list(reward_scale_dict.keys()):
            if reward_scale_dict[key] == 0:
                reward_scale_dict.pop(key)

        # prepare list of functions
        self.reward_names = list(reward_scale_dict.keys())
        self.reward_functions = []
        self.reward_scales = jp.zeros(len(reward_scale_dict))
        for i, (name, scale) in enumerate(reward_scale_dict.items()):
            self.reward_functions.append(get_reward_function(name))
            self.reward_scales = self.reward_scales.at[i].set(scale)


        self.healthy_z_range = self.cfg.rewards.healthy_z_range
        self.max_foot_height = self.cfg.rewards.max_foot_height
        self.tracking_sigma = self.cfg.rewards.tracking_sigma


    def reset(self, rng: jp.ndarray) -> State:
        """ Resets the environment to the initial state"""
        (
        rng,
        rng_cmd,
        rng_gait,
        ) = jax.random.split(rng, 3)

        qpos = self.init_q.copy() 
        qvel = jp.zeros(self.nv)

        #Here we can include some randomizations like joint positions, velocities of base, and position of base and orientation.

        pipeline_state = self.pipeline_init(qpos, qvel) 

        # We can take either a phase based walking approach or a trajectory based walking approach. (ZMP)
        # Phase, freq=U(1.25, 1.5) : Gate cycle
        gait_freq = jax.random.uniform(rng_gait, (1,), minval=1.25, maxval=1.5)
        phase_dt = 2 * jp.pi * self.dt * gait_freq #Change in phase per time step (how gait phase evolves over time)
        phase = jp.array([0, jp.pi]) #One leg starts at beginning of gait cycle, other starts at mid-gait cycle (natural alternation between two legs)

        #Generate a random command
        rng, cmd_rng = jax.random.split(rng_cmd)
        cmd = self.sample_command(cmd_rng) #[lin_vel_x, lin_vel_y, ang_vel_yaw]


        state_info = {
            "rng": rng,
            "phase": phase,
            "phase_dt": phase_dt,
            "command": cmd,
            "first_contact": jp.zeros(2),
            "last_contact": jp.zeros(2),
            "swing_peak": jp.zeros(2),
            "last_act": jp.zeros(self.action_size),
            "last_last_act": jp.zeros(self.action_size),
            "feet_air_time": jp.zeros(2),
            "done": False,
            "step": 0,
        }

        obs_history = jp.zeros(self.num_obs_history * self.obs_size)
        privileged_obs_history = jp.zeros(
            self.num_privileged_obs_history * self.privileged_obs_size
        )

        obs = self._get_obs(
            pipeline_state,
            state_info,
            obs_history,
            privileged_obs_history
        )

        reward, done, zero = jp.zeros(3)

        metrics = {}
        for k in self.reward_names:
            metrics[k] = zero
        
        return State(
            pipeline_state, obs, reward, done, metrics, state_info
        )

    def step(self, state: State, action: jax.Array) -> State:
        """ Performs a step in the environment"""

        rng, cmd_rng = jax.random.split(
            state.info["rng"], 2
        )
        # apply a push if desired
        motor_targets = self.default_pose + action * self.cfg.action.action_scale

        pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)

        if self.add_domain_rand:
            # add rand
            pass

        torso_height = pipeline_state.xpos[self.torso_body_id, 2] 
        done = jp.logical_or(
            torso_height < self.healthy_z_range[0],
            torso_height > self.healthy_z_range[1],
        )
        state.info["done"] = done

        obs = self._get_obs(
            pipeline_state,
            state.info,
            state.obs['state'],
            state.obs['privileged_state'],
        )

        contact = check_feet_contact(pipeline_state, self.feet_link_ids)

        contact_bool = contact.astype(bool)
        last_contact_bool = state.info["last_contact"].astype(bool)
        contact_filt = contact_bool | last_contact_bool
        state.info["first_contact"] = ((state.info["feet_air_time"] > 0.0) * contact_filt).astype(jp.float32)
        state.info["feet_air_time"] += self.dt
        p_f = pipeline_state.xpos[self.feet_body_ids] 
        p_fz = p_f[...,-1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact.astype(jp.float32)  
        state.info["swing_peak"] *= ~contact

        reward_dict = self._compute_reward(pipeline_state, state.info, action)
        reward = sum(reward_dict.values()) * self.dt

        phase_tp1 = state.info["phase"] + state.info["phase_dt"]
        state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
        state.info["last_last_act"] = state.info["last_act"].copy()
        state.info["last_act"] = action.copy()
        state.info["rng"] = rng
        state.info["step"] += 1

        state.info["command"] = jax.lax.cond(
            state.info["step"] % self.resample_steps == 0,
            lambda: self.sample_command(cmd_rng),
            lambda: state.info["command"],
        )

        # reset the step counter when done
        state.info["step"] = jp.where(
            done | (state.info["step"] > self.reset_steps), 0, state.info["step"]
        )

        state.metrics.update(reward_dict)

        return state.replace(
            pipeline_state = pipeline_state,
            obs = obs,
            reward = reward,
            done = done.astype(jp.float32),
        )

    def _get_obs(
            self,
            pipeline_state: base.State,
            info: dict[str, Any],
            obs_history: jax.Array,
            privileged_obs_history: jax.Array,
    ) -> jp.ndarray:
        """ Returns the observation"""
        rng, obs_rng = jax.random.split(info["rng"], 2)

        gyro = self.get_gyro(pipeline_state)
        gravity = pipeline_state.site_xmat[self.sys.mj_model.site("imu").id] @ jp.array([0,0, -1])
        joint_velocities = pipeline_state.qvel[6:]
        lin_vel = self.get_local_linvel(pipeline_state)
        
        motor_pos = pipeline_state.q[7:]
        motor_pos_delta = (
            motor_pos - self.default_pose
        )

        #tracks the phase of the gait cycle
        cos = jp.cos(info["phase"])
        sin = jp.sin(info["phase"])
        phase = jp.concatenate([cos, sin])

        obs = jp.hstack([
            lin_vel, #(3,)
            gyro, #(3,)
            gravity, #(3,)
            info["command"], #(3,)
            joint_velocities, #(12,)
            info["last_act"], #(12,)
            phase, #(2,)
            motor_pos_delta #(12,)
        ])

        acceleromter = self.get_accelerometer(pipeline_state) #(3,)
        global_ang_vel = self.get_global_angvel(pipeline_state) #(3,)
        feet_vel = pipeline_state.sensordata[self._foot_linvel_sensor_adr].ravel()
        root_height = pipeline_state.qpos[2]

        actuator_forces = pipeline_state.actuator_force
        #

    
        privileged_obs = jp.hstack([
            obs, #(32,)
            acceleromter, #(3,)
            global_ang_vel, #(3,)
            feet_vel, #(2,)
            root_height, #(1,)
            info["last_contact"], #(2,)
            actuator_forces, #(12,)
            info["command"], #(3,)
            info["last_act"], #(12,)
        ])


        if self.add_noise:
            obs += self.obs_noise_scale * jax.random.normal(obs_rng, obs.shape)
        
        #check this
        # obs = jp.roll(obs_history, obs.size).at[: obs.size].set(obs)

        # privileged_obs = (
        #     jp.roll(privileged_obs_history, privileged_obs.size)
        #     .at[: privileged_obs.size]
        #     .set(privileged_obs)
        # )
        
        return {
            "state": obs, 
            "privileged_state": privileged_obs
        }
    
    def _compute_reward(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Computes a dictionary of rewards based on the current pipeline state, additional information, and the action taken.

        Args:
            pipeline_state (base.State): The current state of the pipeline.
            info (dict[str, Any]): Additional information that may be required for reward computation.
            action (jax.Array): The action taken, which influences the reward calculation.

        Returns:
            Dict[str, jax.Array]: A dictionary where keys are reward names and values are the computed rewards as JAX arrays.
        """
        # Create an array of indices to map over
        indices = jp.arange(len(self.reward_names))
        
        # Create a list of partial functions that each include self as the first argument
        reward_fns_with_self = [
            lambda ps, inf, act, fn=fn: fn(self, ps, inf, act)
            for fn in self.reward_functions
        ]
        
        # Use jax.lax.map to compute rewards
        reward_arr = jax.lax.map(
            lambda i: jax.lax.switch(
                i,
                reward_fns_with_self,
                pipeline_state,
                info,
                action,
            ) * self.reward_scales[i],
            indices,
        )

        reward_dict = {}
        for i, name in enumerate(self.reward_names):
            reward_dict[name] = reward_arr[i]

        return reward_dict            

    def sample_command(self, rng: jax.Array) -> jax.Array:
        """ 
        Sample random commands with 10% chance of setting everything to 0. 
        Used for exploration, robust learning, and generalization. 
        """
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

        #x,y velocity should be between [-1, 1]
        lin_vel_x = jax.random.uniform(
            rng1, minval=self.lin_vel_x[0], maxval=self.lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            rng2, minval=self.lin_vel_y[0], maxval=self.lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            rng3,
            minval=self.ang_vel_yaw[0],
            maxval=self.ang_vel_yaw[1],
        )

        # With 10% chance, set everything to zero.
        return jp.where(
            jax.random.bernoulli(rng4, p=0.1),
            jp.zeros(3),
            jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw]),
        )

