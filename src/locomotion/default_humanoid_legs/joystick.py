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
from src.utils.math_utils import quat2euler, quat_inv, rotate_vec

class Joystick(PipelineEnv):
    def __init__(
            self,
            name: str,
            robot: any,
            scene: str,
            cfg,
            **kwargs: Any,
    ):
        """ Initalizes the environment with the specified configuration and robot parameters"""
        self.name = name
        self.robot = robot
        self.cfg = cfg
        self.add_noise = cfg.noise.add_noise
        self.add_domain_rand = cfg.domain_rand.add_domain_rand

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
            if scale < 0:
                self.reward_functions.append(getattr(self, "_cost_" + name))
            else:
                self.reward_functions.append(getattr(self, "_reward_" + name))
            self.reward_scales = self.reward_scales.at[i].set(scale)

        self.healthy_z_range = self.cfg.rewards.healthy_z_range
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
            # "contact": jp.zeros(2),
            "last_act": jp.zeros(self.action_size),
            "last_last_act": jp.zeros(self.action_size),
            # "feet_air_time": jp.zeros(2),
            "last_torso_euler": jp.zeros(3),
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
        # for k in self.reward_names:
        #     metrics[f"reward/{k}"] = zero
        
        return State(
            pipeline_state, obs, reward, done, metrics, state_info
        )

    def step(self, state: State, action: jp.ndarray) -> State:
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

        torso_height = pipeline_state.x.pos[0, 2] 
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

        torso_euler = quat2euler(pipeline_state.x.rot[0])
        torso_euler_delta = torso_euler - state.info["last_torso_euler"]
        torso_euler_delta = (torso_euler_delta + jp.pi) % (2 * jp.pi) - jp.pi
        torso_euler = state.info["last_torso_euler"] + torso_euler_delta

        reward_dict = self._compute_reward(pipeline_state, state.info, action)
        reward = sum(reward_dict.values()) * self.dt


        feet_z_delta = (
            pipeline_state.x.pos[self.feet_link_ids, 2]
            - state.info["feet_height_init"]
        )

        phase_tp1 = state.info["phase"] + state.info["phase_dt"]
        state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
        state.info["last_last_act"] = state.info["last_act"].copy()
        state.info["last_act"] = action.copy()
        state.info["last_torso_euler"] = torso_euler
        state.info["rng"] = rng
        state.info["step"] += 1

        state.info["command"] = jax.lax.cond(
            state.info["step"] % self.resample_steps == 0,
            lambda: self._sample_command(cmd_rng, state.info["command"]),
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
        
        motor_pos = pipeline_state.q[7:]
        motor_pos_delta = (
            motor_pos - self.default_pose
        )

        motor_vel = pipeline_state.qd[6:]

        torso_quat = pipeline_state.x.rot[0] 

        #equivalent to gyro or linvel. track only torso movement
        torso_lin_vel = rotate_vec(pipeline_state.xd.vel[0], quat_inv(torso_quat)) 
        torso_ang_vel = rotate_vec(pipeline_state.xd.ang[0], quat_inv(torso_quat))

        #tracks the changes in euler position of torso
        torso_euler = quat2euler(torso_quat)
        torso_euler_delta = torso_euler - info["last_torso_euler"]
        torso_euler_delta = (torso_euler_delta + jp.pi) % (2 * jp.pi) - jp.pi
        torso_euler = info["last_torso_euler"] + torso_euler_delta

        #tracks the phase of the gait cycle
        cos = jp.cos(info["phase"])
        sin = jp.sin(info["phase"])
        phase = jp.concatenate([cos, sin])

        obs = jp.hstack([
            torso_lin_vel * self.obs_scales.lin_vel, #(3,)
            torso_ang_vel * self.obs_scales.ang_vel, #(3,)
            torso_euler * self.obs_scales.euler, #(3,)
            motor_pos_delta * self.obs_scales.motor_pos, #(12,)
            motor_vel * self.obs_scales.motor_vel, #(12,)
            phase, #(4,)
            info["command"], #(3,)
            info["last_act"], #(12,)
        ])

        actuator_forces = pipeline_state.actuator_force
        root_height = pipeline_state.q[2]
        feet_vel = pipeline_state.xd.vel[self.feet_link_ids, 0] #I think this is the velocity of the feet in the x direction. Check in case

    
        privileged_obs = jp.hstack([
            torso_lin_vel * self.obs_scales.lin_vel, #(3,)
            torso_ang_vel * self.obs_scales.ang_vel, #(3,)
            torso_euler * self.obs_scales.euler, #(3,)
            motor_pos_delta * self.obs_scales.motor_pos, #(12,)
            motor_vel * self.obs_scales.motor_vel, #(12,)
            phase, #(4,)
            actuator_forces, #(12,)
            root_height, #(1,)
            feet_vel, #(2,)
            # info["feet_air_time"], #(2,)
            # info["contact"], #(2,)
            info["command"], #(3,)
            info["last_act"], #(12,)
        ])


        if self.add_noise:
            obs += self.obs_noise_scale * jax.random.normal(obs_rng, obs.shape)
        
        #check this
        obs = jp.roll(obs_history, obs.size).at[: obs.size].set(obs)

        privileged_obs = (
            jp.roll(privileged_obs_history, privileged_obs.size)
            .at[: privileged_obs.size]
            .set(privileged_obs)
        )
        
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
        # Use jax.lax.map to compute rewards
        reward_arr = jax.lax.map(
            lambda i: jax.lax.switch(
                i,
                self.reward_functions,
                pipeline_state,
                info,
                action,
            )
            * self.reward_scales[i],
            indices,
        )

        reward_dict = {}
        for i, name in enumerate(self.reward_names):
            reward_dict[name] = reward_arr[i]

        return reward_dict
    
    def _get_contact_forces(self, data: mjx.Data):
        """Compute contact forces between colliders and determine foot contact masks.

        This function calculates the contact forces between colliders based on the provided
        simulation data. It also determines whether the left and right foot colliders are in
        contact with the ground by comparing the contact forces against a predefined threshold.

        Args:
            data (mjx.Data): The simulation data containing contact information.

        Returns:
            Tuple[jax.Array, jax.Array, jax.Array]: A tuple containing:
                - A 3D array of shape (num_colliders, num_colliders, 3) representing the global
                  contact forces between colliders.
                - A 1D array indicating whether each left foot collider is in contact.
                - A 1D array indicating whether each right foot collider is in contact.
        """
        # Extract geom1 and geom2 directly
        geom1 = data.contact.geom1
        geom2 = data.contact.geom2

        def get_body_index(geom_id: jax.Array) -> jax.Array:
            return jp.argmax(self.collider_geom_ids == geom_id)

        # Vectorized computation of body indices for geom1 and geom2
        body_indices_1 = jax.vmap(get_body_index)(geom1)
        body_indices_2 = jax.vmap(get_body_index)(geom2)

        contact_forces_global = jp.zeros((self.num_colliders, self.num_colliders, 3))
        for i in range(data.ncon):
            contact_force = self.jit_contact_force(self.sys, data, i, True)[:3]
            # Update the contact forces for both body_indices_1 and body_indices_2
            # Add instead of set to accumulate forces from multiple contacts
            contact_forces_global = contact_forces_global.at[
                body_indices_1[i], body_indices_2[i]
            ].add(contact_force)
            contact_forces_global = contact_forces_global.at[
                body_indices_2[i], body_indices_1[i]
            ].add(contact_force)

        left_foot_contact_mask = (
            contact_forces_global[0, self.left_foot_collider_indices, 2]
            > self.contact_force_threshold
        ).astype(jp.float32)
        right_foot_contact_mask = (
            contact_forces_global[0, self.right_foot_collider_indices, 2]
            > self.contact_force_threshold
        ).astype(jp.float32)

        return contact_forces_global, left_foot_contact_mask, right_foot_contact_mask        

     # Tracking rewards.

    def _reward_lin_vel_xy(
        self,
        pipeline_state: base.State,
        info: dict[str, Any],
        action: jax.Array,
    ) -> jax.Array:
        
        lin_vel_local = rotate_vec(
            pipeline_state.xd.vel[0], quat_inv(pipeline_state.x.rot[0])
        )
        lin_vel_xy = lin_vel_local[:2]
        error = jp.linalg.norm(lin_vel_xy -  info["command"][:2], axis=-1)
        reward = jp.exp(-self.tracking_sigma * error**2)
        return reward

    def _reward_ang_vel_yaw(
        self,
        pipeline_state: base.State,
        info: dict[str, Any],
        action: jax.Array,
    ) -> jax.Array:
        #resume HERE
        ang_vel_local = rotate_vec(
            pipeline_state.xd.ang[0], quat_inv(pipeline_state.x.rot[0])
        )
        ang_vel_yaw = ang_vel_local[2]
        error = jp.linalg.norm(ang_vel_yaw - info["command"][2], axis=-1)
        reward = jp.exp(-self.tracking_sigma / 4 * error**2)
        return reward
    
    
    # Energy related rewards.

    def _cost_torques(self, torques: jax.Array) -> jax.Array:
        return jp.sum(jp.abs(torques))

    def _cost_energy(
        self, qvel: jax.Array, qfrc_actuator: jax.Array
    ) -> jax.Array:
        return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

    def _cost_action_rate(
        self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
    ) -> jax.Array:
        del last_last_act  # Unused.
        c1 = jp.sum(jp.square(act - last_act))
        return c1
    
    def _reward_survival(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates a survival reward based on the pipeline state and action taken.

        The reward is negative if the episode is marked as done before reaching the
        specified number of reset steps, encouraging survival until the reset threshold.

        Args:
            pipeline_state (base.State): The current state of the pipeline.
            info (dict[str, Any]): A dictionary containing episode information, including
                whether the episode is done and the current step count.
            action (jax.Array): The action taken at the current step.

        Returns:
            jax.Array: A float32 array representing the survival reward.
        """
        return -(info["done"] & (info["step"] < self.reset_steps)).astype(jp.float32)
            

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