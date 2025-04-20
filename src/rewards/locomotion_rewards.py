from .registry import register_reward
import jax
import jax.numpy as jp
from src.utils.math_utils import rotate_vec, quat_inv
from brax import base
from typing import Any
 # Tracking rewards.

@register_reward("lin_vel")
def _lin_vel(
    self,
    pipeline_state: base.State,
    info: dict[str, Any],
    action: jax.Array,
) -> jax.Array:
        
    lin_vel_error = jp.sum(jp.square(info["command"][:2] - pipeline_state.cvel[1, 3:-1]))
    return jp.exp(-lin_vel_error / self.tracking_sigma)

@register_reward("ang_vel")
def _ang_vel(
    self,
    pipeline_state: base.State,
    info: dict[str, Any],
    action: jax.Array,
) -> jax.Array:
    ang_vel_error = jp.square(info["command"][2] - pipeline_state.cvel[1, 2])
    return jp.exp(-ang_vel_error / self.tracking_sigma)


# Energy related rewards.

@register_reward("torques")
def _torques(
    self,
    pipeline_state: base.State,
    info: dict[str, Any],
    action: jax.Array,
) -> jax.Array:
    return jp.sum(jp.abs(pipeline_state.actuator_force))

@register_reward("energy")
def _energy(
    self, 
    pipeline_state: base.State,
    info: dict[str, Any],
    action: jax.Array,
) -> jax.Array:
    return jp.sum(jp.abs(pipeline_state.qvel) * jp.abs(pipeline_state.qfrc_actuator))

@register_reward("action_rate")
def _action_rate(
    self, 
    pipeline_state: base.State,
    info: dict[str, Any],
    action: jax.Array,
) -> jax.Array:
    c1 = jp.sum(jp.square(action - info["last_act"]))
    return c1


  # Feet related rewards.

@register_reward("feet_slip")
def _feet_slip(
    self,
    pipeline_state: base.State,
    info: dict[str, Any],
    action: jax.Array,
) -> jax.Array:
    del info  # Unused.
    body_vel = self.get_global_linvel(data)[:2]
    reward = jp.sum(jp.linalg.norm(body_vel, axis=-1) * contact)
    return reward

@register_reward("feet_clearance")
def _feet_clearance(
    self,
    pipeline_state: base.State,
    info: dict[str, Any],
    action: jax.Array,
) -> jax.Array:
    del info  # Unused.
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
    return jp.sum(delta * vel_norm)

@register_reward("feet_height")
def _feet_height(
    self,
    pipeline_state: base.State,
    info: dict[str, Any],
    action: jax.Array,
) -> jax.Array:
    del info  # Unused.
    error = swing_peak / self._config.reward_config.max_foot_height - 1.0
    return jp.sum(jp.square(error) * first_contact)

@register_reward("feet_air_time")
def _feet_air_time(
    self,
    pipeline_state: base.State,
    info: dict[str, Any],
    action: jax.Array,
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    air_time = (air_time - threshold_min) * first_contact
    air_time = jp.clip(air_time, max=threshold_max - threshold_min)
    reward = jp.sum(air_time)
    reward *= cmd_norm > 0.1  # No reward for zero commands.
    return reward

@register_reward("feet_phase")
def _feet_phase(
    self,
    pipeline_state: base.State,
    info: dict[str, Any],
    action: jax.Array,
  ) -> jax.Array:
    # Reward for tracking the desired foot height.
    del commands  # Unused.
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    rz = gait.get_rz(phase, swing_height=foot_height)
    error = jp.sum(jp.square(foot_z - rz))
    reward = jp.exp(-error / 0.01)
    # TODO(kevin): Ensure no movement at 0 command.
    # cmd_norm = jp.linalg.norm(commands)
    # reward *= cmd_norm > 0.1  # No reward for zero commands.
    return reward



@register_reward("survival")
def _survival(
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

