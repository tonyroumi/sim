from .registry import register_reward
import jax
import jax.numpy as jp
from src.utils.math_utils import rotate_vec, quat_inv
from brax import base
from typing import Any
 # Tracking rewards.

@register_reward("lin_vel_xy")
def _lin_vel_xy(
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

@register_reward("ang_vel_yaw")
def _ang_vel_yaw(
    self,
    pipeline_state: base.State,
    info: dict[str, Any],
    action: jax.Array,
) -> jax.Array:
    ang_vel_local = rotate_vec(
        pipeline_state.xd.ang[0], quat_inv(pipeline_state.x.rot[0])
    )
    ang_vel_yaw = ang_vel_local[2]
    error = jp.linalg.norm(ang_vel_yaw - info["command"][2])
    reward = jp.exp(-self.tracking_sigma / 4 * error**2)
    return reward


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

