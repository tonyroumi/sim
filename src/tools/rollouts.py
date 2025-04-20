import os
import jax
import mediapy as media
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks

def get_rollout(policy_path, env, make_networks_factory, num_steps):
    """
    Get a rollout from the policy
    """
    ppo_network = make_networks_factory(
        env.obs_size, env.action_size
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    params = model.load_params(policy_path)
    inference_fn = make_policy(params, deterministic=True)

    # initialize the state
    jit_reset = jax.jit(env.reset)
    # jit_reset = env.reset
    jit_step = jax.jit(env.step)
    # jit_step = env.step
    jit_inference_fn = jax.jit(inference_fn)
    # jit_inference_fn = inference_fn

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)

    rollout = [state.pipeline_state]
    for i in range(num_steps):
        ctrl, _ = jit_inference_fn(state.obs, rng)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)
        if state.done:
            break
       
    return rollout

def save_rollout(save_path, policy_path, env, make_networks_factory, num_steps) -> None:
    """
    Save a rollout to a file
    """
    rollout = get_rollout(policy_path, env, make_networks_factory, num_steps)
    traj = rollout[::2]

    frames = env.render(traj)
    fps = 1.0 / env.dt / 2

    media.write_video(save_path, frames, fps=fps)
