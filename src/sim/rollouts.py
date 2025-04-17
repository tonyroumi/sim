import jax


def generate_rollout(env, inference_fn, num_steps: int):
    """
    Generate a rollout from the environment using the inference function.
    """
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(inference_fn)

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    rollout = [state.pipeline_state]

    for i in range(num_steps):
      act_rng, rng = jax.random.split(rng)
      ctrl, _ = jit_inference_fn(state.obs, act_rng)
      state = jit_step(state, ctrl)
      rollout.append(state.pipeline_state)

      if state.done:
        break
    return rollout