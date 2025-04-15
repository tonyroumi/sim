import jax
import jax.numpy as jnp
from jax import disable_jit
from src.robots.robot import Robot
from src.locomotion import get_env_class

# For illustration, we'll use a built-in env as a stand-in:
from brax.envs import create
from src.locomotion.default_humanoid_legs.joystick import Joystick
from src.config.sim import MJXConfig
from src.config.agents import PPOConfig


# --- STEP 2: Disable JIT + Run a step with raw values ---
def main():
    rng = jax.random.PRNGKey(0)
    robot = Robot('default_humanoid_legs')

    EnvClass = get_env_class('default_humanoid_legs')
    env_cfg = MJXConfig()
    train_cfg = PPOConfig()

    
    env = EnvClass(
        'default_humanoid_legs',
        robot,
        'flat', 
        env_cfg)

    with disable_jit():
        print("Initializing environment (no jit)...")
        state = env.reset(rng)
        print("\nInitial Position:")
    

        # Random action
        action = jnp.zeros(env.nu)
        state = env.step(state, action)

        print("\nPost-Step Position:")
        print(state.q)

        print("\nVelocity:")
        print(state.qd)

        print("\nReward:", state.reward)
        print("Done:", state.done)


if __name__ == "__main__":
    main()