from src.sim.mujoco_sim import MujocoSim
from src.sim.mujoco_utils import MujocoRenderer, MujocoViewer, mj_render
from src.robots.robot import Robot
import time
import random  # Add this import for random probability check
from src.policies.mjx_policy import MJXPolicy
from src.tools.keyboard import KeyboardController
import numpy as np
import mujoco
import glfw
from dataclasses import dataclass, asdict


robot = Robot("default_humanoid_legs")
sim = MujocoSim(robot=robot, xml_path="src/robots/default_humanoid_legs/default_humanoid_legs_flat_scene.xml", vis_type="view", n_frames=1, dt=0.004)  # Replace with your model path
policy = MJXPolicy(robot=robot, run_dir="/home/anthony-roumi/Desktop/Droids/simulation/outputs/2025-04-28/20-13-24", ckpt="193822720")  # Update with actual path/ckpt
controller = KeyboardController()

if not glfw.init():
    raise Exception("GLFW can't be initialized")

window = glfw.create_window(640, 480, "MuJoCo Simulation", None, None)
if not window:
    glfw.terminate()
    raise Exception("Failed to create GLFW window")

all_observations = []
all_actions = []
all_motor_targets = []

glfw.make_context_current(window)
glfw.set_key_callback(window, controller.key_callback)
last_act = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])



def put_obs(observation, last_act, command):
    # Ensure last_act is a numpy array with proper dimensions
    if isinstance(last_act, dict):
        # If last_act is a dictionary, convert its values to an array
        last_act = np.array(list(last_act.values()))
    else:
        # Otherwise ensure it's a numpy array
        last_act = np.atleast_1d(np.array(last_act))
        
    return np.concatenate(
            [
                observation.lin_vel, #4
                observation.gyro, #12
                observation.gravity, #12
                command, #3 # command here
                observation.joint_angles, #12
                observation.joint_vel, #12
                last_act, #12
                observation.phase, #4
            ]
        )

sim.set_qpos(robot.default_joint_angles)
sim.step(robot.default_motor_ctrls)
try:
    while not glfw.window_should_close(window):
        
        glfw.poll_events()
        command = controller.get_command()
        observation = sim.get_obs()
        observation_vec = put_obs(observation, last_act, command)
        time.sleep(0.001)
        action, motor_targets = policy.step(observation_vec, command)
        print("motor_targets", motor_targets)
        print("action", action)
        print("observation", observation_vec)
        all_observations.append(observation_vec)
        all_actions.append(action)
        all_motor_targets.append(motor_targets)
        last_act = motor_targets
        sim.step(motor_targets)
        time.sleep(0.005)
        
finally:
    sim.close()
    glfw.terminate()

import pandas as pd
import numpy as np

records = []
for t in range(len(all_actions)):
    record = {
        'timestep': t,
        'action': all_actions[t],
        'motor_target': all_motor_targets[t]
    }

    # Observation vector can be stored as-is or decomposed into separate keys
    obs = all_observations[t]
    for i, val in enumerate(obs):
        record[f'obs_{i}'] = val

    records.append(record)

df = pd.DataFrame(records)

# Optional: convert arrays to lists for readability
df = df.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

df.to_csv("mujoco_rollout_data.csv", index=False)

# def test_mujoco_classes():
    
#     resets = 0

#     try:
#         for _ in range(1000):
#             sim.step()
#             obs = sim.get_obs()
#             action = policy.step(obs, controller.get_command())
#             sim.step(action)
#             time.sleep(0.01)
#         print('='*30)
#         print(f"Total resets: {resets}")

#     finally:
#         sim.close()

# if __name__ == "__main__":
#    test_mujoco_classes()