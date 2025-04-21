from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class MJXConfig:
    """Configuration class for the MJX environment."""

    @dataclass
    class SimConfig:
        timestep: float = 0.004
        solver: int = 2
        iterations: int = 1 
        ls_iterations: int = 4

    @dataclass
    class ObsConfig:
        stack_obs: bool = False
        frame_stack: int = 15
        c_frame_stack: int = 15
        num_single_obs: int = 52
        num_single_privileged_obs: int = 69

    @dataclass
    class ObsScales:
        lin_vel: float = 1.0
        ang_vel: float = 1.0
        motor_pos: float = 1.0
        motor_vel: float = 1.0
        euler: float = 1.0

    @dataclass
    class ActionConfig:
        action_scale: float = 0.25
        n_frames: int = 5

    @dataclass
    class RewardConfig:
        healthy_z_range: Tuple[float, float] = (0.7, 1.3)
        tracking_sigma: float = 0.5
        max_foot_height: float = 0.1

    @dataclass
    class RewardScales:
        lin_vel: float = 1.0  
        ang_vel: float = 0.5 
        torques: float = -2.5e-5
        action_rate: float = -0.01
        energy: float = 0.0
        feet_slip: float = -0.25
        feet_clearance: float = 0.0
        feet_height: float = 0.0
        feet_phase: float = 1.0
        stand_still: float = -1.0
        survival: float = 3.0

    @dataclass
    class CommandsConfig:
        resample_time: float = 10 
        reset_time: float = 100.0  # No resetting by default

    @dataclass
    class DomainRandConfig:
        add_domain_rand: bool = False
        lin_vel_x: Tuple[float, float] = (-0.5, 1.5)
        lin_vel_y: Tuple[float, float] = (-0.5, 1.5)
        ang_vel_yaw: Tuple[float, float] = (-0.5, 1.5)

    @dataclass
    class NoiseConfig:
        add_noise: bool = False
        action_noise: float = 0.02
        obs_noise_scale: float = 0.05
        motor_pos: float = 1.0
        motor_vel: float = 2.0
        lin_vel: float = 3.0
        ang_vel: float = 5.0
        euler: float = 2.0

    sim: SimConfig = field(default_factory=SimConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    obs_scales: ObsScales = field(default_factory=ObsScales)
    action: ActionConfig = field(default_factory=ActionConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    reward_scales: RewardScales = field(default_factory=RewardScales)
    commands: CommandsConfig = field(default_factory=CommandsConfig)
    domain_rand: DomainRandConfig = field(default_factory=DomainRandConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)