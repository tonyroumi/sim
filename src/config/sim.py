from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class MJXConfig:
    """Configuration class for the MJX environment."""

    @dataclass
    class SimConfig:
        pass
        timestep: float = 0.004
        solver: int = 2
        iterations: int = 1 
        ls_iterations: int = 4

    @dataclass
    class ObsConfig:
        frame_stack: int = 15
        c_frame_stack: int = 15
        num_single_obs: int = 0
        num_single_privileged_obs: int = 0

    @dataclass
    class ActionConfig:
        action_scale: float = 0.25
        n_frames: int = 5

    @dataclass
    class RewardConfig:
        pass

    @dataclass
    class RewardScales:
        pass

    @dataclass
    class CommandsConfig:
        resample_time: float = 3.0
        reset_time: float = 100.0  # No resetting by default

    @dataclass
    class DomainRandConfig:
        add_domain_rand: bool = False
        lin_vel_x: Tuple[float, float] = (-1.0, 1.0)
        lin_vel_y: Tuple[float, float] = (-1.0, 1.0)
        ang_vel_yaw: Tuple[float, float] = (-1.0, 1.0)

    @dataclass
    class NoiseConfig:
        add_noise: bool = True
        motor_pos: float = 0.0
        motor_vel: float = 0.0
        lin_vel: float = 0.0
        ang_vel: float = 0.0
        euler: float = 0.0
        pass

    sim: SimConfig = field(default_factory=SimConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    reward_scales: RewardScales = field(default_factory=RewardScales)
    commands: CommandsConfig = field(default_factory=CommandsConfig)
    domain_rand: DomainRandConfig = field(default_factory=DomainRandConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)