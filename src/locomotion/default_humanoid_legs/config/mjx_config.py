from dataclasses import dataclass, field


@dataclass
class MJXConfig:
    """Configuration class for the MJX environment."""

    @dataclass
    class SimConfig:
        pass
        # timestep: float 
        # solver: int 
        # iterations: int 
        # ls_iterations: int 

    @dataclass
    class RewardConfig:
        pass

    @dataclass
    class RewardScales:
        pass

    @dataclass
    class DomainRandConfig:
        pass

    @dataclass
    class NoiseConfig:
        pass