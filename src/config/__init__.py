from hydra.core.config_store import ConfigStore
from src.config.config import Config
from src.config.envs import HumanoidLegsEnv
from src.config.agents import PPOConfig
from src.config.robots import DefaultHumanoidLegsRobot
from src.config.sim import MJXConfig

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

# env group
cs.store(group="env", name="humanoid_legs", node=HumanoidLegsEnv)

# agent group
cs.store(group="agent", name="ppo", node=PPOConfig)

# robot group
cs.store(group="robot", name="humanoid_legs", node=DefaultHumanoidLegsRobot)

# sim group
cs.store(group="sim", name="mjx", node=MJXConfig)