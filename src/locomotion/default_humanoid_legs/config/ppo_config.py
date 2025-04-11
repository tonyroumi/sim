from dataclasses import dataclass
from typing import Tuple

@dataclass
class PPOConfig:
    """Data class for storing PPO hyperparameters."""
    num_timesteps: int
    num_evals: int
    episode_length: int
    unroll_length: int
    num_minibatches: int
    num_updates_per_batch: int
    discounting: float
    learning_rate: float
    entropy_cost: float
    clipping_epsilon: float
    num_envs: int
    batch_size: int
    seed: int
    render_interval: int
    normalize_observations: bool
    action_repeat: float
    max_grad_norm: float
    policy_hidden_layer_sizes: Tuple[int, ...]
    value_hidden_layer_sizes: Tuple[int, ...]
    num_resets_per_eval: int
