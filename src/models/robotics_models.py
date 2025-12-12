import numpy as np
from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Configuration for the robotics environment.

    Attributes:
        num_agents: Number of robotic agents in the environment
        render_mode: Rendering mode ('rgb_array', 'human', or None)
        target: Target coordinates for the robot (if predetermined)
        max_steps: Maximum steps allowed per episode
    """

    num_agents: int = 1
    render_mode: str | None = None
    target: np.ndarray | None = None
    max_steps: int = 1000
