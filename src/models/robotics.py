import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Configuration for the robotics simulation environment.

    Attributes:
        num_agents: Number of robotic agents in the environment.
        render_mode: Rendering mode ('rgb_array', 'human', or None).
        target: Predetermined target coordinates, or None for dynamic targeting.
        max_steps: Maximum steps allowed per episode before termination.
    """

    num_agents: int = 1
    render_mode: Optional[str] = None
    target: Optional[np.ndarray] = None
    max_steps: int = 1000


@dataclass
class ControllerConfig:
    """Configuration for the robot controller.

    Attributes:
        max_steps: Maximum control steps per trajectory.
        drop_threshold: Velocity threshold below which drop is detected.
        drop_cooldown: Steps to wait after drop before next action.
        dish_size: Diameter of the petri dish in millimeters.
        dt: Simulation timestep in seconds.
    """

    max_steps: int = 5000
    drop_threshold: float = 0.0001
    drop_cooldown: int = 50
    dish_size: float = 150.0
    dt: float = 1.0 / 240.0
