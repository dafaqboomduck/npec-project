"""
Abstract base controller for robotic manipulation.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from src.models import ControllerState


class BaseController(ABC):
    """
    Abstract base class for robot controllers.

    Subclasses must implement `compute_action` which takes the current
    observation and target, and returns the action to execute.
    """

    def __init__(self, drop_threshold: float = 0.001, drop_cooldown: int = 10):
        self.drop_threshold = drop_threshold
        self.drop_cooldown = drop_cooldown
        self.state = ControllerState()

    def reset(self) -> None:
        """Reset controller state for a new episode."""
        self.state = ControllerState()
        self._on_reset()

    def _on_reset(self) -> None:
        """Hook for subclasses to perform additional reset logic."""
        pass

    @abstractmethod
    def compute_action(
        self,
        obs: np.ndarray,
        target: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Compute the control action given current observation and target.

        Args:
            obs: Current observation from environment
            target: Target position [x, y, z]
            dt: Time step

        Returns:
            Action array [control_x, control_y, control_z, drop_flag]
        """
        pass

    def get_position(self, obs: np.ndarray) -> np.ndarray:
        """Extract current position from observation."""
        return obs[6:9]

    def compute_error(self, obs: np.ndarray, target: np.ndarray) -> tuple:
        """Compute position error and distance to target."""
        pos = self.get_position(obs)
        error = target - pos
        distance = np.linalg.norm(error)
        return error, distance

    def step(
        self,
        obs: np.ndarray,
        targets: List[np.ndarray],
        dt: float,
    ) -> tuple[np.ndarray, bool, Optional[int]]:
        """
        Execute one control step.

        Args:
            obs: Current observation
            targets: List of target positions
            dt: Time step

        Returns:
            Tuple of (action, done, dropped_target_idx)
            - action: The action to execute
            - done: Whether all targets are complete
            - dropped_target_idx: Index of target just dropped at, or None
        """
        if self.state.current_target_idx >= len(targets):
            return np.zeros(4, dtype=np.float32), True, None

        target = np.array(targets[self.state.current_target_idx], dtype=np.float32)
        error, distance = self.compute_error(obs, target)
        dropped_target_idx = None

        if not self.state.dropped_at_target:
            if distance <= self.drop_threshold:
                action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                self.state.dropped_at_target = True
                self.state.cooldown_counter = 0
                dropped_target_idx = self.state.current_target_idx
            else:
                action = self.compute_action(obs, target, dt)
        else:
            self.state.cooldown_counter += 1
            action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

            if self.state.cooldown_counter >= self.drop_cooldown:
                self.state.current_target_idx += 1
                self.state.dropped_at_target = False
                self.state.cooldown_counter = 0
                self._on_target_complete()

        self.state.step += 1
        done = self.state.current_target_idx >= len(targets)
        return action, done, dropped_target_idx

    def _on_target_complete(self) -> None:
        """Hook called when a target is completed. Override to reset internal state."""
        pass
