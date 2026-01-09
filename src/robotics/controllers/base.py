"""
Abstract base controller for robotic manipulation.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict
import numpy as np

from src.models import ControllerState


class BaseController(ABC):
    """
    Abstract base class for robot controllers.

    Subclasses must implement `compute_action` which takes the current
    observation, info, and target, and returns the action to execute.
    """

    def __init__(self, target_threshold: float = 0.001, dwell_steps: int = 10):
        self.target_threshold = target_threshold
        self.dwell_steps = dwell_steps
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
        info: Dict,
        target: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Compute the control action given current observation and target.

        Args:
            obs: Current observation from environment
            info: Info dict from environment (contains pipette_positions)
            target: Target position [x, y, z]
            dt: Time step

        Returns:
            Action array [control_x, control_y, control_z]
        """
        pass

    def get_pipette_position(self, info: Dict) -> np.ndarray:
        """Extract current pipette position from info dict."""
        robot_key = sorted(info["pipette_positions"].keys())[0]
        return np.array(info["pipette_positions"][robot_key])

    def compute_error(self, info: Dict, target: np.ndarray) -> tuple:
        """Compute position error and distance to target."""
        pos = self.get_pipette_position(info)
        error = target - pos
        distance = np.linalg.norm(error)
        return error, distance

    def step(
        self,
        obs: np.ndarray,
        info: Dict,
        targets: List[np.ndarray],
        dt: float,
    ) -> tuple[np.ndarray, bool, Optional[int]]:
        """
        Execute one control step.

        Args:
            obs: Current observation
            info: Info dict from environment
            targets: List of target positions
            dt: Time step

        Returns:
            Tuple of (action, done, completed_target_idx)
            - action: The action to execute
            - done: Whether all targets are complete
            - completed_target_idx: Index of target just completed, or None
        """
        if self.state.current_target_idx >= len(targets):
            return np.zeros(3, dtype=np.float32), True, None

        target = np.array(targets[self.state.current_target_idx], dtype=np.float32)
        error, distance = self.compute_error(info, target)
        completed_target_idx = None

        if distance <= self.target_threshold:
            # At target - hold position and increment dwell counter
            action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.state.dwell_counter += 1

            if self.state.dwell_counter >= self.dwell_steps:
                completed_target_idx = self.state.current_target_idx
                self.state.current_target_idx += 1
                self.state.dwell_counter = 0
                self._on_target_complete()
        else:
            # Still approaching - compute control action
            action = self.compute_action(obs, info, target, dt)
            self.state.dwell_counter = 0

        self.state.step += 1
        done = self.state.current_target_idx >= len(targets)
        return action, done, completed_target_idx

    def _on_target_complete(self) -> None:
        """Hook called when a target is completed. Override to reset internal state."""
        pass