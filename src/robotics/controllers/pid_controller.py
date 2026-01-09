"""
PID-based robot controller implementation.
"""

from typing import Optional, Dict
import numpy as np

from .base import BaseController
from src.models import PIDControllerConfig
from src.robotics.pid import PIDController as PID


class PIDRobotController(BaseController):
    """
    PID-based controller for robot manipulation.

    Uses three independent PID controllers for x, y, z axes.
    """

    def __init__(
        self,
        config: Optional[PIDControllerConfig] = None,
        output_limits: tuple = (-1.0, 1.0),
    ):
        self.config = config or PIDControllerConfig()
        self.output_limits = output_limits

        super().__init__(
            target_threshold=getattr(self.config, 'target_threshold', None) or getattr(self.config, 'drop_threshold', 0.001),
            dwell_steps=getattr(self.config, 'dwell_steps', None) or getattr(self.config, 'drop_cooldown', 10),
        )

        self._init_pid_controllers()

    def _init_pid_controllers(self) -> None:
        """Initialize the three PID controllers."""
        self.pid_x = PID(
            kp=self.config.x.kp,
            ki=self.config.x.ki,
            kd=self.config.x.kd,
        )
        self.pid_y = PID(
            kp=self.config.y.kp,
            ki=self.config.y.ki,
            kd=self.config.y.kd,
        )
        self.pid_z = PID(
            kp=self.config.z.kp,
            ki=self.config.z.ki,
            kd=self.config.z.kd,
        )

    def _on_reset(self) -> None:
        """Reset PID controllers."""
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()

    def _on_target_complete(self) -> None:
        """Reset PIDs when moving to next target."""
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()

    def compute_action(
        self,
        obs: np.ndarray,
        info: Dict,
        target: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Compute PID control action."""
        error, _ = self.compute_error(info, target)

        control_x = self.pid_x.compute(error[0], dt, output_limits=self.output_limits)
        control_y = self.pid_y.compute(error[1], dt, output_limits=self.output_limits)
        control_z = self.pid_z.compute(error[2], dt, output_limits=self.output_limits)

        return np.array([control_x, control_y, control_z], dtype=np.float32)