from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ControllerState:
    """State tracked during robotic manipulation.

    Attributes:
        current_target_idx: Index of the current target in the target list.
        dropped_at_target: Whether the robot has dropped at the current target.
        cooldown_counter: Steps elapsed since dropping (for cooldown period).
        step: Total steps executed in the current episode.
    """
    current_target_idx: int = 0
    dropped_at_target: bool = False
    cooldown_counter: int = 0
    step: int = 0


@dataclass
class PIDParams:
    """PID parameters for a single axis.

    Attributes:
        kp: Proportional gain coefficient.
        ki: Integral gain coefficient.
        kd: Derivative gain coefficient.
    """
    kp: float = 9.0
    ki: float = 0.048
    kd: float = 0.0


@dataclass
class PIDControllerConfig:
    """Configuration for the PID robot controller.

    Attributes:
        x: PID parameters for the x-axis.
        y: PID parameters for the y-axis.
        z: PID parameters for the z-axis.
        drop_threshold: Distance threshold (meters) to trigger drop action.
        drop_cooldown: Number of steps to wait after dropping before moving to next target.
    """
    x: PIDParams = field(default_factory=lambda: PIDParams(kp=9.0, ki=0.048, kd=0.0))
    y: PIDParams = field(default_factory=lambda: PIDParams(kp=9.0, ki=0.048, kd=0.0))
    z: PIDParams = field(default_factory=lambda: PIDParams(kp=5.0, ki=0.024, kd=0.0))
    drop_threshold: float = 0.001
    drop_cooldown: int = 10

    @classmethod
    def from_dict(cls, params: Dict) -> "PIDControllerConfig":
        """Create config from dictionary format."""
        return cls(
            x=PIDParams(**params.get("x", {})),
            y=PIDParams(**params.get("y", {})),
            z=PIDParams(**params.get("z", {})),
            drop_threshold=params.get("drop_threshold", 0.001),
            drop_cooldown=params.get("drop_cooldown", 10),
        )