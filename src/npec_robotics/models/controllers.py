from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ControllerState:
    """State tracked during robotic manipulation.

    Attributes:
        current_target_idx: Index of the current target in the target list.
        dwell_counter: Steps spent at current target (for dwell/stability check).
        step: Total steps executed in the current episode.
    """
    current_target_idx: int = 0
    dwell_counter: int = 0
    step: int = 0
    
    # Backward compatibility aliases
    @property
    def dropped_at_target(self) -> bool:
        return self.dwell_counter > 0
    
    @property
    def cooldown_counter(self) -> int:
        return self.dwell_counter


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
        target_threshold: Distance threshold (meters) to consider target reached.
        dwell_steps: Number of steps to stay at target before moving to next.
        drop_threshold: Deprecated alias for target_threshold.
        drop_cooldown: Deprecated alias for dwell_steps.
    """
    x: PIDParams = field(default_factory=lambda: PIDParams(kp=9.0, ki=0.048, kd=0.0))
    y: PIDParams = field(default_factory=lambda: PIDParams(kp=9.0, ki=0.048, kd=0.0))
    z: PIDParams = field(default_factory=lambda: PIDParams(kp=5.0, ki=0.024, kd=0.0))
    target_threshold: float = None
    dwell_steps: int = None
    # Backward compatibility
    drop_threshold: float = None
    drop_cooldown: int = None

    def __post_init__(self):
        # Resolve aliases with defaults
        if self.target_threshold is None:
            self.target_threshold = self.drop_threshold or 0.001
        if self.dwell_steps is None:
            self.dwell_steps = self.drop_cooldown or 10

    @classmethod
    def from_dict(cls, params: Dict) -> "PIDControllerConfig":
        """Create config from dictionary format."""
        return cls(
            x=PIDParams(**params.get("x", {})),
            y=PIDParams(**params.get("y", {})),
            z=PIDParams(**params.get("z", {})),
            target_threshold=params.get("target_threshold"),
            dwell_steps=params.get("dwell_steps"),
            drop_threshold=params.get("drop_threshold"),
            drop_cooldown=params.get("drop_cooldown"),
        )