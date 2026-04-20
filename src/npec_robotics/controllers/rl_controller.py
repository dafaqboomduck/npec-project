"""
Reinforcement Learning robot controller implementation.
"""

import logging
from typing import Optional, Dict
from pathlib import Path
import numpy as np

from .base import BaseController

logger = logging.getLogger(__name__)


class RLController(BaseController):
    """
    RL-based controller for robot manipulation.

    Uses a trained Stable Baselines3 model to predict actions.
    """

    def __init__(
        self,
        model_path: str | Path,
        target_threshold: float = 0.001,
        dwell_steps: int = 10,
        deterministic: bool = True,
    ):
        """
        Initialize the RL controller.

        Args:
            model_path: Path to the trained SB3 model (.zip file)
            target_threshold: Distance threshold to consider target reached.
            dwell_steps: Number of steps to stay at target before moving to next.
            deterministic: Whether to use deterministic action prediction.
        """
        super().__init__(
            target_threshold=target_threshold,
            dwell_steps=dwell_steps,
        )

        self.model_path = Path(model_path)
        self.deterministic = deterministic
        self.model = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the trained RL model."""
        from stable_baselines3 import PPO

        # Handle relative paths from project root
        if not self.model_path.is_absolute():
            # Try relative to current working directory
            if not self.model_path.exists():
                # Try relative to this file's location (src/robotics/controllers/)
                project_root = Path(__file__).parent.parent.parent.parent
                self.model_path = project_root / self.model_path

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = PPO.load(self.model_path)
        logger.info(f"Loaded RL model from {self.model_path}")

    def _on_reset(self) -> None:
        """No additional reset needed for RL controller."""
        pass

    def _on_target_complete(self) -> None:
        """No additional reset needed for RL controller."""
        pass

    def compute_action(
        self,
        obs: np.ndarray,
        info: Dict,
        target: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Compute action using the trained RL model.

        Note: The model expects the full observation from the environment,
        which already includes the target information.
        """
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return np.array(action, dtype=np.float32)