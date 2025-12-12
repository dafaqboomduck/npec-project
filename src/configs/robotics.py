from src.models import EnvConfig

# Environment config
ENV_CONFIG = EnvConfig(render_mode="rgb_array")

# Robot control constants
MAX_STEPS = 5000  # Maximum steps per episode
DROP_THRESHOLD = 0.0001  # Distance threshold (mm) to trigger drop action
DROP_COOLDOWN = 50  # Steps to wait after dropping before moving to next target

# Simulation time step (240 Hz)
DT = 1.0 / 240.0
