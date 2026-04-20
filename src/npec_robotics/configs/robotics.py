from npec_robotics.models import EnvConfig, ControllerConfig

RENDER_MODE = "human"

# Environment config
ENV_CONFIG = EnvConfig(render_mode=RENDER_MODE)

# Robot control constants
MAX_STEPS = 5000  # Maximum steps per episode
DROP_THRESHOLD = 0.0001  # Distance threshold (mm) to trigger drop action
DROP_COOLDOWN = 50  # Steps to wait after dropping before moving to next target
DISH_SIZE = 150.0

# Simulation time step (240 Hz)
DT = 1.0 / 240.0

CONTROLLER_CONFIG = ControllerConfig(
    MAX_STEPS, DROP_THRESHOLD, DROP_COOLDOWN, DISH_SIZE, DT
)
