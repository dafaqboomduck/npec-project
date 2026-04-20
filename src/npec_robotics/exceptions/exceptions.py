class RoboticsPipelineError(Exception):
    """Base exception for robotics pipeline errors."""

    pass


class InitializationError(RoboticsPipelineError):
    """Raised when robotics components fail to initialize."""

    pass
