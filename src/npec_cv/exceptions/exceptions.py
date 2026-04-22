"""Custom exceptions for the pipeline."""


class CVPipelineError(Exception):
    """Base exception for pipeline-specific errors."""

    pass


class InferenceError(CVPipelineError):
    """Raised when model inference fails."""

    pass


class InitializationError(CVPipelineError):
    """Raised when pipeline components fail to initialize."""

    pass


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""

    pass
