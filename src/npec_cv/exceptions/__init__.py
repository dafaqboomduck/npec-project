"""Exceptions module for the npec_cv package."""

from .exceptions import (
    ConfigurationError,
    CVPipelineError,
    InferenceError,
    InitializationError,
)

__all__ = [
    "CVPipelineError",
    "ConfigurationError",
    "InferenceError",
    "InitializationError",
]
