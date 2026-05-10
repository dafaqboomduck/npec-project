"""Environment stats service.

Reads GPU temperature and memory usage from the host system using
psutil (always available) and pynvml (optional - omitted when no
NVIDIA GPU is present).

All reads are wrapped in try/except so a missing GPU or failing sensor
never raises and exception - it returns None for that field instead.
"""

from __future__ import annotations

import logging

from ..configs.overview import MEMORY_CRITICAL_THRESHOLD, MEMORY_DEGRADED_THRESHOLD
from ..schemas.overview import EnvironmentResponse, SystemStatus

logger = logging.getLogger(__name__)


def _get_memory_percent() -> int:
    """Return the current system memory usage as a percentage.

    Returns:
        Memory usage percent: (0-100), or `0` on failure.
    """
    try:
        import psutil

        return int(psutil.virtual_memory().percent)
    except Exception as exc:
        logger.warning("Failed to read memory usage: %s", exc)
        return 0


def _get_gpu_temp() -> float | None:
    """Return the GPU temperature in degrees Celsius, or `None`.

    Returns:
        GPU temperature as a float, or `None`.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        return float(temp)
    except Exception:
        # No GPU, driver not installed, or pynvml not available
        return


def _system_status(memory_percent: int) -> SystemStatus:
    """Derive system status from memory percent using configured thresholds.

    Args:
        memory_percent: Current memory usage (0-100).

    Returns:
        `SystemStatus`
    """
    if memory_percent >= MEMORY_CRITICAL_THRESHOLD:
        return SystemStatus.CRITICAL
    if memory_percent >= MEMORY_DEGRADED_THRESHOLD:
        return SystemStatus.DEGRADED
    return SystemStatus.OPTIMAL


def get_environment_stats() -> EnvironmentResponse:
    """Read current system environment stats.

    Returns:
        `EnvironmentResponse` with `gpu_temp_celsius`,
        `memory_percent`, and `system_status` derived from
        memory usage.
    """
    memory_percent = _get_memory_percent()
    gpu_temp = _get_gpu_temp()

    return EnvironmentResponse(
        gpu_temp_celsius=gpu_temp,
        memory_percent=memory_percent,
        system_status=_system_status(memory_percent),
    )
