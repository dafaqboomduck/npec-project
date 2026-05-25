"""Environment stats service.

Reads GPU temperature and memory usage from the host system using
psutil (always available) and pynvml (optional - omitted when no
NVIDIA GPU is present).

All reads are wrapped in try/except so a missing GPU or failing sensor
never raises and exception - it returns None for that field instead.
"""

from __future__ import annotations

import logging

from api.src.configs.overview import (
    MEMORY_CRITICAL_THRESHOLD,
    MEMORY_DEGRADED_THRESHOLD,
)
from api.src.schemas.overview import EnvironmentResponse, SystemStatus

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


def _get_cpu_percent() -> float | None:
    """Return the current system-wide CPU utilization as a percentage."""
    try:
        import psutil

        return round(psutil.cpu_percent(interval=None), 1)
    except Exception:
        return None


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


def _get_gpu_utilisation() -> float | None:
    """Return GPU utilisation (0-100%) or None."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(rates.gpu)
    except Exception:
        return None


def _get_disk_percent() -> int | None:
    """Returns the disk usage as a percentage.

    Returns:
        Disk usage as an int.
    """
    try:
        import psutil

        return int(psutil.disk_usage("/").percent)
    except Exception:
        return None


def _get_vram_percent() -> float | None:
    """Return VRAM used as a percentage or None."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return round((mem.used / mem.total) * 100, 1)
    except Exception:
        return None


def _cuda_available() -> bool:
    """Returns True if cuda is available, otherwise False."""
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetCount() > 0
    except Exception:
        return False


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
    return EnvironmentResponse(
        gpu_temp_celsius=_get_gpu_temp(),
        gpu_utilisation_percent=_get_gpu_utilisation(),
        vram_percent=_get_vram_percent(),
        cpu_percent=_get_cpu_percent(),
        disk_percent=_get_disk_percent(),
        cuda_available=_cuda_available(),
        memory_percent=memory_percent,
        system_status=_system_status(memory_percent),
    )
