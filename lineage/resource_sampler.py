"""System resource sampling for the inference audit path.

Dependency-guarded so CPU-only deployments and test environments
without psutil/pynvml never raise — the audit write must never break on
a sampling hiccup, so every function returns None (or Nones) on failure.

Two kinds of measurement:
* Instantaneous gauges (RAM, disk, GPU, system CPU%) read once at persist.
* Process CPU consumed *during* inference, from a cumulative CPU-time
  delta bracketed around the prediction (process_cpu_seconds +
  process_cpu_pct). 100.0 == one full core; >100 == multiple cores.

Why a delta, not cpu_percent(interval=...): interval=0.1 would block the
asyncio event loop 100 ms per request, and at persist time it measures
post-inference idle. The delta is non-blocking and brackets the work.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, TypedDict

log = logging.getLogger(__name__)

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

# nvidia-ml-py installs the `pynvml` module. Import may succeed on a
# CPU-only host; nvmlInit() is what fails there, handled below.
try:
    import pynvml  # type: ignore
except Exception:  # pragma: no cover
    pynvml = None  # type: ignore[assignment]

_DISK_PATH = os.environ.get("NPEC_RESOURCE_DISK_PATH", "/")
_gpu_state: Optional[bool] = None  # cache so we don't retry nvmlInit per request


class ResourceSample(TypedDict, total=False):
    """One snapshot of host resource gauges; any field may be None."""

    cpu_pct_system: Optional[float]
    cpu_count: Optional[int]
    ram_used_mb: Optional[int]
    ram_available_mb: Optional[int]
    disk_free_mb: Optional[int]
    gpu_pct: Optional[float]
    vram_used_mb: Optional[int]


def process_cpu_seconds() -> Optional[float]:
    """Cumulative CPU seconds (user+system) for this process, or None."""
    if psutil is None:
        return None
    try:
        t = psutil.Process().cpu_times()
        return float(t.user + t.system)
    except Exception:
        log.debug("process_cpu_seconds failed", exc_info=True)
        return None


def process_cpu_pct(
    start_seconds: Optional[float],
    end_seconds: Optional[float],
    latency_ms: Optional[int],
) -> Optional[float]:
    """CPU used during the window as % of one core. 100.0 == one core."""
    if start_seconds is None or end_seconds is None:
        return None
    if not latency_ms or latency_ms <= 0:
        return None
    delta = end_seconds - start_seconds
    if delta < 0:
        return None
    return round(delta / (latency_ms / 1000.0) * 100.0, 1)


def _gpu_available() -> bool:
    global _gpu_state
    if _gpu_state is not None:
        return _gpu_state
    if pynvml is None:
        _gpu_state = False
        return False
    try:
        pynvml.nvmlInit()
        _gpu_state = pynvml.nvmlDeviceGetCount() > 0
    except Exception:
        _gpu_state = False
    return _gpu_state


def _sample_gpu() -> tuple[Optional[float], Optional[int]]:
    if not _gpu_available():
        return None, None
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        return float(util.gpu), int(mem.used // (1024 * 1024))
    except Exception:
        log.debug("GPU sample failed", exc_info=True)
        return None, None


def sample_system() -> ResourceSample:
    """Read instantaneous gauges. Never raises."""
    out: ResourceSample = {
        "cpu_pct_system": None,
        "cpu_count": None,
        "ram_used_mb": None,
        "ram_available_mb": None,
        "disk_free_mb": None,
        "gpu_pct": None,
        "vram_used_mb": None,
    }
    if psutil is not None:
        try:
            # interval=None: non-blocking, since-last-call. Coarse but
            # safe in the event loop; the real signal is process_cpu_pct.
            out["cpu_pct_system"] = float(psutil.cpu_percent(interval=None))
            out["cpu_count"] = psutil.cpu_count(logical=True)
            vm = psutil.virtual_memory()
            out["ram_used_mb"] = int(vm.used // (1024 * 1024))
            out["ram_available_mb"] = int(vm.available // (1024 * 1024))
            out["disk_free_mb"] = int(psutil.disk_usage(_DISK_PATH).free // (1024 * 1024))
        except Exception:
            log.debug("system sample failed", exc_info=True)
    out["gpu_pct"], out["vram_used_mb"] = _sample_gpu()
    return out
