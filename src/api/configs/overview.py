"""Overview configuration constants."""

from __future__ import annotations

# Drift detection thresholds
#
# These map a raw drift_score metric value to a DriftStatus label.
# Scores are compared as:   score < LOW_THRESHOLD -> none,
#                           score < MEDIUM_THRESHOLD -> low,
#                           score < HIGH_THRESHOLD -> medium,
#                           score >= HIGH_THRESHOLD -> high

DRIFT_LOW_THRESHOLD: float = 0.1
DRIFT_MEDIUM_THRESHOLD: float = 0.3
DRIFT_HIGH_THRESHOLD: float = 0.6

# Environment thresholds
#
# Memory percent above which system_status degrades

MEMORY_DEGRADED_THRESHOLD: int = 80  # %
MEMORY_CRITICAL_THRESHOLD: int = 95  # %

# Storage quota
#
# Total storage capacity used to compute quota_percent
# Override via NPEC_STORAGE_CAPACITY_TB env var

STORAGE_CAPACITY_TB: float = 10.0
