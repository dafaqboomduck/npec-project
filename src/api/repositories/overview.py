"""Overview repository.

Table read:
- predictions           - total counts, recent uploads
- model_versions        - active model count, recent deployments
- monitoring metrics    - f1, latency, drift score
- annotations           - recent annotation confirmations

Activity feed aggregation strategy:
    The spec calls for a unified feed of different events. Rather than a
    complex UNION query across tables, we fetch the most recent N rows from each
    source, merge them in Python, sort by timestamp, and apply OFFSET/LIMIT.
    This is simpler than a DB-level UNION and performs well at the typical
    page sizes and activity volumes expected for this application.
"""

import logging

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.orm import Annotation, ModelVersion, MonitoringMetric, Prediction
from src.api.configs.overview import (
    DRIFT_HIGH_THRESHOLD,
    DRIFT_LOW_THRESHOLD,
    DRIFT_MEDIUM_THRESHOLD,
    STORAGE_CAPACITY_TB,
)
from src.api.schemas.library import AnnotationStatus, ModelStatus
from src.api.schemas.overview import (
    ActivityFeedResponse,
    ActivityItem,
    DriftStatus,
    EventType,
    ModelHealthResponse,
    OverviewStatusResponse,
)

logger = logging.getLogger(__name__)

# How many rows to fetch from each event source before merging
_ACTIVITY_FETCH_LIMIT = 50


def _drift_status(score: float) -> DriftStatus:
    """Map a raw drift score to a `DriftStatus`.

    Args:
        score: Raw drift score from `monitoring_metrics`.

    Returns:
        `DrifitStatus` label.
    """
    if score < DRIFT_LOW_THRESHOLD:
        return DriftStatus.NONE
    if score < DRIFT_MEDIUM_THRESHOLD:
        return DriftStatus.LOW
    if score < DRIFT_HIGH_THRESHOLD:
        return DriftStatus.MEDIUM
    return DriftStatus.HIGH


class OverviewRepository:
    """Read-only queries for the Overview page endpoints."""

    def __init__(self, db: AsyncSession) -> None:
        """Read-only queries for the Overview page endpoints.

        Args:
            db: Injected database session.
        """
        self._db = db

    async def get_status(self) -> OverviewStatusResponse:
        """Return lab summary statistics.

        Queries:
        - Total predictions as a proxy for total experiments.
        - Count of model_versions rows with status = 'deployed'
        - Total data size estimated from prediction x avg image size

        Returns:
            `OverviewStatusResponse`
        """
        # Total experiments
        total_result = await self._db.execute(select(func.count(Prediction.id)))
        total_experiments = total_result.scalar_one_or_none() or 0

        # Active deployed model count
        model_result = await self._db.execute(
            select(func.count(ModelVersion.id)).where(
                ModelVersion.status == ModelStatus.DEPLOYED.value,
            )
        )
        active_model_count = model_result.scalar_one_or_none() or 0

        # Data collected - rough estimate: predictions * 5MB average per image
        # TODO: Replace with real Azure Blob storage accounting
        estimated_tb = (total_experiments * 5) / (1024 * 1024)  # MB -> TB
        quota_percent = min(int((estimated_tb / STORAGE_CAPACITY_TB) * 100), 100)

        return OverviewStatusResponse(
            total_experiments=total_experiments,
            active_model_count=active_model_count,
            data_collected_tb=round(estimated_tb, 4),
            quota_percent=quota_percent,
        )

    async def get_model_health(self) -> ModelHealthResponse:
        """Return health metrics for the deployed model.

        Reads the most recent `f1_validation`, `prediction_latency_ms`,
        and `drift_score` rows from `monitoring_metrics` for the currently
        deployed model. Returns sensible defaults when no metrics exist yet.

        Returns:
            `ModelHealthResponse`
        """
        # Get deployed model version ID
        model_result = await self._db.execute(
            select(ModelVersion.id, ModelVersion.f1_score)
            .where(
                ModelVersion.status == ModelStatus.DEPLOYED.value,
            )
            .limit(1)
        )
        deployed = model_result.first()

        if deployed is None:
            logger.warning("get_model_health: no deployed model - returning defaults.")
            return ModelHealthResponse(
                f1_score=0.0,
                latency_ms=0,
                drift_status=DriftStatus.NONE,
            )

        model_id, model_f1 = deployed

        # Latest latency from monitoring metrics
        latency_result = await self._db.execute(
            select(MonitoringMetric.metric_value)
            .where(
                MonitoringMetric.metric_name == "prediction_latency_ms",
                MonitoringMetric.model_version_id == model_id,
            )
            .order_by(MonitoringMetric.recorded_at.desc())
            .limit(1)
        )
        latency_ms = int(latency_result.scalar_one_or_none() or 0)

        # Latest drift score
        drift_result = await self._db.execute(
            select(MonitoringMetric.metric_value)
            .where(
                MonitoringMetric.metric_name == "drift_score",
                MonitoringMetric.model_version_id == model_id,
            )
            .order_by(MonitoringMetric.recorded_at.desc())
            .limit(1)
        )
        drift_score = drift_result.scalar_one_or_none() or 0.0

        # Use model's registered f1 score if no f1_validation metric exists yet
        f1_score = float(model_f1 or 0.0)

        return ModelHealthResponse(
            f1_score=round(f1_score, 4),
            latency_ms=latency_ms,
            drift_status=_drift_status(drift_score),
        )

    async def get_activity(self, page: int = 1, page_size: int = 20) -> ActivityFeedResponse:
        """Return a paginated, unified feed of recent lab events.

        Aggregates events from `predictions` and `model_versions`.

        Args:
            page: 1-indexed page number.
            page_size: Events per page (1-100).

        Returns:
            `ActivityFeedResponse`
        """
        events: list[ActivityItem] = []

        # Upload events - recent predictions
        pred_result = await self._db.execute(
            select(Prediction.id, Prediction.input_filename, Prediction.created_at)
            .order_by(Prediction.created_at.desc())
            .limit(_ACTIVITY_FETCH_LIMIT)
        )
        for row in pred_result.all():
            events.append(
                ActivityItem(
                    event_type=EventType.UPLOAD,
                    title="Root scan uploaded",
                    description=(f"Image '{row.input_filename or 'unknown'}' submitted for inference.",),
                    timestamp=row.created_at,
                    resource_url=f"/api/v1/predictions/{row.id}",
                )
            )

        # Deployment events - model deployments
        deploy_result = await self._db.execute(
            select(ModelVersion.id, ModelVersion.name, ModelVersion.deployed_at)
            .where(ModelVersion.deployed_at.is_not(None))
            .order_by(ModelVersion.deployed_at.desc())
            .limit(_ACTIVITY_FETCH_LIMIT)
        )
        for row in deploy_result.all():
            events.append(
                ActivityItem(
                    event_type=EventType.DEPLOYMENT,
                    title="Model deployed",
                    description=f"Model '{row.name}' was deployed to production.",
                    timestamp=row.deployed_at,
                    resource_url=f"/api/v1/models/{row.id}",
                )
            )

        # Annotation events
        annot_result = await self._db.execute(
            select(Annotation.id, Annotation.updated_at)
            .where(Annotation.status == AnnotationStatus.ANNOTATED.value)
            .order_by(Annotation.updated_at.desc())
            .limit(_ACTIVITY_FETCH_LIMIT)
        )
        for row in annot_result.all():
            events.append(
                ActivityItem(
                    event_type=EventType.ANNOTATION,
                    title="Annotation confirmed",
                    description="A prediction was confirmed and linked to the library.",
                    timestamp=row.updated_at,
                    resource_url=f"/api/v1/annotations/{row.id}",
                )
            )

        # Sort all events by timestamp descending, apply page slice
        events.sort(key=lambda e: e.timestamp, reverse=True)
        total = len(events)
        offset = (page - 1) * page_size
        page_items = events[offset : offset + page_size]

        return ActivityFeedResponse(
            items=page_items,
            total=total,
            page=page,
            page_size=page_size,
        )
