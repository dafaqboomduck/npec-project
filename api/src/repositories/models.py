"""Model repository.

The GET /models/{model_id} returns additional internal fields
(artifact_uri, deployed_at) for admin tokens. The write operations
are admin only.
"""

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.src.db.orm import ModelVersion
from api.src.schemas.library import ModelDetail, ModelStatus, ModelSummary

logger = logging.getLogger(__name__)


def _row_to_summary(row: ModelVersion) -> ModelSummary:
    """Convert a `ModelVersion` ORM row to a `ModelSummary`.

    Args:
        row: The ORM row from the database.

    Returns:
        ModelSummary
    """
    return ModelSummary(
        model_id=row.id,
        name=row.name,
        status=ModelStatus(row.status),
        created_at=row.created_at,
    )


def _row_to_detail(row: ModelVersion, include_admin_fields: bool = False) -> ModelDetail:
    """Convert a `ModelVersion` ORM row to a `ModelDetail`.

    Args:
        row: The ORM row from the database.
        include_admin_fields: If `True`, populate `artifact_uri` and
            `deployed_at`. Pass `False` for user-role tokens.

    Returns:
        ModelDetail
    """
    return ModelDetail(
        model_id=row.id,
        name=row.name,
        status=ModelStatus(row.status),
        created_at=row.created_at,
        description=row.description,
        f1_score=row.f1_score,
        precision=row.precision,
        recall=row.recall,
        inference_latency_ms=row.inference_latency_ms,
        # Admin-only fields
        artifact_uri=row.artifact_uri if include_admin_fields else None,
        deployed_at=row.deployed_at if include_admin_fields else None,
    )


class ModelRepository:
    """Read access to the `model_versions` table."""

    def __init__(self, db: AsyncSession) -> None:
        """Read access to the `model_versions` table.

        Args:
            db: An open async SQLAlchemy session injected by `get_session`.
        """
        self._db = db

    async def create(
        self,
        version: str,
        name: str,
        artifact_uri: str,
        description: str | None = None,
        f1_score: float | None = None,
        precision: float | None = None,
        recall: float | None = None,
        inference_latency_ms: int | None = None,
    ) -> ModelDetail:
        """Register a new model version with status = 'registered'.

        Args:
            version: Unique version string, e.g. 'unet-npec-v1.2.0'.
            name: Human-readable model name.
            artifact_uri: Path to the .keras file on disk.
            description: Optional description.
            f1_score: Optional validation F1 score (0-1).
            precision: Optional precision score (0-1).
            recall: Optional recall score (0-1).
            inference_latency_ms: Optional benchmark latency in milliseconds.

        Returns:
            The created ModelDetail with admin fields populated.

        Raises:
            ValueError: If a model with the same version string already exists.
        """
        # Check version uniqueness before attempting insert
        existing = await self._db.execute(select(ModelVersion).where(ModelVersion.version == version))
        if existing.scalar_one_or_none() is not None:
            raise ValueError(f"Model version '{version}' already exists.")

        row = ModelVersion(
            version=version,
            name=name,
            artifact_uri=artifact_uri,
            description=description,
            status=ModelStatus.REGISTERED.value,
            f1_score=f1_score,
            precision=precision,
            recall=recall,
            inference_latency_ms=inference_latency_ms,
        )
        self._db.add(row)
        await self._db.commit()
        await self._db.refresh(row)
        logger.info("Registered model version '%s' with id %s", version, row.id)
        return _row_to_detail(row, include_admin_fields=True)

    async def list(self) -> list[ModelSummary]:
        """Return all model version summaries ordered by creation date.

        Returns:
            A list of `ModelSummary`
        """
        result = await self._db.execute(select(ModelVersion).order_by(ModelVersion.created_at.desc()))
        rows = result.scalars().all()
        return [_row_to_summary(r) for r in rows]

    async def get(
        self,
        model_id: uuid.UUID,
        include_admin_fields: bool = False,
    ) -> ModelDetail | None:
        """Return the model version with `model_id`, or `None`.

        Args:
            model_id: UUID to look up.
            include_admin_fields: Whether to populate admin-only fields.

        Returns:
            `ModelDetail` if found, otherwise `None`.
        """
        row = await self._db.get(ModelVersion, model_id)
        if row is None:
            return None
        return _row_to_detail(row, include_admin_fields=include_admin_fields)

    async def get_deployed_artifact_uri(self) -> str | None:
        """Return the artifact_uri of the currently deployed model, or None.

        Used by LocalDeploymentService and the health check. Returns None
        if no omodel has status='deployed' or if artifact_uri is NULL.

        Returns:
            artifact_uri string if a deployed model with a non-null URI exists,
            otherwise None.
        """
        result = await self._db.execute(
            select(ModelVersion.artifact_uri).where(ModelVersion.status == ModelStatus.DEPLOYED.value).limit(1)
        )
        return result.scalar_one_or_none()

    async def get_deployed_f1(self) -> float | None:
        """Return the f1_score of the currently deployed model, or None.

        Used by conditional registration to compare against the new model.

        Returns:
            f1_score float if a deployed model with a non-null f1_score exists,
            otherwise None.
        """
        result = await self._db.execute(
            select(ModelVersion.f1_score).where(ModelVersion.status == ModelStatus.DEPLOYED.value).limit(1)
        )
        return result.scalar_one_or_none()

    async def register_and_promote_if_better(
        self,
        version: str,
        name: str,
        artifact_uri: str,
        final_f1: float,
        min_improvement: float = 0.01,
    ) -> tuple[uuid.UUID | None, bool]:
        """Register a new model version if it improves on the deployed model.

        Compares final_f1 against the deployed model's f1_score. If the
        improvement exceeds min_improvement, inserts a new model_versions row
        with status='registered' and writes a model_promoted monitoring metric.

        Does not deploy the model - deployment is separate step via
        POST /models/{id}/deploy. This keeps promotion (quality gate) and
        deployment (operational step) decoupled.

        Args:
            version: Unique version string for the new model.
            name: Human-readable model name.
            artifact_uri: Path to the .keras file on disk.
            final_f1: F1 score achieved by the new model.
            min_improvement: Minimum absolute F1 improvement required.

        Returns:
            (model_id, was_registered) - model_id is None if not registered.
        """
        deployed_f1 = await self.get_deployed_f1()

        # If no deployed model has an F1 score, register unconditionally
        if deployed_f1 is None or final_f1 >= deployed_f1 + min_improvement:
            row = ModelVersion(
                id=uuid.uuid4(),
                version=version,
                name=name,
                artifact_uri=artifact_uri,
                status=ModelStatus.REGISTERED.value,
                f1_score=final_f1,
            )
            self._db.add(row)

            # Write model_promoted monitoring metric
            from api.src.db.orm import MonitoringMetric

            self._db.add(
                MonitoringMetric(
                    metric_name="model_promoted",
                    metric_value=1.0,
                    model_version_id=None,  # not yet deployed
                    labels={
                        "new_version": version,
                        "new_f1": final_f1,
                        "deployed_f1": deployed_f1 or 0.0,
                        "improvement": round(final_f1 - (deployed_f1 or 0.0), 4),
                    },
                    recorded_at=datetime.now(timezone.utc),
                )
            )

            await self._db.commit()
            await self._db.refresh(row)
            logger.info(
                "Registered new model version '%s' (f1=%.4f, improvement=%.4f over %.4f)",
                version,
                final_f1,
                final_f1 - (deployed_f1 or 0.0),
                deployed_f1 or 0.0,
            )
            return row.id, True

        else:
            logger.info(
                "New model f1=%.4f did not improve over deployed f1=%.4f " "(min_improvement=%.4f) - not registering.",
                final_f1,
                deployed_f1,
                min_improvement,
            )
            return None, False

    async def exists(self, model_id: uuid.UUID) -> bool:
        """Return `True` if a model with `model_id` exists.

        Args:
            model_id: UUID to check.

        Returns:
            `True` if the row exists, `False` otherwise.
        """
        return await self._db.get(ModelVersion, model_id) is not None

    async def delete(self, model_id: uuid.UUID) -> bool:
        """Hard-delete the model version with `model_id`.

        Note:
            The DB FK constraint (ON DELETE RESTRICT) will raise an
            `IntegrityError` if predictions reference this model. The router
            catches this and returns 409. Archive the model instead if it has
            associated predictions.

        Args:
            model_id: UUID of the model to delete.

        Returns:
            `True` if deleted, `False` if not found.
        """
        row = await self._db.get(ModelVersion, model_id)
        if row is None:
            return False
        await self._db.delete(row)
        await self._db.commit()
        logger.info("Deleted model version %s", model_id)
        return True
