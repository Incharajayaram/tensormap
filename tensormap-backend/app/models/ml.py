import uuid as uuid_pkg
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import JSON, Column, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from app.models.project import Project


class ModelBasic(SQLModel, table=True):
    """Persisted ML model configuration (optimizer, loss, epochs, etc.)."""

    __tablename__ = "model_basic"

    id: int | None = Field(default=None, primary_key=True)
    model_name: str = Field(max_length=50, nullable=False, unique=True)
    file_id: uuid_pkg.UUID | None = Field(
        default=None, sa_column=Column(PgUUID(as_uuid=True), ForeignKey("data_file.id"), index=True, nullable=True)
    )
    project_id: uuid_pkg.UUID | None = Field(
        sa_column=Column(PgUUID(as_uuid=True), ForeignKey("project.id", ondelete="CASCADE"), index=True, nullable=True)
    )
    model_type: int | None = Field(default=None, nullable=True)
    target_field: str | None = Field(default=None, max_length=50)
    training_split: float | None = Field(default=None, nullable=True)
    optimizer: str | None = Field(default=None, max_length=50, nullable=True)
    metric: str | None = Field(default=None, max_length=50, nullable=True)
    epochs: int | None = Field(default=None, nullable=True)
    batch_size: int | None = Field(default=None, nullable=True)
    loss: str | None = Field(default=None, max_length=50, nullable=True)
    # sa_column is required here because SQLModel infers nullable from the
    # Python type hint alone, which does not produce the correct DDL for JSON
    # columns.  Explicit Column(JSON, nullable=True) ensures the database
    # column is created with the right type and NULL constraint.
    graph_json: dict | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    created_on: datetime | None = Field(default=None, sa_column=Column(DateTime, server_default=func.now()))
    updated_on: datetime | None = Field(
        default=None, sa_column=Column(DateTime, server_default=func.now(), onupdate=func.now())
    )

    project: Optional["Project"] = Relationship(back_populates="models")
    file: Optional["DataFile"] = Relationship(back_populates="model_basic")
    configs: list["ModelConfigs"] = Relationship(
        back_populates="model",
        sa_relationship_kwargs={"cascade": "all,delete"},
    )


class ModelConfigs(SQLModel, table=True):
    """Flattened key-value pairs capturing the full model graph configuration."""

    __tablename__ = "model_configs"

    id: int | None = Field(default=None, primary_key=True)
    parameter: str = Field(max_length=50, nullable=False)
    value: str = Field(max_length=50, nullable=False)
    model_id: int = Field(foreign_key="model_basic.id", index=True)
    created_on: datetime | None = Field(default=None, sa_column=Column(DateTime, server_default=func.now()))
    updated_on: datetime | None = Field(
        default=None, sa_column=Column(DateTime, server_default=func.now(), onupdate=func.now())
    )

    model: ModelBasic | None = Relationship(back_populates="configs")


# --- Run history + metrics ---
class RunHistory(SQLModel, table=True):
    """Persisted training run metadata."""

    __tablename__ = "run_history"

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(max_length=64, nullable=False, unique=True, index=True)
    model_id: int | None = Field(default=None, foreign_key="model_basic.id", index=True)
    model_name: str = Field(max_length=50, nullable=False)
    status: str = Field(default="queued", max_length=20)
    started_at: datetime | None = Field(default=None, sa_column=Column(DateTime, nullable=True))
    completed_at: datetime | None = Field(default=None, sa_column=Column(DateTime, nullable=True))
    updated_on: datetime | None = Field(
        default=None, sa_column=Column(DateTime, server_default=func.now(), onupdate=func.now())
    )
    summary: dict | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    error_text: str | None = Field(default=None, max_length=500)

    model: ModelBasic | None = Relationship()


class RunMetric(SQLModel, table=True):
    """Persisted per-epoch metrics for a training run."""

    __tablename__ = "run_metric"

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(max_length=64, nullable=False, index=True)
    phase: str = Field(max_length=10, nullable=False)  # train / val / eval
    epoch: int | None = Field(default=None, nullable=True)
    step: int | None = Field(default=None, nullable=True)
    loss: float | None = Field(default=None, nullable=True)
    metric: float | None = Field(default=None, nullable=True)
    metrics: dict | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    timestamp: datetime | None = Field(default=None, sa_column=Column(DateTime, nullable=True))


# --- Export jobs ---
class ExportJob(SQLModel, table=True):
    """Persisted export jobs for model artifacts."""

    __tablename__ = "export_job"

    id: int | None = Field(default=None, primary_key=True)
    export_id: str = Field(max_length=64, nullable=False, unique=True, index=True)
    run_id: str | None = Field(default=None, max_length=64, nullable=True)
    model_id: int | None = Field(default=None, foreign_key="model_basic.id", index=True)
    model_name: str = Field(max_length=50, nullable=False)
    format: str = Field(max_length=20, nullable=False)
    status: str = Field(default="queued", max_length=20)
    path: str | None = Field(default=None, max_length=500)
    error_text: str | None = Field(default=None, max_length=500)
    created_on: datetime | None = Field(default=None, sa_column=Column(DateTime, server_default=func.now()))
    updated_on: datetime | None = Field(
        default=None, sa_column=Column(DateTime, server_default=func.now(), onupdate=func.now())
    )


class InterpretabilityReport(SQLModel, table=True):
    """Persisted interpretability reports."""

    __tablename__ = "interpretability_report"

    id: int | None = Field(default=None, primary_key=True)
    report_id: str = Field(max_length=64, nullable=False, unique=True, index=True)
    run_id: str | None = Field(default=None, max_length=64, nullable=True)
    model_id: int | None = Field(default=None, foreign_key="model_basic.id", index=True)
    model_name: str = Field(max_length=50, nullable=False)
    problem_type: str = Field(max_length=30, nullable=False)
    status: str = Field(default="running", max_length=20)
    summary: dict | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    artifacts: dict | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    error_text: str | None = Field(default=None, max_length=500)
    created_on: datetime | None = Field(default=None, sa_column=Column(DateTime, server_default=func.now()))
    updated_on: datetime | None = Field(
        default=None, sa_column=Column(DateTime, server_default=func.now(), onupdate=func.now())
    )


class TuningJob(SQLModel, table=True):
    """Persisted hyperparameter tuning jobs."""

    __tablename__ = "tuning_job"

    id: int | None = Field(default=None, primary_key=True)
    job_id: str = Field(max_length=64, nullable=False, unique=True, index=True)
    model_id: int | None = Field(default=None, foreign_key="model_basic.id", index=True)
    model_name: str = Field(max_length=50, nullable=False)
    strategy: str = Field(max_length=20, nullable=False)
    objective: str = Field(max_length=50, nullable=False)
    max_trials: int = Field(default=10, nullable=False)
    status: str = Field(default="running", max_length=20)
    created_on: datetime | None = Field(default=None, sa_column=Column(DateTime, server_default=func.now()))
    updated_on: datetime | None = Field(
        default=None, sa_column=Column(DateTime, server_default=func.now(), onupdate=func.now())
    )


class TuningTrial(SQLModel, table=True):
    """Persisted tuning trial results."""

    __tablename__ = "tuning_trial"

    id: int | None = Field(default=None, primary_key=True)
    job_id: str = Field(max_length=64, nullable=False, index=True)
    trial_id: str = Field(max_length=64, nullable=False, unique=True, index=True)
    status: str = Field(default="completed", max_length=20)
    params: dict | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    score: float | None = Field(default=None, nullable=True)
    run_id: str | None = Field(default=None, max_length=64, nullable=True)
    error_text: str | None = Field(default=None, max_length=500)
    created_on: datetime | None = Field(default=None, sa_column=Column(DateTime, server_default=func.now()))
# Resolve forward references
from app.models.data import DataFile  # noqa: E402

ModelBasic.model_rebuild()
