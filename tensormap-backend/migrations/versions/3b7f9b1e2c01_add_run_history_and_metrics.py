"""add run history and metrics tables

Revision ID: 3b7f9b1e2c01
Revises: e8c4d7f91a2b
Create Date: 2026-03-23 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "3b7f9b1e2c01"
down_revision = "e8c4d7f91a2b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "run_history",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.String(length=64), nullable=False, unique=True, index=True),
        sa.Column("model_id", sa.Integer(), sa.ForeignKey("model_basic.id"), nullable=True, index=True),
        sa.Column("model_name", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="queued"),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("updated_on", sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column("summary", sa.JSON(), nullable=True),
        sa.Column("error_text", sa.String(length=500), nullable=True),
    )

    op.create_table(
        "run_metric",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.String(length=64), nullable=False, index=True),
        sa.Column("phase", sa.String(length=10), nullable=False),
        sa.Column("epoch", sa.Integer(), nullable=True),
        sa.Column("step", sa.Integer(), nullable=True),
        sa.Column("loss", sa.Float(), nullable=True),
        sa.Column("metric", sa.Float(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("run_metric")
    op.drop_table("run_history")
