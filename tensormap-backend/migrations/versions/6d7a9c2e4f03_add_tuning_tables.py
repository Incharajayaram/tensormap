"""add tuning job and trial tables

Revision ID: 6d7a9c2e4f03
Revises: 5a9f2b8c7d21
Create Date: 2026-03-23 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "6d7a9c2e4f03"
down_revision = "5a9f2b8c7d21"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tuning_job",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("job_id", sa.String(length=64), nullable=False, unique=True, index=True),
        sa.Column("model_id", sa.Integer(), sa.ForeignKey("model_basic.id"), nullable=True, index=True),
        sa.Column("model_name", sa.String(length=50), nullable=False),
        sa.Column("strategy", sa.String(length=20), nullable=False),
        sa.Column("objective", sa.String(length=50), nullable=False),
        sa.Column("max_trials", sa.Integer(), nullable=False, server_default="10"),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="running"),
        sa.Column("created_on", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_on", sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    op.create_table(
        "tuning_trial",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("job_id", sa.String(length=64), nullable=False, index=True),
        sa.Column("trial_id", sa.String(length=64), nullable=False, unique=True, index=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="completed"),
        sa.Column("params", sa.JSON(), nullable=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("run_id", sa.String(length=64), nullable=True),
        sa.Column("error_text", sa.String(length=500), nullable=True),
        sa.Column("created_on", sa.DateTime(), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("tuning_trial")
    op.drop_table("tuning_job")
