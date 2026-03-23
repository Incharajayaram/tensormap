"""add export job table

Revision ID: 4c6d3c8f2b10
Revises: 3b7f9b1e2c01
Create Date: 2026-03-23 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "4c6d3c8f2b10"
down_revision = "3b7f9b1e2c01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "export_job",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("export_id", sa.String(length=64), nullable=False, unique=True, index=True),
        sa.Column("run_id", sa.String(length=64), nullable=True),
        sa.Column("model_id", sa.Integer(), sa.ForeignKey("model_basic.id"), nullable=True, index=True),
        sa.Column("model_name", sa.String(length=50), nullable=False),
        sa.Column("format", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="queued"),
        sa.Column("path", sa.String(length=500), nullable=True),
        sa.Column("error_text", sa.String(length=500), nullable=True),
        sa.Column("created_on", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_on", sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("export_job")
