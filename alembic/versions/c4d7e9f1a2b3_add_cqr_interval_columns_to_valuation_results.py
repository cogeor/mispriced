"""add cqr interval columns to valuation_results

Revision ID: c4d7e9f1a2b3
Revises: b2f9a4d6e7c1
Create Date: 2026-05-23 18:30:00.000000

Adds three nullable Numeric columns to ``valuation_results`` so the
Conformalized Quantile Regression (CQR) pipeline can persist 90%
prediction interval bounds (``predicted_mcap_lo`` / ``predicted_mcap_hi``)
and the conformal alpha (``conformal_alpha = 1 - target_coverage``).
All existing rows remain intact with NULL for the new columns.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c4d7e9f1a2b3"
down_revision: Union[str, None] = "b2f9a4d6e7c1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("valuation_results") as batch_op:
        batch_op.add_column(sa.Column("predicted_mcap_lo", sa.Numeric(), nullable=True))
        batch_op.add_column(sa.Column("predicted_mcap_hi", sa.Numeric(), nullable=True))
        batch_op.add_column(sa.Column("conformal_alpha", sa.Numeric(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("valuation_results") as batch_op:
        batch_op.drop_column("conformal_alpha")
        batch_op.drop_column("predicted_mcap_hi")
        batch_op.drop_column("predicted_mcap_lo")
