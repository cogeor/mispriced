"""add currency_validated to financial_snapshots

Revision ID: b2f9a4d6e7c1
Revises: a8f3e2b1c4d5
Create Date: 2026-05-21 01:30:00.000000

Adds a boolean flag marking which snapshot rows have monetary fields
verified to be in USD. Rows where the FX provider failed at ingestion
(``stored_currency != 'USD'`` AND ``original_currency != 'USD'``) and rows
with the GBp/financialCurrency yfinance quirk are marked False so the
valuation pipeline and dashboard can exclude them.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b2f9a4d6e7c1"
down_revision: Union[str, None] = "a8f3e2b1c4d5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("financial_snapshots") as batch_op:
        batch_op.add_column(
            sa.Column(
                "currency_validated",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("1"),
            )
        )

    op.execute(
        """
        UPDATE financial_snapshots
        SET currency_validated = 0
        WHERE original_currency = 'GBp'
           OR (original_currency != 'USD' AND stored_currency != 'USD')
        """
    )

    op.create_index(
        "idx_snapshots_currency_validated",
        "financial_snapshots",
        ["currency_validated"],
    )


def downgrade() -> None:
    op.drop_index("idx_snapshots_currency_validated", table_name="financial_snapshots")
    with op.batch_alter_table("financial_snapshots") as batch_op:
        batch_op.drop_column("currency_validated")
