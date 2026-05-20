#!/usr/bin/env python
"""
Repair non-USD financial snapshots whose monetary fields were never
converted to USD because the FX provider was unreachable at ingestion
time. Targets rows where ``original_currency != 'USD'`` AND
``fx_rate_to_usd IS NULL`` — values are still in raw native currency
(e.g. ~5.7T INR for BAJFINANCE.NS), but downstream code reads them as
USD, so the dashboard surfaces them at the wrong magnitude.

Scope:
  Handled here:  INR, CNY, EUR, HKD, CHF — currencies for which yfinance
                 consistently reports both ``marketCap`` and statement
                 fields in the same native currency, so a single
                 currency→USD rate corrects all monetary fields.

  Skipped:       GBp — UK tickers have a separate, source-dependent
                 quirk (yfinance returns ``info['marketCap']`` in GBP
                 but ``info['totalRevenue']`` in GBp pence, while the
                 DataFrame financials are in GBP). A single
                 multiplicative correction can't fix that without
                 source-level metadata. GBp tickers are left in their
                 current state; fixing them requires a corrected
                 schema mapper and a fresh re-ingest.

After this runs, regenerate predictions with:

    python scripts/run_quarterly_update.py --skip-ingestion --reprocess
"""

import logging
import os
import sys
from datetime import date
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.models.snapshot import FinancialSnapshot
from src.db.session import SessionLocal
from src.providers.fx_rates import FX_PROVIDER

logger = logging.getLogger("fix_currency_conversion")

MONETARY_FIELDS = [
    "market_cap_t0",
    "total_revenue",
    "gross_profit",
    "ebitda",
    "operating_income",
    "net_income",
    "total_debt",
    "total_cash",
    "total_assets",
    "free_cash_flow",
    "operating_cash_flow",
    "capex",
    "book_value",
]

SUPPORTED_CURRENCIES = {"INR", "CNY", "EUR", "HKD", "CHF"}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    session = SessionLocal()
    cache: Dict[Tuple[str, date], float] = {}

    try:
        rows = (
            session.query(FinancialSnapshot)
            .filter(
                FinancialSnapshot.original_currency.in_(SUPPORTED_CURRENCIES),
                FinancialSnapshot.fx_rate_to_usd.is_(None),
            )
            .all()
        )
        logger.info(f"Found {len(rows)} broken rows in supported currencies")

        fixed = 0
        skipped = 0
        for snap in rows:
            ts = snap.snapshot_timestamp
            as_of = ts.date() if hasattr(ts, "date") else ts
            currency = snap.original_currency

            key = (currency, as_of)
            if key not in cache:
                try:
                    cache[key] = FX_PROVIDER.get_rate(currency, "USD", as_of)
                    logger.info(f"  Rate {currency}->USD @ {as_of}: {cache[key]:.6f}")
                except Exception as e:
                    logger.warning(f"  Skip {snap.ticker}@{as_of}: FX lookup failed: {e}")
                    skipped += 1
                    continue
            rate = cache[key]

            for field in MONETARY_FIELDS:
                value = getattr(snap, field, None)
                if value is not None:
                    setattr(snap, field, float(value) * rate)

            snap.fx_rate_to_usd = rate
            snap.stored_currency = "USD"
            snap.currency_validated = True
            fixed += 1

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    logger.info(f"Done. fixed={fixed}, skipped={skipped}, distinct_fx_cells={len(cache)}")


if __name__ == "__main__":
    main()
