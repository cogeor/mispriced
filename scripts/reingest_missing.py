"""
Re-ingest tickers with missing financial data.

Usage:
    python scripts/reingest_missing.py --quarters 2024-Q1,2024-Q2,2024-Q3
    python scripts/reingest_missing.py --tickers TSLA,WMT,XOM
    python scripts/reingest_missing.py --file tickers_to_reingest.txt
"""

import asyncio
import sys
import os
import sqlite3
from typing import List, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.db.session import SessionLocal
from src.db.repositories.ticker_repo import TickerRepository
from src.db.repositories.snapshot_repo import SnapshotRepository
from src.providers.yahoo.client import YFinanceProvider
from src.providers.fx_rates import FX_PROVIDER
from src.ingestion.service import IngestionService
from src.ingestion.config import IngestionConfig
from src.ingestion.report import IngestionReport


def get_tickers_with_missing_data(quarters: Optional[List[str]] = None, min_mcap: float = 1e9) -> List[str]:
    """Get tickers that have NULL financials in specified quarters."""
    conn = sqlite3.connect('mispriced.db')

    # Map quarter strings to date ranges
    quarter_dates = {
        '2024-Q1': ('2024-01-01', '2024-03-31'),
        '2024-Q2': ('2024-04-01', '2024-06-30'),
        '2024-Q3': ('2024-07-01', '2024-09-30'),
    }

    if quarters:
        conditions = []
        for q in quarters:
            if q in quarter_dates:
                start, end = quarter_dates[q]
                conditions.append(f"(date(snapshot_timestamp) BETWEEN '{start}' AND '{end}')")
        date_filter = " OR ".join(conditions)
    else:
        # Default: Q1-Q3 2024
        date_filter = "(date(snapshot_timestamp) BETWEEN '2024-01-01' AND '2024-09-30')"

    query = f"""
    SELECT DISTINCT ticker
    FROM financial_snapshots
    WHERE ({date_filter})
    AND total_revenue IS NULL AND net_income IS NULL AND ebitda IS NULL
    AND market_cap_t0 > {min_mcap}
    ORDER BY market_cap_t0 DESC
    """

    cursor = conn.cursor()
    cursor.execute(query)
    tickers = [row[0] for row in cursor.fetchall()]
    conn.close()

    return tickers


async def reingest_tickers(tickers: List[str], batch_size: int = 50) -> IngestionReport:
    """Re-ingest specified tickers."""
    print(f"Re-ingesting {len(tickers)} tickers...")

    db = SessionLocal()
    ticker_repo = TickerRepository(db)
    snapshot_repo = SnapshotRepository(db)
    provider = YFinanceProvider()

    service = IngestionService(
        provider=provider,
        fx_provider=FX_PROVIDER,
        snapshot_repo=snapshot_repo,
        ticker_repo=ticker_repo,
        config=IngestionConfig()
    )

    report = IngestionReport()

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"\nBatch {i // batch_size + 1}: {len(batch)} tickers ({batch[0]}...{batch[-1]})")

        batch_report = await service.ingest_tickers(batch)

        report.attempted.extend(batch_report.attempted)
        report.successes.update(batch_report.successes)
        report.failures.update(batch_report.failures)

    db.close()
    return report


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Re-ingest tickers with missing data")
    parser.add_argument("--quarters", help="Quarters to check (e.g., 2024-Q1,2024-Q2,2024-Q3)")
    parser.add_argument("--tickers", help="Specific tickers to re-ingest (comma-separated)")
    parser.add_argument("--file", help="File with tickers to re-ingest (one per line)")
    parser.add_argument("--min-mcap", type=float, default=1e9, help="Minimum market cap (default: 1B)")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size (default: 50)")
    parser.add_argument("--dry-run", action="store_true", help="Just list tickers, don't ingest")

    args = parser.parse_args()

    # Determine tickers
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]
    elif args.file:
        with open(args.file, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
    else:
        quarters = args.quarters.split(",") if args.quarters else None
        tickers = get_tickers_with_missing_data(quarters, args.min_mcap)

    print(f"Found {len(tickers)} tickers to re-ingest")

    if args.dry_run:
        print("\nTickers (dry run):")
        for i, t in enumerate(tickers[:50]):
            print(f"  {t}")
        if len(tickers) > 50:
            print(f"  ... and {len(tickers) - 50} more")
        return

    if not tickers:
        print("No tickers to re-ingest!")
        return

    report = await reingest_tickers(tickers, args.batch_size)

    print("\n" + "=" * 50)
    print("Re-ingestion Report:")
    print(f"  Successes: {report.success_count}")
    print(f"  Failures: {report.failure_count}")

    if report.failures:
        print("\nFailed tickers (first 10):")
        for ticker, reason in list(report.failures.items())[:10]:
            print(f"  {ticker}: {reason}")


if __name__ == "__main__":
    asyncio.run(main())
