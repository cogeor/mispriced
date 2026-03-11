#!/usr/bin/env python
"""
Unified quarterly update script.

Runs the full pipeline in one command:
  1. Ingest/refresh financial snapshots for all tracked tickers
  2. Re-ingest tickers with missing data
  3. Run valuation pipeline for all unprocessed quarters
  4. Run sector/index backtests
  5. Generate dashboard JSON
  6. Build frontend (optional)
  7. Run validation checks

Handles multiple missed quarters automatically.

Usage:
    python scripts/run_quarterly_update.py
    python scripts/run_quarterly_update.py --quarter 2025-09-30
    python scripts/run_quarterly_update.py --dry-run
    python scripts/run_quarterly_update.py --skip-ingestion --skip-build
"""

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, date
from typing import Generator, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import func

from src.config.pipeline import (
    CRITICAL_FEATURES,
    MIN_MARKET_CAP,
    MIN_SAMPLES_FOR_CV,
    MIN_SNAPSHOTS_FOR_QUARTER,
    N_CV_FOLDS,
    N_CV_REPEATS,
    TRACKED_INDICES,
    WEB_DIR,
)
from src.db.config import DATABASE_URL
from src.db.models import FinancialSnapshot, ValuationResult
from src.db.session import SessionLocal

logger = logging.getLogger("quarterly_update")


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for the pipeline."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@contextmanager
def step_context(name: str, step_num: int, dry_run: bool = False) -> Generator[None, None, None]:
    """Context manager that logs step start/end with timing."""
    prefix = "[DRY RUN] " if dry_run else ""
    logger.info(f"{prefix}[STEP {step_num} START] {name}")
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"{prefix}[STEP {step_num} DONE] {name} in {elapsed:.1f}s")


def detect_current_quarter(ref_date: Optional[date] = None) -> date:
    """Detect the most recent completed calendar quarter-end date.

    Returns the last day of the most recently completed quarter.
    For example, if today is 2026-03-11, returns 2025-12-31.
    """
    if ref_date is None:
        ref_date = date.today()

    year = ref_date.year
    month = ref_date.month

    # Find the most recent completed quarter
    if month <= 3:
        # We're in Q1, last completed is Q4 of previous year
        return date(year - 1, 12, 31)
    elif month <= 6:
        return date(year, 3, 31)
    elif month <= 9:
        return date(year, 6, 30)
    else:
        return date(year, 9, 30)


def get_quarters_with_snapshots(min_count: int = MIN_SNAPSHOTS_FOR_QUARTER) -> List[datetime]:
    """Get all quarter dates in DB that have enough snapshots for valuation."""
    session = SessionLocal()
    try:
        result = (
            session.query(
                FinancialSnapshot.snapshot_timestamp,
                func.count(FinancialSnapshot.ticker).label("count"),
            )
            .group_by(FinancialSnapshot.snapshot_timestamp)
            .having(func.count(FinancialSnapshot.ticker) >= min_count)
            .order_by(FinancialSnapshot.snapshot_timestamp)
            .all()
        )
        return [r[0] for r in result]
    finally:
        session.close()


def get_quarters_with_valuations() -> List[datetime]:
    """Get all quarter dates that already have valuation results."""
    session = SessionLocal()
    try:
        result = (
            session.query(ValuationResult.snapshot_timestamp)
            .group_by(ValuationResult.snapshot_timestamp)
            .having(func.count(ValuationResult.ticker) >= MIN_SNAPSHOTS_FOR_QUARTER)
            .order_by(ValuationResult.snapshot_timestamp)
            .all()
        )
        return [r[0] for r in result]
    finally:
        session.close()


def find_unprocessed_quarters(target_quarter: Optional[date] = None) -> List[datetime]:
    """Find quarters that have snapshots but no valuations yet.

    If target_quarter is specified, also includes it if it has snapshots.
    Returns all unprocessed quarters sorted chronologically.
    """
    snapshot_quarters = get_quarters_with_snapshots()
    valuation_quarters = set(get_quarters_with_valuations())

    unprocessed = [q for q in snapshot_quarters if q not in valuation_quarters]

    if target_quarter and not unprocessed:
        # If a specific quarter was requested but all are processed,
        # check if it exists in snapshots and force re-processing
        target_dt = datetime(target_quarter.year, target_quarter.month, target_quarter.day)
        for sq in snapshot_quarters:
            if sq.date() == target_quarter or sq == target_dt:
                unprocessed.append(sq)
                break

    return sorted(unprocessed)


# ──────────────────────────────────────────────────────────────
# Steps
# ──────────────────────────────────────────────────────────────


def step_ingest_snapshots(dry_run: bool = False) -> None:
    """Step 1: Ingest/refresh financial snapshots for all tracked tickers."""
    if dry_run:
        logger.info("  Would run: populate_snapshots.py")
        return

    from src.db.repositories.ticker_repo import TickerRepository
    from src.db.repositories.snapshot_repo import SnapshotRepository
    from src.providers.yahoo.client import YFinanceProvider
    from src.providers.fx_rates import FX_PROVIDER
    from src.ingestion.service import IngestionService
    from src.ingestion.config import IngestionConfig
    from src.ingestion.report import IngestionReport

    db = SessionLocal()
    try:
        from src.db.models.ticker import Ticker
        tickers_query = db.query(Ticker.ticker).all()
        tickers = [t[0] for t in tickers_query]
        logger.info(f"  Found {len(tickers)} tickers in database")

        if not tickers:
            logger.warning("  No tickers found. Run ingest_tickers.py first.")
            return

        ticker_repo = TickerRepository(db)
        snapshot_repo = SnapshotRepository(db)
        provider = YFinanceProvider()

        service = IngestionService(
            provider=provider,
            fx_provider=FX_PROVIDER,
            snapshot_repo=snapshot_repo,
            ticker_repo=ticker_repo,
            config=IngestionConfig(),
        )

        report = IngestionReport()
        batch_size = 100

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            logger.info(f"  Batch {i // batch_size + 1}/{(len(tickers) + batch_size - 1) // batch_size}: {len(batch)} tickers")
            batch_report = asyncio.get_event_loop().run_until_complete(
                service.ingest_tickers(batch)
            )
            report.attempted.extend(batch_report.attempted)
            report.successes.update(batch_report.successes)
            report.failures.update(batch_report.failures)

        logger.info(f"  Ingestion complete: {report.success_count} successes, {report.failure_count} failures")
        if report.failures:
            failed_sample = list(report.failures.items())[:5]
            logger.warning(f"  Failed tickers (first 5): {failed_sample}")
    finally:
        db.close()


def step_reingest_missing(quarter_date: date, dry_run: bool = False) -> None:
    """Step 2: Re-ingest tickers with missing financial data."""
    if dry_run:
        logger.info(f"  Would re-ingest missing tickers for quarter {quarter_date}")
        return

    from scripts.reingest_missing import get_tickers_with_missing_data, reingest_tickers

    # Determine which quarter string to use
    q_num = (quarter_date.month - 1) // 3 + 1
    quarter_str = f"{quarter_date.year}-Q{q_num}"

    tickers = get_tickers_with_missing_data([quarter_str])
    if not tickers:
        logger.info(f"  No tickers with missing data for {quarter_str}")
        return

    logger.info(f"  Found {len(tickers)} tickers with missing data for {quarter_str}")
    report = asyncio.get_event_loop().run_until_complete(reingest_tickers(tickers))
    logger.info(f"  Re-ingestion: {report.success_count} successes, {report.failure_count} failures")


def step_run_valuation(quarters: List[datetime], dry_run: bool = False) -> int:
    """Step 3: Run valuation pipeline for specified quarters."""
    if dry_run:
        logger.info(f"  Would run valuation for {len(quarters)} quarters: {[q.date() for q in quarters]}")
        return 0

    from scripts.run_quarterly_valuation import run_all_quarters

    total = run_all_quarters(quarters=quarters)
    logger.info(f"  Valuation complete: {total} total valuations across {len(quarters)} quarters")
    return total


def step_run_backtest(dry_run: bool = False) -> None:
    """Step 4: Run sector/index backtests."""
    if dry_run:
        logger.info("  Would run sector backtest")
        return

    # Run as subprocess to avoid import conflicts with logging setup
    result = subprocess.run(
        [sys.executable, "scripts/run_sector_backtest.py"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"  Backtest failed:\n{result.stderr}")
        raise RuntimeError("Backtest script failed")
    logger.info("  Backtest complete")


def step_generate_dashboard(dry_run: bool = False) -> None:
    """Step 5: Generate dashboard JSON files."""
    if dry_run:
        logger.info("  Would generate dashboard data")
        return

    # Import and run directly
    from scripts.generate_dashboard import main as generate_main

    generate_main()
    logger.info("  Dashboard data generated")


def step_build_frontend(dry_run: bool = False) -> None:
    """Step 6: Build the frontend."""
    if dry_run:
        logger.info(f"  Would build frontend in {WEB_DIR}/")
        return

    web_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), WEB_DIR)
    if not os.path.isdir(web_dir):
        logger.warning(f"  Web directory not found: {web_dir}")
        return

    # Check if npm is available
    result = subprocess.run(
        ["npm", "run", "build"],
        cwd=web_dir,
        capture_output=True,
        text=True,
        shell=True,
    )
    if result.returncode != 0:
        logger.error(f"  Frontend build failed:\n{result.stderr}")
        raise RuntimeError("Frontend build failed")
    logger.info("  Frontend built successfully")


def step_validate(quarters: List[datetime], dry_run: bool = False) -> List[str]:
    """Step 7: Run basic validation checks."""
    if dry_run:
        logger.info("  Would run validation checks")
        return []

    warnings: List[str] = []
    session = SessionLocal()
    try:
        for quarter in quarters:
            # Check valuation count
            count = (
                session.query(func.count(ValuationResult.ticker))
                .filter(ValuationResult.snapshot_timestamp == quarter)
                .scalar()
            )
            if count is None or count < MIN_SNAPSHOTS_FOR_QUARTER:
                msg = f"Quarter {quarter.date()}: only {count or 0} valuations (expected >= {MIN_SNAPSHOTS_FOR_QUARTER})"
                warnings.append(msg)
                logger.warning(f"  WARN: {msg}")
            else:
                logger.info(f"  Quarter {quarter.date()}: {count} valuations OK")

        # Check dashboard files exist
        public_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "web", "public",
        )
        for fname in ["core.json", "scatter.json"]:
            fpath = os.path.join(public_dir, fname)
            if not os.path.exists(fpath):
                msg = f"Missing dashboard file: {fname}"
                warnings.append(msg)
                logger.warning(f"  WARN: {msg}")

    finally:
        session.close()

    if not warnings:
        logger.info("  All validation checks passed")
    return warnings


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run the full quarterly update pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_quarterly_update.py                     # Auto-detect quarters
  python scripts/run_quarterly_update.py --quarter 2025-09-30
  python scripts/run_quarterly_update.py --dry-run
  python scripts/run_quarterly_update.py --skip-ingestion --skip-build
  python scripts/run_quarterly_update.py --reprocess         # Force re-run all quarters
        """,
    )
    parser.add_argument(
        "--quarter",
        type=str,
        help="Specific quarter-end date (YYYY-MM-DD). If omitted, auto-detects.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without executing.",
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip data ingestion steps (useful for re-runs).",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip backtest computation.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip frontend build.",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Force re-process all quarters (not just unprocessed ones).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code on validation warnings.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return parser


def main() -> None:
    """Run the full quarterly update pipeline."""
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)
    pipeline_start = time.time()

    logger.info("=" * 70)
    logger.info("QUARTERLY UPDATE PIPELINE")
    logger.info("=" * 70)

    # Determine target quarter(s)
    if args.quarter:
        try:
            target = datetime.strptime(args.quarter, "%Y-%m-%d").date()
        except ValueError:
            logger.error(f"Invalid date format: {args.quarter}. Use YYYY-MM-DD.")
            sys.exit(1)
    else:
        target = detect_current_quarter()

    logger.info(f"Target quarter: {target}")
    logger.info(f"Flags: dry_run={args.dry_run}, skip_ingestion={args.skip_ingestion}, "
                f"skip_backtest={args.skip_backtest}, skip_build={args.skip_build}, "
                f"reprocess={args.reprocess}, strict={args.strict}")

    step_num = 0

    # ── Step 1: Ingest snapshots ──
    if not args.skip_ingestion:
        step_num += 1
        with step_context("Ingest financial snapshots", step_num, args.dry_run):
            step_ingest_snapshots(dry_run=args.dry_run)

        # ── Step 2: Re-ingest missing ──
        step_num += 1
        with step_context("Re-ingest tickers with missing data", step_num, args.dry_run):
            step_reingest_missing(target, dry_run=args.dry_run)
    else:
        logger.info("[SKIP] Ingestion steps (--skip-ingestion)")

    # ── Step 3: Determine quarters to process ──
    step_num += 1
    with step_context("Determine quarters to process", step_num, args.dry_run):
        if args.reprocess:
            quarters_to_process = get_quarters_with_snapshots()
            logger.info(f"  Reprocessing ALL {len(quarters_to_process)} quarters with sufficient snapshots")
        else:
            quarters_to_process = find_unprocessed_quarters(target)
            if quarters_to_process:
                logger.info(f"  Found {len(quarters_to_process)} unprocessed quarters: "
                            f"{[q.date() for q in quarters_to_process]}")
            else:
                # If no unprocessed quarters, process the target
                target_dt = datetime(target.year, target.month, target.day)
                all_snapshot_quarters = get_quarters_with_snapshots()
                # Find matching quarter in DB
                matched = [q for q in all_snapshot_quarters if q.date() == target]
                if matched:
                    quarters_to_process = matched
                    logger.info(f"  No new quarters found. Re-processing target: {target}")
                else:
                    logger.warning(f"  No quarters found with >= {MIN_SNAPSHOTS_FOR_QUARTER} snapshots matching {target}")
                    logger.info(f"  Available snapshot quarters: {[q.date() for q in all_snapshot_quarters]}")

    if not quarters_to_process and not args.dry_run:
        logger.error("No quarters to process. Ensure data has been ingested.")
        sys.exit(1)

    # ── Step 4: Run valuation ──
    step_num += 1
    with step_context(f"Run valuation for {len(quarters_to_process)} quarter(s)", step_num, args.dry_run):
        total_valuations = step_run_valuation(quarters_to_process, dry_run=args.dry_run)

    # ── Step 5: Run backtest ──
    if not args.skip_backtest:
        step_num += 1
        with step_context("Run sector/index backtest", step_num, args.dry_run):
            step_run_backtest(dry_run=args.dry_run)
    else:
        logger.info("[SKIP] Backtest (--skip-backtest)")

    # ── Step 6: Generate dashboard ──
    step_num += 1
    with step_context("Generate dashboard data", step_num, args.dry_run):
        step_generate_dashboard(dry_run=args.dry_run)

    # ── Step 7: Build frontend ──
    if not args.skip_build:
        step_num += 1
        with step_context("Build frontend", step_num, args.dry_run):
            step_build_frontend(dry_run=args.dry_run)
    else:
        logger.info("[SKIP] Frontend build (--skip-build)")

    # ── Step 8: Validate ──
    step_num += 1
    with step_context("Validate results", step_num, args.dry_run):
        warnings = step_validate(quarters_to_process, dry_run=args.dry_run)

    # ── Report ──
    elapsed = time.time() - pipeline_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Total time: {elapsed:.1f}s")
    logger.info(f"  Quarters processed: {len(quarters_to_process)}")
    if not args.dry_run:
        logger.info(f"  Valuations created: {total_valuations}")
    if warnings:
        logger.info(f"  Warnings: {len(warnings)}")
    logger.info("=" * 70)

    if warnings and args.strict:
        logger.error("Exiting with error due to --strict and validation warnings")
        sys.exit(1)


if __name__ == "__main__":
    main()
