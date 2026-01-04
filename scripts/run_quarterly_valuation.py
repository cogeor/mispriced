"""
Run quarterly valuation pipeline with fair out-of-fold predictions.

This script:
1. Identifies all valid historical quarters (dates with >1000 snapshots)
2. For each quarter:
   a. Loads financial snapshot data for THAT QUARTER ONLY
   b. Trains XGBoost using repeated K-fold CV with fixed params
   c. Generates fair out-of-fold predictions (mean + std)
   d. Stores valuations in the database

Key principles:
- Uses log(market_cap) as target for numerical stability
- Fixed hyperparameters (no tuning) for consistency
- Out-of-fold predictions ensure fair evaluation
- Each quarter is evaluated independently (no data leakage)
- Each CV repeat uses DIFFERENT random splits

Usage:
    python scripts/run_quarterly_valuation.py
"""

import logging
import sys
import os
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session, sessionmaker
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.config import DATABASE_URL
from src.db.models import FinancialSnapshot, ValuationResult
from src.valuation.feature_builder import build_feature_matrix
from src.valuation.config import gbr_baseline_model
from src.valuation.size_correction import estimate_size_coefficient, compute_residual_mispricing
from src.valuation.currency_fix import normalize_snapshots
from src.evaluation import SimpleRepeatedCV, DEFAULT_MODEL_PARAMS

# Configuration
MIN_SNAPSHOTS_FOR_QUARTER = 1000
MIN_MARKET_CAP = 100e6  # $100M minimum
MIN_SAMPLES_FOR_CV = 50
N_CV_REPEATS = 10
N_CV_FOLDS = 5

# Critical features - stocks missing ALL of these are excluded
# At least one financial metric is needed for a meaningful prediction
CRITICAL_FEATURES = ["total_revenue", "net_income", "ebitda"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_quarter_dates(session: Session) -> List[datetime]:
    """
    Get quarter-end dates with substantial snapshot counts.

    Returns dates that have at least MIN_SNAPSHOTS_FOR_QUARTER tickers,
    indicating a quarterly filing spike.
    """
    result = (
        session.query(
            FinancialSnapshot.snapshot_timestamp,
            func.count(FinancialSnapshot.ticker).label("count"),
        )
        .group_by(FinancialSnapshot.snapshot_timestamp)
        .having(func.count(FinancialSnapshot.ticker) >= MIN_SNAPSHOTS_FOR_QUARTER)
        .order_by(FinancialSnapshot.snapshot_timestamp)
        .all()
    )

    return [r[0] for r in result]


def clear_valuations_for_quarter(
    session: Session, quarter_date: datetime, model_version: str
) -> int:
    """
    Clear existing valuations for a specific quarter and model version.

    This ensures clean state before inserting new valuations.
    """
    deleted = (
        session.query(ValuationResult)
        .filter(
            ValuationResult.snapshot_timestamp == quarter_date,
            ValuationResult.model_version == model_version,
        )
        .delete(synchronize_session=False)
    )
    session.commit()
    return deleted


def load_quarter_data(
    session: Session, 
    quarter_date: datetime, 
    tickers: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load latest financial snapshot data as of the quarter date.
    
    Implements Point-in-Time logic:
    - Selects the latest snapshot for each ticker where snapshot_timestamp <= quarter_date
    - Handles fiscal year mismatches (e.g., NVDA).
    - Optional: Filter by specific tickers.

    Applies:
    - Market cap filter (>= MIN_MARKET_CAP)
    """
    logger.info(f"  Loading stats as of {quarter_date.date()}...")
    
    # Subquery to find latest timestamp per ticker
    # Logic: Max timestamp <= quarter_date
    stmt = session.query(
        FinancialSnapshot.ticker,
        func.max(FinancialSnapshot.snapshot_timestamp).label('max_ts')
    ).filter(
        FinancialSnapshot.snapshot_timestamp <= quarter_date
    )
    
    # Optimization: Filter tickers early if specified
    if tickers:
        stmt = stmt.filter(FinancialSnapshot.ticker.in_(tickers))
        
    # Valid window: Don't fetch snapshots older than 100 days (staleness check)
    # This prevents using very old data for dead tickers.
    cutoff_stale = quarter_date - pd.Timedelta(days=100)
    stmt = stmt.filter(FinancialSnapshot.snapshot_timestamp >= cutoff_stale)
    
    subq = stmt.group_by(FinancialSnapshot.ticker).subquery()

    # Main query joining back to get full row
    query = session.query(FinancialSnapshot).join(
        subq,
        (FinancialSnapshot.ticker == subq.c.ticker) & 
        (FinancialSnapshot.snapshot_timestamp == subq.c.max_ts)
    )

    df = pd.read_sql(query.statement, session.bind)
    
    original_count = len(df)
    logger.info(f"  Found {original_count} raw snapshots")

    # Filter by market cap
    if "market_cap_t0" in df.columns:
        df["market_cap_t0"] = pd.to_numeric(df["market_cap_t0"], errors="coerce")
        df = df[df["market_cap_t0"] >= MIN_MARKET_CAP].copy()

    mcap_filtered_count = len(df)

    # Filter out stocks missing ALL critical features
    # These would get median-filled and produce meaningless predictions
    if len(df) > 0:
        # Convert critical feature columns to numeric
        for col in CRITICAL_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Check which stocks have at least one non-null critical feature
        critical_cols = [c for c in CRITICAL_FEATURES if c in df.columns]
        if critical_cols:
            has_any_critical = df[critical_cols].notna().any(axis=1)
            excluded_count = (~has_any_critical).sum()
            if excluded_count > 0:
                excluded_tickers = df[~has_any_critical]["ticker"].tolist()
                logger.info(f"  Excluding {excluded_count} tickers missing ALL critical features: {excluded_tickers[:5]}...")
            df = df[has_any_critical].copy()

    logger.info(
        f"  Loaded {len(df)} valid tickers (from {original_count} raw, "
        f"{mcap_filtered_count} after mcap filter, mcap >= ${MIN_MARKET_CAP/1e6:.0f}M)"
    )

    return df


def run_valuation_for_quarter(
    session: Session,
    quarter_date: datetime,
    model_config,
    model_version: str,
    debug_tickers: Optional[List[str]] = None,
) -> int:
    """
    Run the full valuation pipeline for a single quarter.

    Steps:
    1. Clear existing valuations for this quarter/model
    2. Load and filter quarter data
    3. Build feature matrix
    4. Run repeated CV with fixed params on log(mcap)
    5. Convert predictions back and store to database

    Returns:
        Number of valuations saved
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing Quarter: {quarter_date.date()}")
    logger.info(f"{'='*60}")

    # 1. Clear stale data
    deleted = clear_valuations_for_quarter(session, quarter_date, model_version)
    if deleted > 0:
        logger.info(f"  Cleared {deleted} existing valuations")

    # 2. Load data
    # Load FULL market to ensure valid cross-sectional training
    df = load_quarter_data(session, quarter_date)

    if debug_tickers:
        for t in debug_tickers:
            if t in df["ticker"].values:
                logger.info(f"  [DEBUG] {t} found in loaded data (pre-feature build).")
            else:
                logger.warning(f"  [DEBUG] {t} NOT found in loaded data! (Filtered by mcap or missing in snapshot?)")
                # Attempt to diagnose if it was in snapshot but filtered
                # This would require querying raw snapshot again, effectively done outside.

    if len(df) < MIN_SAMPLES_FOR_CV:
        logger.warning(f"  Insufficient data ({len(df)} < {MIN_SAMPLES_FOR_CV}), skipping")
        return 0

    # 2.5. Normalize currency issues (book_value per-share, ADR FX, anomalies)
    logger.info("  Normalizing currency issues...")
    df, anomalies = normalize_snapshots(
        df,
        fix_bv=True,
        fix_adrs=True,
        exclude_anomalies=True,
    )
    if len(anomalies) > 0:
        logger.info(f"  Excluded {len(anomalies)} tickers with currency anomalies: {anomalies['ticker'].tolist()[:10]}...")

    if len(df) < MIN_SAMPLES_FOR_CV:
        logger.warning(f"  Insufficient data after currency normalization ({len(df)} < {MIN_SAMPLES_FOR_CV}), skipping")
        return 0

    # 3. Build features
    logger.info("  Building feature matrix...")
    try:
        X_df = build_feature_matrix(df, model_config.features)
        X = X_df.values

        # Target: log(Market Cap) for numerical stability
        # This is critical - raw market cap has huge variance ($100M to $4T)
        mcap_raw = pd.to_numeric(df["market_cap_t0"], errors="coerce").fillna(0).values
        mcap_raw = np.clip(mcap_raw, 1e6, None)  # Floor at $1M to avoid log(0)
        y_log = np.log(mcap_raw)  # Target in log space

    except Exception as e:
        logger.error(f"  Feature building failed: {e}")
        return 0

    # 4. Run repeated CV with FIXED params on log target
    logger.info(f"  Running {N_CV_REPEATS}x {N_CV_FOLDS}-fold CV (XGBoost, log target)...")

    cv = SimpleRepeatedCV(
        n_repeats=N_CV_REPEATS,
        n_folds=N_CV_FOLDS,
        model_params=DEFAULT_MODEL_PARAMS,
        random_seed=model_config.random_seed,
    )

    try:
        results = cv.fit_predict(X, y_log)
    except Exception as e:
        logger.error(f"  CV failed: {e}")
        return 0

    # 5. Convert predictions from log space back to linear
    pred_log_mean = results["mean"]
    pred_log_std = results["std"]

    # Convert mean prediction back to linear space
    # For log-normal: E[X] = exp(mu + sigma^2/2), but we use simple exp(mean) for point estimate
    pred_mcap = np.exp(pred_log_mean)

    # For std in linear space, use delta method approximation: std_linear ≈ exp(mu) * std_log
    pred_mcap_std = pred_mcap * pred_log_std

    # Compute relative metrics
    actual_mcap = mcap_raw
    relative_error = (pred_mcap - actual_mcap) / actual_mcap
    relative_std = pred_mcap_std / actual_mcap

    # 6. Compute size-corrected (residual) mispricing
    logger.info("  Computing size premium correction...")
    residual_error, _ = compute_residual_mispricing(relative_error, actual_mcap)
    logger.info(f"  Residual error: mean={np.mean(residual_error):.3f}, median={np.median(residual_error):.3f}")

    logger.info("  Saving valuations to database...")
    valuations = []

    for i in range(len(df)):
        ticker = df.iloc[i]["ticker"]

        # DEBUG check
        if debug_tickers and ticker in debug_tickers:
             logger.info(f"  [DEBUG] Saving valuation for {ticker}: Pred=${pred_mcap[i]/1e9:.1f}B, Actual=${actual_mcap[i]/1e9:.1f}B")

        val = ValuationResult(
            ticker=ticker,
            snapshot_timestamp=quarter_date,
            model_version=model_version,
            predicted_mcap_mean=float(pred_mcap[i]),
            predicted_mcap_std=float(pred_mcap_std[i]),
            actual_mcap=float(actual_mcap[i]),
            relative_error=float(relative_error[i]),
            residual_error=float(residual_error[i]),
            relative_std=float(relative_std[i]),
            n_experiments=N_CV_REPEATS,
        )
        valuations.append(val)

    session.bulk_save_objects(valuations)
    session.commit()

    # Log summary stats
    logger.info(f"  Saved {len(valuations)} valuations")
    logger.info(
        f"  Relative error: mean={np.mean(relative_error):.3f}, "
        f"median={np.median(relative_error):.3f}, "
        f"std={np.std(relative_error):.3f}"
    )

    return len(valuations)


def run_all_quarters(quarters: Optional[List[datetime]] = None, debug_tickers: Optional[List[str]] = None) -> int:
    """
    Run valuation pipeline for all (or specified) quarters.

    Args:
        quarters: Optional list of specific quarters to process.
                  If None, processes all valid quarters.
        debug_tickers: List of tickers to debug.

    Returns:
        Total number of valuations created
    """
    logger.info("=" * 70)
    logger.info("QUARTERLY VALUATION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  - Model: XGBoost")
    logger.info(f"  - Target: log(market_cap)")
    logger.info(f"  - CV Repeats: {N_CV_REPEATS}")
    logger.info(f"  - CV Folds: {N_CV_FOLDS}")
    logger.info(f"  - Min Market Cap: ${MIN_MARKET_CAP/1e6:.0f}M")
    logger.info(f"  - Model Params: {DEFAULT_MODEL_PARAMS}")
    if debug_tickers:
        logger.info(f"  - Debug Tickers: {debug_tickers}")

    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    try:
        # Load model config (for features only, params are fixed)
        model_config = gbr_baseline_model()
        model_version = f"{model_config.name}_v{model_config.version}"

        # Get all valid quarters from DB to ensure timestamp alignment
        available_quarters = get_quarter_dates(session)
        logger.info(f"DB has {len(available_quarters)} valid quarters having >= {MIN_SNAPSHOTS_FOR_QUARTER} snapshots.")

        # If specific quarters requested, filter the DB list
        quarters_to_process = []
        if quarters:
            # Match requested quarters to available DB quarters
            # Precision: Match by YYYY-MM-DD
            available_map = {q.strftime("%Y-%m-%d"): q for q in available_quarters}
            logger.info(f"  Available quarters in DB: {list(available_map.keys())}")
            
            for req_q in quarters:
                req_str = req_q.strftime("%Y-%m-%d")
                if req_str in available_map:
                    quarters_to_process.append(available_map[req_str])
                    logger.info(f"  Matched request {req_str} to DB timestamp {available_map[req_str]}")
                else:
                    logger.warning(f"  Requested quarter {req_str} NOT found in valid DB quarters! Attempting to process anyway...")
                    # Fallback: Use the requested datetime
                    quarters_to_process.append(req_q)
        else:
            quarters_to_process = available_quarters

        logger.info(f"Processing {len(quarters_to_process)} quarters")

        total_valuations = 0
        for quarter_date in quarters_to_process:
            count = run_valuation_for_quarter(
                session, quarter_date, model_config, model_version, debug_tickers
            )
            total_valuations += count

        logger.info(f"\n{'='*70}")
        logger.info(f"PIPELINE COMPLETE")
        logger.info(f"Total valuations: {total_valuations}")
        logger.info(f"{'='*70}")

        return total_valuations

    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run quarterly valuation pipeline")
    parser.add_argument("--tickers", type=str, help="Comma-separated list of tickers to debug")
    parser.add_argument("--quarter", type=str, help="Specific quarter date (YYYY-MM-DD)")
    args = parser.parse_args()

    debug_tickers_list = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    
    specific_quarters = None
    if args.quarter:
        try:
            q_date = datetime.strptime(args.quarter, "%Y-%m-%d")
            specific_quarters = [q_date]
        except ValueError:
            logger.error("Invalid date format. Use YYYY-MM-DD")
            sys.exit(1)

    run_all_quarters(quarters=specific_quarters, debug_tickers=debug_tickers_list)
