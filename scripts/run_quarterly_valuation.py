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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.config import DATABASE_URL
from src.db.models import FinancialSnapshot, ValuationResult
from src.valuation.feature_builder import build_feature_matrix
from src.valuation.config import gbr_baseline_model
from src.valuation.size_correction import estimate_size_coefficient, compute_residual_mispricing
from src.evaluation import SimpleRepeatedCV, DEFAULT_MODEL_PARAMS

# Configuration
MIN_SNAPSHOTS_FOR_QUARTER = 1000
MIN_MARKET_CAP = 100e6  # $100M minimum
MIN_SAMPLES_FOR_CV = 50
N_CV_REPEATS = 10
N_CV_FOLDS = 5

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


def load_quarter_data(session: Session, quarter_date: datetime) -> pd.DataFrame:
    """
    Load and filter financial snapshot data for a specific quarter.

    Applies:
    - Market cap filter (>= MIN_MARKET_CAP)
    - Deduplication by ticker
    """
    query = session.query(FinancialSnapshot).filter(
        FinancialSnapshot.snapshot_timestamp == quarter_date
    )
    df = pd.read_sql(query.statement, session.bind)

    original_count = len(df)

    # Filter by market cap
    if "market_cap_t0" in df.columns:
        df["market_cap_t0"] = pd.to_numeric(df["market_cap_t0"], errors="coerce")
        df = df[df["market_cap_t0"] >= MIN_MARKET_CAP].copy()

    # Deduplicate by ticker (keep first occurrence)
    df = df.drop_duplicates(subset=["ticker"])

    logger.info(
        f"  Loaded {len(df)} tickers (filtered from {original_count}, "
        f"mcap >= ${MIN_MARKET_CAP/1e6:.0f}M)"
    )

    return df


def run_valuation_for_quarter(
    session: Session,
    quarter_date: datetime,
    model_config,
    model_version: str,
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
    df = load_quarter_data(session, quarter_date)

    if len(df) < MIN_SAMPLES_FOR_CV:
        logger.warning(f"  Insufficient data ({len(df)} < {MIN_SAMPLES_FOR_CV}), skipping")
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


def run_all_quarters(quarters: Optional[List[datetime]] = None) -> int:
    """
    Run valuation pipeline for all (or specified) quarters.

    Args:
        quarters: Optional list of specific quarters to process.
                  If None, processes all valid quarters.

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

    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    try:
        # Load model config (for features only, params are fixed)
        model_config = gbr_baseline_model()
        model_version = f"{model_config.name}_v{model_config.version}"

        # Get quarters to process
        if quarters is None:
            quarters = get_quarter_dates(session)

        logger.info(f"Found {len(quarters)} quarters to process")

        total_valuations = 0
        for quarter_date in quarters:
            count = run_valuation_for_quarter(
                session, quarter_date, model_config, model_version
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
    run_all_quarters()
