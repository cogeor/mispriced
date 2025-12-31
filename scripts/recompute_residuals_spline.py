"""
Recompute residual_error column in valuation_results table using current Size Correction logic (Spline).

Run this to update existing valuations with the new cubic spline correction.

Usage:
    python scripts/recompute_residuals_spline.py
"""

import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.config import DATABASE_URL
from src.db.models import ValuationResult
from src.valuation.size_correction import compute_residual_mispricing

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def backfill_residual_errors(session):
    """Compute and store residual_error for all quarters."""
    quarters = session.query(
        ValuationResult.snapshot_timestamp
    ).group_by(
        ValuationResult.snapshot_timestamp
    ).order_by(
        ValuationResult.snapshot_timestamp
    ).all()
    
    logger.info(f"Found {len(quarters)} quarters to update")
    
    for (quarter_date,) in quarters:
        valuations = session.query(ValuationResult).filter(
            ValuationResult.snapshot_timestamp == quarter_date
        ).all()
        
        if len(valuations) < 50:
            logger.info(f"Skipping {quarter_date.date()} ({len(valuations)} valuations, too few)")
            continue
        
        # Compute size correction (force re-compute)
        mispricing = np.array([float(v.relative_error) for v in valuations])
        market_cap = np.array([float(v.actual_mcap) for v in valuations])
        
        residual, _ = compute_residual_mispricing(mispricing, market_cap)
        
        for i, val in enumerate(valuations):
            val.residual_error = float(residual[i])
        
        session.commit()
        logger.info(f"Updated {quarter_date.date()} ({len(valuations)} valuations)")
    
    logger.info("Recompute complete")

def main():
    logger.info("=" * 60)
    logger.info("RECOMPUTE: Updating residual_error using Spline")
    logger.info("=" * 60)
    
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        backfill_residual_errors(session)
    finally:
        session.close()
    
    logger.info("Update complete!")

if __name__ == "__main__":
    main()
