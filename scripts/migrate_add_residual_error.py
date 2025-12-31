"""
Add residual_error column to valuation_results table and backfill existing data.

Run this once to add the new column and compute size-corrected mispricing for 
all existing valuations.

Usage:
    python scripts/migrate_add_residual_error.py
"""

import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.db.config import DATABASE_URL
from src.db.models import ValuationResult
from src.valuation.size_correction import compute_residual_mispricing

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def add_column_if_not_exists(engine):
    """Add residual_error column if it doesn't exist."""
    with engine.connect() as conn:
        # Check if using SQLite
        is_sqlite = 'sqlite' in str(engine.url)
        
        column_exists = False
        if is_sqlite:
            # SQLite check
            result = conn.execute(text("PRAGMA table_info(valuation_results)"))
            for row in result:
                # row is (cid, name, type, notnull, dflt_value, pk)
                if row[1] == 'residual_error':
                    column_exists = True
                    break
        else:
            # Postgres check
            result = conn.execute(text("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'valuation_results' AND column_name = 'residual_error'
            """))
            if result.fetchone() is not None:
                column_exists = True

        if not column_exists:
            logger.info("Adding residual_error column...")
            # SQLite supports ADD COLUMN with basic types
            conn.execute(text("""
                ALTER TABLE valuation_results 
                ADD COLUMN residual_error FLOAT
            """))
            conn.commit()
            logger.info("Column added successfully")
        else:
            logger.info("residual_error column already exists")


def backfill_residual_errors(session):
    """Compute and store residual_error for all quarters."""
    from sqlalchemy import func
    
    # Get all quarters with valuations
    quarters = session.query(
        ValuationResult.snapshot_timestamp
    ).group_by(
        ValuationResult.snapshot_timestamp
    ).order_by(
        ValuationResult.snapshot_timestamp
    ).all()
    
    logger.info(f"Found {len(quarters)} quarters to backfill")
    
    for (quarter_date,) in quarters:
        # Load valuations for this quarter
        valuations = session.query(ValuationResult).filter(
            ValuationResult.snapshot_timestamp == quarter_date
        ).all()
        
        if len(valuations) < 50:
            logger.info(f"Skipping {quarter_date.date()} ({len(valuations)} valuations, too few)")
            continue
        
        # Check if already backfilled
        has_residual = any(v.residual_error is not None for v in valuations)
        if has_residual:
            logger.info(f"Skipping {quarter_date.date()} (already has residual_error)")
            continue
        
        # Compute size correction
        mispricing = np.array([float(v.relative_error) for v in valuations])
        market_cap = np.array([float(v.actual_mcap) for v in valuations])
        
        residual, _ = compute_residual_mispricing(mispricing, market_cap)
        
        # Update valuations
        for i, val in enumerate(valuations):
            val.residual_error = float(residual[i])
        
        session.commit()
        logger.info(f"Backfilled {quarter_date.date()} ({len(valuations)} valuations)")
    
    logger.info("Backfill complete")


def main():
    logger.info("=" * 60)
    logger.info("MIGRATION: Add residual_error column")
    logger.info("=" * 60)
    
    engine = create_engine(DATABASE_URL)
    
    # Add column
    add_column_if_not_exists(engine)
    
    # Backfill data
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        backfill_residual_errors(session)
    finally:
        session.close()
    
    logger.info("Migration complete!")


if __name__ == "__main__":
    main()
