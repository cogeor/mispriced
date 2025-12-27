#!/usr/bin/env python
"""Run backtest to validate index overpriciness correlation with future returns.

Usage:
    python scripts/run_backtest.py --start 2020-01-01 --end 2023-12-31

This script:
1. Loads existing valuation results from the database
2. Computes index overpriciness at each cutoff date
3. Fetches forward returns from Yahoo Finance
4. Analyzes correlation between overpriciness and returns
"""

import argparse
import logging
import sys
import os
from datetime import date, datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.session import SessionLocal
from src.backtest.service import BacktestService
from src.backtest.models import BacktestConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_date(date_str: str) -> date:
    """Parse date string in YYYY-MM-DD format."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def main() -> None:
    """Run the backtest."""
    parser = argparse.ArgumentParser(
        description="Run backtest to validate index overpriciness signals"
    )
    parser.add_argument(
        "--start",
        type=parse_date,
        default=date(2020, 1, 1),
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=parse_date,
        default=date(2023, 12, 31),
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=90,
        help="Days between cutoff dates (default: 90 for quarterly)",
    )
    parser.add_argument(
        "--indices",
        type=str,
        nargs="+",
        default=["SP500"],
        help="Indices to test (default: SP500)",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[63, 126, 252],
        help="Forward horizons in trading days (default: 63 126 252)",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.5,
        help="Minimum coverage ratio (default: 0.5)",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Filter by model version (default: use all)",
    )
    parser.add_argument(
        "--use-latest",
        action="store_true",
        help="Use latest valuations (ignore cutoff for valuations). "
             "Useful when historical valuations are not available.",
    )
    
    args = parser.parse_args()
    
    # Create config
    config = BacktestConfig(
        indices=args.indices,
        start_date=args.start,
        end_date=args.end,
        step_days=args.step,
        forward_horizons=args.horizons,
        min_coverage=args.min_coverage,
        model_version=args.model_version,
        use_latest_valuations=args.use_latest,
    )
    
    logger.info("=" * 60)
    logger.info("BACKTEST CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"  Indices:     {config.indices}")
    logger.info(f"  Date range:  {config.start_date} to {config.end_date}")
    logger.info(f"  Step:        {config.step_days} days")
    logger.info(f"  Horizons:    {config.forward_horizons} days")
    logger.info(f"  Min coverage: {config.min_coverage:.0%}")
    logger.info("=" * 60)
    
    # Run backtest
    session = SessionLocal()
    try:
        service = BacktestService(session)
        result = service.run_backtest(config)
        
        # Print results
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"  Observations:       {result.n_observations}")
        print(f"  Mean overpriciness: {result.mean_overpriciness:.2%}" if result.mean_overpriciness else "  Mean overpriciness: N/A")
        print(f"  Std overpriciness:  {result.std_overpriciness:.2%}" if result.std_overpriciness else "  Std overpriciness:  N/A")
        print(f"  Mean coverage:      {result.mean_coverage:.1%}" if result.mean_coverage else "  Mean coverage:      N/A")
        
        print("\n" + "-" * 60)
        print("CORRELATION ANALYSIS")
        print("-" * 60)
        
        for horizon, metrics in sorted(result.correlations.items()):
            print(f"\n  Horizon: {horizon} days ({horizon // 21} months)")
            print(f"    Pearson r:     {metrics.pearson_r:+.3f} (p={metrics.pearson_p_value:.4f})")
            print(f"    Spearman rho:  {metrics.spearman_rho:+.3f} (p={metrics.spearman_p_value:.4f})")
            print(f"    Hit rate:      {metrics.hit_rate:.1%} (p={metrics.hit_rate_p_value:.4f})")
            print(f"    N observations: {metrics.n_observations}")
            
            if metrics.bootstrap_ci_pearson:
                ci_low, ci_high = metrics.bootstrap_ci_pearson
                print(f"    95% CI (r):    [{ci_low:+.3f}, {ci_high:+.3f}]")
            
            if metrics.quintile_mean_returns:
                print(f"    Quintile returns:")
                for q, ret in sorted(metrics.quintile_mean_returns.items()):
                    label = "underpriced" if q == 1 else "overpriced" if q == 5 else ""
                    print(f"      Q{q}: {ret:+.2%} {label}")
        
        print("\n" + "=" * 60)
        
        # Interpretation
        if result.correlations:
            first_horizon = min(result.correlations.keys())
            r = result.correlations[first_horizon].pearson_r
            p = result.correlations[first_horizon].pearson_p_value
            hit = result.correlations[first_horizon].hit_rate
            
            print("\nINTERPRETATION")
            print("-" * 60)
            if r < 0 and p < 0.05:
                print("✅ THESIS SUPPORTED: Significant negative correlation")
                print("   Overpriced indices tend to underperform.")
            elif r < 0 and p < 0.10:
                print("⚠️ WEAK SUPPORT: Marginally significant negative correlation")
            elif r < 0:
                print("❓ INCONCLUSIVE: Negative correlation but not significant")
            else:
                print("❌ THESIS NOT SUPPORTED: No negative correlation found")
            
            if hit > 0.55 and result.correlations[first_horizon].hit_rate_p_value < 0.05:
                print(f"✅ Hit rate {hit:.1%} is significantly better than random")
            
        print("=" * 60)
        
    finally:
        session.close()


if __name__ == "__main__":
    main()
