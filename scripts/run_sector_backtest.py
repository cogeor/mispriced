#!/usr/bin/env python
"""Signal quality backtest by sector AND index.

Usage:
    python scripts/run_sector_backtest.py

Computes signal quality metrics (IC, quantile spread, hit rate, L/S returns)
grouped by both sector and index.

Output: signal_backtest_results.csv (combined results)
"""

import sys
import os
import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.session import SessionLocal
from src.db.models import ValuationResult, Ticker, IndexMembership
from src.backtest.price_cache import fetch_and_cache_prices, get_ticker_return
from src.backtest.signal_metrics import (
    compute_ic,
    compute_quantile_returns,
    compute_quantile_spread,
    compute_hit_rate,
    compute_long_short_returns,
)
from sqlalchemy import func

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

HORIZONS = [10, 30, 60, 90]
OUTPUT_DIR = "data"


def get_available_quarters(session) -> List[Tuple[date, int]]:
    """Get quarters with substantial valuation data (>100 valuations)."""
    quarters = session.query(
        ValuationResult.snapshot_timestamp,
        func.count(ValuationResult.ticker)
    ).group_by(
        ValuationResult.snapshot_timestamp
    ).having(
        func.count(ValuationResult.ticker) > 100
    ).order_by(
        ValuationResult.snapshot_timestamp
    ).all()
    
    return [(q[0].date(), q[1]) for q in quarters]


def get_valuations_with_metadata(session, snapshot_date: date) -> pd.DataFrame:
    """Get valuations with sector and index membership info."""
    # Get valuations with sector
    query = session.query(
        ValuationResult.ticker,
        Ticker.sector,
        ValuationResult.actual_mcap,
        ValuationResult.predicted_mcap_mean,
    ).join(
        Ticker, ValuationResult.ticker == Ticker.ticker
    ).filter(
        ValuationResult.snapshot_timestamp == datetime.combine(snapshot_date, datetime.min.time())
    )
    
    df = pd.read_sql(query.statement, session.bind)
    df["overpriciness"] = (df["actual_mcap"] - df["predicted_mcap_mean"]) / df["predicted_mcap_mean"]
    
    # Get index memberships for each ticker
    memberships = session.query(
        IndexMembership.ticker,
        IndexMembership.index_id
    ).filter(
        IndexMembership.is_member == True
    ).distinct().all()
    
    # Create ticker -> list of indices mapping
    ticker_indices = {}
    for ticker, index_id in memberships:
        if ticker not in ticker_indices:
            ticker_indices[ticker] = []
        ticker_indices[ticker].append(index_id)
    
    df["indices"] = df["ticker"].apply(lambda t: ticker_indices.get(t, []))
    
    return df


def compute_group_metrics(
    tickers: List[str],
    signal: np.ndarray,
    prices: Dict[str, Dict[date, float]],
    snapshot_date: date,
    horizons: List[int],
) -> List[Dict]:
    """Compute signal metrics for a group of tickers."""
    results = []
    
    for horizon in horizons:
        returns_list = []
        signal_list = []
        
        for i, ticker in enumerate(tickers):
            if ticker not in prices or not prices[ticker]:
                continue
            
            ret = get_ticker_return(prices[ticker], snapshot_date, horizon)
            if ret is not None and not np.isnan(ret) and abs(ret) < 2.0:
                returns_list.append(ret)
                signal_list.append(signal[i])
        
        n_obs = len(returns_list)
        if n_obs < 10:
            continue
        
        signal_arr = np.array(signal_list)
        returns_arr = np.array(returns_list)
        
        ic, ic_pval = compute_ic(signal_arr, returns_arr)
        quantile_returns = compute_quantile_returns(signal_arr, returns_arr)
        spread = compute_quantile_spread(signal_arr, returns_arr)
        hit_rate, _ = compute_hit_rate(signal_arr, returns_arr)
        ls_returns = compute_long_short_returns(signal_arr, returns_arr)
        
        results.append({
            "horizon": horizon,
            "n_obs": n_obs,
            "ic": ic,
            "ic_pval": ic_pval,
            "q1_return": quantile_returns.get(1, np.nan),
            "q5_return": quantile_returns.get(5, np.nan),
            "spread": spread,
            "hit_rate": hit_rate,
            "ls_return": ls_returns.get("return", np.nan),
        })
    
    return results


def main():
    session = SessionLocal()
    
    try:
        logger.info("=" * 70)
        logger.info("SIGNAL QUALITY ANALYSIS: BY SECTOR AND INDEX")
        logger.info("=" * 70)
        
        # 1. Get available quarters
        quarters = get_available_quarters(session)
        logger.info(f"Found {len(quarters)} quarters with data")
        
        if not quarters:
            logger.error("No data found!")
            return
        
        all_results = []
        
        for q_date, count in quarters:
            logger.info(f"\n--- {q_date} ({count} valuations) ---")
            
            df = get_valuations_with_metadata(session, q_date)
            if df.empty:
                continue
            
            tickers = df["ticker"].tolist()
            
            # Fetch prices
            start_date = q_date - timedelta(days=5)
            end_date = q_date + timedelta(days=max(HORIZONS) + 10)
            
            logger.info(f"  Fetching prices...")
            prices = fetch_and_cache_prices(session, tickers, start_date, end_date, batch_size=100)
            logger.info(f"  Got prices for {len(prices)} tickers")
            
            # 2. Compute by SECTOR
            for sector in df["sector"].dropna().unique():
                sector_df = df[df["sector"] == sector]
                sector_tickers = sector_df["ticker"].tolist()
                sector_signal = sector_df["overpriciness"].values
                
                metrics = compute_group_metrics(sector_tickers, sector_signal, prices, q_date, HORIZONS)
                for m in metrics:
                    m["group_type"] = "sector"
                    m["group_name"] = sector
                    m["quarter"] = q_date
                    all_results.append(m)
            
            # 3. Compute by INDEX
            # Get all indices
            indices = session.query(IndexMembership.index_id).distinct().all()
            indices = [i[0] for i in indices]
            
            for index_id in indices:
                # Filter to tickers in this index
                idx_df = df[df["indices"].apply(lambda x: index_id in x)]
                if len(idx_df) < 10:
                    continue
                
                idx_tickers = idx_df["ticker"].tolist()
                idx_signal = idx_df["overpriciness"].values
                
                metrics = compute_group_metrics(idx_tickers, idx_signal, prices, q_date, HORIZONS)
                for m in metrics:
                    m["group_type"] = "index"
                    m["group_name"] = index_id
                    m["quarter"] = q_date
                    all_results.append(m)
        
        if not all_results:
            logger.error("No results!")
            return
        
        # 4. Create combined DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Aggregate across quarters
        summary = results_df.groupby(["group_type", "group_name", "horizon"]).agg({
            "n_obs": "sum",
            "ic": "mean",
            "ic_pval": "mean",
            "q1_return": "mean",
            "q5_return": "mean",
            "spread": "mean",
            "hit_rate": "mean",
            "ls_return": "mean",
        }).reset_index()
        
        # 5. Display results
        print("\n" + "=" * 95)
        print("SIGNAL QUALITY BY SECTOR")
        print("=" * 95)
        _print_group_table(summary[summary["group_type"] == "sector"])
        
        print("\n" + "=" * 95)
        print("SIGNAL QUALITY BY INDEX")
        print("=" * 95)
        _print_group_table(summary[summary["group_type"] == "index"])
        
        # 6. Find strongest signals overall
        print("\n" + "=" * 95)
        print("STRONGEST SIGNALS (|IC| > 0.03)")
        print("=" * 95)
        strong = summary[abs(summary["ic"]) > 0.03].sort_values("ic_pval")
        for _, row in strong.head(20).iterrows():
            sig = "*" if row["ic_pval"] < 0.05 else ("~" if row["ic_pval"] < 0.10 else "")
            print(f"  [{row['group_type']:<6}] {row['group_name']:<22} @ {int(row['horizon']):>2}d: IC={row['ic']:+.3f}{sig}")
        
        # Save
        import os
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        results_df.to_csv(f"{OUTPUT_DIR}/signal_backtest_detailed.csv", index=False)
        summary.to_csv(f"{OUTPUT_DIR}/signal_backtest_summary.csv", index=False)
        logger.info(f"\n✅ Saved to {OUTPUT_DIR}/")
        
    finally:
        session.close()


def _print_group_table(df: pd.DataFrame) -> None:
    """Print formatted table for a group type."""
    for horizon in HORIZONS:
        h_df = df[df["horizon"] == horizon].sort_values("ic")
        if h_df.empty:
            continue
        
        print(f"\n--- {horizon}-day horizon ---")
        print(f"{'Name':<22} {'N':>5} {'IC':>8} {'p-val':>7} {'Q1':>7} {'Q5':>7} {'Spread':>7} {'Hit%':>5} {'L/S':>7}")
        print("-" * 85)
        
        for _, row in h_df.iterrows():
            sig = "*" if row["ic_pval"] < 0.05 else ("~" if row["ic_pval"] < 0.10 else " ")
            print(
                f"{row['group_name']:<22} "
                f"{int(row['n_obs']):>5} "
                f"{row['ic']:>+7.3f}{sig} "
                f"{row['ic_pval']:>6.3f} "
                f"{row['q1_return']:>+6.1%} "
                f"{row['q5_return']:>+6.1%} "
                f"{row['spread']:>+6.1%} "
                f"{row['hit_rate']:>4.0%} "
                f"{row['ls_return']:>+6.1%}"
            )


if __name__ == "__main__":
    main()
