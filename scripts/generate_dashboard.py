"""
Generate a professional financial dashboard combining index and valuation analysis.

Outputs: plots/dashboard.html
"""

import sys
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np

# Approximate FX rates to USD (for currencies not stored in USD)
# These are used to convert actual_mcap to USD for comparison with USD predictions
FX_TO_USD = {
    "USD": 1.0,
    "GBP": 1.27,      # 1 GBP = 1.27 USD
    "GBp": 0.0127,    # 1 pence = 0.0127 USD (GBP/100)
    "GBX": 0.0127,    # Alternative code for pence
    "EUR": 1.08,      # 1 EUR = 1.08 USD
    "INR": 0.012,     # 1 INR = 0.012 USD (1/84)
    "HKD": 0.128,     # 1 HKD = 0.128 USD (1/7.8)
    "JPY": 0.0067,    # 1 JPY = 0.0067 USD (1/150)
    "CNY": 0.14,      # 1 CNY = 0.14 USD (1/7.2)
    "CHF": 1.13,      # 1 CHF = 1.13 USD
    "CAD": 0.74,      # 1 CAD = 0.74 USD
    "AUD": 0.65,      # 1 AUD = 0.65 USD
    "KRW": 0.00072,   # 1 KRW = 0.00072 USD (1/1400)
    "TWD": 0.031,     # 1 TWD = 0.031 USD (1/32)
    "SGD": 0.74,      # 1 SGD = 0.74 USD
    "SEK": 0.095,     # 1 SEK = 0.095 USD
    "NOK": 0.091,     # 1 NOK = 0.091 USD
    "DKK": 0.145,     # 1 DKK = 0.145 USD
    "ILS": 0.27,      # 1 ILS = 0.27 USD
    "BRL": 0.17,      # 1 BRL = 0.17 USD
    "MXN": 0.058,     # 1 MXN = 0.058 USD
    "ZAR": 0.054,     # 1 ZAR = 0.054 USD
}

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.db.session import SessionLocal
from src.db.models import Ticker, ValuationResult, FinancialSnapshot
from src.index import IndexService, IndexAnalysis
from src.config.metrics import (
    ACTIVE_MISPRICING_METRIC,
    MispricingMetric,
    get_metric_info,
    compute_mispricing,
)
from src.valuation.size_correction import (
    apply_size_correction, 
    SizeCorrectionResult,
    estimate_size_coefficient,
    SizeCoefficient,
)

# Backtest data path
BACKTEST_SUMMARY_PATH = "data/signal_backtest_summary.csv"


def get_available_quarters() -> List[datetime]:
    """Get all quarters with significant valuation data (>1000 tickers)."""
    from sqlalchemy import func
    session = SessionLocal()
    try:
        quarters = (
            session.query(ValuationResult.snapshot_timestamp)
            .group_by(ValuationResult.snapshot_timestamp)
            .having(func.count(func.distinct(ValuationResult.ticker)) > 1000)
            .order_by(ValuationResult.snapshot_timestamp.desc())
            .all()
        )
        return [q[0] for q in quarters]
    finally:
        session.close()


# Professional color palette for sectors (Bloomberg-inspired)
SECTOR_COLORS: Dict[str, str] = {
    "Technology": "#2563eb",
    "Healthcare": "#059669",
    "Financial Services": "#7c3aed",
    "Consumer Cyclical": "#db2777",
    "Communication Services": "#ea580c",
    "Industrials": "#64748b",
    "Consumer Defensive": "#0891b2",
    "Energy": "#dc2626",
    "Utilities": "#65a30d",
    "Real Estate": "#a855f7",
    "Basic Materials": "#d97706",
    "Unknown": "#94a3b8",
}


def get_valuation_data_with_sectors(quarter_date: datetime = None) -> pd.DataFrame:
    """
    Fetch valuation results joined with ticker sector info.

    If quarter_date is None, fetches the LATEST quarter's data.
    Returns one row per ticker for the specified quarter.
    """
    from sqlalchemy import func

    session = SessionLocal()
    try:
        # DEBUG: List all quarterly counts
        counts = (
            session.query(ValuationResult.snapshot_timestamp, func.count(func.distinct(ValuationResult.ticker)))
            .group_by(ValuationResult.snapshot_timestamp)
            .order_by(ValuationResult.snapshot_timestamp.desc())
            .all()
        )
        print(f"DEBUG: Found {len(counts)} quarters in DB:")
        for q, c in counts[:5]:
             print(f"  {q}: {c} tickers")

        # If no quarter specified, find the latest one with significant data
        # Use 1000 threshold to match pipeline's MIN_SNAPSHOTS_FOR_QUARTER
        if quarter_date is None:
            latest = (
                session.query(ValuationResult.snapshot_timestamp)
                .group_by(ValuationResult.snapshot_timestamp)
                .having(func.count(func.distinct(ValuationResult.ticker)) > 1000)
                .order_by(ValuationResult.snapshot_timestamp.desc())
                .first()
            )
            if latest:
                quarter_date = latest[0]
                print(f"  Using latest quarter: {quarter_date.date()}")
            else:
                print("  Warning: No quarters with significant data found")
                return pd.DataFrame()

        # Find the dominant model version for this quarter
        most_common_version = (
            session.query(ValuationResult.model_version)
            .filter(ValuationResult.snapshot_timestamp == quarter_date)
            .group_by(ValuationResult.model_version)
            .order_by(func.count(ValuationResult.ticker).desc())
            .first()
        )
        model_version = most_common_version[0] if most_common_version else None

        if model_version:
            print(f"  Using model version: {model_version}")

        # Query valuations for this specific quarter and model
        # Also join with FinancialSnapshot to get price_t0 and shares_outstanding
        # for calculating historical market cap
        # Query valuations for this specific quarter and model
        # Also join with FinancialSnapshot to get price_t0 and shares_outstanding
        # for calculating historical market cap
        base_query = session.query(
            ValuationResult.ticker,
            ValuationResult.actual_mcap,
            ValuationResult.predicted_mcap_mean,
            ValuationResult.predicted_mcap_std,
            ValuationResult.relative_error,
            ValuationResult.residual_error,
            ValuationResult.relative_std,
            ValuationResult.snapshot_timestamp,
            Ticker.sector,
            Ticker.company_name,
            Ticker.industry,
            Ticker.original_currency,  # More reliable than outer-joined stored_currency
            FinancialSnapshot.price_t0,
            FinancialSnapshot.shares_outstanding,
            FinancialSnapshot.stored_currency,
        ).join(
            Ticker, Ticker.ticker == ValuationResult.ticker
        ).outerjoin(
            # Use OUTER JOIN for FinancialSnapshot since valuation timestamps may not
            # exactly match snapshot timestamps (e.g., for stocks with different fiscal years
            # like NVDA). Missing snapshot data is handled gracefully with defaults.
            FinancialSnapshot,
            (FinancialSnapshot.ticker == ValuationResult.ticker) &
            (FinancialSnapshot.snapshot_timestamp == ValuationResult.snapshot_timestamp)
        )

        query = base_query.filter(
            ValuationResult.snapshot_timestamp == quarter_date
        )

        if model_version:
            query = query.filter(ValuationResult.model_version == model_version)

        df = pd.read_sql(query.statement, session.bind)
        
        if df.empty:
            print(f"  Exact timestamp match failed. Trying date match for {quarter_date.date()}...")
            query = base_query.filter(
                func.date(ValuationResult.snapshot_timestamp) == quarter_date.strftime("%Y-%m-%d")
            )
            if model_version:
                 query = query.filter(ValuationResult.model_version == model_version)
            df = pd.read_sql(query.statement, session.bind)

        # Deduplicate by ticker (shouldn't be needed but safety check)
        df = df.drop_duplicates(subset=["ticker"])

        df["predicted_mcap_mean"] = pd.to_numeric(df["predicted_mcap_mean"], errors="coerce")
        df["actual_mcap"] = pd.to_numeric(df["actual_mcap"], errors="coerce")

        # Currency conversion: convert non-USD values to USD
        # Prefer stored_currency from FinancialSnapshot, fallback to Ticker.original_currency if NULL
        # This handles cases where the OUTER JOIN fails (no matching snapshot)
        currency_col = df["stored_currency"].fillna(df.get("original_currency")).fillna("USD")
        df["effective_currency"] = currency_col
        df["fx_rate"] = df["effective_currency"].map(FX_TO_USD).fillna(1.0)
        needs_fx = (df["effective_currency"] != "USD") & df["fx_rate"].notna()
        
        # Convert both actual and predicted to USD
        df.loc[needs_fx, "actual_mcap"] = df.loc[needs_fx, "actual_mcap"] * df.loc[needs_fx, "fx_rate"]
        df.loc[needs_fx, "predicted_mcap_mean"] = df.loc[needs_fx, "predicted_mcap_mean"] * df.loc[needs_fx, "fx_rate"]
        
        print(f"  Applied FX conversion to {needs_fx.sum()} non-USD stocks")

        # Recalculate relative_error with USD-normalized mcaps
        valid_mask = df["actual_mcap"].notna() & (df["actual_mcap"] > 0)
        df.loc[valid_mask, "relative_error"] = (
            (df.loc[valid_mask, "predicted_mcap_mean"] - df.loc[valid_mask, "actual_mcap"])
            / df.loc[valid_mask, "actual_mcap"]
        )

        # Recalculate residual_error (size-neutral mispricing)
        from src.valuation.size_correction import compute_residual_mispricing
        if valid_mask.sum() > 10:  # Need enough data for regression
            residual_err, _ = compute_residual_mispricing(
                df.loc[valid_mask, "relative_error"].values,
                df.loc[valid_mask, "actual_mcap"].values
            )
            df.loc[valid_mask, "residual_error"] = residual_err

        # Clean sector data
        df["sector"] = df["sector"].fillna("Unknown")
        df["sector"] = df["sector"].replace("", "Unknown")
        
        # Handle null residual_error (fallback to relative_error if not computed)
        if "residual_error" in df.columns:
            df["residual_error"] = df["residual_error"].fillna(df["relative_error"])

        return df
    finally:
        session.close()


def get_index_analysis_data() -> List[IndexAnalysis]:
    """Get index analysis directly from the database."""
    session = SessionLocal()
    try:
        service = IndexService(session)
        return service.analyze_all_indices()
    finally:
        session.close()


def get_index_mispricing_timeseries() -> List[Dict[str, Any]]:
    """Get index mispricing over time for each index, using per-quarter data."""
    session = SessionLocal()
    try:
        from src.db.models.index import IndexMembership, Index
        from sqlalchemy import func
        import pandas as pd
        
        # Get all quarter dates with significant valuations (>1000 unique tickers)
        quarter_dates = session.query(
            ValuationResult.snapshot_timestamp
        ).group_by(
            ValuationResult.snapshot_timestamp
        ).having(
            func.count(func.distinct(ValuationResult.ticker)) > 1000
        ).order_by(
            ValuationResult.snapshot_timestamp
        ).all()
        
        # Get all indices
        indices = session.query(Index.index_id).all()
        index_ids = [idx[0] for idx in indices]
        
        results = []
        for (quarter_date,) in quarter_dates:
            # Find dominant model version for this quarter
            most_common_version = (
                session.query(ValuationResult.model_version)
                .filter(ValuationResult.snapshot_timestamp == quarter_date)
                .group_by(ValuationResult.model_version)
                .order_by(func.count(ValuationResult.ticker).desc())
                .first()
            )
            model_version = most_common_version[0] if most_common_version else None
            
            for index_id in index_ids:
                # Get valuations for this index at this specific quarter
                # Use pandas for proper deduplication per ticker
                query = session.query(
                    ValuationResult.ticker,
                    ValuationResult.actual_mcap,
                    ValuationResult.predicted_mcap_mean,
                    ValuationResult.relative_error,
                    ValuationResult.residual_error,
                ).join(
                    IndexMembership,
                    IndexMembership.ticker == ValuationResult.ticker,
                ).filter(
                    IndexMembership.index_id == index_id,
                    ValuationResult.snapshot_timestamp == quarter_date,
                )
                
                # Filter by model version if found
                if model_version:
                    query = query.filter(ValuationResult.model_version == model_version)
                
                df = pd.read_sql(query.statement, session.bind)
                
                if df.empty:
                    continue
                
                # Deduplicate by ticker (in case membership has multiple entries)
                df = df.drop_duplicates(subset=['ticker'])
                
                total_actual = float(df['actual_mcap'].sum())
                total_predicted = float(df['predicted_mcap_mean'].sum())
                count = len(df)
                
                if total_actual > 0 and count >= 5:  # At least 5 stocks for meaningful data
                    mispricing = (total_predicted - total_actual) / total_actual
                    
                    # Compute residual mispricing (cap-weighted average of residual errors)
                    residual_mispricing = mispricing # Fallback
                    if 'residual_error' in df.columns:
                        # Fill NA with relative_error where residual_error is missing
                        df['residual_error'] = df['residual_error'].fillna(df['relative_error'])
                        weights = df['actual_mcap'] / total_actual
                        residual_mispricing = float((weights * df['residual_error']).sum())
                    
                    results.append({
                        "index": index_id,
                        "date": quarter_date.strftime("%Y-%m-%d"),
                        "mispricing": mispricing,
                        "residualMispricing": residual_mispricing,
                        "count": count,
                    })
        
        return results
    finally:
        session.close()


def get_sector_mispricing_timeseries() -> List[Dict[str, Any]]:
    """Get sector mispricing over time, using per-quarter data grouped by sector."""
    session = SessionLocal()
    try:
        from sqlalchemy import func

        # Get all quarter dates with significant valuations (>1000 unique tickers)
        quarter_dates = session.query(
            ValuationResult.snapshot_timestamp
        ).group_by(
            ValuationResult.snapshot_timestamp
        ).having(
            func.count(func.distinct(ValuationResult.ticker)) > 1000
        ).order_by(
            ValuationResult.snapshot_timestamp
        ).all()

        results = []
        for (quarter_date,) in quarter_dates:
            # Find dominant model version for this quarter
            most_common_version = (
                session.query(ValuationResult.model_version)
                .filter(ValuationResult.snapshot_timestamp == quarter_date)
                .group_by(ValuationResult.model_version)
                .order_by(func.count(ValuationResult.ticker).desc())
                .first()
            )
            model_version = most_common_version[0] if most_common_version else None

            # Get valuations with sector info for this quarter
            query = session.query(
                ValuationResult.ticker,
                ValuationResult.actual_mcap,
                ValuationResult.predicted_mcap_mean,
                ValuationResult.relative_error,
                ValuationResult.residual_error,
                Ticker.sector,
            ).join(
                Ticker, Ticker.ticker == ValuationResult.ticker,
            ).filter(
                ValuationResult.snapshot_timestamp == quarter_date,
            )

            if model_version:
                query = query.filter(ValuationResult.model_version == model_version)

            df = pd.read_sql(query.statement, session.bind)

            if df.empty:
                continue

            # Clean sector data
            df["sector"] = df["sector"].fillna("Unknown")
            df["sector"] = df["sector"].replace("", "Unknown")

            # Deduplicate by ticker
            df = df.drop_duplicates(subset=['ticker'])

            # Handle null residual_error
            df['residual_error'] = df['residual_error'].fillna(df['relative_error'])

            # Group by sector and compute aggregates
            for sector in df['sector'].unique():
                sector_df = df[df['sector'] == sector]

                total_actual = float(sector_df['actual_mcap'].sum())
                total_predicted = float(sector_df['predicted_mcap_mean'].sum())
                count = len(sector_df)

                if total_actual > 0 and count >= 5:  # At least 5 stocks for meaningful data
                    mispricing = (total_predicted - total_actual) / total_actual

                    # Compute residual mispricing (cap-weighted average)
                    weights = sector_df['actual_mcap'] / total_actual
                    residual_mispricing = float((weights * sector_df['residual_error']).sum())

                    results.append({
                        "sector": sector,
                        "date": quarter_date.strftime("%Y-%m-%d"),
                        "mispricing": mispricing,
                        "residualMispricing": residual_mispricing,
                        "count": count,
                    })

        return results
    finally:
        session.close()


def get_per_quarter_size_coefficients() -> List[Dict[str, Any]]:
    """
    Compute size premium coefficients for each quarter.
    
    Returns list of dicts with quarter, slope, SE, t-stat, p-value.
    """
    from sqlalchemy import func
    
    session = SessionLocal()
    try:
        # Get all quarters with significant valuations
        quarter_dates = session.query(
            ValuationResult.snapshot_timestamp
        ).group_by(
            ValuationResult.snapshot_timestamp
        ).having(
            func.count(func.distinct(ValuationResult.ticker)) > 100
        ).order_by(
            ValuationResult.snapshot_timestamp
        ).all()
        
        coefficients = []
        for (quarter_date,) in quarter_dates:
            # Get valuations for this quarter
            query = session.query(
                ValuationResult.actual_mcap,
                ValuationResult.relative_error,
            ).filter(
                ValuationResult.snapshot_timestamp == quarter_date,
                ValuationResult.actual_mcap > 0,
            )
            
            df = pd.read_sql(query.statement, session.bind)
            
            if len(df) < 50:
                continue
            
            # Estimate coefficient
            coef = estimate_size_coefficient(
                mispricing=df["relative_error"].values,
                market_cap=df["actual_mcap"].values,
                quarter=quarter_date.strftime("%Y-%m-%d"),
            )
            
            if coef:
                coefficients.append(coef.to_dict())
        
        return coefficients
    finally:
        session.close()

def get_backtest_data() -> Dict[str, Any]:
    """Load backtest detailed data for time-series IC visualization."""
    detailed_path = "data/signal_backtest_detailed.csv"
    if not os.path.exists(detailed_path):
        print(f"  Warning: {detailed_path} not found. Run scripts/run_sector_backtest.py first.")
        return {"sector_ts": [], "index_ts": [], "sector_summary": [], "index_summary": [], "horizon": []}

    df = pd.read_csv(detailed_path)

    # Build time-series data for scatter plots (individual    # Build time-series data points for scatter plot
    def build_timeseries(group_df: pd.DataFrame, min_samples_per_point: int = 10) -> List[Dict]:
        """Build time-series data points for scatter plot."""
        results = []
        for _, row in group_df.iterrows():
            # Only include 10, 30, 60 day horizons
            if row["horizon"] not in [10, 30, 60]:
                continue
            if row["n_obs"] < min_samples_per_point:
                continue
            
            pval = max(row["ic_pval"], 1e-10)  # Floor to avoid log(0)
            log_pval = -np.log10(pval)
            
            results.append({
                "name": row["group_name"],
                "quarter": row["quarter"],
                "horizon": int(row["horizon"]),
                "ic": float(row["ic"]),
                "pval": float(row["ic_pval"]),
                "log_pval": float(log_pval),
                "n_obs": int(row["n_obs"]),
                "spread": float(row["spread"]) if pd.notna(row.get("spread")) else 0.0,
                "hit_rate": float(row["hit_rate"]) if pd.notna(row.get("hit_rate")) else 0.5,
                "metric": row.get("metric", "raw"),
            })
        
        # Sort by quarter for proper time ordering
        results.sort(key=lambda x: x["quarter"])
        return results

    # Aggregate for summary statistics (used in Signal Quality table)
    def aggregate_group(group_df: pd.DataFrame, min_samples: int = 100) -> List[Dict]:
        """Aggregate across quarters, keep only groups with enough data."""
        results = []
        
        # Get available metrics (default to ['raw'] if column missing)
        metrics = group_df["metric"].unique() if "metric" in group_df.columns else ["raw"]
        
        for metric in metrics:
            m_df = group_df[group_df.get("metric", "raw") == metric] if "metric" in group_df.columns else group_df
            
            for horizon in [10, 30, 60, 90]:
                h_df = m_df[m_df["horizon"] == horizon]
                if h_df.empty:
                    continue

                for name in h_df["group_name"].unique():
                    name_df = h_df[h_df["group_name"] == name]
                    total_n = name_df["n_obs"].sum()

                    if total_n < min_samples:
                        continue

                    weights = name_df["n_obs"].values
                    avg_ic = np.average(name_df["ic"].values, weights=weights)
                    avg_spread = np.average(name_df["spread"].values, weights=weights) if "spread" in name_df.columns else 0
                    avg_hit_rate = np.average(name_df["hit_rate"].values, weights=weights) if "hit_rate" in name_df.columns else 0.5
                    median_pval = name_df["ic_pval"].median()

                    results.append({
                        "name": name,
                        "horizon": horizon,
                        "ic": float(avg_ic),
                        "pval": float(median_pval),
                        "n_obs": int(total_n),
                        "spread": float(avg_spread),
                        "hit_rate": float(avg_hit_rate),
                        "significant": bool(median_pval < 0.05),
                        "marginal": bool(0.05 <= median_pval < 0.10),
                        "metric": metric,
                    })

        return results

    # Build time-series for sectors and indices
    sector_df = df[df["group_type"] == "sector"]
    sector_ts = build_timeseries(sector_df)
    sector_all = aggregate_group(sector_df, min_samples=100)
    # Return all summaries, filtering by metric happens in JS
    sector_summary = sector_all 

    index_df = df[df["group_type"] == "index"]
    index_ts = build_timeseries(index_df)
    index_all = aggregate_group(index_df, min_samples=100)
    index_summary = index_all

    # Horizon comparison (IC decay) - Compute for RAW metric only to avoid noise
    horizon_data = []
    # Filter for Raw only
    all_raw = [d for d in sector_all + index_all if d.get("metric", "raw") == "raw"]
    
    for horizon in [10, 30, 60, 90]:
        h_items = [d for d in all_raw if d["horizon"] == horizon]
        if h_items:
            weights = [d["n_obs"] for d in h_items]
            avg_ic = np.average([d["ic"] for d in h_items], weights=weights)
            avg_spread = np.average([d["spread"] for d in h_items], weights=weights)
            horizon_data.append({
                "horizon": horizon,
                "avg_ic": float(avg_ic),
                "avg_spread": float(avg_spread),
            })

    return {
        "sector_ts": sector_ts,
        "index_ts": index_ts,
        "sector_summary": sector_summary,
        "index_summary": index_summary,
        "horizon": horizon_data,
    }


def compute_summary_stats(
    valuation_df: pd.DataFrame,
    index_data: List[IndexAnalysis],
    size_correction: Optional[SizeCorrectionResult] = None,
) -> Dict[str, Any]:
    """Compute dashboard summary statistics."""
    total_tickers = len(valuation_df)
    total_actual_mcap = float(valuation_df["actual_mcap"].sum())
    total_predicted_mcap = float(valuation_df["predicted_mcap_mean"].sum())

    undervalued = valuation_df[valuation_df["relative_error"] > 0]
    overvalued = valuation_df[valuation_df["relative_error"] < 0]

    avg_mispricing = float(valuation_df["relative_error"].mean())
    median_mispricing = float(valuation_df["relative_error"].median())

    # Get quarter date from dataframe if available
    quarter_date_str = "N/A"
    if "snapshot_timestamp" in valuation_df.columns and len(valuation_df) > 0:
        quarter_date = valuation_df["snapshot_timestamp"].iloc[0]
        if hasattr(quarter_date, "strftime"):
            quarter_date_str = quarter_date.strftime("%Y-%m-%d")
        else:
            quarter_date_str = str(quarter_date)[:10]

    # Get active metric info
    metric_info = get_metric_info()
    
    # Size premium info if available
    size_premium_info = None
    if size_correction:
        size_premium_info = {
            "method": size_correction.size_premium_estimate.estimation_method,
            "n_obs": size_correction.size_premium_estimate.n_observations,
            "smoothing": size_correction.size_premium_estimate.smoothing_frac,
        }

    return {
        "total_tickers": total_tickers,
        "total_actual_mcap_t": total_actual_mcap / 1e12,
        "total_predicted_mcap_t": total_predicted_mcap / 1e12,
        "undervalued_count": len(undervalued),
        "overvalued_count": len(overvalued),
        "avg_mispricing_pct": avg_mispricing * 100,
        "median_mispricing_pct": median_mispricing * 100,
        "indices_tracked": len(index_data),
        "quarter_date": quarter_date_str,
        # Metric info
        "metric_name": metric_info.name,
        "metric_formula": metric_info.formula,
        "metric_description": metric_info.description,
        # Size premium
        "size_premium": size_premium_info,
    }


def build_scatter_data(
    valuation_df: pd.DataFrame, 
    min_mcap_b: float = 0.1,
    exclude_smallest: int = 700
) -> List[Dict[str, Any]]:
    """Build scatter data from a valuation dataframe.

    Args:
        valuation_df: DataFrame with valuation results
        min_mcap_b: Minimum market cap in billions to include (default 0.1 = $100M)
        exclude_smallest: Number of smallest companies by mcap to exclude (default 500)
                         This improves frontend rendering performance.
    """
    # First filter by min mcap
    df = valuation_df.copy()
    df["actual_mcap_b"] = pd.to_numeric(df["actual_mcap"], errors="coerce") / 1e9
    df = df[df["actual_mcap_b"] >= min_mcap_b]
    
    # Exclude the smallest N companies by market cap
    if exclude_smallest > 0 and len(df) > exclude_smallest:
        # Sort by mcap ascending, drop the smallest N
        df = df.nlargest(len(df) - exclude_smallest, "actual_mcap_b")
        print(f"  Excluded {exclude_smallest} smallest companies. Remaining: {len(df)}")
    
    scatter_data = []
    for _, row in df.iterrows():
        actual_b = float(row["actual_mcap_b"])
        predicted_b = float(row["predicted_mcap_mean"]) / 1e9
        mispricing = float(row["relative_error"])
        residual_mispricing = float(row["residual_error"]) if pd.notna(row["residual_error"]) else mispricing
        rel_std = float(row["relative_std"]) if pd.notna(row["relative_std"]) else 0.0
        sector = row["sector"] if row["sector"] else "Unknown"

        scatter_data.append({
            "ticker": row["ticker"],
            "actual": actual_b,
            "predicted": predicted_b,
            "mispricing": mispricing,
            "residualMispricing": residual_mispricing,
            "relStd": rel_std,
            "mispricingPct": f"{mispricing*100:.1f}%",
            "residualMispricingPct": f"{residual_mispricing*100:.1f}%",
            "sector": sector,
            "company": row["company_name"] or row["ticker"],
            "industry": row["industry"] or "N/A",
            "sectorColor": SECTOR_COLORS.get(sector, SECTOR_COLORS["Unknown"]),
        })
    return scatter_data


def build_sector_breakdown(valuation_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Build sector breakdown from a valuation dataframe."""
    sector_stats = valuation_df.groupby("sector").agg({
        "ticker": "count",
        "relative_error": "mean",
        "actual_mcap": "sum"
    }).reset_index()
    sector_stats.columns = ["sector", "count", "avg_mispricing", "total_mcap"]
    sector_breakdown = []
    for _, row in sector_stats.iterrows():
        sector_breakdown.append({
            "sector": row["sector"],
            "count": int(row["count"]),
            "avgMispricing": float(row["avg_mispricing"]),
            "totalMcap": float(row["total_mcap"]) / 1e12,
            "color": SECTOR_COLORS.get(row["sector"], SECTOR_COLORS["Unknown"]),
        })
    return sorted(sector_breakdown, key=lambda x: x["totalMcap"], reverse=True)


def build_and_save_dashboard_data(
    valuation_df: pd.DataFrame,
    index_data: List[IndexAnalysis],
    stats: Dict[str, Any],
    backtest_data: Dict[str, Any],
    index_timeseries: List[Dict[str, Any]] = None,
    sector_timeseries: List[Dict[str, Any]] = None,
    size_correction: Optional[SizeCorrectionResult] = None,
    size_coefficients: List[Dict[str, Any]] = None,
    available_quarters: List[str] = None,
) -> None:
    """Build dashboard data dictionary and save to JSON.

    Only saves the latest quarter's data in the main file.
    Historical quarters are saved separately for lazy loading.
    """

    # Prepare size premium curve data
    size_premium_curve = []
    size_correction_model = None

    if size_correction:
        est = size_correction.size_premium_estimate
        
        # Save model coefficients if available
        if hasattr(est, 'coefficients') and est.coefficients:
            size_correction_model = {
                "method": est.estimation_method,
                "coefficients": est.coefficients, # [a, b, c]
                "equation": "log(1 + mispricing) = a*logMcap^2 + b*logMcap + c"
            }

        # Sample points along the curve for plotting
        if hasattr(est, 'log_mcap_grid'):
             for i in range(0, len(est.log_mcap_grid), max(1, len(est.log_mcap_grid) // 50)):
                size_premium_curve.append({
                    "logMcap": float(est.log_mcap_grid[i]),
                    "mcapB": float(np.exp(est.log_mcap_grid[i])) / 1e9,
                    # Convert log-space expectation back to percentage: exp(y) - 1
                    "expectedMispricing": float(np.expm1(est.expected_mispricing[i])),
                })

    # Prepare valuation scatter data (for latest quarter only)
    scatter_data = build_scatter_data(valuation_df)

    # Prepare index bar chart data
    index_chart_data = []
    for analysis in index_data:
        res_misc = analysis.residual_mispricing if analysis.residual_mispricing is not None else analysis.mispricing
        index_chart_data.append({
            "index": analysis.index,
            "mispricing": analysis.mispricing,
            "residualMispricing": res_misc,
            "mispricingPct": f"{analysis.mispricing*100:.2f}%",
            "residualMispricingPct": f"{res_misc*100:.2f}%",
            "color": "#10b981" if analysis.mispricing > 0 else "#ef4444",
            "status": analysis.status,
            "totalActual": f"${analysis.total_actual/1e9:,.1f}B",
            "totalPredicted": f"${analysis.total_predicted/1e9:,.1f}B",
            "count": analysis.count,
            "officialCount": analysis.official_count,
        })

    # Sector breakdown (for latest quarter)
    sector_breakdown = build_sector_breakdown(valuation_df)

    # Construct final payload - NO valuation_by_quarter (loaded lazily)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "stats": stats,
        "scatter_data": scatter_data,
        "index_chart_data": index_chart_data,
        "sector_breakdown": sector_breakdown,
        "backtest_data": backtest_data,
        "index_timeseries": index_timeseries,
        "sector_timeseries": sector_timeseries,
        "size_premium_curve": size_premium_curve,
        "size_correction_model": size_correction_model,
        "size_coefficients": size_coefficients,
        "available_quarters": available_quarters or [],
    }

    # Save to web/public/dashboard_data.json
    output_dir = "web/public"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dashboard_data.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Dashboard data saved to {output_path}")


def save_quarter_data(quarter_str: str, quarter_df: pd.DataFrame) -> None:
    """Save a single quarter's data to a separate JSON file for lazy loading."""
    output_dir = "web/public/quarters"
    os.makedirs(output_dir, exist_ok=True)

    quarter_data = {
        "quarter": quarter_str,
        "scatter_data": build_scatter_data(quarter_df),
        "sector_breakdown": build_sector_breakdown(quarter_df),
    }

    output_path = os.path.join(output_dir, f"{quarter_str}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(quarter_data, f, indent=2)

    return output_path


def main() -> None:
    """Generate the financial dashboard data."""
    with open("dashboard_debug.log", "w") as log:
        log.write("Generating Financial Dashboard Data...\n")
        print("Generating Financial Dashboard Data...")

    # Load data
    with open("dashboard_debug.log", "a") as log:
        log.write("Fetching valuation data...\n")
    print("Fetching valuation data...")
    valuation_df = get_valuation_data_with_sectors()
    
    with open("dashboard_debug.log", "a") as log:
        log.write(f"Valuation DataFrame shape: {valuation_df.shape}\n")
    print(f"Valuation DataFrame shape: {valuation_df.shape}")
    
    if valuation_df.empty:
        print("ERROR: Valuation DataFrame is empty!")
        # Proceeding will save empty JSON
        
    print(f"  Loaded {len(valuation_df)} valuations")

    print("Fetching index analysis...")
    index_data = get_index_analysis_data()
    print(f"  Loaded {len(index_data)} indices")

    print("  Loading index mispricing time series...")
    index_timeseries = get_index_mispricing_timeseries()
    print(f"  Loaded {len(index_timeseries)} index-quarter data points")

    print("  Loading sector mispricing time series...")
    sector_timeseries = get_sector_mispricing_timeseries()
    print(f"  Loaded {len(sector_timeseries)} sector-quarter data points")

    print("  Loading backtest data...")
    backtest_data = get_backtest_data()
    print(f"  Loaded {len(backtest_data['sector_summary'])} sectors, {len(backtest_data['index_summary'])} indices with signal data")

    if valuation_df.empty:
        print("Error: No valuation data found. Run the valuation pipeline first.")
        return

    # Re-estimate size correction model to get coefficients and curve for dashboard
    # (The residual_error values are already in valuation_df, but we need the model params)
    print("  Estimating size premium model for visualization...")
    size_correction = None
    if not valuation_df.empty:
        # Use simple arrays
        mispricing_vals = valuation_df["relative_error"].values
        mcap_vals = valuation_df["actual_mcap"].values
        
        # apply_size_correction handles filtering internally
        size_correction = apply_size_correction(mispricing_vals, mcap_vals)

    # Load per-quarter size coefficients for time-series chart
    print("  Computing per-quarter size coefficients...")
    size_coefficients = get_per_quarter_size_coefficients()
    print(f"    Loaded {len(size_coefficients)} quarters with coefficients")

    # Compute stats
    stats = compute_summary_stats(valuation_df, index_data, size_correction)

    # Generate per-quarter valuation data as separate files for lazy loading
    print("  Generating per-quarter valuation data (lazy loading)...")
    available_quarters = get_available_quarters()
    quarter_strings = []
    for q in available_quarters:
        q_str = q.strftime("%Y-%m-%d") if hasattr(q, 'strftime') else str(q)[:10]
        q_df = get_valuation_data_with_sectors(q)
        if not q_df.empty:
            save_quarter_data(q_str, q_df)
            quarter_strings.append(q_str)
    print(f"    Saved {len(quarter_strings)} quarter files to web/public/quarters/")

    # Generate main JSON (latest quarter data only)
    build_and_save_dashboard_data(
        valuation_df,
        index_data,
        stats,
        backtest_data,
        index_timeseries,
        sector_timeseries,
        size_correction,
        size_coefficients,
        quarter_strings,
    )

if __name__ == "__main__":
    main()
