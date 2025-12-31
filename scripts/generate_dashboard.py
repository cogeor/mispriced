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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.db.session import SessionLocal
from src.db.models import Ticker, ValuationResult
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
        query = session.query(
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
        ).join(
            Ticker, Ticker.ticker == ValuationResult.ticker
        ).filter(
            ValuationResult.snapshot_timestamp == quarter_date
        )

        if model_version:
            query = query.filter(ValuationResult.model_version == model_version)

        df = pd.read_sql(query.statement, session.bind)

        # Deduplicate by ticker (shouldn't be needed but safety check)
        df = df.drop_duplicates(subset=["ticker"])

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
    def build_timeseries(group_df: pd.DataFrame, min_samples_per_point: int = 30) -> List[Dict]:
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


def build_and_save_dashboard_data(
    valuation_df: pd.DataFrame,
    index_data: List[IndexAnalysis],
    stats: Dict[str, Any],
    backtest_data: Dict[str, Any],
    index_timeseries: List[Dict[str, Any]] = None,
    size_correction: Optional[SizeCorrectionResult] = None,
    size_coefficients: List[Dict[str, Any]] = None,
) -> None:
    """Build dashboard data dictionary and save to JSON."""

    # Prepare size premium curve data
    size_premium_curve = []
    if size_correction:
        est = size_correction.size_premium_estimate
        # Sample points along the curve for plotting
        if hasattr(est, 'log_mcap_grid'):
             for i in range(0, len(est.log_mcap_grid), max(1, len(est.log_mcap_grid) // 50)):
                size_premium_curve.append({
                    "logMcap": float(est.log_mcap_grid[i]),
                    "mcapB": float(np.exp(est.log_mcap_grid[i])) / 1e9,
                    "expectedMispricing": float(est.expected_mispricing[i]),
                })

    # Prepare valuation scatter data
    scatter_data = []
    for idx, (_, row) in enumerate(valuation_df.iterrows()):
        actual_b = float(row["actual_mcap"]) / 1e9
        predicted_b = float(row["predicted_mcap_mean"]) / 1e9
        mispricing = float(row["relative_error"])
        # Get residual (size-neutral) mispricing from DB
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

    # Sector breakdown
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
    sector_breakdown = sorted(sector_breakdown, key=lambda x: x["totalMcap"], reverse=True)


    # Construct final payload
    payload = {
        "generated_at": datetime.now().isoformat(),
        "stats": stats,
        "scatter_data": scatter_data,
        "index_chart_data": index_chart_data,
        "sector_breakdown": sector_breakdown,
        "backtest_data": backtest_data,
        "index_timeseries": index_timeseries,
        "size_premium_curve": size_premium_curve,
        "size_coefficients": size_coefficients,
    }

    # Save to web/public/dashboard_data.json
    output_dir = "web/public"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dashboard_data.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    
    print(f"Dashboard data saved to {output_path}")


def main() -> None:
    """Generate the financial dashboard data."""
    print("Generating Financial Dashboard Data...")

    # Load data
    print("  Loading valuation data with sectors...")
    valuation_df = get_valuation_data_with_sectors()
    print(f"  Loaded {len(valuation_df)} valuations")

    print("  Loading index analysis data...")
    index_data = get_index_analysis_data()
    print(f"  Loaded {len(index_data)} indices")

    print("  Loading index mispricing time series...")
    index_timeseries = get_index_mispricing_timeseries()
    print(f"  Loaded {len(index_timeseries)} index-quarter data points")

    print("  Loading backtest data...")
    backtest_data = get_backtest_data()
    print(f"  Loaded {len(backtest_data['sector_summary'])} sectors, {len(backtest_data['index_summary'])} indices with signal data")

    if valuation_df.empty:
        print("Error: No valuation data found. Run the valuation pipeline first.")
        return

    # Size correction is now applied at valuation time and stored in DB (residual_error)
    size_correction = None

    # Load per-quarter size coefficients for time-series chart
    print("  Computing per-quarter size coefficients...")
    size_coefficients = get_per_quarter_size_coefficients()
    print(f"    Loaded {len(size_coefficients)} quarters with coefficients")

    # Compute stats
    stats = compute_summary_stats(valuation_df, index_data, size_correction)

    # Generate JSON
    build_and_save_dashboard_data(
        valuation_df, 
        index_data, 
        stats, 
        backtest_data, 
        index_timeseries, 
        size_correction, 
        size_coefficients
    )

if __name__ == "__main__":
    main()
