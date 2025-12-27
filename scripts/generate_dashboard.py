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
                    results.append({
                        "index": index_id,
                        "date": quarter_date.strftime("%Y-%m-%d"),
                        "mispricing": mispricing,
                        "count": count,
                    })
        
        return results
    finally:
        session.close()


def get_backtest_data() -> Dict[str, Any]:
    """Load backtest detailed data for time-series IC visualization."""
    detailed_path = "data/signal_backtest_detailed.csv"
    if not os.path.exists(detailed_path):
        print(f"  Warning: {detailed_path} not found. Run scripts/run_sector_backtest.py first.")
        return {"sector_ts": [], "index_ts": [], "sector_summary": [], "index_summary": [], "horizon": []}

    df = pd.read_csv(detailed_path)

    # Build time-series data for scatter plots (individual quarter-horizon-group points)
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
            })
        
        # Sort by quarter for proper time ordering
        results.sort(key=lambda x: x["quarter"])
        return results

    # Aggregate for summary statistics (used in Signal Quality table)
    def aggregate_group(group_df: pd.DataFrame, min_samples: int = 100) -> List[Dict]:
        """Aggregate across quarters, keep only groups with enough data."""
        results = []
        for horizon in [10, 30, 60, 90]:
            h_df = group_df[group_df["horizon"] == horizon]
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
                })

        return results

    # Build time-series for sectors and indices
    sector_df = df[df["group_type"] == "sector"]
    sector_ts = build_timeseries(sector_df)
    sector_all = aggregate_group(sector_df, min_samples=100)
    sector_summary = [s for s in sector_all if s["horizon"] == 30]
    sector_summary = sorted(sector_summary, key=lambda x: x["ic"], reverse=True)

    index_df = df[df["group_type"] == "index"]
    index_ts = build_timeseries(index_df)
    index_all = aggregate_group(index_df, min_samples=100)
    index_summary = [s for s in index_all if s["horizon"] == 30]
    index_summary = sorted(index_summary, key=lambda x: x["ic"], reverse=True)

    # Horizon comparison (IC decay)
    horizon_data = []
    all_data = sector_all + index_all
    for horizon in [10, 30, 60, 90]:
        h_items = [d for d in all_data if d["horizon"] == horizon]
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
    }


def generate_dashboard_html(
    valuation_df: pd.DataFrame,
    index_data: List[IndexAnalysis],
    stats: Dict[str, Any],
    backtest_data: Dict[str, Any],
    index_timeseries: List[Dict[str, Any]] = None,
) -> str:
    """Generate the complete dashboard HTML."""

    # Prepare valuation scatter data
    scatter_data = []
    for _, row in valuation_df.iterrows():
        actual_b = float(row["actual_mcap"]) / 1e9
        predicted_b = float(row["predicted_mcap_mean"]) / 1e9
        mispricing = float(row["relative_error"])
        sector = row["sector"] if row["sector"] else "Unknown"

        scatter_data.append({
            "ticker": row["ticker"],
            "actual": actual_b,
            "predicted": predicted_b,
            "mispricing": mispricing,
            "mispricingPct": f"{mispricing*100:.1f}%",
            "sector": sector,
            "company": row["company_name"] or row["ticker"],
            "industry": row["industry"] or "N/A",
            "sectorColor": SECTOR_COLORS.get(sector, SECTOR_COLORS["Unknown"]),
        })

    # Prepare index bar chart data from IndexAnalysis objects
    index_chart_data = []
    for analysis in index_data:
        index_chart_data.append({
            "index": analysis.index,
            "mispricing": analysis.mispricing,
            "mispricingPct": f"{analysis.mispricing*100:.2f}%",
            "color": "#10b981" if analysis.mispricing > 0 else "#ef4444",
            "status": analysis.status,
            "totalActual": f"${analysis.total_actual/1e9:,.1f}B",
            "totalPredicted": f"${analysis.total_predicted/1e9:,.1f}B",
            "count": analysis.count,
            "officialCount": analysis.official_count,
        })

    # Get top movers
    top_undervalued = sorted(scatter_data, key=lambda x: x["mispricing"], reverse=True)[:10]
    top_overvalued = sorted(scatter_data, key=lambda x: x["mispricing"])[:10]

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

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mispriced - Market Valuation Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        :root {{
            --bg-primary: #0a0c10;
            --bg-secondary: #12151a;
            --bg-card: #1a1d24;
            --border-color: #2a2e38;
            --text-primary: #e8eaed;
            --text-secondary: #8b919a;
            --text-muted: #5a6069;
            --accent-green: #00a86b;
            --accent-red: #d94545;
            --accent-blue: #4a90d9;
            --accent-amber: #c9a227;
        }}
        body {{
            background: var(--bg-primary);
            color: var(--text-primary);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        .card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 4px;
        }}
        .stat-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
        }}
        .positive {{ color: var(--accent-green); }}
        .negative {{ color: var(--accent-red); }}
        .table-row:hover {{ background: rgba(255,255,255,0.03); }}
        .toggle-btn {{
            transition: all 0.15s ease;
            border: 1px solid var(--border-color);
        }}
        .toggle-btn.active {{
            background: var(--accent-blue);
            border-color: var(--accent-blue);
            color: white;
        }}
        .toggle-btn:not(.active) {{
            background: var(--bg-card);
            color: var(--text-secondary);
        }}
        .toggle-btn:not(.active):hover {{
            background: var(--bg-secondary);
        }}
        h1, h2, h3 {{
            font-weight: 500;
            letter-spacing: -0.01em;
        }}
        .legend-item {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 8px;
            background: var(--bg-card);
            border-radius: 3px;
            font-size: 11px;
        }}
    </style>
</head>
<body class="min-h-screen">
    <!-- Header -->
    <header class="border-b border-gray-800 px-6 py-4">
        <div class="max-w-7xl mx-auto flex items-center justify-between">
            <div>
                <h1 class="text-2xl font-bold text-white">Mispriced</h1>
                <p class="text-sm text-gray-400">Market Valuation Dashboard</p>
            </div>
            <div class="text-right text-sm text-gray-400">
                <div>Quarter: <span class="text-white font-medium">{stats["quarter_date"]}</span></div>
                <div>{stats["total_tickers"]:,} tickers analyzed</div>
                <div class="text-xs text-gray-500">Generated: <span id="timestamp"></span></div>
            </div>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-6 py-8">
        <!-- Summary Stats -->
        <section class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <div class="stat-card card p-4">
                <div class="text-sm text-gray-400 mb-1">Total Market Cap (Actual)</div>
                <div class="text-2xl font-bold">${stats["total_actual_mcap_t"]:.2f}T</div>
            </div>
            <div class="stat-card card p-4">
                <div class="text-sm text-gray-400 mb-1">Total Market Cap (Predicted)</div>
                <div class="text-2xl font-bold">${stats["total_predicted_mcap_t"]:.2f}T</div>
            </div>
            <div class="stat-card card p-4">
                <div class="text-sm text-gray-400 mb-1">Undervalued / Overvalued</div>
                <div class="text-2xl font-bold">
                    <span class="positive">{stats["undervalued_count"]:,}</span>
                    <span class="text-gray-500">/</span>
                    <span class="negative">{stats["overvalued_count"]:,}</span>
                </div>
            </div>
            <div class="stat-card card p-4">
                <div class="text-sm text-gray-400 mb-1">Avg Mispricing</div>
                <div class="text-2xl font-bold {'positive' if stats["avg_mispricing_pct"] > 0 else 'negative'}">
                    {stats["avg_mispricing_pct"]:+.2f}%
                </div>
            </div>
        </section>

        <!-- Charts Row -->
        <section class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <!-- Index Mispricing Chart -->
            <div class="card p-4">
                <h2 class="text-lg font-semibold mb-4">Index Mispricing (Current)</h2>
                <div id="indexChart" style="height: 350px;"></div>
            </div>

            <!-- Index Mispricing Over Time -->
            <div class="card p-4">
                <h2 class="text-lg font-semibold mb-4">Index Mispricing Over Time</h2>
                <div id="indexTimeSeriesChart" style="height: 350px;"></div>
            </div>
        </section>

        <!-- Valuation Charts Row -->
        <section class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <div class="card p-4">
                <div class="flex items-center justify-between mb-4">
                    <h2 class="text-lg font-semibold">Stock Valuations</h2>
                    <div class="flex gap-2">
                        <button id="colorByMispricing" class="toggle-btn active px-3 py-1 rounded text-sm">
                            Color by Mispricing
                        </button>
                        <button id="colorBySector" class="toggle-btn px-3 py-1 rounded text-sm">
                            Color by Sector
                        </button>
                    </div>
                </div>
                <div id="valuationChart" style="height: 400px;"></div>
            </div>
        </section>

        <!-- Sector Breakdown with Legend -->
        <section class="card p-4 mb-8">
            <h2 class="text-lg font-semibold mb-4">Sector Breakdown</h2>
            <div id="sectorChart" style="height: 300px;"></div>
            <div class="mt-4 pt-4 border-t border-gray-700">
                <div class="flex flex-wrap gap-4" id="sectorLegend"></div>
            </div>
        </section>

        <!-- Tables Row -->
        <section class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <!-- Top Undervalued -->
            <div class="card p-4">
                <h2 class="text-lg font-semibold mb-4 positive">Top Undervalued Stocks</h2>
                <div class="overflow-x-auto">
                    <table class="w-full text-sm">
                        <thead>
                            <tr class="text-left text-gray-400 border-b border-gray-700">
                                <th class="pb-2">Ticker</th>
                                <th class="pb-2">Sector</th>
                                <th class="pb-2 text-right">Actual</th>
                                <th class="pb-2 text-right">Predicted</th>
                                <th class="pb-2 text-right">Upside</th>
                            </tr>
                        </thead>
                        <tbody id="undervaluedTable"></tbody>
                    </table>
                </div>
            </div>

            <!-- Top Overvalued -->
            <div class="card p-4">
                <h2 class="text-lg font-semibold mb-4 negative">Top Overvalued Stocks</h2>
                <div class="overflow-x-auto">
                    <table class="w-full text-sm">
                        <thead>
                            <tr class="text-left text-gray-400 border-b border-gray-700">
                                <th class="pb-2">Ticker</th>
                                <th class="pb-2">Sector</th>
                                <th class="pb-2 text-right">Actual</th>
                                <th class="pb-2 text-right">Predicted</th>
                                <th class="pb-2 text-right">Downside</th>
                            </tr>
                        </thead>
                        <tbody id="overvaluedTable"></tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- Signal Quality Section -->
        <section class="mb-8">
            <h2 class="text-xl font-medium mb-2 text-white">Signal Quality Analysis</h2>
            <p class="text-sm text-gray-500 mb-6">Information Coefficient over time by horizon. Larger points indicate higher statistical significance (-log10 p-value).</p>

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <!-- IC by Sector Time Series -->
                <div class="card p-4">
                    <div class="flex items-center justify-between mb-4">
                        <h3 class="text-base font-medium">Sector IC Time Series</h3>
                        <div class="flex gap-1">
                            <span class="legend-item"><span style="width:8px;height:8px;border-radius:50%;background:#4a90d9;"></span>10d</span>
                            <span class="legend-item"><span style="width:8px;height:8px;border-radius:50%;background:#00a86b;"></span>30d</span>
                            <span class="legend-item"><span style="width:8px;height:8px;border-radius:50%;background:#c9a227;"></span>60d</span>
                        </div>
                    </div>
                    <div id="icSectorChart" style="height: 400px;"></div>
                </div>

                <!-- IC by Index Time Series -->
                <div class="card p-4">
                    <div class="flex items-center justify-between mb-4">
                        <h3 class="text-base font-medium">Index IC Time Series</h3>
                        <div class="flex gap-1">
                            <span class="legend-item"><span style="width:8px;height:8px;border-radius:50%;background:#4a90d9;"></span>10d</span>
                            <span class="legend-item"><span style="width:8px;height:8px;border-radius:50%;background:#00a86b;"></span>30d</span>
                            <span class="legend-item"><span style="width:8px;height:8px;border-radius:50%;background:#c9a227;"></span>60d</span>
                        </div>
                    </div>
                    <div id="icIndexChart" style="height: 400px;"></div>
                </div>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <!-- IC Decay by Horizon -->
                <div class="card p-4">
                    <h3 class="text-lg font-semibold mb-4">IC Decay by Horizon</h3>
                    <div id="icHorizonChart" style="height: 300px;"></div>
                </div>

                <!-- Signal Quality Table -->
                <div class="card p-4">
                    <h3 class="text-lg font-semibold mb-4">Signal Quality Summary</h3>
                    <div class="overflow-x-auto">
                        <table class="w-full text-sm">
                            <thead>
                                <tr class="text-left text-gray-400 border-b border-gray-700">
                                    <th class="pb-2">Group</th>
                                    <th class="pb-2 text-right">N</th>
                                    <th class="pb-2 text-right">IC</th>
                                    <th class="pb-2 text-right">Spread</th>
                                    <th class="pb-2 text-right">Hit Rate</th>
                                </tr>
                            </thead>
                            <tbody id="signalQualityTable"></tbody>
                        </table>
                    </div>
                </div>
            </div>
        </section>
    </main>

    <!-- Footer -->
    <footer class="border-t border-gray-800 px-6 py-4 mt-8">
        <div class="max-w-7xl mx-auto text-center text-sm text-gray-500">
            Data sourced from public filings. Valuations are model estimates and not financial advice.
        </div>
    </footer>

    <script>
        // Data
        const scatterData = {json.dumps(scatter_data)};
        const indexData = {json.dumps(index_chart_data)};
        const topUndervalued = {json.dumps(top_undervalued)};
        const topOvervalued = {json.dumps(top_overvalued)};
        const sectorBreakdown = {json.dumps(sector_breakdown)};
        const sectorColors = {json.dumps(SECTOR_COLORS)};

        // Backtest data
        const backtestSectorTS = {json.dumps(backtest_data["sector_ts"])};
        const backtestIndexTS = {json.dumps(backtest_data["index_ts"])};
        const backtestSectorSummary = {json.dumps(backtest_data["sector_summary"])};
        const backtestIndexSummary = {json.dumps(backtest_data["index_summary"])};
        const backtestHorizon = {json.dumps(backtest_data["horizon"])};

        // Index mispricing time series
        const indexTimeSeries = {json.dumps(index_timeseries or [])};

        // Timestamp
        document.getElementById('timestamp').textContent = new Date().toLocaleString();

        // Plotly professional theme config
        const darkLayout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#8b919a', size: 11 }},
            xaxis: {{ gridcolor: '#2a2e38', zerolinecolor: '#3a3e48', linecolor: '#2a2e38' }},
            yaxis: {{ gridcolor: '#2a2e38', zerolinecolor: '#3a3e48', linecolor: '#2a2e38' }},
            margin: {{ t: 30, r: 20, b: 50, l: 55 }},
        }};

        const config = {{ displayModeBar: false, responsive: true }};

        // Index Bar Chart
        Plotly.newPlot('indexChart', [{{
            x: indexData.map(d => d.index),
            y: indexData.map(d => d.mispricing),
            type: 'bar',
            marker: {{ color: indexData.map(d => d.color) }},
            text: indexData.map(d => d.mispricingPct),
            textposition: 'auto',
            textfont: {{ color: '#f1f5f9' }},
            hovertemplate:
                '<b>%{{x}}</b><br>' +
                'Mispricing: %{{text}}<br>' +
                'Status: %{{customdata[0]}}<br>' +
                'Coverage: %{{customdata[3]}} / %{{customdata[4]}}<br>' +
                'Actual: %{{customdata[1]}}<br>' +
                'Predicted: %{{customdata[2]}}<extra></extra>',
            customdata: indexData.map(d => [d.status, d.totalActual, d.totalPredicted, d.count, d.officialCount])
        }}], {{
            ...darkLayout,
            yaxis: {{ ...darkLayout.yaxis, title: 'Mispricing %', tickformat: '.1%' }},
            xaxis: {{ ...darkLayout.xaxis, title: 'Index' }},
        }}, config);

        // Index Mispricing Over Time Chart
        if (indexTimeSeries && indexTimeSeries.length > 0) {{
            // Group by index
            const indexGroups = {{}};
            indexTimeSeries.forEach(d => {{
                if (!indexGroups[d.index]) {{
                    indexGroups[d.index] = {{ dates: [], values: [], counts: [] }};
                }}
                indexGroups[d.index].dates.push(d.date);
                indexGroups[d.index].values.push(d.mispricing);
                indexGroups[d.index].counts.push(d.count);
            }});

            // Color palette for indices
            const indexColors = {{
                'SP500': '#3b82f6',
                'NASDAQ100': '#8b5cf6', 
                'RUSSELL2000': '#f59e0b',
                'EUROSTOXX50': '#10b981',
                'NIKKEI225': '#ef4444',
                'FTSE100': '#ec4899',
                'DAX': '#14b8a6',
                'CAC40': '#f97316',
                'SMI': '#6366f1',
                'NIFTY50': '#84cc16',
                'SSE50': '#06b6d4',
            }};

            const timeSeriesTraces = Object.entries(indexGroups).map(([indexName, data]) => ({{
                x: data.dates,
                y: data.values,
                type: 'scatter',
                mode: 'lines+markers',
                name: indexName,
                line: {{ color: indexColors[indexName] || '#8b919a', width: 2 }},
                marker: {{ size: 6 }},
                hovertemplate: '<b>' + indexName + '</b><br>Date: %{{x}}<br>Mispricing: %{{y:.1%}}<br>Stocks: %{{customdata}}<extra></extra>',
                customdata: data.counts,
            }}));

            Plotly.newPlot('indexTimeSeriesChart', timeSeriesTraces, {{
                ...darkLayout,
                yaxis: {{ ...darkLayout.yaxis, title: 'Mispricing %', tickformat: '.1%', zeroline: true, zerolinewidth: 2 }},
                xaxis: {{ ...darkLayout.xaxis, title: 'Quarter' }},
                legend: {{ orientation: 'h', y: -0.15, font: {{ size: 10 }} }},
                showlegend: true,
            }}, config);
        }}

        // Valuation Scatter Plot
        let currentColorMode = 'mispricing';

        function getMispricingColor(mispricing) {{
            // Gradient from red (negative) through neutral to green (positive)
            const absVal = Math.min(Math.abs(mispricing), 0.5);
            const intensity = absVal / 0.5;
            if (mispricing > 0) {{
                return `rgba(16, 185, 129, ${{0.3 + intensity * 0.7}})`;
            }} else {{
                return `rgba(239, 68, 68, ${{0.3 + intensity * 0.7}})`;
            }}
        }}

        function renderValuationChart(colorBy) {{
            let colors, showLegend = false;
            let traces = [];

            if (colorBy === 'sector') {{
                // Group by sector for legend
                const sectors = [...new Set(scatterData.map(d => d.sector))];
                sectors.forEach(sector => {{
                    const sectorPoints = scatterData.filter(d => d.sector === sector);
                    traces.push({{
                        x: sectorPoints.map(d => d.actual),
                        y: sectorPoints.map(d => d.mispricing * 100),
                        mode: 'markers',
                        type: 'scatter',
                        name: sector,
                        marker: {{
                            color: sectorColors[sector] || sectorColors['Unknown'],
                            size: 6,
                            opacity: 0.7
                        }},
                        text: sectorPoints.map(d => d.ticker),
                        customdata: sectorPoints.map(d => [d.company, d.sector, d.industry, d.mispricingPct, d.predicted]),
                        hovertemplate:
                            '<b>%{{text}}</b><br>' +
                            '%{{customdata[0]}}<br>' +
                            'Sector: %{{customdata[1]}}<br>' +
                            'Actual: $%{{x:.2f}}B<br>' +
                            'Predicted: $%{{customdata[4]:.2f}}B<br>' +
                            'Mispricing: %{{customdata[3]}}<extra></extra>'
                    }});
                }});
                showLegend = true;
            }} else {{
                colors = scatterData.map(d => getMispricingColor(d.mispricing));
                traces.push({{
                    x: scatterData.map(d => d.actual),
                    y: scatterData.map(d => d.mispricing * 100),
                    mode: 'markers',
                    type: 'scatter',
                    marker: {{
                        color: colors,
                        size: 6,
                    }},
                    text: scatterData.map(d => d.ticker),
                    customdata: scatterData.map(d => [d.company, d.sector, d.industry, d.mispricingPct, d.predicted]),
                    hovertemplate:
                        '<b>%{{text}}</b><br>' +
                        '%{{customdata[0]}}<br>' +
                        'Sector: %{{customdata[1]}}<br>' +
                        'Actual: $%{{x:.2f}}B<br>' +
                        'Predicted: $%{{customdata[4]:.2f}}B<br>' +
                        'Mispricing: %{{customdata[3]}}<extra></extra>'
                }});
            }}

            // Add zero reference line (fair value)
            const maxActual = Math.max(...scatterData.map(d => d.actual));
            const minActual = Math.min(...scatterData.filter(d => d.actual > 0).map(d => d.actual));
            traces.push({{
                x: [minActual, maxActual],
                y: [0, 0],
                mode: 'lines',
                type: 'scatter',
                line: {{ color: 'rgba(255,255,255,0.4)', dash: 'dash', width: 1 }},
                hoverinfo: 'skip',
                showlegend: false
            }});

            Plotly.newPlot('valuationChart', traces, {{
                ...darkLayout,
                xaxis: {{ ...darkLayout.xaxis, title: 'Market Cap ($B)', type: 'log' }},
                yaxis: {{ ...darkLayout.yaxis, title: 'Mispricing %', ticksuffix: '%' }},
                showlegend: showLegend,
                legend: {{
                    orientation: 'v',
                    x: 1.02,
                    y: 1,
                    font: {{ size: 10 }}
                }}
            }}, config);
        }}

        renderValuationChart('mispricing');

        // Toggle buttons
        document.getElementById('colorByMispricing').addEventListener('click', () => {{
            document.getElementById('colorByMispricing').classList.add('active');
            document.getElementById('colorBySector').classList.remove('active');
            renderValuationChart('mispricing');
        }});

        document.getElementById('colorBySector').addEventListener('click', () => {{
            document.getElementById('colorBySector').classList.add('active');
            document.getElementById('colorByMispricing').classList.remove('active');
            renderValuationChart('sector');
        }});

        // Sector Breakdown Chart
        Plotly.newPlot('sectorChart', [{{
            x: sectorBreakdown.map(d => d.sector),
            y: sectorBreakdown.map(d => d.avgMispricing * 100),
            type: 'bar',
            marker: {{ color: sectorBreakdown.map(d => d.color) }},
            text: sectorBreakdown.map(d => (d.avgMispricing * 100).toFixed(1) + '%'),
            textposition: 'auto',
            textfont: {{ color: '#f1f5f9' }},
            hovertemplate:
                '<b>%{{x}}</b><br>' +
                'Avg Mispricing: %{{text}}<br>' +
                'Stocks: %{{customdata[0]}}<br>' +
                'Total MCap: $%{{customdata[1]:.2f}}T<extra></extra>',
            customdata: sectorBreakdown.map(d => [d.count, d.totalMcap])
        }}], {{
            ...darkLayout,
            yaxis: {{ ...darkLayout.yaxis, title: 'Avg Mispricing %', tickformat: '.1f', ticksuffix: '%' }},
            xaxis: {{ ...darkLayout.xaxis, tickangle: -45 }},
            margin: {{ ...darkLayout.margin, b: 100 }}
        }}, config);

        // Populate tables
        function formatMcap(b) {{
            if (b >= 1000) return '$' + (b/1000).toFixed(1) + 'T';
            if (b >= 1) return '$' + b.toFixed(1) + 'B';
            return '$' + (b * 1000).toFixed(0) + 'M';
        }}

        function renderTable(data, tableId, isPositive) {{
            const tbody = document.getElementById(tableId);
            tbody.innerHTML = data.map(d => `
                <tr class="table-row border-b border-gray-800">
                    <td class="py-2 font-medium">${{d.ticker}}</td>
                    <td class="py-2 text-gray-400">${{d.sector}}</td>
                    <td class="py-2 text-right">${{formatMcap(d.actual)}}</td>
                    <td class="py-2 text-right">${{formatMcap(d.predicted)}}</td>
                    <td class="py-2 text-right ${{isPositive ? 'positive' : 'negative'}}">${{d.mispricingPct}}</td>
                </tr>
            `).join('');
        }}

        renderTable(topUndervalued, 'undervaluedTable', true);
        renderTable(topOvervalued, 'overvaluedTable', false);

        // Sector Legend
        const legendDiv = document.getElementById('sectorLegend');
        legendDiv.innerHTML = Object.entries(sectorColors).map(([sector, color]) => `
            <div class="flex items-center gap-2">
                <div class="w-3 h-3 rounded" style="background: ${{color}}"></div>
                <span class="text-sm text-gray-400">${{sector}}</span>
            </div>
        `).join('');

        // Signal Quality Charts
        function getICColor(ic, pval) {{
            // Green for positive IC, red for negative, opacity based on significance
            const opacity = pval < 0.05 ? 1 : (pval < 0.10 ? 0.7 : 0.4);
            if (ic > 0) {{
                return `rgba(16, 185, 129, ${{opacity}})`;
            }} else {{
                return `rgba(239, 68, 68, ${{opacity}})`;
            }}
        }}

        function getSignificanceMarker(pval) {{
            if (pval < 0.05) return '*';
            if (pval < 0.10) return '~';
            return '';
        }}

        // Horizon colors for IC time series
        const horizonColors = {{
            10: '#4a90d9',  // Blue
            30: '#00a86b',  // Green
            60: '#c9a227',  // Amber
        }};

        // Build IC time series scatter plot
        function buildICTimeSeriesChart(data, chartId) {{
            if (!data || data.length === 0) {{
                document.getElementById(chartId).innerHTML = '<div style="color:#5a6069;text-align:center;padding:40px;">No backtest data available</div>';
                return;
            }}

            // Group by sector/index name
            const groups = [...new Set(data.map(d => d.name))];
            const traces = [];

            // For each horizon, create a trace for each group
            [10, 30, 60].forEach(horizon => {{
                const horizonData = data.filter(d => d.horizon === horizon);
                if (horizonData.length === 0) return;

                groups.forEach(groupName => {{
                    const groupData = horizonData.filter(d => d.name === groupName);
                    if (groupData.length === 0) return;

                    // Scale point sizes based on -log10(pvalue)
                    // log_pval typically ranges from 0 (p=1) to ~10+ (very significant)
                    const sizes = groupData.map(d => {{
                        const baseSize = 6;
                        const maxSize = 22;
                        const scaledSize = baseSize + (d.log_pval / 5) * (maxSize - baseSize);
                        return Math.min(maxSize, Math.max(baseSize, scaledSize));
                    }});

                    traces.push({{
                        x: groupData.map(d => d.quarter),
                        y: groupData.map(d => d.ic),
                        mode: 'markers',
                        type: 'scatter',
                        name: `${{groupName}} (${{horizon}}d)`,
                        legendgroup: groupName,
                        showlegend: horizon === 30,  // Only show legend for 30d horizon
                        marker: {{
                            color: horizonColors[horizon],
                            size: sizes,
                            opacity: 0.75,
                            line: {{ color: 'rgba(0,0,0,0.3)', width: 1 }}
                        }},
                        text: groupData.map(d => d.name),
                        customdata: groupData.map(d => [d.pval, d.n_obs, d.spread, d.hit_rate, d.horizon]),
                        hovertemplate:
                            '<b>%{{text}}</b> (%{{customdata[4]}}d)<br>' +
                            'Quarter: %{{x}}<br>' +
                            'IC: %{{y:.3f}}<br>' +
                            'p-value: %{{customdata[0]:.4f}}<br>' +
                            'N: %{{customdata[1]:,}}<extra></extra>'
                    }});
                }});
            }});

            // Add zero reference line
            traces.push({{
                x: [...new Set(data.map(d => d.quarter))].sort(),
                y: Array([...new Set(data.map(d => d.quarter))].length).fill(0),
                mode: 'lines',
                type: 'scatter',
                line: {{ color: '#3a3e48', dash: 'dash', width: 1 }},
                hoverinfo: 'skip',
                showlegend: false
            }});

            Plotly.newPlot(chartId, traces, {{
                ...darkLayout,
                yaxis: {{ 
                    ...darkLayout.yaxis, 
                    title: 'Information Coefficient',
                    zeroline: true, 
                    zerolinecolor: '#3a3e48',
                    range: [-0.3, 0.3]
                }},
                xaxis: {{ 
                    ...darkLayout.xaxis, 
                    title: 'Quarter',
                    tickangle: -45
                }},
                showlegend: true,
                legend: {{ 
                    orientation: 'v', 
                    x: 1.02, 
                    y: 1,
                    font: {{ size: 9 }},
                    bgcolor: 'rgba(0,0,0,0)'
                }},
                margin: {{ ...darkLayout.margin, r: 120, b: 80 }}
            }}, config);
        }}

        // Render IC time series charts
        buildICTimeSeriesChart(backtestSectorTS, 'icSectorChart');
        buildICTimeSeriesChart(backtestIndexTS, 'icIndexChart');

        // IC Decay by Horizon Chart
        if (backtestHorizon.length > 0) {{
            Plotly.newPlot('icHorizonChart', [
                {{
                    x: backtestHorizon.map(d => d.horizon + 'd'),
                    y: backtestHorizon.map(d => d.avg_ic),
                    type: 'bar',
                    name: 'Avg IC',
                    marker: {{ color: '#3b82f6' }},
                    text: backtestHorizon.map(d => (d.avg_ic * 100).toFixed(2) + '%'),
                    textposition: 'auto',
                    textfont: {{ color: '#f1f5f9' }},
                }},
                {{
                    x: backtestHorizon.map(d => d.horizon + 'd'),
                    y: backtestHorizon.map(d => d.avg_spread),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Avg Spread',
                    yaxis: 'y2',
                    line: {{ color: '#f59e0b', width: 2 }},
                    marker: {{ size: 8 }},
                }}
            ], {{
                ...darkLayout,
                yaxis: {{ ...darkLayout.yaxis, title: 'Average IC', side: 'left' }},
                yaxis2: {{ title: 'Quantile Spread', overlaying: 'y', side: 'right', gridcolor: 'rgba(0,0,0,0)', tickformat: '.1%', color: '#f59e0b' }},
                xaxis: {{ ...darkLayout.xaxis, title: 'Horizon' }},
                legend: {{ orientation: 'h', y: 1.1, x: 0.5, xanchor: 'center' }},
                showlegend: true,
            }}, config);
        }}

        // Signal Quality Table
        function renderSignalQualityTable() {{
            const allData = [
                ...backtestSectorSummary.map(d => ({{...d, type: 'Sector'}})), 
                ...backtestIndexSummary.map(d => ({{...d, type: 'Index'}}))
            ];
            // Sort by absolute IC
            allData.sort((a, b) => Math.abs(b.ic) - Math.abs(a.ic));

            const tbody = document.getElementById('signalQualityTable');
            tbody.innerHTML = allData.slice(0, 15).map(d => {{
                const sig = d.pval < 0.05 ? '*' : (d.pval < 0.10 ? '~' : '');
                const icClass = d.ic > 0 ? 'positive' : 'negative';
                return `
                    <tr class="table-row border-b border-gray-800">
                        <td class="py-2">
                            <span class="text-xs text-gray-500">${{d.type}}</span>
                            <span class="font-medium ml-1">${{d.name}}</span>
                        </td>
                        <td class="py-2 text-right text-gray-400">${{d.n_obs.toLocaleString()}}</td>
                        <td class="py-2 text-right ${{icClass}}">${{(d.ic >= 0 ? '+' : '') + (d.ic * 100).toFixed(2)}}%${{sig}}</td>
                        <td class="py-2 text-right">${{(d.spread * 100).toFixed(1)}}%</td>
                        <td class="py-2 text-right">${{(d.hit_rate * 100).toFixed(0)}}%</td>
                    </tr>
                `;
            }}).join('');
        }}
        renderSignalQualityTable();
    </script>
</body>
</html>'''

    return html


def main() -> None:
    """Generate the financial dashboard."""
    print("Generating Financial Dashboard...")

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

    # Compute stats
    stats = compute_summary_stats(valuation_df, index_data)

    # Generate HTML
    html = generate_dashboard_html(valuation_df, index_data, stats, backtest_data, index_timeseries)

    # Ensure output directory exists
    os.makedirs("plots", exist_ok=True)
    output_file = "plots/dashboard.html"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Dashboard saved to {output_file}")

    # Open in browser
    try:
        os.startfile(output_file)
    except Exception:
        print(f"  Open {output_file} in your browser to view.")


if __name__ == "__main__":
    main()
