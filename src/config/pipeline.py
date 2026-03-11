"""
Shared pipeline configuration constants.

Single source of truth for thresholds and settings used across
the valuation, backtest, and dashboard generation scripts.
"""

from typing import List

# Minimum number of snapshots/valuations for a quarter to be considered valid
MIN_SNAPSHOTS_FOR_QUARTER: int = 1000

# Minimum market cap for inclusion in valuation pipeline
MIN_MARKET_CAP: float = 100e6  # $100M

# Indices tracked by the pipeline
TRACKED_INDICES: List[str] = [
    "SP500",
    "NASDAQ100",
    "FTSE100",
    "DAX",
    "EUROSTOXX50",
    "CAC40",
    "SMI",
    "NIFTY50",
    "SSE50",
    "HSI",
    "ASX200",
    "KOSPI",
    "TAIEX",
]

# Cross-validation settings
N_CV_REPEATS: int = 10
N_CV_FOLDS: int = 5
MIN_SAMPLES_FOR_CV: int = 50

# Critical features - stocks missing ALL of these are excluded
CRITICAL_FEATURES: List[str] = ["total_revenue", "net_income", "ebitda"]

# Web output directory (relative to project root)
WEB_DIR: str = "web"
WEB_PUBLIC_DIR: str = "web/public"
DATA_DIR: str = "data"
