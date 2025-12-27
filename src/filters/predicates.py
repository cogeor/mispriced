from typing import Callable
import pandas as pd
import numpy as np

# Type alias for filter functions
FilterPredicate = Callable[[pd.DataFrame], pd.Series]


def market_cap_min(min_cap: float) -> FilterPredicate:
    """Filter to tickers with market cap >= min_cap."""
    def predicate(df: pd.DataFrame) -> pd.Series:
        return df["market_cap_t0"] >= min_cap
    return predicate


def market_cap_range(min_cap: float, max_cap: float) -> FilterPredicate:
    """Filter to tickers with market cap in range."""
    def predicate(df: pd.DataFrame) -> pd.Series:
        return (df["market_cap_t0"] >= min_cap) & (df["market_cap_t0"] <= max_cap)
    return predicate


    return predicate

def currency_filter(currency: str = "USD") -> FilterPredicate:
    """Filter to specific stored currency."""
    def predicate(df: pd.DataFrame) -> pd.Series:
        # Check 'stored_currency' if it exists (canonical), else fallback to 'currency' for tests/legacy
        col = "stored_currency" if "stored_currency" in df.columns else "currency"
        return df[col] == currency
    return predicate

# ... (existing predicates omitted for brevity in replace call if contiguous, but here we append new ones) ...

def sector_filter(sectors: list) -> FilterPredicate:
    """Filter to specific sectors."""
    def predicate(df: pd.DataFrame) -> pd.Series:
        return df["sector"].isin(sectors)
    return predicate

def exclude_sectors(sectors: list) -> FilterPredicate:
    """Exclude specific sectors (e.g., financials for DCF)."""
    def predicate(df: pd.DataFrame) -> pd.Series:
        return ~df["sector"].isin(sectors)
    return predicate

def min_data_quality(min_score: float = 0.5) -> FilterPredicate:
    """Filter to snapshots with data quality score >= threshold."""
    def predicate(df: pd.DataFrame) -> pd.Series:
        if "data_quality_score" not in df.columns:
             return pd.Series(False, index=df.index)
        return df["data_quality_score"] >= min_score
    return predicate

def has_required_fields(fields: list) -> FilterPredicate:
    """Filter to snapshots that have non-null values for all required fields."""
    def predicate(df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        for field in fields:
            if field in df.columns:
                mask &= df[field].notna()
            else:
                return pd.Series(False, index=df.index)
        return mask
    return predicate

def max_nan_ratio(max_ratio: float = 0.3) -> FilterPredicate:
    """Filter to rows with no more than max_ratio of NaN values."""
    def predicate(df: pd.DataFrame) -> pd.Series:
        nan_ratio = df.isnull().sum(axis=1) / len(df.columns)
        return nan_ratio <= max_ratio
    return predicate

def positive_revenue() -> FilterPredicate:
    """Filter to companies with positive revenue."""
    def predicate(df: pd.DataFrame) -> pd.Series:
        return df["total_revenue"] > 0
    return predicate

def profitable() -> FilterPredicate:
    """Filter to profitable companies (positive net income)."""
    def predicate(df: pd.DataFrame) -> pd.Series:
        return df["net_income"] > 0
    return predicate

# --- New Time-Based Filters ---

def is_latest() -> FilterPredicate:
    """
    Filter to keep only the latest snapshot per ticker.
    Useful when dataset contains full history but we only want to train on latest.
    """
    def predicate(df: pd.DataFrame) -> pd.Series:
        if "snapshot_timestamp" not in df.columns:
             # If no timestamp, assume all are valid? Or fail? 
             # Let's return all if no timestamp col (safer for subsets)
             return pd.Series(True, index=df.index)
        
        # Group by ticker and find max timestamp
        # We need to return a text boolean series aligned with original index
        # transform('max') gives the max date for each group broadcasted to all rows
        max_dates = df.groupby("ticker")["snapshot_timestamp"].transform("max")
        return df["snapshot_timestamp"] == max_dates
    return predicate

def date_range(start_date, end_date) -> FilterPredicate:
    """Filter snapshots within a specific period_end_date range."""
    def predicate(df: pd.DataFrame) -> pd.Series:
        # Convert to datetime if needed, or assume column is datetime/date
        # handling string or object
        dates = pd.to_datetime(df["period_end_date"])
        
        # Parse inputs
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        return (dates >= start) & (dates <= end)
    return predicate

def frequency_filter(frequency: str = "quarterly") -> FilterPredicate:
    """Filter by reporting frequency (e.g. 'quarterly' or 'annual')."""
    def predicate(df: pd.DataFrame) -> pd.Series:
        if "frequency" not in df.columns:
            return pd.Series(False, index=df.index)
        return df["frequency"] == frequency
    return predicate


def latest_before(cutoff_date) -> FilterPredicate:
    """
    Filter to keep only the latest snapshot per ticker BEFORE a cutoff date.
    
    Useful for backtesting: ensures we only use data that was available
    before the cutoff, avoiding look-ahead bias.
    
    Args:
        cutoff_date: date or datetime - only snapshots before this date are considered
        
    Returns:
        FilterPredicate that keeps the latest snapshot per ticker before cutoff
    """
    def predicate(df: pd.DataFrame) -> pd.Series:
        if "snapshot_timestamp" not in df.columns:
            return pd.Series(True, index=df.index)
        
        # Convert cutoff to datetime
        cutoff = pd.to_datetime(cutoff_date)
        
        # First filter to only rows before cutoff
        before_cutoff = df["snapshot_timestamp"] < cutoff
        
        # Create a copy to avoid SettingWithCopyWarning
        df_before = df[before_cutoff].copy()
        
        if df_before.empty:
            return pd.Series(False, index=df.index)
        
        # For each ticker, find the max timestamp among those before cutoff
        max_dates = df_before.groupby("ticker")["snapshot_timestamp"].transform("max")
        
        # Now mark which rows in original df are the latest before cutoff
        result = pd.Series(False, index=df.index)
        
        for idx in df_before.index:
            if df_before.loc[idx, "snapshot_timestamp"] == max_dates.loc[idx]:
                result.loc[idx] = True
        
        return result
    return predicate

