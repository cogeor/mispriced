"""
Currency normalization for financial snapshots.

Fixes known issues with the ingested data:
1. book_value is per-share (from YFinance bookValue), not total stockholders' equity
2. ADRs may have financial statements in local currency but stored as USD
3. Some fields (working_capital, eps) were not converted during ingestion

This module provides preprocessing to normalize data before model training.
"""

import logging
from typing import Tuple, Optional, List, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Known ADRs that report in local currency but trade in USD
# These have financialCurrency != currency but YFinance may not detect it
KNOWN_ADR_CURRENCIES: Dict[str, str] = {
    # Argentina (ARS)
    "GGAL": "ARS",
    "YPF": "ARS",
    "PAM": "ARS",
    "TGS": "ARS",
    "BMA": "ARS",
    "EDN": "ARS",
    "IRS": "ARS",
    "CRESY": "ARS",
    "SUPV": "ARS",
    "LOMA": "ARS",
    # Brazil (BRL)
    "PBR": "BRL",
    "VALE": "BRL",
    "ITUB": "BRL",
    "BBD": "BRL",
    "ABEV": "BRL",
    "SBS": "BRL",
    # Chile (CLP)
    "CCU": "CLP",
    "SQM": "CLP",
    # Mexico (MXN)
    "AMX": "MXN",
    "CEMEX": "MXN",
    "FMX": "MXN",
}

# Approximate FX rates to USD (for normalization when historical rate unavailable)
# These are rough current rates - better than nothing for ADRs
FX_RATES_TO_USD: Dict[str, float] = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.27,
    "GBp": 0.0127,  # Pence
    "GBX": 0.0127,
    "JPY": 0.0067,
    "CNY": 0.14,
    "HKD": 0.128,
    "INR": 0.012,
    "CHF": 1.13,
    "CAD": 0.74,
    "AUD": 0.65,
    "KRW": 0.00072,
    "TWD": 0.032,
    "BRL": 0.20,
    "ARS": 0.001,  # ~1000 ARS per USD as of late 2024
    "CLP": 0.00105,
    "MXN": 0.058,
    "ZAR": 0.053,
}

# Monetary fields that should be in USD
MONETARY_FIELDS = [
    "total_revenue",
    "gross_profit",
    "ebitda",
    "operating_income",
    "net_income",
    "total_debt",
    "total_cash",
    "total_assets",
    "free_cash_flow",
    "operating_cash_flow",
    "capex",
    "book_value",
    "working_capital",
    "eps",
]


def fix_book_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix book_value from per-share to total equity value.

    YFinance's bookValue is per-share. We need total book value for comparability.
    Computes: total_book_value = book_value * shares_outstanding

    Args:
        df: DataFrame with book_value and shares_outstanding columns

    Returns:
        DataFrame with corrected book_value (total, not per-share)
    """
    df = df.copy()

    if "book_value" not in df.columns or "shares_outstanding" not in df.columns:
        logger.warning("Missing book_value or shares_outstanding column, skipping fix")
        return df

    # Convert to numeric
    bv = pd.to_numeric(df["book_value"], errors="coerce")
    shares = pd.to_numeric(df["shares_outstanding"], errors="coerce")

    # Compute total book value
    # Only fix where book_value looks like per-share (small absolute value)
    # Heuristic: if book_value < 1000, it's likely per-share
    is_per_share = (bv.abs() < 10000) & (shares > 0) & bv.notna() & shares.notna()

    total_bv = bv.copy()
    total_bv[is_per_share] = bv[is_per_share] * shares[is_per_share]

    fixed_count = is_per_share.sum()
    if fixed_count > 0:
        logger.info(f"  Fixed {fixed_count} book_value entries (per-share -> total)")

    df["book_value"] = total_bv

    return df


def detect_currency_anomalies(df: pd.DataFrame) -> pd.Series:
    """
    Detect rows that likely have currency issues.

    Uses ratio-based heuristics to find data where monetary values
    appear to be in different currencies (e.g., ARS stored as USD).

    Red flags (only when both values are present and valid):
    - revenue / assets > 50 (extremely suspicious)
    - debt / assets > 20 (extremely suspicious)
    - book_value / market_cap < 1e-9 AND mcap > 10B (indicates per-share issue not fixed)

    Args:
        df: DataFrame with financial data

    Returns:
        Boolean series: True = likely currency issue
    """
    anomaly = pd.Series(False, index=df.index)

    # Convert columns to numeric
    revenue = pd.to_numeric(df.get("total_revenue"), errors="coerce")
    assets = pd.to_numeric(df.get("total_assets"), errors="coerce")
    debt = pd.to_numeric(df.get("total_debt"), errors="coerce")
    mcap = pd.to_numeric(df.get("market_cap_t0"), errors="coerce")
    bv = pd.to_numeric(df.get("book_value"), errors="coerce")

    # Only flag when we have both values and ratio is extreme
    # Check revenue/assets ratio (should be < 10 for virtually all companies)
    has_both_rev = revenue.notna() & assets.notna() & (assets > 0)
    rev_ratio = revenue / assets.replace(0, np.nan)
    anomaly |= has_both_rev & (rev_ratio > 50)

    # Check debt/assets ratio (should be < 5 normally)
    has_both_debt = debt.notna() & assets.notna() & (assets > 0)
    debt_ratio = debt / assets.replace(0, np.nan)
    anomaly |= has_both_debt & (debt_ratio > 20)

    # Check book_value / market_cap for large caps (should be > 1e-6 after fix)
    # This catches cases where book_value wasn't properly fixed
    has_both_bv = bv.notna() & mcap.notna() & (mcap > 10e9)  # Only for >$10B companies
    bv_ratio = bv / mcap.replace(0, np.nan)
    anomaly |= has_both_bv & (bv_ratio.abs() < 1e-9)

    return anomaly


def detect_known_adrs(df: pd.DataFrame) -> pd.Series:
    """
    Identify known ADRs that need currency correction.

    Args:
        df: DataFrame with ticker column

    Returns:
        Boolean series: True = known ADR with local currency financials
    """
    if "ticker" not in df.columns:
        return pd.Series(False, index=df.index)

    return df["ticker"].isin(KNOWN_ADR_CURRENCIES.keys())


def get_adr_fx_rate(ticker: str) -> Optional[float]:
    """Get the FX rate for a known ADR's financial currency."""
    currency = KNOWN_ADR_CURRENCIES.get(ticker)
    if currency:
        return FX_RATES_TO_USD.get(currency)
    return None


def normalize_adr_financials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply FX correction to known ADRs.

    For ADRs where we know the financials are in local currency,
    apply the appropriate FX rate to convert to USD.

    The key insight: ADRs trade in USD but often report financials in local currency.
    YFinance returns currency="USD" for the trading currency, but the financial
    statements are still in ARS/BRL/etc. So we ALWAYS apply FX correction for
    known ADRs, regardless of what fx_rate_to_usd says.

    Args:
        df: DataFrame with financial data

    Returns:
        DataFrame with corrected monetary values for ADRs
    """
    df = df.copy()

    is_adr = detect_known_adrs(df)
    adr_count = is_adr.sum()

    if adr_count == 0:
        return df

    logger.info(f"  Found {adr_count} known ADRs needing currency correction")

    # Apply FX correction per ADR
    corrected = 0
    for idx in df[is_adr].index:
        ticker = df.loc[idx, "ticker"]
        fx_rate = get_adr_fx_rate(ticker)

        if fx_rate is None:
            continue

        # Check if this ticker's stored fx_rate already matches our expected rate
        # (meaning it was already properly converted)
        stored_fx = df.loc[idx, "fx_rate_to_usd"] if "fx_rate_to_usd" in df.columns else None
        if stored_fx and stored_fx is not None and not pd.isna(stored_fx):
            if abs(stored_fx - fx_rate) < 0.0001:
                # Already converted with correct rate
                continue

        # Apply FX rate to monetary fields
        # For ADRs with fx_rate ~1.0, this means the data was stored in local currency
        # mistakenly labeled as USD. We need to convert to actual USD.
        fields_converted = 0
        for field in MONETARY_FIELDS:
            if field in df.columns:
                val = df.loc[idx, field]
                if pd.notna(val) and val != 0:
                    df.loc[idx, field] = float(val) * fx_rate
                    fields_converted += 1

        if fields_converted > 0:
            corrected += 1
            logger.debug(f"    Applied FX rate {fx_rate:.6f} to {ticker} ({fields_converted} fields)")

    if corrected > 0:
        logger.info(f"  Applied FX correction to {corrected} ADRs")

    return df


def normalize_snapshots(
    df: pd.DataFrame,
    fix_bv: bool = True,
    fix_adrs: bool = True,
    exclude_anomalies: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full normalization pipeline for financial snapshots.

    Applies all currency fixes and optionally excludes anomalous rows.

    Args:
        df: Raw DataFrame from database
        fix_bv: If True, fix book_value from per-share to total
        fix_adrs: If True, apply FX correction to known ADRs
        exclude_anomalies: If True, return anomalous rows separately

    Returns:
        Tuple of (cleaned_df, anomalies_df)
        anomalies_df contains rows with detected currency issues
    """
    logger.info("Normalizing snapshots for currency issues...")
    original_count = len(df)

    df = df.copy()

    # Step 1: Fix book_value (per-share -> total)
    if fix_bv:
        df = fix_book_value(df)

    # Step 2: Apply FX correction to known ADRs
    if fix_adrs:
        df = normalize_adr_financials(df)

    # Step 3: Detect remaining anomalies
    anomaly_mask = detect_currency_anomalies(df)
    anomaly_count = anomaly_mask.sum()

    if exclude_anomalies and anomaly_count > 0:
        anomalies = df[anomaly_mask].copy()
        df = df[~anomaly_mask].copy()
        logger.info(f"  Excluded {anomaly_count} rows with currency anomalies")
        logger.info(f"  Remaining: {len(df)} / {original_count} snapshots")
    else:
        anomalies = pd.DataFrame()
        if anomaly_count > 0:
            logger.warning(f"  {anomaly_count} rows have currency anomalies (not excluded)")

    return df, anomalies


def create_currency_neutral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ratio-based features that are currency-neutral.

    Since market_cap is correctly converted to USD, we can create
    ratios like revenue/mcap which are currency-neutral (both in local currency,
    so ratio is unaffected).

    This is useful when raw values may have currency issues but ratios are valid.

    Args:
        df: DataFrame with financial data

    Returns:
        DataFrame with additional ratio features
    """
    df = df.copy()

    mcap = pd.to_numeric(df.get("market_cap_t0"), errors="coerce")
    mcap_safe = mcap.replace(0, np.nan)

    # Revenue multiples (P/S reciprocals)
    if "total_revenue" in df.columns:
        rev = pd.to_numeric(df["total_revenue"], errors="coerce")
        df["revenue_to_mcap"] = rev / mcap_safe

    # EBITDA multiple
    if "ebitda" in df.columns:
        ebitda = pd.to_numeric(df["ebitda"], errors="coerce")
        df["ebitda_to_mcap"] = ebitda / mcap_safe

    # Book-to-market
    if "book_value" in df.columns:
        bv = pd.to_numeric(df["book_value"], errors="coerce")
        df["book_to_mcap"] = bv / mcap_safe

    # Debt-to-market
    if "total_debt" in df.columns:
        debt = pd.to_numeric(df["total_debt"], errors="coerce")
        df["debt_to_mcap"] = debt / mcap_safe

    # Cash-to-market
    if "total_cash" in df.columns:
        cash = pd.to_numeric(df["total_cash"], errors="coerce")
        df["cash_to_mcap"] = cash / mcap_safe

    # FCF yield
    if "free_cash_flow" in df.columns:
        fcf = pd.to_numeric(df["free_cash_flow"], errors="coerce")
        df["fcf_to_mcap"] = fcf / mcap_safe

    return df
