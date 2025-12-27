"""Signal quality metrics for alpha signal evaluation.

Industry-standard quantitative finance metrics to evaluate predictive power
of the overpriciness signal.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SignalMetrics(BaseModel):
    """Comprehensive signal quality metrics for one sector/horizon."""
    
    # Core metrics
    ic: float = Field(description="Information Coefficient (Spearman rank correlation)")
    ic_pvalue: float = Field(description="p-value for IC")
    
    # Quantile analysis
    quantile_returns: Dict[int, float] = Field(
        default_factory=dict,
        description="Mean return per signal quintile (1=lowest, 5=highest)"
    )
    quantile_spread: float = Field(description="Q5-Q1 return spread")
    
    # Directional
    hit_rate: float = Field(description="Fraction where signal direction predicts return direction")
    
    # Sample info
    n_observations: int
    
    # Optional advanced metrics
    ic_tstat: Optional[float] = Field(default=None, description="t-stat if computed across periods")
    sharpe_ratio: Optional[float] = Field(default=None, description="Long/short portfolio Sharpe")


class ICDecay(BaseModel):
    """IC decay analysis across forward horizons."""
    
    horizons: List[int]
    ic_values: List[float]
    ic_pvalues: List[float]
    half_life_days: Optional[int] = Field(
        default=None,
        description="Estimated days until IC decays to half its peak value"
    )


# =============================================================================
# Core Metric Functions
# =============================================================================

def compute_ic(
    signal: np.ndarray,
    returns: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute Information Coefficient (rank correlation).
    
    IC is the Spearman correlation between signal ranks and return ranks.
    It's the standard measure of signal quality in quant finance.
    
    Args:
        signal: Signal values (e.g., overpriciness)
        returns: Forward returns
        
    Returns:
        Tuple of (IC, p-value)
        
    Interpretation:
        - IC > 0: Higher signal → higher returns
        - IC < 0: Higher signal → lower returns (mean reversion)
        - |IC| > 0.05: Meaningful signal
        - |IC| > 0.10: Strong signal
    """
    if len(signal) < 5:
        return (np.nan, 1.0)
    
    # Remove NaN pairs
    valid = ~(np.isnan(signal) | np.isnan(returns))
    signal = signal[valid]
    returns = returns[valid]
    
    if len(signal) < 5:
        return (np.nan, 1.0)
    
    try:
        rho, pvalue = stats.spearmanr(signal, returns)
        return (float(rho), float(pvalue))
    except Exception:
        return (np.nan, 1.0)


def compute_quantile_returns(
    signal: np.ndarray,
    returns: np.ndarray,
    n_quantiles: int = 5,
) -> Dict[int, float]:
    """
    Compute mean return for each signal quantile.
    
    Quantile 1 = lowest signal values (most underpriced)
    Quantile 5 = highest signal values (most overpriced)
    
    Args:
        signal: Signal values
        returns: Forward returns
        n_quantiles: Number of buckets (default 5 for quintiles)
        
    Returns:
        Dict mapping quantile number (1-5) to mean return
    """
    # Remove NaN pairs
    valid = ~(np.isnan(signal) | np.isnan(returns))
    signal = signal[valid]
    returns = returns[valid]
    
    if len(signal) < n_quantiles * 2:
        return {}
    
    try:
        # Create quantile labels (0 to n_quantiles-1)
        quantile_labels = pd.qcut(signal, n_quantiles, labels=False, duplicates='drop')
        
        result = {}
        for q in range(n_quantiles):
            mask = quantile_labels == q
            if mask.sum() > 0:
                result[q + 1] = float(np.mean(returns[mask]))
            else:
                result[q + 1] = np.nan
        
        return result
        
    except Exception as e:
        logger.warning(f"Quantile computation failed: {e}")
        return {}


def compute_quantile_spread(
    signal: np.ndarray,
    returns: np.ndarray,
    n_quantiles: int = 5,
) -> float:
    """
    Compute Q5-Q1 return spread.
    
    Positive spread: High signal → higher returns (momentum)
    Negative spread: High signal → lower returns (mean reversion)
    
    Returns:
        Return difference between top and bottom quantiles
    """
    quantile_returns = compute_quantile_returns(signal, returns, n_quantiles)
    
    if not quantile_returns or n_quantiles not in quantile_returns or 1 not in quantile_returns:
        return np.nan
    
    return quantile_returns[n_quantiles] - quantile_returns[1]


def compute_hit_rate(
    signal: np.ndarray,
    returns: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute directional hit rate.
    
    A "hit" occurs when:
    - Signal > 0 (overpriced) AND return < 0 (underperformed), OR
    - Signal < 0 (underpriced) AND return > 0 (outperformed)
    
    This tests the mean-reversion hypothesis.
    
    Returns:
        Tuple of (hit_rate, binomial_pvalue)
    """
    valid = ~(np.isnan(signal) | np.isnan(returns))
    signal = signal[valid]
    returns = returns[valid]
    
    n = len(signal)
    if n < 5:
        return (np.nan, 1.0)
    
    hits = np.sum(
        ((signal > 0) & (returns < 0)) |
        ((signal < 0) & (returns > 0))
    )
    
    hit_rate = hits / n
    
    # Binomial test vs 50%
    result = stats.binomtest(int(hits), n, 0.5, alternative='two-sided')
    
    return (float(hit_rate), float(result.pvalue))


def compute_ic_tstat(ic_values: List[float]) -> float:
    """
    Compute t-statistic for IC across multiple periods.
    
    Tests whether mean IC is significantly different from zero.
    
    Args:
        ic_values: List of IC values from different periods
        
    Returns:
        t-statistic (|t| > 2 is significant at 95%)
    """
    ic_values = [ic for ic in ic_values if not np.isnan(ic)]
    
    if len(ic_values) < 2:
        return np.nan
    
    mean_ic = np.mean(ic_values)
    std_ic = np.std(ic_values, ddof=1)
    
    if std_ic == 0:
        return np.nan
    
    return mean_ic / (std_ic / np.sqrt(len(ic_values)))


# =============================================================================
# IC Decay Analysis
# =============================================================================

def compute_ic_decay(
    signal: np.ndarray,
    prices: Dict[str, Dict],  # ticker -> {date -> price}
    tickers: List[str],
    base_date,
    horizons: List[int] = [1, 5, 10, 20, 40, 60],
) -> ICDecay:
    """
    Compute IC at multiple forward horizons to analyze decay.
    
    Args:
        signal: Signal values (aligned with tickers)
        prices: Price cache {ticker: {date: price}}
        tickers: List of tickers (aligned with signal)
        base_date: Start date for forward returns
        horizons: List of forward horizons in days
        
    Returns:
        ICDecay object with IC at each horizon
    """
    from datetime import timedelta
    from .price_cache import get_ticker_return
    
    ic_values = []
    ic_pvalues = []
    
    for horizon in horizons:
        returns = []
        valid_signal = []
        
        for i, ticker in enumerate(tickers):
            if ticker not in prices or not prices[ticker]:
                continue
            
            ret = get_ticker_return(prices[ticker], base_date, horizon)
            if ret is not None and not np.isnan(ret) and abs(ret) < 2.0:
                returns.append(ret)
                valid_signal.append(signal[i])
        
        if len(returns) >= 10:
            ic, pval = compute_ic(np.array(valid_signal), np.array(returns))
        else:
            ic, pval = np.nan, 1.0
        
        ic_values.append(ic)
        ic_pvalues.append(pval)
    
    # Estimate half-life (when IC drops to half of max)
    half_life = _estimate_half_life(horizons, ic_values)
    
    return ICDecay(
        horizons=horizons,
        ic_values=ic_values,
        ic_pvalues=ic_pvalues,
        half_life_days=half_life,
    )


def _estimate_half_life(horizons: List[int], ic_values: List[float]) -> Optional[int]:
    """Estimate half-life of IC decay."""
    valid_ics = [(h, ic) for h, ic in zip(horizons, ic_values) if not np.isnan(ic)]
    
    if len(valid_ics) < 2:
        return None
    
    # Find peak IC (by absolute value)
    peak_idx = max(range(len(valid_ics)), key=lambda i: abs(valid_ics[i][1]))
    peak_ic = abs(valid_ics[peak_idx][1])
    peak_horizon = valid_ics[peak_idx][0]
    
    if peak_ic < 0.01:
        return None
    
    # Find first horizon after peak where |IC| drops below half
    half_target = peak_ic / 2
    
    for h, ic in valid_ics[peak_idx:]:
        if abs(ic) < half_target:
            return h
    
    return None


# =============================================================================
# Portfolio Simulation
# =============================================================================

def compute_long_short_returns(
    signal: np.ndarray,
    returns: np.ndarray,
    long_quantile: int = 1,   # Lowest signal (most underpriced)
    short_quantile: int = 5,  # Highest signal (most overpriced)
    n_quantiles: int = 5,
) -> Dict[str, float]:
    """
    Compute returns for a long/short portfolio based on signal quintiles.
    
    Default: Long the bottom quintile (underpriced), short top quintile (overpriced).
    
    Args:
        signal: Signal values
        returns: Forward returns
        long_quantile: Quantile to go long (1-5)
        short_quantile: Quantile to go short (1-5)
        n_quantiles: Number of quantiles
        
    Returns:
        Dict with portfolio metrics
    """
    valid = ~(np.isnan(signal) | np.isnan(returns))
    signal = signal[valid]
    returns = returns[valid]
    
    if len(signal) < n_quantiles * 2:
        return {"return": np.nan, "sharpe": np.nan}
    
    try:
        quantile_labels = pd.qcut(signal, n_quantiles, labels=False, duplicates='drop') + 1
        
        # Get returns for long and short legs
        long_mask = quantile_labels == long_quantile
        short_mask = quantile_labels == short_quantile
        
        if long_mask.sum() == 0 or short_mask.sum() == 0:
            return {"return": np.nan, "sharpe": np.nan}
        
        long_return = np.mean(returns[long_mask])
        short_return = np.mean(returns[short_mask])
        
        # Long/short return (assuming equal weight, no leverage)
        portfolio_return = long_return - short_return
        
        # For Sharpe, we'd need time series data, so just return spread
        return {
            "return": float(portfolio_return),
            "long_return": float(long_return),
            "short_return": float(short_return),
            "long_count": int(long_mask.sum()),
            "short_count": int(short_mask.sum()),
        }
        
    except Exception as e:
        logger.warning(f"Long/short computation failed: {e}")
        return {"return": np.nan, "sharpe": np.nan}


# =============================================================================
# Aggregate Analysis Function
# =============================================================================

def analyze_signal(
    signal: np.ndarray,
    returns: np.ndarray,
) -> SignalMetrics:
    """
    Comprehensive signal quality analysis.
    
    Args:
        signal: Signal values (e.g., overpriciness)
        returns: Forward returns
        
    Returns:
        SignalMetrics with all computed metrics
    """
    ic, ic_pval = compute_ic(signal, returns)
    quantile_returns = compute_quantile_returns(signal, returns)
    spread = compute_quantile_spread(signal, returns)
    hit_rate, _ = compute_hit_rate(signal, returns)
    
    # Count valid observations
    valid = ~(np.isnan(signal) | np.isnan(returns))
    n_obs = int(valid.sum())
    
    return SignalMetrics(
        ic=ic,
        ic_pvalue=ic_pval,
        quantile_returns=quantile_returns,
        quantile_spread=spread,
        hit_rate=hit_rate,
        n_observations=n_obs,
    )
