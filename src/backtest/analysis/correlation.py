"""Correlation analysis for backtesting."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats

from ..models import CorrelationMetrics

logger = logging.getLogger(__name__)


def analyze_correlation(
    overpriciness: np.ndarray,
    returns: np.ndarray,
    horizon_days: int,
    n_bootstrap: int = 1000,
) -> CorrelationMetrics:
    """
    Analyze relationship between overpriciness and future returns.
    
    Expected relationship: NEGATIVE correlation (overpriced → underperforms).
    
    Args:
        overpriciness: Array of overpriciness values
        returns: Array of forward returns
        horizon_days: Forward horizon in days (for labeling)
        n_bootstrap: Number of bootstrap samples for CI
        
    Returns:
        CorrelationMetrics with all computed statistics
    """
    n = len(overpriciness)
    
    if n < 5:
        raise ValueError(f"Insufficient observations: {n} (need at least 5)")
    
    # Pearson (linear relationship)
    pearson_r, pearson_p = stats.pearsonr(overpriciness, returns)
    
    # Spearman (monotonic relationship, robust to outliers)
    spearman_rho, spearman_p = stats.spearmanr(overpriciness, returns)
    
    # Directional accuracy (hit rate)
    # Hit = overpriced (>0) and negative return, OR underpriced (<0) and positive return
    hits = np.sum(
        ((overpriciness > 0) & (returns < 0)) |
        ((overpriciness < 0) & (returns > 0))
    )
    hit_rate = float(hits / n)
    
    # Binomial test: is hit rate significantly different from 50%?
    # Use alternative='two-sided' for general test
    binom_result = stats.binomtest(int(hits), n, 0.5, alternative='two-sided')
    hit_rate_p = float(binom_result.pvalue)
    
    # Quintile analysis
    quintile_returns = _compute_quintile_returns(overpriciness, returns)
    
    # Bootstrap confidence interval for Pearson
    bootstrap_ci = bootstrap_correlation_ci(overpriciness, returns, n_bootstrap)
    
    return CorrelationMetrics(
        horizon_days=horizon_days,
        pearson_r=float(pearson_r),
        pearson_p_value=float(pearson_p),
        spearman_rho=float(spearman_rho),
        spearman_p_value=float(spearman_p),
        hit_rate=hit_rate,
        hit_rate_p_value=hit_rate_p,
        quintile_mean_returns=quintile_returns,
        n_observations=n,
        bootstrap_ci_pearson=bootstrap_ci,
    )


def _compute_quintile_returns(
    overpriciness: np.ndarray,
    returns: np.ndarray,
) -> Dict[int, float]:
    """
    Compute mean returns for each overpriciness quintile.
    
    Quintile 1 = lowest overpriciness (most underpriced)
    Quintile 5 = highest overpriciness (most overpriced)
    
    If thesis is correct, Q1 should have higher returns than Q5.
    """
    quintile_returns: Dict[int, float] = {}
    
    try:
        quintiles = np.percentile(overpriciness, [20, 40, 60, 80])
        bounds = [-np.inf] + list(quintiles) + [np.inf]
        
        for i in range(5):
            low, high = bounds[i], bounds[i + 1]
            mask = (overpriciness >= low) & (overpriciness < high)
            if mask.any():
                quintile_returns[i + 1] = float(np.mean(returns[mask]))
            else:
                quintile_returns[i + 1] = float('nan')
                
    except Exception as e:
        logger.warning(f"Error computing quintile returns: {e}")
        
    return quintile_returns


def bootstrap_correlation_ci(
    overpriciness: np.ndarray,
    returns: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for Pearson correlation.
    
    Args:
        overpriciness: Overpriciness values
        returns: Forward returns
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (default 95%)
        
    Returns:
        Tuple of (lower, upper) confidence bounds
    """
    n = len(overpriciness)
    correlations = []
    
    rng = np.random.default_rng(42)  # Reproducible
    
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            r, _ = stats.pearsonr(overpriciness[idx], returns[idx])
            if not np.isnan(r):
                correlations.append(r)
        except Exception:
            continue
    
    if len(correlations) < n_bootstrap * 0.5:
        logger.warning("Many bootstrap samples failed, CI may be unreliable")
    
    if not correlations:
        return (float('nan'), float('nan'))
    
    alpha = 1 - ci
    lower = float(np.percentile(correlations, 100 * alpha / 2))
    upper = float(np.percentile(correlations, 100 * (1 - alpha / 2)))
    
    return (lower, upper)
