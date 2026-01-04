"""
Size Premium Correction Module.

Estimates and removes the structural size premium from mispricing signals.
Raw predictions are NOT modified - this module computes corrections that
can be applied on demand.

The size premium is estimated cross-sectionally using LOWESS (locally weighted
scatterplot smoothing) which captures non-linear size effects.
"""

import numpy as np
from typing import Tuple, Optional, NamedTuple, List
from dataclasses import dataclass
from scipy import stats as scipy_stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class SizeCoefficient:
    """
    Linear regression coefficient for size premium at one timepoint.
    
    Model: mispricing = intercept + slope * log(market_cap) + error
    """
    quarter: str
    slope: float  # Coefficient on log(market_cap)
    slope_se: float  # Standard error of slope
    slope_tstat: float  # t-statistic
    slope_pval: float  # p-value (two-tailed)
    intercept: float
    r_squared: float
    n_obs: int
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "quarter": self.quarter,
            "slope": self.slope,
            "slopeSE": self.slope_se,
            "slopeTstat": self.slope_tstat,
            "slopePval": self.slope_pval,
            "intercept": self.intercept,
            "rSquared": self.r_squared,
            "nObs": self.n_obs,
        }


def estimate_size_coefficient(
    mispricing: np.ndarray,
    market_cap: np.ndarray,
    quarter: str = "",
) -> Optional[SizeCoefficient]:
    """
    Estimate size premium coefficient using OLS with standard errors.
    
    Model: mispricing = α + β * log(market_cap) + ε
    
    Args:
        mispricing: Raw mispricing values
        market_cap: Market cap values (will be log-transformed)
        quarter: Label for this timepoint
        
    Returns:
        SizeCoefficient with slope, SE, t-stat, p-value, or None if insufficient data
    """
    # Filter valid data
    valid_mask = (
        np.isfinite(mispricing) & 
        np.isfinite(market_cap) & 
        (market_cap > 0) &
        (mispricing > -0.99)
    )
    y = np.log1p(mispricing[valid_mask])
    x = np.log(market_cap[valid_mask])
    
    if len(y) < 10:
        return None
    
    # OLS regression
    n = len(y)
    x_mean = x.mean()
    y_mean = y.mean()
    
    # Slope and intercept
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    
    if ss_xx < 1e-10:
        return None
    
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    
    # Predictions and residuals
    y_pred = intercept + slope * x
    residuals = y - y_pred
    
    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Standard error of slope
    mse = ss_res / (n - 2) if n > 2 else 0
    se_slope = np.sqrt(mse / ss_xx) if ss_xx > 0 else 0
    
    # t-statistic and p-value
    if se_slope > 0:
        t_stat = slope / se_slope
        p_val = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=n-2))
    else:
        t_stat = 0
        p_val = 1.0
    
    return SizeCoefficient(
        quarter=quarter,
        slope=slope,
        slope_se=se_slope,
        slope_tstat=t_stat,
        slope_pval=p_val,
        intercept=intercept,
        r_squared=r_squared,
        n_obs=n,
    )


@dataclass
class SizePremiumEstimate:
    """
    Holds the estimated size premium for a given dataset.
    
    Coefficients are kept in memory (not persisted).
    """
    # Fitted LOWESS points (x=log_mcap, y=expected_mispricing)
    log_mcap_grid: np.ndarray
    expected_mispricing: np.ndarray
    
    # Summary statistics
    n_observations: int
    estimation_method: str = "lowess"
    smoothing_frac: float = 0.3
    
    # Fitted coefficients (for polynomial models)
    # [a, b, c] for ax^2 + bx + c
    coefficients: Optional[List[float]] = None
    
    def get_expected_mispricing(self, log_mcap: np.ndarray) -> np.ndarray:
        """
        Interpolate expected mispricing for given log market caps.
        
        Args:
            log_mcap: Log market cap values to get expectations for
            
        Returns:
            Expected mispricing based on size (the "size premium")
        """
        # If coefficients available, use them directly for smoother interpolation
        if self.coefficients and self.estimation_method == "quadratic":
            # Note: Model fitted on log(1+mispricing), so result is in log-space
            # But this function is expected to return the correction in the SAME space as input fit
            # If the user of this class expects percentage, we need to be careful.
            # However, for consistency, this returns the fitted value Y_hat.
            # If fitted on log(1+MP), this returns log(1+MP_expected).
            return np.polyval(self.coefficients, log_mcap)
            
        return np.interp(log_mcap, self.log_mcap_grid, self.expected_mispricing)


def estimate_size_premium(
    mispricing: np.ndarray,
    market_cap: np.ndarray,
    frac: float = 0.3,  # Unused, kept for API compatibility
    winsorize_pct: Tuple[float, float] = (2.0, 98.0),
) -> SizePremiumEstimate:
    """
    Estimate the size premium using quadratic polynomial fit.

    Model: log(1 + mispricing) = α + β*log(mcap) + γ*log(mcap)²

    Uses log-transformed mispricing (approx log(Predicted/Actual)) for better
    statistical properties and stability.
    
    NO clipping/winsorization is applied.

    Args:
        mispricing: Raw mispricing values (relative_error)
        market_cap: Market cap values
        frac: Unused
        winsorize_pct: Unused (winsorization removed)

    Returns:
        SizePremiumEstimate with fitted quadratic curve
    """
    # Filter valid data
    valid_mask = (
        np.isfinite(mispricing) &
        np.isfinite(market_cap) &
        (market_cap > 0) &
        (mispricing > -0.99) # Ensure log(1+mp) is valid
    )
    mispricing_valid = mispricing[valid_mask]
    mcap_valid = market_cap[valid_mask]

    if len(mispricing_valid) < 20:
        logger.warning(f"Only {len(mispricing_valid)} valid observations for size premium estimation")
        # Return zero correction
        grid = np.linspace(15, 30, 100)
        return SizePremiumEstimate(
            log_mcap_grid=grid,
            expected_mispricing=np.zeros_like(grid),
            n_observations=len(mispricing_valid),
            estimation_method="constant_zero",
            smoothing_frac=0.0,
            coefficients=[0.0, 0.0, 0.0]
        )

    # Use log(1 + mispricing) as target variable
    # mispricing = (Pred - Actual) / Actual = Pred/Actual - 1
    # 1 + mispricing = Pred/Actual
    # log(1 + mispricing) = log(Pred) - log(Actual)
    y_log = np.log1p(mispricing_valid)
    
    # Log transform market cap
    log_mcap = np.log(mcap_valid)

    # Fit quadratic polynomial: log(1+mp) = α + β*log(mcap) + γ*log(mcap)²
    coeffs = np.polyfit(log_mcap, y_log, 2)

    # Create grid for expected values
    grid_x = np.linspace(log_mcap.min(), log_mcap.max(), 100)
    grid_y = np.polyval(coeffs, grid_x)

    logger.info(f"Size premium (quadratic): a={coeffs[0]:.6f}, b={coeffs[1]:.4f}, c={coeffs[2]:.4f}")

    return SizePremiumEstimate(
        log_mcap_grid=grid_x,
        expected_mispricing=grid_y,
        n_observations=len(mispricing_valid),
        estimation_method="quadratic",
        smoothing_frac=0.0,
        coefficients=coeffs.tolist(),
    )


def _estimate_size_premium_linear(
    mispricing: np.ndarray,
    market_cap: np.ndarray,
) -> SizePremiumEstimate:
    """
    Fallback: estimate size premium using simple linear regression.
    """
    valid_mask = (
        np.isfinite(mispricing) & 
        np.isfinite(market_cap) & 
        (market_cap > 0)
    )
    mispricing_valid = mispricing[valid_mask]
    mcap_valid = market_cap[valid_mask]
    
    log_mcap = np.log(mcap_valid)
    
    # Simple linear fit: E[mispricing | log_mcap] = a + b * log_mcap
    if len(log_mcap) < 2:
        # No data - return zeros
        grid = np.linspace(15, 30, 100)  # Typical log mcap range
        return SizePremiumEstimate(
            log_mcap_grid=grid,
            expected_mispricing=np.zeros_like(grid),
            n_observations=0,
            estimation_method="constant_zero",
            smoothing_frac=0.0,
        )
    
    coeffs = np.polyfit(log_mcap, mispricing_valid, 1)
    grid = np.linspace(log_mcap.min(), log_mcap.max(), 100)
    expected = np.polyval(coeffs, grid)
    
    return SizePremiumEstimate(
        log_mcap_grid=grid,
        expected_mispricing=expected,
        n_observations=len(mispricing_valid),
        estimation_method="linear",
        smoothing_frac=0.0,
    )


def compute_residual_mispricing(
    mispricing: np.ndarray,
    market_cap: np.ndarray,
    size_premium: Optional[SizePremiumEstimate] = None,
) -> Tuple[np.ndarray, SizePremiumEstimate]:
    """
    Compute size-neutral (residual) mispricing.

    Uses log-space decomposition:
    log(1 + residual) = log(1 + mispricing) - E[log(1 + mispricing) | size]
    
    This ensures proper multiplicative scaling:
    (1 + mispricing) = (1 + residual) * (1 + size_premium)

    Args:
        mispricing: Raw mispricing values
        market_cap: Market cap values
        size_premium: Pre-computed size premium. If None, will be estimated.

    Returns:
        Tuple of (residual_mispricing, size_premium_estimate)
    """
    if size_premium is None:
        size_premium = estimate_size_premium(mispricing, market_cap)

    # Get expected mispricing (in log space) for each observation
    valid_mask = (market_cap > 0) & np.isfinite(market_cap)
    log_mcap = np.zeros_like(market_cap)
    log_mcap[valid_mask] = np.log(market_cap[valid_mask])

    # Expected value from model (fitted on log(1+m))
    expected_log = size_premium.get_expected_mispricing(log_mcap)

    # Actual value in log space
    # Handle potentially small negative values safely (though inputs should be > -1)
    # We clip slightly above -1 for stability if needed, but here we trust inputs or return NaN
    actual_log = np.log1p(mispricing)
    
    # Residual in log space
    residual_log = actual_log - expected_log

    # Convert back to regular percentage space
    # residual_pct = exp(residual_log) - 1
    residual = np.expm1(residual_log)

    # NO clipping applied (user request)
    return residual, size_premium


class SizeCorrectionResult(NamedTuple):
    """Complete result of size correction analysis."""
    raw_mispricing: np.ndarray
    residual_mispricing: np.ndarray
    size_premium_estimate: SizePremiumEstimate
    
    @property
    def size_premium_values(self) -> np.ndarray:
        """
        The size premium component for each observation.
        
        Using multiplicative decomposition:
        (1 + mispricing) = (1 + residual) * (1 + size_premium)
        => (1 + size_premium) = (1 + mispricing) / (1 + residual)
        => size_premium = ((1 + mispricing) / (1 + residual)) - 1
        """
        # Note: simplistic approximation for now, or could compute exact
        return (1 + self.raw_mispricing) / (1 + self.residual_mispricing) - 1


def apply_size_correction(
    mispricing: np.ndarray,
    market_cap: np.ndarray,
    frac: float = 0.3,
) -> SizeCorrectionResult:
    """
    Full size correction pipeline.
    
    Args:
        mispricing: Raw mispricing values
        market_cap: Market cap values
        frac: LOWESS smoothing fraction
        
    Returns:
        SizeCorrectionResult with all components
    """
    size_premium = estimate_size_premium(mispricing, market_cap, frac=frac)
    residual, _ = compute_residual_mispricing(mispricing, market_cap, size_premium)
    
    return SizeCorrectionResult(
        raw_mispricing=mispricing,
        residual_mispricing=residual,
        size_premium_estimate=size_premium,
    )
