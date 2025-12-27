"""Pydantic models for backtest module."""

from datetime import date
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class BacktestConfig(BaseModel):
    """Configuration for a backtest run."""
    
    indices: List[str] = Field(
        description="List of index IDs to test (e.g., ['SP500', 'NASDAQ100'])"
    )
    start_date: date = Field(description="First cutoff date")
    end_date: date = Field(description="Last cutoff date (must leave room for forward horizon)")
    step_days: int = Field(default=90, description="Days between cutoff dates (e.g., 90 for quarterly)")
    forward_horizons: List[int] = Field(
        default=[63, 126, 252],
        description="Forward horizons in trading days (e.g., 63=1Q, 126=2Q, 252=4Q)"
    )
    model_version: Optional[str] = Field(
        default=None,
        description="Model version to use (None = use latest)"
    )
    min_coverage: float = Field(
        default=0.5,
        description="Minimum coverage ratio required (constituents with valuations / total)"
    )
    use_latest_valuations: bool = Field(
        default=False,
        description="If True, use latest valuations ignoring cutoff_date for valuation lookup. "
                    "Only forward returns use cutoff_date. Useful when historical valuations not available."
    )

    class Config:
        """Pydantic config."""
        
        frozen = True


class BacktestPoint(BaseModel):
    """Single observation: overpriciness at cutoff vs forward return."""
    
    cutoff_date: date
    index_id: str
    
    # Signal at cutoff
    actual_index_value: float
    estimated_index_value: float
    estimated_index_std: float
    overpriciness: float = Field(description="(actual - estimated) / estimated")
    overpriciness_z_score: Optional[float] = Field(
        default=None,
        description="overpriciness / (estimated_std / estimated)"
    )
    
    # Forward outcomes (by horizon in days)
    forward_returns: Dict[int, float] = Field(
        default_factory=dict,
        description="Mapping of horizon_days -> return"
    )
    
    # Coverage metrics
    n_constituents: int
    n_constituents_with_valuation: int
    coverage_ratio: float
    
    # Model metadata
    model_version: Optional[str] = None


class CorrelationMetrics(BaseModel):
    """Detailed correlation analysis for one horizon."""
    
    horizon_days: int
    
    # Correlation coefficients
    pearson_r: float
    pearson_p_value: float
    spearman_rho: float
    spearman_p_value: float
    
    # Directional accuracy
    hit_rate: float = Field(description="% where sign(overpriciness) opposite to sign(return)")
    hit_rate_p_value: float = Field(description="vs random 50% (binomial test)")
    
    # Magnitude analysis
    quintile_mean_returns: Dict[int, float] = Field(
        default_factory=dict,
        description="{1: lowest overpriciness quintile mean return, ...}"
    )
    
    # Robustness
    n_observations: int
    bootstrap_ci_pearson: Optional[Tuple[float, float]] = Field(
        default=None,
        description="95% confidence interval for Pearson r"
    )


class BacktestResult(BaseModel):
    """Aggregated backtest results across multiple time points."""
    
    # Configuration
    start_date: date
    end_date: date
    indices_tested: List[str]
    forward_horizons: List[int]
    model_version: Optional[str] = None
    
    # Raw observations
    observations: List[BacktestPoint] = Field(default_factory=list)
    n_observations: int = 0
    
    # Correlation analysis (by horizon)
    correlations: Dict[int, CorrelationMetrics] = Field(
        default_factory=dict,
        description="Mapping of horizon_days -> CorrelationMetrics"
    )
    
    # Summary statistics
    mean_overpriciness: Optional[float] = None
    std_overpriciness: Optional[float] = None
    mean_coverage: Optional[float] = None
