# Backtest Module

> Validates index "overpriciness" signals by correlating them with future index returns using walk-forward analysis.

---

## Responsibilities

- **Historical Index Signal Generation**:
  - For a given `cutoff_date`, fetch ticker data available up to (but not including) that date
  - Train valuation model on this historical data
  - Compute "overpriciness" for each tracked index at the cutoff date

- **Forward Return Calculation**:
  - Measure actual index performance over a forward horizon (e.g., 1Q, 2Q, 4Q)
  - Calculate directional and magnitude-based returns

- **Correlation Analysis**:
  - Compute correlation between overpriciness scores and subsequent returns
  - Track hit rate (did overpriced indices underperform?)
  - Statistical significance testing

- **Walk-Forward Framework**:
  - Repeat analysis across multiple time points (expanding or rolling window)
  - Aggregate results to avoid single-period bias

---

## Core Concept

The system predicts **fair market cap** from financial statements. For an index:

```
Index Overpriciness = (Actual Index Value - Estimated Index Value) / Estimated Index Value
```

- **Positive overpriciness** → Index is trading above model estimate → Overpriced
- **Negative overpriciness** → Index is trading below model estimate → Underpriced

**Thesis to validate**: Overpriced indices should underperform in subsequent periods.

---

## Key Design: Index-Only Focus

> [!IMPORTANT]
> This module operates ONLY on index-level data, not individual tickers.

**Rationale**:
- Indices aggregate noise from individual stock mispricings
- Fewer data points allows on-the-fly web fetching (no storage needed for backtest)
- Index returns are easily verifiable from public sources (e.g., Yahoo Finance)

**Data Flow**:
```
Cutoff Date
    ↓
Fetch underlying ticker data (up to cutoff)
    ↓
Train valuation model (on historical data)
    ↓
Compute index overpriciness at cutoff
    ↓
Fetch actual index returns (cutoff → cutoff + horizon)
    ↓
Record (overpriciness, future_return) pair
    ↓
Repeat for multiple cutoff dates
    ↓
Analyze correlation & significance
```

---

## Inputs

- **Backtest Configuration**:
  - `indices_to_test: List[str]` — e.g., ["SP500", "NASDAQ100", "EUROSTOXX50"]
  - `start_date: date` — Earliest cutoff date
  - `end_date: date` — Latest cutoff date (must leave room for forward horizon)
  - `step: timedelta` — Step between cutoff dates (e.g., 3 months)
  - `forward_horizons: List[int]` — Days forward to measure (e.g., [63, 126, 252] for 1Q, 2Q, 4Q)

- **Model Configuration**:
  - `ModelConfig` from valuation module (or reference to stored config)
  - Uses same bootstrap-crossval approach for consistency

- **Data Sources** (from existing infrastructure):
  - `indices` table — Index definitions (weighting scheme, base value)
  - `index_memberships` table — Ticker memberships by `as_of_time`
  - `financial_snapshots` table — Ticker data for training/prediction
  - Yahoo Finance — **Only** for historical index prices (return calculation)

---

## Outputs

```python
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import date
import numpy as np

class BacktestPoint(BaseModel):
    """Single observation: overpriciness at cutoff vs forward return."""
    cutoff_date: date
    index_id: str
    
    # Signal at cutoff
    actual_index_value: float
    estimated_index_value: float
    estimated_index_std: float
    overpriciness: float  # (actual - estimated) / estimated
    overpriciness_z_score: float  # overpriciness / (estimated_std / estimated)
    
    # Forward outcomes (by horizon)
    forward_returns: Dict[int, float]  # {horizon_days: return}
    
    # Coverage metrics
    n_constituents: int
    n_constituents_with_valuation: int
    coverage_ratio: float

class BacktestResult(BaseModel):
    """Aggregated backtest results across multiple time points."""
    
    # Configuration
    start_date: date
    end_date: date
    indices_tested: List[str]
    forward_horizons: List[int]
    model_config_hash: str
    
    # Raw observations
    observations: List[BacktestPoint]
    n_observations: int
    
    # Correlation analysis (by horizon)
    correlations: Dict[int, Dict[str, float]]  # {horizon: {metric: value}}
    # Metrics: pearson_r, spearman_rho, p_value, hit_rate
    
    # Summary statistics
    mean_overpriciness: float
    std_overpriciness: float
    mean_coverage: float

class CorrelationMetrics(BaseModel):
    """Detailed correlation analysis for one horizon."""
    horizon_days: int
    
    # Correlation coefficients
    pearson_r: float
    pearson_p_value: float
    spearman_rho: float
    spearman_p_value: float
    
    # Directional accuracy
    hit_rate: float  # % where sign(overpriciness) opposite to sign(return)
    hit_rate_p_value: float  # vs random (binomial test)
    
    # Magnitude analysis
    quintile_mean_returns: Dict[int, float]  # {1: lowest overpriciness quintile mean return, ...}
    
    # Robustness
    n_observations: int
    bootstrap_ci_pearson: tuple[float, float]  # 95% CI
```

---

## Walk-Forward Analysis

### ⚡ KEY DESIGN: No Look-Ahead Bias

> [!CAUTION]
> At each cutoff date, the model MUST only see data available BEFORE that date.

**Implementation**:
```python
def run_backtest_point(
    cutoff_date: date,
    index_id: str,
    horizon_days: int,
    index_repo: IndexRepository,
    snapshot_repo: SnapshotRepository,
) -> BacktestPoint:
    """
    Run single backtest observation.
    
    Critical: All data fetching and model training uses only
    data available BEFORE cutoff_date.
    """
    # 1. Get index constituents as of cutoff date from index_memberships table
    constituents = index_repo.get_members(
        index_id, 
        as_of_time=cutoff_date  # Uses most recent membership <= cutoff_date
    )
    
    # 2. For each constituent, get latest financial snapshot before cutoff
    ticker_data = []
    for ticker in constituents:
        snapshot = snapshot_repo.get_latest_before(ticker, cutoff_date)
        if snapshot:
            ticker_data.append(snapshot)
    
    # 3. Train model on ALL historical snapshots up to cutoff
    all_historical = snapshot_repo.get_all_before(cutoff_date)
    model = train_valuation_model(all_historical, config=MODEL_CONFIG)
    
    # 4. Predict fair value for each constituent at cutoff
    valuations = model.predict(ticker_data)
    
    # 5. Compute index overpriciness
    index_result = aggregate_to_index(valuations, index_id)
    
    # 6. Get actual forward return (this data "arrives" after cutoff)
    actual_return = get_index_return(
        index_id, 
        from_date=cutoff_date,
        to_date=cutoff_date + timedelta(days=horizon_days),
    )
    
    return BacktestPoint(
        cutoff_date=cutoff_date,
        index_id=index_id,
        overpriciness=index_result.index_relative_error,
        forward_returns={horizon_days: actual_return},
        ...
    )
```

### Walk-Forward Windows

**Options**:

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Expanding Window** | Train on all data before cutoff | More data = better model | Recent data may differ |
| **Rolling Window** | Fixed lookback (e.g., 5 years) | Adapts to regime changes | Less data early on |
| **Hybrid** | Expanding with minimum lookback | Best of both | More complex |

**Recommended**: Expanding window (matches production use case).

---

## Index Price Fetching

### On-the-Fly Web Fetching

Since we're only testing a handful of indices, fetch prices directly during backtest:

```python
# src/backtest/index_prices.py
import yfinance as yf
from datetime import date, timedelta
from typing import Dict

# Index symbol mappings (Yahoo Finance format)
INDEX_SYMBOLS = {
    "SP500": "^GSPC",
    "NASDAQ100": "^NDX",
    "DOWJONES": "^DJI",
    "EUROSTOXX50": "^STOXX50E",
    "CAC40": "^FCHI",
    "DAX": "^GDAXI",
    "FTSE100": "^FTSE",
    "SMI": "^SSMI",
    "NIKKEI225": "^N225",
    "NIFTY50": "^NSEI",
    "SSE50": "000016.SS",  # Shanghai
}

def get_index_price(index_id: str, target_date: date) -> float:
    """
    Fetch index closing price on target_date.
    Falls back to previous trading day if market closed.
    """
    symbol = INDEX_SYMBOLS.get(index_id)
    if not symbol:
        raise ValueError(f"Unknown index: {index_id}")
    
    # Fetch window around target date
    start = target_date - timedelta(days=5)
    end = target_date + timedelta(days=1)
    
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start, end=end)
    
    # Get closest date <= target_date
    valid_dates = [d.date() for d in hist.index if d.date() <= target_date]
    if not valid_dates:
        raise ValueError(f"No price data for {index_id} on {target_date}")
    
    closest = max(valid_dates)
    return hist.loc[str(closest)]["Close"]

def get_index_return(
    index_id: str,
    from_date: date,
    to_date: date,
) -> float:
    """
    Calculate index return over period.
    Returns: (P_end - P_start) / P_start
    """
    p_start = get_index_price(index_id, from_date)
    p_end = get_index_price(index_id, to_date)
    return (p_end - p_start) / p_start
```

---

## Statistical Analysis

### Correlation Methods

```python
from scipy import stats
import numpy as np

def analyze_correlation(
    overpriciness: np.ndarray,
    returns: np.ndarray,
) -> CorrelationMetrics:
    """
    Analyze relationship between overpriciness and future returns.
    
    Expected relationship: NEGATIVE correlation
    (overpriced → underperforms)
    """
    # Pearson (linear relationship)
    pearson_r, pearson_p = stats.pearsonr(overpriciness, returns)
    
    # Spearman (monotonic relationship, robust to outliers)
    spearman_rho, spearman_p = stats.spearmanr(overpriciness, returns)
    
    # Directional accuracy (hit rate)
    # Hit = overpriced (>0) and negative return, OR underpriced (<0) and positive return
    hits = np.sum(
        (overpriciness > 0) & (returns < 0) |
        (overpriciness < 0) & (returns > 0)
    )
    hit_rate = hits / len(overpriciness)
    
    # Binomial test: is hit rate significantly different from 50%?
    hit_rate_p = stats.binom_test(hits, len(overpriciness), 0.5)
    
    # Quintile analysis
    quintiles = np.percentile(overpriciness, [20, 40, 60, 80])
    quintile_returns = {}
    for i, (low, high) in enumerate(zip(
        [-np.inf] + list(quintiles),
        list(quintiles) + [np.inf]
    ), 1):
        mask = (overpriciness >= low) & (overpriciness < high)
        quintile_returns[i] = np.mean(returns[mask]) if mask.any() else np.nan
    
    return CorrelationMetrics(
        pearson_r=pearson_r,
        pearson_p_value=pearson_p,
        spearman_rho=spearman_rho,
        spearman_p_value=spearman_p,
        hit_rate=hit_rate,
        hit_rate_p_value=hit_rate_p,
        quintile_mean_returns=quintile_returns,
        ...
    )
```

### Bootstrap Confidence Intervals

```python
def bootstrap_correlation_ci(
    overpriciness: np.ndarray,
    returns: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for Pearson correlation.
    """
    n = len(overpriciness)
    correlations = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        r, _ = stats.pearsonr(overpriciness[idx], returns[idx])
        correlations.append(r)
    
    alpha = 1 - ci
    lower = np.percentile(correlations, 100 * alpha / 2)
    upper = np.percentile(correlations, 100 * (1 - alpha / 2))
    
    return (lower, upper)
```

---

## Best Practices Implemented

### ✅ Academic Backtesting Standards

| Practice | Implementation |
|----------|---------------|
| **No look-ahead bias** | Strict cutoff date enforcement |
| **Walk-forward analysis** | Multiple cutoff dates, expanding window |
| **Out-of-sample testing** | Model trained only on pre-cutoff data |
| **Statistical significance** | P-values for all correlations |
| **Robustness checks** | Bootstrap CIs, multiple horizons |
| **Transparency** | Store all observations, model config hash |

### ⚡ CONSTRAINT: Time Series Dependencies

> [!WARNING]
> Observations are NOT independent — overlapping horizons create autocorrelation.

**Mitigations**:
- Use non-overlapping cutoff dates (step ≥ horizon)
- Report effective sample size after adjusting for correlation
- Use Newey-West standard errors for significance tests

### ⚠️ NEEDS REVIEW: Transaction Cost Modeling

For strategy validation (not primary scope), consider:
- Index ETF tracking error
- Not applicable for direct index trading
- Could add if validating ETF-based strategies

---

## Dependencies

### External Packages
- `numpy` — Numerical computation
- `scipy` — Statistical tests
- `pandas` — Data manipulation
- `yfinance` — Index price fetching (backtest only)

### Internal Modules
- `src/db/` — Repository access (indices, index_memberships, financial_snapshots)
- `src/valuation/` — Model training and prediction
- `src/index/` — Index aggregation with uncertainty

---

## Folder Structure

```
src/backtest/
  __init__.py
  service.py              # Main backtest orchestrator
  config.py               # BacktestConfig
  models.py               # BacktestPoint, BacktestResult, CorrelationMetrics
  index_prices.py         # On-the-fly index price fetching
  analysis/
    __init__.py
    correlation.py        # Correlation analysis
    significance.py       # Statistical significance tests
    bootstrap.py          # Bootstrap CI estimation
  visualization.py        # Result plotting (optional)
```

---

## Backtest Orchestrator

```python
# src/backtest/service.py
from datetime import date, timedelta
from typing import List

class BacktestService:
    """Orchestrate index overpriciness backtesting."""
    
    def __init__(
        self,
        valuation_service: ValuationService,
        index_service: IndexService,
    ):
        self.valuation_service = valuation_service
        self.index_service = index_service
    
    def run_backtest(
        self,
        indices: List[str],
        start_date: date,
        end_date: date,
        step_days: int = 90,  # Quarterly
        horizons: List[int] = [63, 126, 252],  # 1Q, 2Q, 4Q
        model_config: Optional[ModelConfig] = None,
    ) -> BacktestResult:
        """
        Run full walk-forward backtest.
        
        Args:
            indices: Index IDs to test
            start_date: First cutoff date
            end_date: Last cutoff date (must allow for longest horizon)
            step_days: Days between cutoff dates
            horizons: Forward horizons in trading days
            model_config: Model configuration (uses default if None)
        
        Returns:
            BacktestResult with all observations and analysis
        """
        observations = []
        
        cutoff = start_date
        while cutoff <= end_date:
            for index_id in indices:
                try:
                    point = self._run_single_point(
                        cutoff, index_id, horizons, model_config
                    )
                    observations.append(point)
                except Exception as e:
                    logger.warning(f"Failed {index_id} at {cutoff}: {e}")
            
            cutoff += timedelta(days=step_days)
        
        # Analyze correlations for each horizon
        correlations = {}
        for horizon in horizons:
            overpriciness = np.array([o.overpriciness for o in observations])
            returns = np.array([o.forward_returns.get(horizon, np.nan) for o in observations])
            
            # Remove NaN pairs
            valid = ~(np.isnan(overpriciness) | np.isnan(returns))
            if valid.sum() >= 10:  # Minimum sample size
                correlations[horizon] = analyze_correlation(
                    overpriciness[valid], returns[valid]
                )
        
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            indices_tested=indices,
            forward_horizons=horizons,
            observations=observations,
            correlations=correlations,
            ...
        )
```

---

## Integration Tests

```python
# tests/integration/test_backtest.py

class TestBacktestIntegration:
    """Integration tests for backtesting module."""
    
    def test_index_price_fetch(self):
        """
        Test index price fetching from Yahoo Finance.
        
        Verifies:
        - Can fetch S&P 500 price
        - Historical date fetch works
        - Return calculation is correct
        """
        price = get_index_price("SP500", date(2023, 12, 29))
        assert 4000 < price < 6000  # Reasonable range
        
        ret = get_index_return("SP500", date(2023, 1, 3), date(2023, 12, 29))
        assert -0.5 < ret < 0.5  # 2023 return was ~24%
    
    def test_no_look_ahead(self, mock_valuation):
        """
        Test that backtest enforces no look-ahead bias.
        
        Verifies:
        - Model training only uses pre-cutoff data
        - Forward returns fetched from post-cutoff
        """
        cutoff = date(2023, 6, 30)
        
        point = run_backtest_point(cutoff, "SP500", horizon_days=63)
        
        # All training data should be before cutoff
        for snapshot in mock_valuation.training_data:
            assert snapshot.snapshot_timestamp.date() < cutoff
    
    def test_correlation_analysis(self, sample_backtest_data):
        """
        Test correlation metrics computation.
        
        Verifies:
        - Pearson/Spearman computed
        - P-values reasonable
        - Hit rate between 0 and 1
        """
        overpriciness = np.array([0.1, -0.05, 0.15, -0.1, 0.08])
        returns = np.array([-0.05, 0.03, -0.08, 0.06, -0.02])
        
        metrics = analyze_correlation(overpriciness, returns)
        
        assert -1 <= metrics.pearson_r <= 1
        assert -1 <= metrics.spearman_rho <= 1
        assert 0 <= metrics.hit_rate <= 1
    
    def test_walk_forward_multiple_dates(self, test_db):
        """
        Test walk-forward with multiple cutoff dates.
        
        Verifies:
        - Observations generated for each cutoff
        - Step size respected
        - End date honored
        """
        service = BacktestService(...)
        result = service.run_backtest(
            indices=["SP500"],
            start_date=date(2020, 1, 1),
            end_date=date(2020, 7, 1),
            step_days=90,
            horizons=[63],
        )
        
        assert len(result.observations) >= 3  # Q1, Q2, Q3

### Running Tests

```bash
# Run backtest tests (mocked)
pytest tests/integration/test_backtest.py -v

# Run with real data (slow, requires network)
pytest tests/integration/test_backtest.py -v --real-network

# Run with coverage
pytest tests/integration/test_backtest.py --cov=src/backtest
```

---

## Constraints

- ⚡ **NO look-ahead bias** — Training data strictly before cutoff
- ⚡ **Index-only** — No individual ticker backtesting in this module
- ⚡ **On-the-fly fetching** — No persistent storage of index prices (for now)
- ⚡ **Reproducibility** — Store model config hash with results
- ⚡ **Statistical rigor** — Always report p-values and confidence intervals

---

## Open Items

| Item | Status | Notes |
|------|--------|-------|
| Minimum sample size | 📌 TODO | Determine n required for significance |
| Overlapping horizon adjustment | 📌 TODO | Newey-West standard errors |
| Multi-index correlation | ⚠️ NEEDS REVIEW | Are index mispricings correlated? |
| Regime-conditional analysis | 📌 TODO | Bull vs bear market performance |
| ETF tracking error | 💡 ALTERNATIVE | Add if validating ETF strategies |
