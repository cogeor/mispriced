# Strategy (Portfolio Construction)

> Converts snapshot-level mispricing results into portfolio weights.

---

## Responsibilities

- Convert snapshot-level mispricing results into portfolio weights
- Apply portfolio construction rules based on:
  - Mispricing magnitude
  - Confidence score
  - Stability across multiple statements
  - Sector neutrality constraints
- Generate rebalancing signals on statement cadence

---

## Inputs

- **Universe Selection**:
  - Latest snapshot per ticker (or a given date cut)
- **Per Ticker**:
  - `mispricing_pct`
  - `intrinsic_value`
  - `price_t0`
  - `confidence_score`
- **Strategy Config**:
  - Long-only vs long/short
  - Max weight per name
  - Min confidence threshold
  - Sector exposure caps
  - Rebalance frequency by snapshot cadence (quarterly/annual)

---

## Outputs

- `PortfolioTarget`:
  - List of `(ticker, target_weight)`
  - Rationale fields: mispricing, confidence, data quality
- **DB Persistence**:
  - `portfolio_runs(run_id, as_of_timestamp, config)`
  - `portfolio_weights(run_id, ticker, weight, signals...)`

---

## Dependencies

### External Packages
- `numpy` - Array operations
- `cvxpy` (optional) - Optimization for v2
- `pandas` - DataFrame operations

### Internal Modules
- `src/db/` - Repository access
- `src/valuation/` - Valuation results

---

## Folder Structure

```
src/strategy/
  __init__.py
  service.py              # Main strategy orchestrator
  universe.py             # Universe selection logic
  weighting/
    __init__.py
    base.py               # Abstract weighting scheme
    rank_based.py         # Simple ranking + caps
    optimization.py       # cvxpy-based optimization
  constraints.py          # Portfolio constraints
  config.py               # Strategy configuration
  result.py               # PortfolioTarget dataclass
```

---

## Weighting Schemes

### ⚠️ NEEDS REVIEW: Which Weighting Strategy for v1

| Scheme | Complexity | Best For |
|--------|------------|----------|
| **Equal Weight** | Low | Baseline comparison |
| **Mispricing Ranked** | Low | Simple value tilt |
| **Confidence Weighted** | Medium | Quality-aware |
| **Mean-Variance Optimized** | High | Risk-adjusted |

### Rank-Based Weighting (Proposed v1)

```python
def rank_weight(snapshots: List[SnapshotWithValuation], config: StrategyConfig) -> PortfolioTarget:
    """
    Simple ranking strategy:
    1. Filter by min confidence
    2. Rank by mispricing (most undervalued first)
    3. Select top N
    4. Equal weight with caps
    """
    # Filter
    eligible = [s for s in snapshots if s.confidence_score >= config.min_confidence]
    
    # Rank (negative mispricing = undervalued = buy signal)
    ranked = sorted(eligible, key=lambda s: s.mispricing_pct)
    
    # Select top N
    selected = ranked[:config.max_positions]
    
    # Equal weight with max cap
    raw_weight = 1.0 / len(selected)
    capped_weight = min(raw_weight, config.max_weight_per_name)
    
    return PortfolioTarget(
        weights=[(s.ticker, capped_weight) for s in selected],
        as_of=max(s.snapshot_timestamp for s in selected),
        config_hash=config.hash(),
    )
```

### Long/Short Strategy

```python
def long_short_weight(snapshots, config):
    """
    Long most undervalued, short most overvalued.
    """
    ranked = sorted(snapshots, key=lambda s: s.mispricing_pct)
    
    long_candidates = ranked[:config.long_count]   # Most undervalued
    short_candidates = ranked[-config.short_count:]  # Most overvalued
    
    long_weight = 0.5 / len(long_candidates)
    short_weight = -0.5 / len(short_candidates)
    
    weights = (
        [(s.ticker, long_weight) for s in long_candidates] +
        [(s.ticker, short_weight) for s in short_candidates]
    )
    
    return PortfolioTarget(weights=weights, ...)
```

---

## Constraint Enforcement

### ⚠️ NEEDS REVIEW: Sector Neutrality

**Options**:
1. **Sector caps** - Max 20% per sector
2. **Sector neutral** - Net zero exposure per sector
3. **Market weight relative** - Max ±5% from benchmark weight

### Position Constraints

```python
@dataclass
class StrategyConstraints:
    max_positions: int = 30
    max_weight_per_name: float = 0.10  # 10%
    max_sector_weight: float = 0.20    # 20%
    min_confidence: float = 0.5
    long_only: bool = True
```

---

## Rebalancing Logic

### ⚡ KEY DESIGN: Statement Cadence Rebalancing

> Rebalance happens on **new statement snapshots**, NOT daily.

**Trigger conditions**:
1. New quarterly/annual statements available for universe
2. Manual rebalance request
3. Configurable minimum elapsed time since last rebalance

### Turnover Tracking

```python
def compute_turnover(old_weights: Dict, new_weights: Dict) -> float:
    """
    Sum of absolute weight changes / 2.
    """
    all_tickers = set(old_weights.keys()) | set(new_weights.keys())
    total_change = sum(
        abs(new_weights.get(t, 0) - old_weights.get(t, 0))
        for t in all_tickers
    )
    return total_change / 2
```

---

## Design Decisions

### ⚠️ NEEDS REVIEW: Optimization Framework

**Options for v2**:
1. **cvxpy** - General convex optimization (Python native)
2. **scipy.optimize** - Simpler but less flexible
3. **Custom solver** - For specific constraints

### ⚠️ NEEDS REVIEW: Benchmark Integration

Should strategy track against a benchmark?

**Options**:
1. **No benchmark** - Absolute return focus
2. **S&P 500** - relative comparison
3. **Custom universe** - Equal weight of full universe

### 📌 TODO: Transaction Cost Model

For realistic evaluation, need to model:
- Spread costs
- Market impact (for position sizing)
- Commission costs

---

## Constraints

- ⚡ Rebalance on **statement cadence**, not daily
- ⚡ Must persist full portfolio state for audit trail
- ⚡ Must respect all position/sector constraints
- ⚡ Must handle tickers dropping out of universe (rebalance to zero)

---

## Integration Tests

### Test Scope

Integration tests verify portfolio construction logic with real valuation results and database.

### Test Cases

```python
# tests/integration/test_strategy.py

class TestStrategyIntegration:
    """Integration tests for portfolio strategy."""
    
    def test_rank_based_weighting(self, test_db, sample_valuations):
        """
        Test rank-based portfolio construction.
        
        Verifies:
        - Tickers sorted by mispricing
        - Top N selected
        - Weights sum to ~1.0
        - Max weight constraint enforced
        """
        config = StrategyConstraints(max_positions=10, max_weight_per_name=0.15)
        portfolio = rank_weight(sample_valuations, config)
        
        total_weight = sum(w for _, w in portfolio.weights)
        assert abs(total_weight - 1.0) < 0.01
        assert all(w <= 0.15 for _, w in portfolio.weights)
    
    def test_long_short_portfolio(self, test_db, sample_valuations):
        """
        Test long/short portfolio construction.
        
        Verifies:
        - Long positions are undervalued
        - Short positions are overvalued
        - Net exposure is neutral
        """
        portfolio = long_short_weight(sample_valuations, config)
        
        long_weight = sum(w for _, w in portfolio.weights if w > 0)
        short_weight = sum(w for _, w in portfolio.weights if w < 0)
        
        assert abs(long_weight + short_weight) < 0.01  # Net neutral
    
    def test_sector_constraints(self, test_db, sample_valuations):
        """
        Test sector exposure constraints.
        
        Verifies:
        - No sector exceeds max weight
        - Sector weights computed correctly
        """
        config = StrategyConstraints(max_sector_weight=0.25)
        portfolio = rank_weight(sample_valuations, config)
        
        sector_weights = compute_sector_weights(portfolio)
        assert all(w <= 0.25 for w in sector_weights.values())
    
    def test_turnover_calculation(self, test_db):
        """
        Test turnover tracking between rebalances.
        
        Verifies:
        - Turnover computed correctly
        - New positions count fully
        - Closed positions count fully
        """
        old = {"AAPL": 0.5, "MSFT": 0.5}
        new = {"AAPL": 0.5, "GOOGL": 0.5}  # Swap MSFT for GOOGL
        
        turnover = compute_turnover(old, new)
        assert turnover == 0.5  # 50% turnover
```

### Running Tests

```bash
# Run strategy integration tests
pytest tests/integration/test_strategy.py -v

# Run with real valuation data
pytest tests/integration/test_strategy.py -v --use-db

# Run with coverage
pytest tests/integration/test_strategy.py --cov=src/strategy
```
