# Index Module

> Aggregates per-ticker valuations into index-level estimates with uncertainty propagation.

---

## Responsibilities

- Pull existing valuation results from DB for tickers in an index
- Compute **estimated index value** (weighted sum of predicted market caps) **with standard deviation**
- Compute **actual index value** (weighted sum of actual market caps)
- Return results in-memory (no persistence for now — evaluation workflow under validation)

---

## Core Concept

Given an index (e.g., S&P 500) and valuation results for its constituents:

1. **Actual Index**: Weighted sum of actual market caps
2. **Estimated Index**: Weighted sum of predicted market caps (with uncertainty)

This lets you answer: "If all our valuations are correct, what should this index be worth?"

---

## Inputs

- **Index definition** (from `indices` table):
  - `index_id`: e.g., "SP500", "NASDAQ100"
  - `weighting_scheme`: `equal` | `market_cap` | `custom`
  - `base_value`: default 100.0

- **Index membership** (from `index_memberships` table):
  - List of tickers in index at `as_of_time`

- **Valuation results** (from `valuation_results` table):
  - `predicted_mcap_mean`, `predicted_mcap_std` per ticker
  - `actual_mcap` per ticker

---

## Outputs

```python
from pydantic import BaseModel
from typing import Dict, Optional
from datetime import datetime

class IndexResult(BaseModel):
    """Result of index calculation at a single point in time."""
    index_id: str
    as_of_time: datetime
    
    # Actual index value (weighted actual market caps)
    actual_index: float
    
    # Estimated index value (weighted predicted market caps)
    estimated_index: float
    estimated_index_std: float  # Propagated uncertainty
    
    # Relative mispricing of the index
    index_relative_error: float  # (estimated - actual) / actual
    
    # Constituents used
    n_tickers: int
    n_tickers_with_valuation: int
    
    # Weights used
    weights: Dict[str, float]
    
    # Model metadata
    model_version: Optional[str] = None
```

---

## Index Calculation

### Weighting Schemes

| Scheme | Formula | Use Case |
|--------|---------|----------|
| `equal` | w_i = 1/N | Equal contribution from all tickers |
| `market_cap` | w_i = actual_mcap_i / Σ actual_mcap | Cap-weighted like S&P 500 |
| `custom` | Passed as Dict[ticker, weight] | User-defined weights |

### Index Value Formulas

**Base value** is set at first timestamp (default: 100).

For subsequent timestamps (same base for comparability):

```python
# Actual index value
actual_index = base × Σ(w_i × actual_mcap_i / actual_mcap_i_base)

# Estimated index value  
estimated_index = base × Σ(w_i × predicted_mcap_i / actual_mcap_i_base)

# Propagated uncertainty (assuming independent errors)
estimated_index_std = base × √Σ(w_i² × (predicted_std_i / actual_mcap_i_base)²)
```

---

## Key Methods

```python
from typing import Dict, List, Optional
from datetime import datetime

class IndexService:
    """Compute index-level values from per-ticker valuations."""
    
    def __init__(
        self,
        index_repo: IndexRepository,
        valuation_repo: ValuationRepository,
    ):
        self.index_repo = index_repo
        self.valuation_repo = valuation_repo
    
    def compute_index(
        self,
        index_id: str,
        as_of_time: datetime,
        model_version: Optional[str] = None,
        custom_weights: Optional[Dict[str, float]] = None,
    ) -> IndexResult:
        """
        Compute index values at a specific time.
        
        Args:
            index_id: Index to compute
            as_of_time: Snapshot time
            model_version: Filter valuations by model version
            custom_weights: Override index weighting scheme
        
        Returns:
            IndexResult with actual, estimated, and uncertainty
        """
        # Get index definition and members
        index_def = self.index_repo.get_index(index_id)
        tickers = self.index_repo.get_members(index_id, as_of_time)
        
        # Get valuations for members
        valuations = self.valuation_repo.get_valuations(
            tickers=tickers,
            as_of_time=as_of_time,
            model_version=model_version,
        )
        
        # Compute weights
        weights = custom_weights or self._compute_weights(
            valuations, index_def.weighting_scheme
        )
        
        # Compute index values
        return self._aggregate_index(valuations, weights, index_def)
    
    def _compute_weights(
        self,
        valuations: List[ValuationResult],
        scheme: str,
    ) -> Dict[str, float]:
        """Compute ticker weights based on scheme."""
        if scheme == "equal":
            n = len(valuations)
            return {v.ticker: 1.0 / n for v in valuations}
        
        elif scheme == "market_cap":
            total_mcap = sum(v.actual_mcap for v in valuations)
            return {v.ticker: v.actual_mcap / total_mcap for v in valuations}
        
        else:
            raise ValueError(f"Unknown weighting scheme: {scheme}")
    
    def _aggregate_index(
        self,
        valuations: List[ValuationResult],
        weights: Dict[str, float],
        index_def: Index,
    ) -> IndexResult:
        """Aggregate valuations into index-level metrics."""
        base = index_def.base_value
        
        actual_index = 0.0
        estimated_index = 0.0
        variance_sum = 0.0
        
        for v in valuations:
            w = weights.get(v.ticker, 0.0)
            if w == 0:
                continue
            
            # Normalize to base
            actual_index += w * v.actual_mcap
            estimated_index += w * v.predicted_mcap_mean
            variance_sum += (w * v.predicted_mcap_std) ** 2
        
        estimated_std = variance_sum ** 0.5
        
        return IndexResult(
            index_id=index_def.index_id,
            as_of_time=valuations[0].snapshot_timestamp,
            actual_index=actual_index,
            estimated_index=estimated_index,
            estimated_index_std=estimated_std,
            index_relative_error=(estimated_index - actual_index) / actual_index,
            n_tickers=len(weights),
            n_tickers_with_valuation=len(valuations),
            weights=weights,
        )
```

---

## Dependencies

### External Packages
- `numpy` - Numerical computation (optional, for efficiency)

### Internal Modules
- `src/db/` - Repository access (indices, memberships, valuations)

---

## Folder Structure

```
src/index/
  __init__.py
  service.py              # IndexService implementation
  models.py               # IndexResult, IndexAnalysis, and related models
  weights.py              # Weighting scheme implementations
```

---

## Dashboard Integration

### IndexAnalysis Model

Lightweight model for dashboard display:

```python
class IndexAnalysis(BaseModel):
    """Dashboard-friendly index analysis result."""
    index: str              # e.g., "SP500"
    mispricing: float       # Relative mispricing as decimal
    status: str             # "UNDERPRICED" or "OVERPRICED"
    total_actual: float     # Total actual market cap (USD)
    total_predicted: float  # Total predicted market cap (USD)
    count: int              # Tickers with valuations
    official_count: int     # Official constituent count
```

### Analysis Methods

```python
class IndexService:
    def analyze_index(
        self,
        index_id: str,
        official_count: int = 0,
    ) -> Optional[IndexAnalysis]:
        """
        Analyze single index mispricing using cap-weighted aggregation.
        Returns None if no valuation data available.
        """

    def analyze_all_indices(
        self,
        index_ids: Optional[List[str]] = None,
    ) -> List[IndexAnalysis]:
        """
        Analyze mispricing for multiple indices.
        If index_ids is None, analyzes all indices in database.
        """
```

### Usage in Dashboard

```python
from src.index import IndexService
from src.db.session import SessionLocal

session = SessionLocal()
service = IndexService(session)

# Get all index analyses for dashboard
analyses = service.analyze_all_indices()

for a in analyses:
    print(f"{a.index}: {a.mispricing:+.2%} ({a.status})")
```

---

## Design Decisions

### ✅ RESOLVED: No Persistence (For Now)

**Decision**: Index results are computed in-memory and returned, not stored to DB.

**Rationale**: Evaluation workflow is still under validation. Once stable, we may add:
- `index_series` table for historical index values
- Caching layer for frequently-requested indices

### ✅ RESOLVED: Uncertainty Propagation

**Decision**: Use standard error propagation assuming independent ticker errors.

```
σ_index = √Σ(w_i² × σ_i²)
```

This gives a lower bound on index uncertainty. In reality, prediction errors may be correlated (e.g., sector-wide bias), which would increase true uncertainty.

### ⚠️ NEEDS REVIEW: Missing Valuations

What to do when some index members don't have valuations?

**Options**:
1. **Skip** - Only use tickers with valuations, note in result
2. **Error** - Fail if any ticker missing
3. **Impute** - Use actual_mcap as predicted (zero error assumption)

**Current**: Option 1 (skip), with `n_tickers_with_valuation` tracking coverage.

---

## Constraints

- ⚡ Read-only from valuation results — does NOT modify predictions
- ⚡ No persistence for now — returns in-memory results only
- ⚡ Must propagate uncertainty from per-ticker to index level
- ⚡ Must support multiple weighting schemes
- ⚡ Must handle missing valuations gracefully

---

## Integration Tests

### Test Cases

```python
# tests/integration/test_index.py

class TestIndexIntegration:
    """Integration tests for index computation."""
    
    def test_equal_weight_index(self, test_db, sample_valuations):
        """
        Test equal-weight index computation.
        
        Verifies:
        - All tickers get 1/N weight
        - Index values computed correctly
        - Uncertainty propagated
        """
        service = IndexService(...)
        result = service.compute_index("TEST_EQUAL", as_of_time)
        
        assert len(result.weights) == len(sample_valuations)
        assert all(abs(w - 1/len(result.weights)) < 0.001 
                   for w in result.weights.values())
        assert result.estimated_index_std > 0
    
    def test_market_cap_weight_index(self, test_db, sample_valuations):
        """
        Test market-cap-weighted index.
        
        Verifies:
        - Weights proportional to actual market cap
        - Larger companies have more influence
        """
        service = IndexService(...)
        result = service.compute_index("TEST_MCAP", as_of_time)
        
        # Verify weights sum to ~1
        assert abs(sum(result.weights.values()) - 1.0) < 0.001
    
    def test_uncertainty_propagation(self, test_db):
        """
        Test that uncertainty is correctly propagated.
        
        Verifies:
        - estimated_index_std is computed
        - Higher per-ticker uncertainty → higher index uncertainty
        """
        # Create valuations with known std
        result = service.compute_index("TEST", as_of_time)
        
        assert result.estimated_index_std > 0
    
    def test_missing_valuations(self, test_db, partial_valuations):
        """
        Test handling of missing valuations.
        
        Verifies:
        - Index still computes with partial data
        - n_tickers_with_valuation < n_tickers when some missing
        """
        result = service.compute_index("TEST", as_of_time)
        
        assert result.n_tickers_with_valuation < result.n_tickers
```

### Running Tests

```bash
# Run index integration tests
pytest tests/integration/test_index.py -v

# Run with coverage
pytest tests/integration/test_index.py --cov=src/index
```