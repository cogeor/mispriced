# Dataset Filters Module

> Provides dataset filtering rules applied BEFORE valuation model training.

---

## Responsibilities

- Define reusable filter predicates for snapshots
- Apply filters at the **ingestion→valuation boundary** (post-normalization, pre-model)
- Allow combining multiple filters
- Store filter configuration in Python for flexibility

---

## Component Ownership: Who Filters?

### ⚠️ Design Decision: Filters in Valuation (not Ingestion)

**Recommendation**: Filtering is owned by the **valuation module**, applied after data is normalized but before model training.

**Rationale**:
- Ingestion should fetch **all** available data (maximize raw data)
- Normalization should process **all** data (ensure consistency)
- Filtering is a **modeling decision**, not a data decision
- Same normalized data can be used with different filter configs

```
ingestion → normalize → [FILTERS HERE] → valuation → evaluation
              ↓
           raw data
           stored
```

---

## Folder Structure

```
src/filters/
  __init__.py
  predicates.py           # Individual filter functions
  composer.py             # Combine multiple filters
  config.py               # Active filter configuration
```

---

## Filter Predicates

```python
# src/filters/predicates.py
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


def currency_filter(currency: str = "USD") -> FilterPredicate:
    """Filter to specific currency (after conversion, this filters original currency)."""
    def predicate(df: pd.DataFrame) -> pd.Series:
        return df["currency"] == currency
    return predicate


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
        return df["data_quality_score"] >= min_score
    return predicate


def has_required_fields(fields: list) -> FilterPredicate:
    """Filter to snapshots that have non-null values for all required fields."""
    def predicate(df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        for field in fields:
            if field in df.columns:
                mask &= df[field].notna()
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


# --- Time-Based Filters ---

def is_latest() -> FilterPredicate:
    """
    Filter to keep only the latest snapshot per ticker.
    Useful for training on the most recent data point available.
    """
    def predicate(df: pd.DataFrame) -> pd.Series:
        max_dates = df.groupby("ticker")["snapshot_timestamp"].transform("max")
        return df["snapshot_timestamp"] == max_dates
    return predicate


def date_range(start_date: str, end_date: str) -> FilterPredicate:
    """Filter snapshots within a specific period_end_date range."""
    def predicate(df: pd.DataFrame) -> pd.Series:
        dates = pd.to_datetime(df["period_end_date"])
        return (dates >= pd.to_datetime(start_date)) & (dates <= pd.to_datetime(end_date))
    return predicate


def frequency_filter(frequency: str = "quarterly") -> FilterPredicate:
    """Filter by reporting frequency (e.g., 'quarterly' or 'annual')."""
    def predicate(df: pd.DataFrame) -> pd.Series:
        return df["frequency"] == frequency
    return predicate
```

---

## Filter Composer

```python
# src/filters/composer.py
from typing import List
import pandas as pd
from .predicates import FilterPredicate


class DatasetFilter:
    """
    Compose multiple filter predicates.
    Applies all filters with AND logic (intersection).
    """
    
    def __init__(self, predicates: List[FilterPredicate] = None):
        self.predicates = predicates or []
    
    def add(self, predicate: FilterPredicate) -> "DatasetFilter":
        """Add a filter predicate. Returns self for chaining."""
        self.predicates.append(predicate)
        return self
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all filters to dataframe.
        Returns filtered dataframe.
        """
        if not self.predicates:
            return df
        
        mask = pd.Series(True, index=df.index)
        for predicate in self.predicates:
            mask &= predicate(df)
        
        return df[mask].copy()
    
    def summary(self, df: pd.DataFrame) -> dict:
        """
        Get filtering summary without actually filtering.
        Useful for debugging/logging.
        """
        original_count = len(df)
        final_mask = pd.Series(True, index=df.index)
        
        step_counts = []
        for i, predicate in enumerate(self.predicates):
            mask = predicate(df)
            remaining = (final_mask & mask).sum()
            step_counts.append({
                "step": i + 1,
                "predicate": predicate.__name__ if hasattr(predicate, "__name__") else str(predicate),
                "remaining": remaining,
                "dropped": (final_mask.sum() - remaining),
            })
            final_mask &= mask
        
        return {
            "original_count": original_count,
            "final_count": final_mask.sum(),
            "total_dropped": original_count - final_mask.sum(),
            "steps": step_counts,
        }
```

---

## Filter Configuration

```python
# src/filters/config.py
from .predicates import (
    market_cap_min,
    currency_filter,
    min_data_quality,
    max_nan_ratio,
    positive_revenue,
)
from .composer import DatasetFilter


# === DEFAULT FILTER CONFIG ===
# Matches current value_analysis.py behavior
DEFAULT_FILTER = (
    DatasetFilter()
    .add(market_cap_min(1e9))       # Market cap > $1B
    .add(currency_filter("USD"))     # USD stocks only
    .add(max_nan_ratio(0.3))         # Max 30% missing values
)


# === ALTERNATIVE CONFIGS ===
# For experimentation

# All stocks (minimal filtering)
MINIMAL_FILTER = (
    DatasetFilter()
    .add(market_cap_min(0))
    .add(positive_revenue())
)

# Large cap only
LARGE_CAP_FILTER = (
    DatasetFilter()
    .add(market_cap_min(10e9))       # Market cap > $10B
    .add(min_data_quality(0.7))
)

# Profitable companies only
PROFITABLE_FILTER = (
    DatasetFilter()
    .add(market_cap_min(1e9))
    .add(positive_revenue())
    .add(profitable())
)


# === ACTIVE FILTER ===
# Change this to swap filter configs
ACTIVE_FILTER = DEFAULT_FILTER
```

---

## Usage in Valuation Module

```python
# src/valuation/service.py
from src.filters.config import ACTIVE_FILTER
from src.config.models import ACTIVE_MODEL
from src.config.features import get_feature_list

class ValuationService:
    def __init__(self, dataset_filter=None, model_adapter=None):
        self.filter = dataset_filter or ACTIVE_FILTER
        self.model = model_adapter or ACTIVE_MODEL
    
    def run_valuation(self, snapshots: pd.DataFrame) -> ValuationResults:
        # Step 1: Apply dataset filters
        filtered = self.filter.apply(snapshots)
        
        # Log filter summary
        summary = self.filter.summary(snapshots)
        print(f"Filtered {summary['total_dropped']} rows, {summary['final_count']} remaining")
        
        # Step 2: Get features
        feature_cols = get_feature_list(filtered.columns.tolist())
        
        # Step 3: Build X, y
        X = filtered[feature_cols].fillna(0).values
        y = filtered["market_cap_t0"].values / 1e6  # In millions
        
        # Step 4: Run model
        # ... bootstrap-crossval
```

---

## Design Decisions

### ✅ RESOLVED: Filters Owned by Valuation

**Decision**: Filtering happens at valuation stage, not ingestion.

**Flow**:
```
ingestion (fetch all) → normalize (process all) → filters (reduce) → valuation (model)
```

### ✅ RESOLVED: Composable Predicates

**Decision**: Use function-based predicates that can be chained.

**Benefits**:
- Easy to add new filter types
- Combine in any order
- Python-native, no DSL

### ⚠️ NEEDS REVIEW: Filter in Train vs Test

Should filters be applied to both train AND test sets identically?

**Options**:
1. **Same filter** - Apply identical filter to train and test
2. **Train-only** - Filter train set, evaluate on full test set
3. **Configurable** - Separate train/test filter configs

---

## Constraints

- ⚡ Filters are Python functions, not config files
- ⚡ Filters are composable with AND logic
- ⚡ Raw data is always preserved (filters don't delete)
- ⚡ Filter summary must be loggable for reproducibility

---

## Integration Tests

### Test Scope

Integration tests verify filter predicates and composition with real dataframes.

### Test Cases

```python
# tests/integration/test_filters.py

class TestFiltersIntegration:
    """Integration tests for dataset filters."""
    
    def test_market_cap_filter(self, sample_df):
        """
        Test market cap minimum filter.
        
        Verifies:
        - Tickers below threshold removed
        - Tickers above threshold kept
        """
        filter_obj = DatasetFilter().add(market_cap_min(1e9))
        result = filter_obj.apply(sample_df)
        
        assert all(result["market_cap_t0"] >= 1e9)
    
    def test_currency_filter(self, sample_df_with_currencies):
        """
        Test currency filter.
        
        Verifies:
        - Only specified currency kept
        - Other currencies removed
        """
        filter_obj = DatasetFilter().add(currency_filter("USD"))
        result = filter_obj.apply(sample_df_with_currencies)
        
        assert all(result["currency"] == "USD")
    
    def test_filter_composition(self, sample_df):
        """
        Test composing multiple filters.
        
        Verifies:
        - All filters applied (AND logic)
        - Order doesn't affect result
        """
        filter1 = (
            DatasetFilter()
            .add(market_cap_min(1e9))
            .add(min_data_quality(0.5))
        )
        filter2 = (
            DatasetFilter()
            .add(min_data_quality(0.5))
            .add(market_cap_min(1e9))
        )
        
        result1 = filter1.apply(sample_df)
        result2 = filter2.apply(sample_df)
        
        assert len(result1) == len(result2)
    
    def test_filter_summary(self, sample_df):
        """
        Test filter summary for logging.
        
        Verifies:
        - Summary includes step counts
        - Original and final counts correct
        """
        filter_obj = DatasetFilter().add(market_cap_min(1e9))
        summary = filter_obj.summary(sample_df)
        
        assert "original_count" in summary
        assert "final_count" in summary
        assert summary["final_count"] <= summary["original_count"]
```

### Running Tests

```bash
# Run filter integration tests
pytest tests/integration/test_filters.py -v

# Run with sample data
pytest tests/integration/test_filters.py -v

# Run with coverage
pytest tests/integration/test_filters.py --cov=src/filters
```
