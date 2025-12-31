# Configuration Module

> Python-based configuration for model parameters, loss functions, and feature selection with adapter pattern.

---

## Responsibilities

- Store model hyperparameters, loss functions, feature selection config in **Python files** (not YAML)
- Provide **adapter pattern** for easy model swapping with minimal abstraction
- Allow tinkering without code changes to core modules
- Expose clear interfaces that valuation module consumes

---

## Design Philosophy

**Low abstraction, high flexibility**:
- Config is Python code, not data files
- Adapters wrap models with consistent interface
- Easy to experiment with new models/parameters

---

## Folder Structure

```
src/config/
  __init__.py
  models.py               # Model adapters and hyperparameters
  features.py             # Feature selection configuration
  loss_functions.py       # Custom loss functions
  filters.py              # Dataset filtering rules
```

---

## Model Adapter Pattern

```python
# src/config/models.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

class ModelAdapter(ABC):
    """
    Abstract adapter for ML models.
    Low abstraction - just wraps sklearn-compatible estimators.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier."""
        pass
    
    @property
    @abstractmethod
    def estimator(self) -> BaseEstimator:
        """Return the sklearn-compatible estimator."""
        pass
    
    @property
    @abstractmethod
    def param_grid(self) -> Dict[str, list]:
        """Hyperparameter grid for GridSearchCV."""
        pass
    
    @property
    def cv_config(self) -> Dict[str, Any]:
        """Cross-validation configuration."""
        return {
            "outer_splits": 4,
            "inner_splits": 4,
            "n_experiments": 100,
            "shuffle": True,
        }


class XGBoostAdapter(ModelAdapter):
    """XGBoost regressor adapter - DEFAULT MODEL."""
    
    name = "xgboost_v1"
    
    @property
    def estimator(self) -> XGBRegressor:
        return XGBRegressor(
            objective=self.loss_function,
            random_state=42,
        )
    
    @property
    def loss_function(self) -> str:
        # Easily changeable loss function
        return "reg:squarederror"  # Options: reg:absoluteerror, custom
    
    @property
    def param_grid(self) -> Dict[str, list]:
        return {
            "n_estimators": [100, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.8],
        }


class RandomForestAdapter(ModelAdapter):
    """Random Forest regressor adapter - ALTERNATIVE."""
    
    name = "rf_v1"
    
    @property
    def estimator(self) -> RandomForestRegressor:
        return RandomForestRegressor(random_state=42)
    
    @property
    def loss_function(self) -> str:
        return "squared_error"  # RF uses criterion, not objective
    
    @property
    def param_grid(self) -> Dict[str, list]:
        return {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
        }


# === ACTIVE MODEL SELECTION ===
# Change this to swap models easily
ACTIVE_MODEL: ModelAdapter = XGBoostAdapter()
```

---

## Feature Selection Config

```python
# src/config/features.py
from typing import List, Callable, Optional
from dataclasses import dataclass

@dataclass
class FeatureConfig:
    """Configuration for feature selection."""
    
    # Core features always included
    core_features: List[str]
    
    # Optional features (may be missing)
    optional_features: List[str]
    
    # Features to exclude from model
    excluded_features: List[str]
    
    # Minimum non-null ratio to include feature (0.0 - 1.0)
    min_coverage: float = 0.5
    
    # Custom feature filter function
    custom_filter: Optional[Callable] = None


# === ACTIVE FEATURE CONFIG ===
FEATURE_CONFIG = FeatureConfig(
    core_features=[
        "total_revenue",
        "gross_profit",
        "ebitda",
        "total_debt",
        "total_cash",
        "free_cash_flow",
        "operating_cash_flow",
        "shares_outstanding",
        "book_value",
        "profit_margins",
        "debt_to_equity",
        "roa",
        "roe",
        "current_ratio",
        "quick_ratio",
    ],
    optional_features=[
        "dividend_yield",
        "dividend_rate",
        "audit_risk",
        "board_risk",
        "compensation_risk",
        "shares_short",
        "short_ratio",
        "held_percent_insiders",
        "held_percent_institutions",
        "full_time_employees",
    ],
    excluded_features=[
        "ticker",
        "snapshot_timestamp",
        "market_cap_t0",  # This is the target, not a feature
        "price_t0",
        "price_t_minus_1",
        "price_t_plus_1",
    ],
    min_coverage=0.5,
)


def get_feature_list(available_columns: List[str]) -> List[str]:
    """
    Get final feature list based on config and available columns.
    
    Args:
        available_columns: Columns available in the dataset
        
    Returns:
        List of feature names to use
    """
    config = FEATURE_CONFIG
    
    features = []
    
    # Add core features (must be present)
    for f in config.core_features:
        if f in available_columns:
            features.append(f)
    
    # Add optional features (if present)
    for f in config.optional_features:
        if f in available_columns and f not in config.excluded_features:
            features.append(f)
    
    # Apply custom filter if provided
    if config.custom_filter:
        features = config.custom_filter(features)
    
    return features
```

---

## Custom Loss Functions

```python
# src/config/loss_functions.py
import numpy as np

def relative_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Relative MSE: penalizes relative error, not absolute.
    Better for market cap prediction across different scales.
    """
    relative_errors = (y_pred - y_true) / y_true
    return np.mean(relative_errors ** 2)


def relative_mse_obj(y_pred: np.ndarray, dtrain) -> tuple:
    """
    XGBoost custom objective for relative MSE.
    Returns gradient and hessian.
    """
    y_true = dtrain.get_label()
    
    # Gradient: d/dy_pred of ((y_pred - y_true) / y_true)^2
    grad = 2 * (y_pred - y_true) / (y_true ** 2)
    
    # Hessian: second derivative
    hess = 2 / (y_true ** 2)
    
    return grad, hess


# === ACTIVE LOSS FUNCTION ===
# Set to None to use XGBoost default, or provide custom objective
CUSTOM_OBJECTIVE = None  # Options: relative_mse_obj, None
```

---

## Usage in Valuation Module

```python
# src/valuation/service.py
from src.config.models import ACTIVE_MODEL
from src.config.features import get_feature_list, FEATURE_CONFIG
from src.config.loss_functions import CUSTOM_OBJECTIVE

class ValuationService:
    def __init__(self, model_adapter=None):
        self.model = model_adapter or ACTIVE_MODEL
        
    def run_valuation(self, snapshots: pd.DataFrame) -> ValuationResults:
        # Get features based on config
        feature_cols = get_feature_list(snapshots.columns.tolist())
        
        X = snapshots[feature_cols].values
        y = snapshots["market_cap_t0"].values
        
        # Get model from adapter
        estimator = self.model.estimator
        param_grid = self.model.param_grid
        cv_config = self.model.cv_config
        
        # Run bootstrap-crossval
        # ... rest of implementation
```

---

## Design Decisions

### ✅ RESOLVED: Python Config over YAML

**Decision**: Use Python files for configuration instead of YAML/JSON.

**Rationale**:
- Supports complex types (functions, classes)
- IDE autocomplete and type checking
- Easy to add custom logic
- No parsing/validation layer needed

### ✅ RESOLVED: Adapter Pattern

**Decision**: Wrap models in lightweight adapters with consistent interface.

**Key properties**:
- `estimator` - sklearn-compatible model
- `param_grid` - hyperparameter search space
- `cv_config` - cross-validation settings

---

## Constraints

- ⚡ All config in Python files, not YAML/JSON
- ⚡ Adapters must be sklearn-compatible
- ⚡ Easy to swap `ACTIVE_MODEL` for experiments
- ⚡ Feature config decoupled from model config

---

## Integration Tests

### Test Scope

Integration tests verify model adapters work with sklearn and feature selection is correct.

### Test Cases

```python
# tests/integration/test_config.py

class TestConfigIntegration:
    """Integration tests for model configuration."""
    
    def test_xgboost_adapter(self, sample_data):
        """
        Test XGBoost adapter with sklearn GridSearchCV.
        
        Verifies:
        - Adapter returns valid estimator
        - Estimator works with GridSearchCV
        - Can fit and predict
        """
        adapter = XGBoostAdapter()
        estimator = adapter.estimator
        
        X, y = sample_data
        estimator.fit(X, y)
        predictions = estimator.predict(X)
        
        assert len(predictions) == len(y)
    
    def test_feature_selection(self):
        """
        Test feature list generation.
        
        Verifies:
        - Core features included
        - Optional features conditionally included
        - Excluded features not present
        """
        available = ["total_revenue", "market_cap_t0", "ebitda", "unknown_field"]
        features = get_feature_list(available)
        
        assert "total_revenue" in features
        assert "market_cap_t0" not in features  # Excluded (target)
    
    def test_model_swap(self, sample_data):
        """
        Test swapping between adapters.
        
        Verifies:
        - Can swap from XGBoost to RF
        - Both produce valid predictions
        """
        X, y = sample_data
        
        xgb = XGBoostAdapter()
        rf = RandomForestAdapter()
        
        xgb.estimator.fit(X, y)
        rf.estimator.fit(X, y)
        
        assert xgb.name != rf.name
    
    def test_param_grid_validity(self):
        """
        Test param grid format for GridSearchCV.
        
        Verifies:
        - All values are lists
        - Keys match estimator parameters
        """
        adapter = XGBoostAdapter()
        grid = adapter.param_grid
        
        assert all(isinstance(v, list) for v in grid.values())
```

### Running Tests

```bash
# Run config integration tests
pytest tests/integration/test_config.py -v

# Run with XGBoost and RF
pytest tests/integration/test_config.py -v

# Run with coverage
pytest tests/integration/test_config.py --cov=src/config
```
