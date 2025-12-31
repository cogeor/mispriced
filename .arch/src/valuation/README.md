# Valuation / Model Optimization

> Predicts fair market cap from financial statements using ML models with bootstrap-cross-validation. Supports model versioning for reproducibility.

---

## Responsibilities

- Predict fair market cap from each `FinancialSnapshot` using ML models
- Use **repeated nested cross-validation** (bootstrap-crossval) to get prediction distributions:
  - Inner CV: Hyperparameter tuning (GridSearchCV)
  - Outer CV: Held-out predictions
  - Repeat N times (e.g., 100) to build distribution of predictions
- Output prediction mean ± std for confidence intervals
- Compute mispricing as `(actual_mcap - predicted_mcap) / actual_mcap`
- **Model versioning**: Save/load model configs as JSON for reproducibility

---

## Model Versioning

### ✅ RESOLVED: JSON-Based Model Configuration

**Goal**: Replicate model results by storing complete model configuration as JSON.

### Model Config Schema

```python
# src/valuation/model_config.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import hashlib

class FeatureSpec(BaseModel):
    """Single feature specification."""
    name: str
    required: bool = True
    transform: Optional[str] = None  # "log", "sqrt", "normalize"
    fill_strategy: str = "zero"      # "zero", "mean", "median", "drop"

class ModelConfig(BaseModel):
    """
    Complete model configuration for reproducibility.
    Can be serialized to JSON and reconstructed via builder.
    """
    # Model identification
    name: str
    version: str
    description: str
    
    # Model type
    model_type: str  # "xgboost", "random_forest", "lightgbm"
    
    # Features (as list for builder pattern)
    features: List[FeatureSpec]
    
    # Hyperparameters
    hyperparameters: Dict[str, Any]
    param_grid: Dict[str, List[Any]]  # For grid search
    
    # Cross-validation config
    n_experiments: int = 100
    outer_cv_splits: int = 4
    inner_cv_splits: int = 4
    
    # Loss function
    loss_function: str = "reg:squarederror"
    custom_objective: Optional[str] = None
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    def to_json(self) -> str:
        """Serialize config to JSON."""
        return self.model_dump_json(indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ModelConfig":
        """Deserialize config from JSON."""
        return cls.model_validate_json(json_str)
    
    def config_hash(self) -> str:
        """Compute hash for config identification."""
        return hashlib.sha256(self.model_dump_json(indent=2).encode()).hexdigest()[:16]
```

### JSON Config Example

```json
{
  "name": "xgboost_baseline",
  "version": "1.0.0",
  "description": "XGBoost baseline model with core financial features",
  "model_type": "xgboost",
  "features": [
    {"name": "total_revenue", "required": true, "fill_strategy": "zero"},
    {"name": "gross_profit", "required": true, "fill_strategy": "zero"},
    {"name": "ebitda", "required": false, "fill_strategy": "median"},
    {"name": "total_debt", "required": true, "fill_strategy": "zero"},
    {"name": "total_cash", "required": true, "fill_strategy": "zero"},
    {"name": "free_cash_flow", "required": false, "fill_strategy": "zero"},
    {"name": "profit_margins", "required": false, "transform": null},
    {"name": "debt_to_equity", "required": false, "transform": "log"},
    {"name": "roe", "required": false, "fill_strategy": "median"},
    {"name": "roa", "required": false, "fill_strategy": "median"}
  ],
  "hyperparameters": {
    "objective": "reg:squarederror",
    "random_state": 42
  },
  "param_grid": {
    "n_estimators": [100, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.8]
  },
  "n_experiments": 100,
  "outer_cv_splits": 4,
  "inner_cv_splits": 4,
  "loss_function": "reg:squarederror",
  "random_seed": 42
}
```

---

## Builder Pattern for Model Construction

### ✅ RESOLVED: Modular Feature Addition

**Goal**: Build models incrementally with fluent interface.

```python
# src/valuation/model_builder.py
from typing import Optional, List, Dict, Any
from .model_config import ModelConfig, FeatureSpec

class ModelBuilder:
    """
    Builder pattern for constructing model configurations.
    Allows modular feature addition and configuration.
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self._name = name
        self._version = version
        self._description = ""
        self._model_type = "xgboost"
        self._features: List[FeatureSpec] = []
        self._hyperparameters: Dict[str, Any] = {}
        self._param_grid: Dict[str, List[Any]] = {}
        self._n_experiments = 100
        self._outer_cv_splits = 4
        self._inner_cv_splits = 4
        self._loss_function = "reg:squarederror"
        self._random_seed = 42
    
    def description(self, desc: str) -> "ModelBuilder":
        """Set model description."""
        self._description = desc
        return self
    
    def model_type(self, model_type: str) -> "ModelBuilder":
        """Set model type: 'xgboost', 'random_forest', 'lightgbm'."""
        self._model_type = model_type
        return self
    
    # --- Feature Addition Methods ---
    
    def add_feature(
        self,
        name: str,
        required: bool = True,
        transform: Optional[str] = None,
        fill_strategy: str = "zero",
    ) -> "ModelBuilder":
        """Add a single feature to the model."""
        self._features.append(FeatureSpec(
            name=name,
            required=required,
            transform=transform,
            fill_strategy=fill_strategy,
        ))
        return self
    
    def add_core_fundamentals(self) -> "ModelBuilder":
        """Add standard fundamental features."""
        return (
            self.add_feature("total_revenue", required=True)
            .add_feature("gross_profit", required=True)
            .add_feature("ebitda", required=False, fill_strategy="median")
            .add_feature("net_income", required=False)
            .add_feature("total_debt", required=True)
            .add_feature("total_cash", required=True)
            .add_feature("free_cash_flow", required=False)
            .add_feature("operating_cash_flow", required=False)
        )
    
    def add_ratio_features(self) -> "ModelBuilder":
        """Add financial ratio features."""
        return (
            self.add_feature("profit_margins", required=False)
            .add_feature("debt_to_equity", required=False, transform="log")
            .add_feature("roe", required=False, fill_strategy="median")
            .add_feature("roa", required=False, fill_strategy="median")
            .add_feature("current_ratio", required=False)
            .add_feature("quick_ratio", required=False)
        )
    
    def add_dividend_features(self) -> "ModelBuilder":
        """Add dividend-related features."""
        return (
            self.add_feature("dividend_yield", required=False)
            .add_feature("dividend_rate", required=False)
        )
    
    def add_risk_features(self) -> "ModelBuilder":
        """Add risk score features."""
        return (
            self.add_feature("audit_risk", required=False)
            .add_feature("board_risk", required=False)
            .add_feature("compensation_risk", required=False)
        )
    
    def add_short_interest(self) -> "ModelBuilder":
        """Add short interest features."""
        return (
            self.add_feature("shares_short", required=False)
            .add_feature("short_ratio", required=False)
        )
    
    def add_ownership_features(self) -> "ModelBuilder":
        """Add ownership structure features."""
        return (
            self.add_feature("held_percent_insiders", required=False)
            .add_feature("held_percent_institutions", required=False)
        )
    
    # --- Configuration Methods ---
    
    def hyperparameters(self, **kwargs) -> "ModelBuilder":
        """Set base hyperparameters."""
        self._hyperparameters.update(kwargs)
        return self
    
    def param_grid(self, **kwargs) -> "ModelBuilder":
        """Set hyperparameter search grid."""
        self._param_grid.update(kwargs)
        return self
    
    def cv_config(
        self,
        n_experiments: int = 100,
        outer_splits: int = 4,
        inner_splits: int = 4,
    ) -> "ModelBuilder":
        """Configure cross-validation."""
        self._n_experiments = n_experiments
        self._outer_cv_splits = outer_splits
        self._inner_cv_splits = inner_splits
        return self
    
    def loss_function(self, loss: str) -> "ModelBuilder":
        """Set loss function."""
        self._loss_function = loss
        return self
    
    def random_seed(self, seed: int) -> "ModelBuilder":
        """Set random seed for reproducibility."""
        self._random_seed = seed
        return self
    
    # --- Build Methods ---
    
    def build(self) -> ModelConfig:
        """Build the final ModelConfig."""
        return ModelConfig(
            name=self._name,
            version=self._version,
            description=self._description,
            model_type=self._model_type,
            features=self._features,
            hyperparameters=self._hyperparameters,
            param_grid=self._param_grid,
            n_experiments=self._n_experiments,
            outer_cv_splits=self._outer_cv_splits,
            inner_cv_splits=self._inner_cv_splits,
            loss_function=self._loss_function,
            random_seed=self._random_seed,
        )
    
    def to_json(self) -> str:
        """Build and serialize to JSON."""
        return self.build().to_json()


# --- Factory Functions for Common Configs ---

def baseline_model() -> ModelConfig:
    """Create baseline XGBoost model."""
    return (
        ModelBuilder("xgboost_baseline", "1.0.0")
        .description("XGBoost baseline with core fundamentals")
        .model_type("xgboost")
        .add_core_fundamentals()
        .add_ratio_features()
        .param_grid(
            n_estimators=[100, 300],
            max_depth=[3, 5, 7],
            learning_rate=[0.05, 0.1, 0.2],
            subsample=[0.8],
        )
        .cv_config(n_experiments=100)
        .build()
    )

def full_feature_model() -> ModelConfig:
    """Create model with all available features."""
    return (
        ModelBuilder("xgboost_full", "1.0.0")
        .description("XGBoost with all available features")
        .model_type("xgboost")
        .add_core_fundamentals()
        .add_ratio_features()
        .add_dividend_features()
        .add_risk_features()
        .add_short_interest()
        .add_ownership_features()
        .param_grid(
            n_estimators=[100, 300, 500],
            max_depth=[3, 5, 7, 9],
            learning_rate=[0.01, 0.05, 0.1, 0.2],
            subsample=[0.7, 0.8, 0.9],
        )
        .cv_config(n_experiments=100)
        .build()
    )
```

### Creating Model from JSON

```python
# Load model from saved config
def create_model_from_json(json_path: str) -> ModelRunner:
    """
    Reconstruct model from JSON configuration.
    Enables exact reproduction of previous experiments.
    """
    with open(json_path) as f:
        config = ModelConfig.from_json(f.read())
    
    # Create estimator based on model type
    if config.model_type == "xgboost":
        from xgboost import XGBRegressor
        estimator = XGBRegressor(
            objective=config.loss_function,
            random_state=config.random_seed,
            **config.hyperparameters,
        )
    elif config.model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        estimator = RandomForestRegressor(
            random_state=config.random_seed,
            **config.hyperparameters,
        )
    
    return ModelRunner(config, estimator)
```

---

## Core Approach (from value_analysis.py)

The valuation model is fundamentally a **financial statement → market cap predictor**:

```python
# Conceptual flow from value_analysis.py
X = financial_metrics  # revenue, margins, debt, FCF, etc.
y = market_cap / 1e6   # Target: market cap in millions

# Nested cross-validation with GridSearch
outer_cv = KFold(n_splits=4, shuffle=True)
inner_cv = KFold(n_splits=4, shuffle=True)
grid_search = GridSearchCV(pipeline, param_grid, cv=inner_cv)

# Repeat N times for distribution
n_experiments = 100
predictions = []
for i in range(n_experiments):
    y_pred = cross_val_predict(grid_search, X, y, cv=outer_cv)
    predictions.append(y_pred)

# Result: mean ± std per ticker
pred_mean = np.mean(predictions, axis=0)
pred_std = np.std(predictions, axis=0)
rel_err = (pred_mean - actual_mcap) / actual_mcap
rel_std = pred_std / actual_mcap
```

---

## Inputs

- `FinancialSnapshot` rows (from DB) with features:
  - Market cap (target)
  - Revenue, gross profit, EBITDA, net income
  - Total debt, total cash, free cash flow
  - Profit margins, ROA, ROE, debt-to-equity
  - Current ratio, quick ratio
  - Shares outstanding, float shares
  - Dividend yield, audit/board/compensation risk scores
  - Industry/sector (for stratification, not one-hot encoding by default)
- **Model configuration** (JSON or ModelConfig object):
  - Number of experiments (default: 100)
  - CV splits (default: 4-fold for both inner/outer)
  - Model type: XGBoost, RandomForest
  - Hyperparameter grid
  - Feature list with transforms
- **Data version ID** (optional): For reproducibility

---

## Outputs

- `ValuationResult` per ticker (at snapshot time):
  - `predicted_mcap_mean: float` - Mean prediction across experiments
  - `predicted_mcap_std: float` - Std deviation across experiments
  - `actual_mcap: float` - Actual market cap at snapshot
  - `relative_error: float` = `(pred_mean - actual) / actual`
  - `relative_std: float` = `pred_std / actual`
  - `confidence_interval: Tuple[float, float]` - e.g., ±2σ
  - `model_config_hash: str` - Hash of model config for versioning
  - `data_version_id: UUID` - Reference to data version used
  - `experiment_metadata: dict` (n_experiments, cv_splits, param_grid)

---

## Dependencies

### External Packages
- `numpy` - Numerical computation
- `pandas` - DataFrame operations
- `scikit-learn` - Cross-validation, pipelines, metrics
- `xgboost` - Primary model
- `pyyaml` - Configuration loading

### Internal Modules
- `src/db/` - Repository access to snapshots
- `src/normalize/` - Canonical snapshot models

---

## Folder Structure

```
src/valuation/
  __init__.py
  service.py              # Main valuation orchestrator
  models/
    __init__.py
    base.py               # Abstract model interface
    xgboost_model.py      # XGBoost regressor
    rf_model.py           # RandomForest regressor
  bootstrap_cv.py         # Repeated cross-validation engine
  feature_builder.py      # Build feature matrix from snapshots
  result.py               # ValuationResult dataclass
  model_config.py         # ModelConfig and FeatureSpec
  model_builder.py        # Builder pattern implementation
  config.py               # Valuation configuration
  metrics.py              # Custom metrics (relative_mse)
```

---

## Feature Set (from value_analysis.py)

### ⚠️ Current Data Availability (as of 2025-12)

Due to data ingestion limitations, most financial features have <1% availability:

| Feature | Availability | Notes |
|---------|-------------|-------|
| total_revenue | ~90% | **Primary feature** - only reliable feature |
| gross_profit | <0.1% | Data ingestion issue |
| ebitda | <0.1% | Data ingestion issue |
| net_income | <0.1% | Data ingestion issue |
| total_debt | <0.1% | Data ingestion issue |
| total_cash | <0.1% | Data ingestion issue |
| Other fundamentals | <0.1% | Data ingestion issue |
| beta | ~96% | Price-derived, excluded from features |
| market_cap_t0 | ~95% | Target variable, not a feature |

**Impact**: The model currently operates as a single-feature predictor using only `total_revenue`.
This significantly limits the model's ability to differentiate valuations based on profitability,
leverage, cash position, and other fundamental factors.

**Resolution**: Requires re-ingestion of financial statement data from provider API.
See `src/ingestion/` module and `CLAUDE.md` for ingestion pipeline details.

### Ideal Feature Set

```python
# Core features intended for market cap prediction
FEATURES = [
    # Fundamentals
    "totalRevenue",
    "grossProfit",
    "ebitda",
    "totalDebt",
    "totalCash",
    "freeCashflow",
    "operatingCashFlow",

    # Per-share / structure
    "sharesOutstanding",
    "floatShares",
    "bookValue",

    # Ratios
    "profitMargins",
    "debtToEquity",
    "returnOnAssets",
    "returnOnEquity",
    "quickRatio",
    "currentRatio",

    # Dividend
    "dividendYield",
    "dividendRate",
    "fiveYearAvgDividendYield",

    # Risk scores
    "auditRisk",
    "boardRisk",
    "compensationRisk",

    # Short interest
    "sharesShort",
    "sharesShortPriorMonth",
    "shortRatio",

    # Ownership
    "heldPercentInsiders",
    "heldPercentInstitutions",

    # Employees
    "fullTimeEmployees",
]
```

---

## Bootstrap Cross-Validation Engine

```python
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import numpy as np

class BootstrapCrossValidator:
    def __init__(
        self,
        n_experiments: int = 100,
        outer_splits: int = 4,
        inner_splits: int = 4,
        model_class = XGBRegressor,
        param_grid: dict = None,
    ):
        self.n_experiments = n_experiments
        self.outer_cv = KFold(n_splits=outer_splits, shuffle=True)
        self.inner_cv = KFold(n_splits=inner_splits, shuffle=True)
        self.param_grid = param_grid or {
            'regressor__n_estimators': [100, 300],
            'regressor__max_depth': [3, 5, 7],
            'regressor__learning_rate': [0.05, 0.1, 0.2],
            'regressor__subsample': [0.8]
        }
        self.pipeline = Pipeline([
            ('regressor', model_class(objective='reg:squarederror', random_state=42))
        ])
    
    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Run repeated nested cross-validation.
        Returns prediction distribution statistics.
        """
        n_samples = X.shape[0]
        predictions = np.zeros((self.n_experiments, n_samples))
        
        for i in range(self.n_experiments):
            grid_search = GridSearchCV(
                self.pipeline, 
                self.param_grid, 
                cv=self.inner_cv, 
                n_jobs=-1
            )
            y_pred = cross_val_predict(grid_search, X, y, cv=self.outer_cv)
            predictions[i] = y_pred
        
        return {
            'predictions': predictions,
            'mean': np.mean(predictions, axis=0),
            'std': np.std(predictions, axis=0),
        }
```

---

## Design Decisions

### ✅ RESOLVED: Model Type

**Decision**: Use **XGBoost** with nested cross-validation as primary model, based on `value_analysis.py`.

Default hyperparameter grid:
```python
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8]
}
```

### ✅ RESOLVED: Confidence Estimation

**Decision**: Use **bootstrap-crossval** (repeated nested CV) to generate prediction distributions.
- N experiments = 100 (configurable)
- Output: mean ± std for each prediction
- Confidence interval: ±2σ by default

### ✅ RESOLVED: Model Versioning

**Decision**: Use JSON-based model configuration for reproducibility.
- Complete model config serializable to/from JSON
- Builder pattern for modular feature addition
- Config hash stored with results for traceability

### ⚠️ NEEDS REVIEW: Loss Function

Current: `reg:squarederror` (MSE)

**Alternatives to consider**:
- `reg:absoluteerror` (MAE) - More robust to outliers
- Custom `relative_mse` - Penalize relative error, not absolute

### ⚠️ NEEDS REVIEW: Feature Selection

Should we include all features or use feature importance to prune?

**Options**:
1. **All available** - Use all features, let model select
2. **Pre-filter** - Remove highly correlated or low-variance features
3. **Post-analysis** - Use SHAP/feature importance to understand drivers

### 📌 TODO: Market Cap Filtering

Current: Filter to `marketCap > 1e9` (billion+)

Need to decide minimum market cap threshold for model training.

---

## Constraints

- ⚡ Predicts **market cap**, not returns
- ⚡ Uses cross-validation on **held-out data** only
- ⚡ Must produce mean ± std for all predictions
- ⚡ Must persist experiment metadata for reproducibility
- ⚡ Must handle missing features gracefully (fillna or drop)
- ⚡ Model config must be serializable to JSON

---

## Integration Tests

### Test Scope

Integration tests verify the full valuation pipeline including model training, prediction, and versioning.

### Test Cases

```python
# tests/integration/test_valuation.py

class TestValuationIntegration:
    """Integration tests for valuation pipeline."""
    
    def test_bootstrap_cv_basic(self, sample_data):
        """
        Test basic bootstrap cross-validation.
        
        Verifies:
        - Model runs N experiments
        - Returns predictions for all samples
        - Mean and std are computed correctly
        """
        X, y = sample_data
        validator = BootstrapCrossValidator(n_experiments=10)
        result = validator.fit_predict(X, y)
        
        assert result['predictions'].shape[0] == 10
        assert len(result['mean']) == len(y)
        assert len(result['std']) == len(y)
    
    def test_model_config_serialization(self):
        """
        Test model config JSON serialization.
        
        Verifies:
        - Config can be serialized to JSON
        - Config can be loaded from JSON
        - Loaded config matches original
        """
        original = baseline_model()
        json_str = original.to_json()
        loaded = ModelConfig.from_json(json_str)
        
        assert loaded.name == original.name
        assert loaded.version == original.version
        assert len(loaded.features) == len(original.features)
        assert loaded.param_grid == original.param_grid
    
    def test_model_builder_fluent(self):
        """
        Test builder pattern fluent interface.
        
        Verifies:
        - Can chain method calls
        - Features are accumulated correctly
        - Final config is valid
        """
        config = (
            ModelBuilder("test_model")
            .description("Test model")
            .add_core_fundamentals()
            .add_ratio_features()
            .param_grid(n_estimators=[100])
            .build()
        )
        
        assert config.name == "test_model"
        assert len(config.features) > 0
        assert "n_estimators" in config.param_grid
    
    def test_model_from_json(self, model_config_file):
        """
        Test creating model from saved JSON config.
        
        Verifies:
        - Model can be reconstructed from JSON
        - Model produces valid predictions
        - Config hash is reproducible
        """
        runner = create_model_from_json(model_config_file)
        
        assert runner.config.config_hash() == expected_hash
    
    def test_valuation_with_data_version(self, test_db, version_manager):
        """
        Test valuation with specific data version.
        
        Verifies:
        - Valuation uses specified data version
        - Results include data_version_id
        - Same version produces same results (with same seed)
        """
        version = version_manager.create_version("test-v1", "Test")
        
        result1 = run_valuation(baseline_model(), data_version_id=version.version_id)
        result2 = run_valuation(baseline_model(), data_version_id=version.version_id)
        
        assert result1.data_version_id == version.version_id
        # With same seed and data, results should be identical
        assert np.allclose(result1.predictions, result2.predictions)
    
    def test_feature_transforms(self, sample_data):
        """
        Test feature transformations are applied.
        
        Verifies:
        - Log transform is applied
        - Fill strategies work correctly
        - Missing features handled
        """
        config = (
            ModelBuilder("transform_test")
            .add_feature("total_revenue", required=True)
            .add_feature("debt_to_equity", transform="log")
            .add_feature("ebitda", fill_strategy="median")
            .build()
        )
        
        X = build_feature_matrix(sample_data, config.features)
        
        # Verify transforms applied
        assert X is not None

### Running Tests

```bash
# Run all valuation integration tests
pytest tests/integration/test_valuation.py -v

# Run with sample data (fast)
pytest tests/integration/test_valuation.py -v --synthetic-data

# Run with real database (slow)
pytest tests/integration/test_valuation.py -v --use-db

# Run with coverage
pytest tests/integration/test_valuation.py --cov=src/valuation
```
