# Evaluation Engine

> Financial audit: compare predicted market cap to actual and track prediction accuracy over time.

---

## Responsibilities

Evaluate the thesis: **"Given financial statements, can we predict fair market cap?"**

This module performs a **financial audit** by:

1. **Out-of-Fold Predictions** - Generate fair predictions using repeated CV
2. **Prediction Accuracy** - Compare predicted market cap to actual market cap at snapshot time
3. **Error Distribution** - Analyze relative error distribution across universe
4. **Temporal Stability** - Track how predictions evolve as new statements arrive

---

## Evaluation Methodology & Pipeline

### Core Principle: Fair Out-of-Fold Predictions

To ensure rigorous evaluation, predictions **MUST** be out-of-fold (held-out) predictions:

```
For each quarter:
1. Load data for THAT QUARTER ONLY (no future data)
2. Run repeated K-fold CV with random shuffling
3. Each sample is predicted when it's in the held-out fold
4. Repeat N times with different random seeds
5. Output: mean and std of predictions per ticker
```

### Why This Matters

- **No Data Leakage**: Model never sees the sample it's predicting
- **Fair Evaluation**: Predictions represent true out-of-sample performance
- **Uncertainty Quantification**: Std across repeats measures prediction confidence
- **Same-Quarter**: Uses only financials from the same period

---

## Pipeline Steps

### 1. Data Preparation
- **Input**: Financial snapshots for a specific quarter
- **Filtering**:
  - Market cap >= $100M (configurable)
  - Deduplicate by ticker
  - Exclude direct price/mcap features from X
- **Target**: Market Cap in $Millions (for numerical stability)
- **Features**: Core fundamentals + ratio features

### 2. Repeated Cross-Validation (Fixed Params)
- **Fixed Parameters**: No hyperparameter tuning
- **Process**:
  1. For each of N repeats (default: 10):
     - Shuffle data with different random seed
     - Split into K folds (default: 5)
     - Train on K-1 folds, predict on held-out fold
     - Collect out-of-fold predictions
  2. Aggregate across repeats

### 3. Result Aggregation
- **Output per ticker**:
  - `predicted_mcap_mean`: Mean of OOF predictions across repeats
  - `predicted_mcap_std`: Std of predictions (uncertainty)
  - `actual_mcap`: Actual market cap at snapshot time
  - `relative_error`: (predicted - actual) / actual
  - `relative_std`: std / actual

---

## Core Metrics

### Relative Error (Primary)

```python
relative_error = (predicted_mcap - actual_mcap) / actual_mcap
```

- Negative → Predicted < Actual → Stock is "overvalued" by market
- Positive → Predicted > Actual → Stock is "undervalued" by market

### Relative Standard Deviation

```python
relative_std = prediction_std / actual_mcap
```

Measures confidence in the prediction. High std → uncertain prediction.

---

## Inputs

- **Financial Snapshots** per quarter from database
- **Configuration**:
  - `n_repeats`: Number of CV repetitions (default: 10)
  - `n_folds`: Number of K-fold splits (default: 5)
  - `min_market_cap`: Minimum market cap filter
  - `model_params`: Fixed hyperparameters (no grid search)

---

## Outputs

- **ValuationResult table** per ticker/quarter:
  - `predicted_mcap_mean`, `predicted_mcap_std`
  - `actual_mcap`, `relative_error`, `relative_std`
  - `n_experiments`: Number of CV repeats used

---

## Dependencies

### External Packages
- `numpy` - Numerical computation
- `scikit-learn` - Cross-validation, HistGradientBoostingRegressor
- `pandas` - Data manipulation
- `sqlalchemy` - Database access
- `pydantic` - Configuration models

### Internal Modules
- `src/db/` - Repository access
- `src/valuation/` - Feature building, model config

---

## Folder Structure

```
src/evaluation/
  __init__.py             # Module exports
  simple_cv.py            # SimpleRepeatedCV - core CV engine
```

Key script:
```
scripts/run_quarterly_valuation.py  # Run pipeline for all quarters
```

---

## Default Configuration

```python
# Fixed model parameters (no tuning)
DEFAULT_MODEL_PARAMS = {
    "max_iter": 200,
    "max_depth": 5,
    "learning_rate": 0.1,
    "early_stopping": False,
    "random_state": 42,
}

# CV configuration
N_CV_REPEATS = 10
N_CV_FOLDS = 5
MIN_MARKET_CAP = 100e6  # $100M
```

---

## Design Decisions

### ✅ RESOLVED: Fixed Parameters (No Tuning)

**Decision**: Use fixed default hyperparameters, no GridSearchCV.

**Rationale**:
- Simpler, faster pipeline
- Consistent results across quarters
- Hyperparameter tuning should be done separately as a one-time exercise

### ✅ RESOLVED: Repeated CV for Uncertainty

**Decision**: Run CV multiple times with different random seeds.

**Rationale**:
- Single CV run is sensitive to random split
- Multiple repeats give distribution of predictions
- Std across repeats quantifies prediction confidence

### ✅ RESOLVED: Same-Quarter Predictions

**Decision**: Predictions use only data from the same quarter.

**Rationale**:
- No look-ahead bias
- Fair evaluation of model's ability to value from financials
- Future price tracking is done separately in backtest module

---

## Constraints

- ⚡ Predictions must be **out-of-fold** (held-out)
- ⚡ Use **fixed parameters** (no tuning during prediction)
- ⚡ Data from **same quarter only** (no future information)
- ⚡ Store both mean and std for uncertainty quantification

---

## Usage

```bash
# Run valuation pipeline for all quarters
python scripts/run_quarterly_valuation.py

# Generate dashboard with results
python scripts/generate_dashboard.py
```
