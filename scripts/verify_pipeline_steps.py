"""
Verify pipeline steps explicitly.
"""
import sys, os
from datetime import datetime
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.config import DATABASE_URL
from src.db.models import FinancialSnapshot, ValuationResult
from src.valuation.feature_builder import build_feature_matrix
from src.valuation.repeated_cv import RepeatedCrossValidator
from src.valuation.config import gbr_baseline_model

# Setup
quarter_date = datetime(2025, 9, 30)
model_config = gbr_baseline_model()
model_config.n_experiments = 5 # Small number for test

print(f"STEP 1: CONNECT TO DB")
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

try:
    print(f"STEP 2: LOAD DATA for {quarter_date.date()}")
    query = session.query(FinancialSnapshot).filter(
        FinancialSnapshot.snapshot_timestamp == quarter_date
    )
    df = pd.read_sql(query.statement, session.bind)
    print(f"  Loaded shape: {df.shape}")
    
    # Filter
    df['market_cap_t0'] = pd.to_numeric(df['market_cap_t0'], errors='coerce')
    df = df[df['market_cap_t0'] >= 100e6].copy()
    
    # Ensure unique tickers
    n_before = len(df)
    df = df.drop_duplicates(subset=['ticker'])
    n_after = len(df)
    if n_before != n_after:
        print(f"  Removed {n_before - n_after} duplicate tickers!")
    
    print(f"  Filtered shape: {df.shape}")
    
    assert len(df) > 1000, "Data load failed or too small"

    print(f"STEP 3: BUILD FEATURES")
    X_df = build_feature_matrix(df, model_config.features)
    X = X_df.values
    
    y_raw = pd.to_numeric(df['market_cap_t0'], errors='coerce').fillna(0)
    y = (y_raw / 1e6).values # Millions
    
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    
    # Check NaNs
    nan_count = np.isnan(X).sum()
    print(f"  NaN count: {nan_count}")
    
    print(f"STEP 4: INIT REPEATED CV")
    # Using HistGBR
    from sklearn.ensemble import HistGradientBoostingRegressor
    model_params = {
        'loss': 'squared_error',
        'max_iter': 50,
        'early_stopping': True
    }
    
    cv = RepeatedCrossValidator(
        n_experiments=model_config.n_experiments,
        outer_splits=2, # Fast
        inner_splits=2,
        model_class=HistGradientBoostingRegressor,
        model_init_params=model_params
    )
    
    print(f"STEP 5: RUN FIT_PREDICT")
    results = cv.fit_predict(X, y)
    
    preds = results['predictions']
    print(f"  Predictions shape: {preds.shape}") # Should be (n_experiments, n_samples)
    
    expected_shape = (model_config.n_experiments, len(df))
    assert preds.shape == expected_shape, f"Shape match failed: {preds.shape} != {expected_shape}"
    
    print(f"STEP 6: COMPUTE STATS")
    means = results['mean']
    stds = results['std']
    
    print(f"  Mean shape: {means.shape}")
    print(f"  Sample mean[0]: {means[0]:.2f}")
    
    print(f"STEP 7: DB OPERATIONS")
    # Clear old
    model_version = "TEST_VERIFY"
    deleted = session.query(ValuationResult).filter(
        ValuationResult.snapshot_timestamp == quarter_date,
        ValuationResult.model_version == model_version
    ).delete()
    print(f"  Cleared {deleted} items")
    
    # Save 1 item as test
    row = df.iloc[0]
    val = ValuationResult(
        ticker=row['ticker'],
        snapshot_timestamp=quarter_date,
        model_version=model_version,
        predicted_mcap_mean=float(means[0]) * 1e6,
        predicted_mcap_std=float(stds[0]) * 1e6,
        actual_mcap=float(y[0]) * 1e6,
        relative_error=0.0,
        relative_std=0.0,
        n_experiments=model_config.n_experiments
    )
    session.add(val)
    session.commit()
    print("  Saved 1 test valuation to DB")

    print("\n✅ VERIFICATION SUCCESSFUL")

except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
finally:
    session.close()
