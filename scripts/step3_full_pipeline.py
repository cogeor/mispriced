
import sys
import os
import pandas as pd
import numpy as np
import traceback
from collections import defaultdict
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import GradientBoostingRegressor

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.session import SessionLocal
from src.db.models import FinancialSnapshot, ValuationResult

def full_pipeline():
    session = SessionLocal()
    try:
        print("--- Step 3: Full Pipeline ---")
        
        # 1. Data Prep
        print("Loading data...")
        query = session.query(FinancialSnapshot)
        df = pd.read_sql(query.statement, session.bind)
        df['snapshot_timestamp'] = pd.to_datetime(df['snapshot_timestamp'])
        df_sorted = df.sort_values(['ticker', 'snapshot_timestamp'], ascending=[True, False])
        df_latest = df_sorted.drop_duplicates(subset=['ticker'], keep='first').copy()
        
        df_clean = df_latest.dropna(subset=['market_cap_t0'])
        df_clean = df_clean[df_clean['market_cap_t0'] > 0]
        
        df_clean['target_log_mcap'] = np.log(df_clean['market_cap_t0'])
        
        exclude_cols = [
            'ticker', 'snapshot_timestamp', 'period_end_date', 'filing_date', 'release_date',
            'market_cap_t0', 'price_t0', 'price_t_minus_1', 'price_t_plus_1', 
            'original_currency', 'stored_currency', 'fx_rate_to_usd',
            'created_at', 'updated_at', 'validation_warnings', 'data_quality_score'
        ]
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        X = df_clean[feature_cols].fillna(0).values
        y = df_clean['target_log_mcap'].values
        tickers = df_clean['ticker'].values
        current_snapshots = df_clean['snapshot_timestamp'].tolist() # Convert to list of Timestamps
        actual_mcaps = df_clean['market_cap_t0'].values
        
        print(f"Data Prepared: {X.shape[0]} samples. Features: {len(feature_cols)}")
        
        # 2. Repeated Cross-Validation
        n_splits = 3 
        n_repeats = 3 
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
        
        ticker_predictions = defaultdict(list)
        
        print(f"Running Repeated K-Fold ({n_splits} splits, {n_repeats} repeats)...")
        
        fold_i = 0
        for train_index, test_index in rkf.split(X):
            fold_i += 1
            if fold_i % n_splits == 0:
                print(f"  Fold {fold_i}...")
                
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            test_tickers_fold = tickers[test_index]
            
            model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred_log = model.predict(X_test)
            y_pred_actual = np.exp(y_pred_log)
            
            for i, ticker in enumerate(test_tickers_fold):
                ticker_predictions[ticker].append(y_pred_actual[i])
        
        # 3. Aggregate Results
        print("Aggregating results...")
        valuation_results = []
        
        model_version = "GBR_Baseline_v1"
        
        for i, ticker in enumerate(tickers):
            preds = ticker_predictions.get(ticker, [])
            if not preds:
                continue
                
            mean_pred = np.mean(preds)
            std_pred = np.std(preds)
            actual = actual_mcaps[i]
            
            relative_error = (mean_pred - actual) / actual
            relative_std = std_pred / actual if actual > 0 else 0
            
            # Convert timestamp to python datetime just in case
            ts = current_snapshots[i].to_pydatetime()
            
            result = ValuationResult(
                ticker=ticker,
                snapshot_timestamp=ts, 
                model_version=model_version,
                predicted_mcap_mean=float(mean_pred),
                predicted_mcap_std=float(std_pred),
                actual_mcap=float(actual),
                relative_error=float(relative_error),
                relative_std=float(relative_std),
                n_experiments=len(preds)
            )
            valuation_results.append(result)
            
        print(f"Generated {len(valuation_results)} valuation results.")
        
        # 4. Save to DB
        print("Saving to database...")
        try:
            for res in valuation_results:
                session.merge(res)
            session.commit()
            print("Saved successfully.")
            
             # Summary
            errors = [r.relative_error for r in valuation_results]
            print(f"Mean Relative Error: {np.mean(errors):.2%}")
            print(f"Median Relative Error: {np.median(errors):.2%}")

        except Exception as e:
            session.rollback()
            print(f"Save Failed: {e}")
            traceback.print_exc()

    except Exception as e:
        print(f"Pipeline Failed: {e}")
        traceback.print_exc()
    finally:
        session.close()

if __name__ == "__main__":
    full_pipeline()
