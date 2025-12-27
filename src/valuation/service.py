
import pandas as pd
import numpy as np
import logging
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
from .model_config import ModelConfig
from .feature_builder import build_feature_matrix
from .repeated_cv import RepeatedCrossValidator
from ..db.models.snapshot import FinancialSnapshot
from ..filters.composer import DatasetFilter

logger = logging.getLogger(__name__)

class ValuationService:
    """
    Orchestrator for the valuation pipeline.
    Loads data, filters it, builds features, runs model, and returns results.
    """
    
    def __init__(self, db_session: Session):
        self.session = db_session
        
    def run_valuation(
        self, 
        model_config: ModelConfig, 
        dataset_filter: Optional[DatasetFilter] = None
    ) -> Dict[str, Any]:
        """
        Execute full valuation pipeline.
        
        Args:
            model_config: Configuration for model and features
            dataset_filter: Optional filter pipeline
            
        Returns:
            Dict with results and metrics
        """
        logger.info(f"Starting valuation with model: {model_config.name} (v{model_config.version})")
        
        # 1. Load Data
        # In a real scenario, we might batch this or allow filtering at SQL level
        # For now, load all snapshots. 
        # TODO: Optimize to load only needed columns based on feature specs + filters
        logger.info("Loading snapshots from database...")
        query = self.session.query(FinancialSnapshot)
        
        # Convert to DataFrame
        # This is memory intensive for full DB, but fine for prototype/subset
        snapshots_df = pd.read_sql(query.statement, self.session.bind)
        logger.info(f"Loaded {len(snapshots_df)} snapshots.")
        
        if snapshots_df.empty:
            raise ValueError("No data found in database.")
            
        # 2. Apply Filters
        if dataset_filter:
            logger.info("Applying dataset filters...")
            summary = dataset_filter.summary(snapshots_df)
            logger.info(f"Filter summary: {summary}")
            snapshots_df = dataset_filter.apply(snapshots_df)
            logger.info(f"Remaining snapshots: {len(snapshots_df)}")
            
        if snapshots_df.empty:
            raise ValueError("All snapshots were filtered out. Adjust filters.")
            
        # 3. Build Feature Matrix (X) and Target (y)
        logger.info("Building feature matrix...")
        X_df = build_feature_matrix(snapshots_df, model_config.features)
        X = X_df.values
        
        # Target: Market Cap in Billions (or Millions? convention check)
        # Usually standardized to Billions for readability, or Millions.
        # Let's use Millions as per Design Doc default? 
        # Design doc said "market_cap / 1e6 # In millions"
        if "market_cap_t0" not in snapshots_df.columns:
            raise ValueError("Target 'market_cap_t0' missing from data.")
            
        # Ensure target is numeric
        y_raw = pd.to_numeric(snapshots_df["market_cap_t0"], errors='coerce').fillna(0)
        y = (y_raw / 1e6).values # Target in Millions
        
        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        
        # 4. Run Bootstrap Cross-Validation
        logger.info("Running bootstrap cross-validation...")
        # Prepare model params for default XGBoost (since ValuationService uses default)
        model_params = {
            'objective': model_config.loss_function,
            'n_jobs': -1
        }
        
        cv_engine = RepeatedCrossValidator(
            n_experiments=model_config.n_experiments,
            outer_splits=model_config.outer_cv_splits,
            inner_splits=model_config.inner_cv_splits,
            param_grid=model_config.param_grid,
            model_init_params=model_params,
            random_seed=model_config.random_seed
        )
        
        results = cv_engine.fit_predict(X, y)
        
        # 5. Enrich Results
        results['tickers'] = snapshots_df['ticker'].values
        results['model_config'] = model_config
        results['snapshot_timestamps'] = snapshots_df['snapshot_timestamp'].values
        
        # Calculate derived metrics
        # Mean Predicted Market Cap (Millions)
        pred_mean = results['mean']
        
        # Mispricing = (Predicted - Actual) / Actual
        # Positive = Undervalued (Predicted > Actual) ? No, typically:
        # If Predicted > Actual, it's Undervalued (Price should rise).
        # Mispricing % often defined as upside potential.
        
        # Avoid division by zero
        safe_y = np.where(y == 0, 1e-6, y)
        mispricing = (pred_mean - y) / safe_y
        
        results['mispricing'] = mispricing
        results['actual_mcap_mm'] = y
        
        logger.info("Valuation completed.")
        return results
