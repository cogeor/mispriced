
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class RepeatedCrossValidator:
    """
    Executes repeated nested cross-validation to estimate prediction confidence intervals.
    """
    
    def __init__(
        self,
        n_experiments: int = 100,
        outer_splits: int = 4,
        inner_splits: int = 4,
        model_class: Any = XGBRegressor,
        model_init_params: Dict[str, Any] = None,
        param_grid: dict = None,
        random_seed: int = 42
    ):
        self.n_experiments = n_experiments
        self.outer_splits = outer_splits
        self.inner_splits = inner_splits
        self.param_grid = param_grid or {}
        
        # Base model configuration
        self.model_class = model_class
        self.model_init_params = model_init_params or {}
        self.random_seed = random_seed

    def fit_predict(self, X: np.ndarray, target_vector: np.ndarray) -> Dict[str, Any]:
        """
        Run repeated nested cross-validation.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            target_vector: Target vector (n_samples,)
        """
        n_samples = X.shape[0]
        predictions = np.zeros((self.n_experiments, n_samples))
        
        logger.info(f"Starting {self.n_experiments} bootstrap experiments with {n_samples} samples.")
        
        for i in range(self.n_experiments):
            # Vary random state for CV folds to get distribution
            seed = self.random_seed + i
            
            outer_cv = KFold(n_splits=self.outer_splits, shuffle=True, random_state=seed)
            inner_cv = KFold(n_splits=self.inner_splits, shuffle=True, random_state=seed)
            
            # Create pipeline with fresh model
            # Merge init params with random state
            model_params = self.model_init_params.copy()
            model_params['random_state'] = seed
            
            pipeline = Pipeline([
                ('regressor', self.model_class(**model_params))
            ])
            
            # Prefix param grid keys with 'regressor__' for pipeline
            pipeline_grid = {f"regressor__{k}": v for k, v in self.param_grid.items()}
            
            grid_search = GridSearchCV(
                pipeline, 
                pipeline_grid, 
                cv=inner_cv, 
                n_jobs=-1,
                scoring='neg_mean_squared_error' 
            )
            
            # Predictions on held-out data
            try:
                # Use target_vector here
                y_pred = cross_val_predict(grid_search, X, target_vector, cv=outer_cv, n_jobs=-1)
                predictions[i] = y_pred
            except Exception as e:
                logger.error(f"Experiment {i} failed: {e}")
                raise e
                
        return {
            'predictions': predictions,
            'mean': np.mean(predictions, axis=0),
            'std': np.std(predictions, axis=0),
        }
