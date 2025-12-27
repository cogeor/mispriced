"""
Simple repeated cross-validation with fixed parameters.

This module provides fair out-of-fold predictions for valuation,
using fixed hyperparameters (no tuning) and repeated random CV splits.

Key design:
- Uses XGBoost for predictions
- Target is log(market_cap) for numerical stability
- Each CV repeat uses different random splits
"""

import logging
from typing import Dict, Any, Optional, Type

import numpy as np
from pydantic import BaseModel
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


# Fixed default model parameters (no tuning)
DEFAULT_MODEL_PARAMS: Dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "verbosity": 0,
}


class CVResult(BaseModel):
    """Result of cross-validation for a single dataset."""

    predictions_mean: list[float]
    predictions_std: list[float]
    n_samples: int
    n_repeats: int
    n_folds: int

    class Config:
        arbitrary_types_allowed = True


class SimpleRepeatedCV:
    """
    Simple repeated K-fold cross-validation with fixed parameters.

    For each repetition:
    1. Shuffle data with different random seed
    2. Split into K folds
    3. Train on K-1 folds, predict on held-out fold
    4. Collect out-of-fold predictions

    Final output: mean and std of predictions across repetitions.

    Key properties:
    - Uses XGBoost with fixed default params (no tuning)
    - Each repeat uses a DIFFERENT random split
    - Predictions are always out-of-fold (fair evaluation)
    - Target should be log(market_cap) for stability
    """

    def __init__(
        self,
        n_repeats: int = 10,
        n_folds: int = 5,
        model_class: Optional[Type] = None,
        model_params: Optional[Dict[str, Any]] = None,
        random_seed: int = 42,
    ):
        """
        Initialize the cross-validator.

        Args:
            n_repeats: Number of times to repeat CV with different random splits
            n_folds: Number of folds for K-fold CV
            model_class: Sklearn-compatible regressor class (default: XGBRegressor)
            model_params: Fixed hyperparameters for the model
            random_seed: Base random seed for reproducibility
        """
        self.n_repeats = n_repeats
        self.n_folds = n_folds
        self.model_class = model_class or XGBRegressor
        self.model_params = model_params or DEFAULT_MODEL_PARAMS.copy()
        self.random_seed = random_seed

    def fit_predict(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Run repeated cross-validation and return out-of-fold predictions.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) - should be log(market_cap)

        Returns:
            Dictionary with:
                - 'mean': Mean prediction for each sample across repeats
                - 'std': Std deviation of predictions across repeats
                - 'all_predictions': Full (n_repeats, n_samples) matrix
        """
        n_samples = X.shape[0]

        if n_samples < self.n_folds:
            raise ValueError(
                f"Need at least {self.n_folds} samples for {self.n_folds}-fold CV, "
                f"got {n_samples}"
            )

        # Store predictions from each repeat
        all_predictions = np.zeros((self.n_repeats, n_samples))

        logger.info(
            f"Starting {self.n_repeats} CV repeats with {self.n_folds} folds "
            f"on {n_samples} samples"
        )

        for rep in range(self.n_repeats):
            # Each repeat uses a DIFFERENT random seed for shuffling
            seed = self.random_seed + rep * 1000
            kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=seed)

            # Out-of-fold predictions for this repeat
            oof_predictions = np.zeros(n_samples)

            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y[train_idx]

                # Create model with fixed params and unique seed per fold
                params = self.model_params.copy()
                params["random_state"] = seed + fold_idx

                model = self.model_class(**params)
                model.fit(X_train, y_train)

                # Predict on held-out fold
                oof_predictions[val_idx] = model.predict(X_val)

            all_predictions[rep] = oof_predictions

            if (rep + 1) % 5 == 0 or rep == 0:
                logger.info(f"  Completed repeat {rep + 1}/{self.n_repeats}")

        # Compute statistics across repeats
        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0)

        logger.info(
            f"CV complete. Mean prediction range: [{mean_predictions.min():.2f}, "
            f"{mean_predictions.max():.2f}]"
        )

        return {
            "mean": mean_predictions,
            "std": std_predictions,
            "all_predictions": all_predictions,
        }

    def get_cv_result(self, X: np.ndarray, y: np.ndarray) -> CVResult:
        """
        Run CV and return structured result.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            CVResult with predictions and metadata
        """
        results = self.fit_predict(X, y)

        return CVResult(
            predictions_mean=results["mean"].tolist(),
            predictions_std=results["std"].tolist(),
            n_samples=len(y),
            n_repeats=self.n_repeats,
            n_folds=self.n_folds,
        )
