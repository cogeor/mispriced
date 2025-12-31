
import pandas as pd
import numpy as np
import logging
from typing import List, Tuple
from .model_config import FeatureSpec

logger = logging.getLogger(__name__)


def build_feature_matrix(
    snapshots: pd.DataFrame,
    feature_specs: List[FeatureSpec]
) -> pd.DataFrame:
    """
    Convert pandas DataFrame of snapshots into numeric feature matrix X.
    Applies transforms and fill strategies defined in specs.
    Handles missing values gracefully with fallback strategies.

    Args:
        snapshots: Input dataframe containing raw features
        feature_specs: List of features to extract and process

    Returns:
        pd.DataFrame: Processed numeric feature matrix
    """
    X = pd.DataFrame(index=snapshots.index)

    for spec in feature_specs:
        if spec.name not in snapshots.columns:
            if spec.required:
                raise ValueError(f"Required feature '{spec.name}' missing from input data.")
            else:
                # If optional and missing, create column of zeros
                col_data = pd.Series(0.0, index=snapshots.index)
                logger.debug(f"Feature '{spec.name}' not in data, filling with zeros")
        else:
            col_data = snapshots[spec.name].copy()

        # Convert to numeric, forcing errors to NaN
        col_data = pd.to_numeric(col_data, errors='coerce')

        # Handle infinite values
        col_data = col_data.replace([np.inf, -np.inf], np.nan)

        # Apply Transforms (before fill to avoid log(0))
        if spec.transform == "log":
            # Use log1p for safety: log(1 + x)
            # Clip negative values to 0 before transform
            col_data = np.log1p(col_data.clip(lower=0))
        elif spec.transform == "sqrt":
            col_data = np.sqrt(col_data.clip(lower=0))

        # Apply Fill Strategy with fallback chain
        null_count = col_data.isna().sum()
        if null_count > 0:
            if spec.fill_strategy == "zero":
                col_data = col_data.fillna(0.0)
            elif spec.fill_strategy == "mean":
                fill_val = col_data.mean()
                if pd.isna(fill_val):  # All values are NaN
                    fill_val = 0.0
                col_data = col_data.fillna(fill_val)
            elif spec.fill_strategy == "median":
                fill_val = col_data.median()
                if pd.isna(fill_val):  # All values are NaN
                    fill_val = 0.0
                col_data = col_data.fillna(fill_val)
            else:
                # Default fallback: fill with zero
                col_data = col_data.fillna(0.0)

            if null_count > len(col_data) * 0.5:
                logger.debug(
                    f"Feature '{spec.name}': {null_count}/{len(col_data)} "
                    f"({100*null_count/len(col_data):.1f}%) values filled"
                )

        # Final safety check: ensure no NaN/inf remains
        col_data = col_data.fillna(0.0).replace([np.inf, -np.inf], 0.0)

        X[spec.name] = col_data

    return X
