
from .model_builder import ModelBuilder, ModelConfig

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

def gbr_baseline_model() -> ModelConfig:
    """Create baseline Gradient Boosting Regressor model.

    Uses core financial features with good data coverage (>60%).
    Missing values are handled gracefully via fill strategies.
    """
    return (
        ModelBuilder("GBR_Baseline", "3")
        .description("Gradient Boosting Regressor with core fundamentals")
        .model_type("gbr")
        # Core features with good coverage
        .add_feature("total_revenue", required=True, transform="log")
        .add_feature("net_income", required=False, transform="log", fill_strategy="zero")
        .add_feature("ebitda", required=False, transform="log", fill_strategy="zero")
        .add_feature("total_debt", required=False, transform="log", fill_strategy="zero")
        .add_feature("total_cash", required=False, transform="log", fill_strategy="zero")
        .add_feature("book_value", required=False, fill_strategy="median")
        .add_feature("free_cash_flow", required=False, fill_strategy="zero")
        # Ratio features
        .add_feature("profit_margins", required=False, fill_strategy="median")
        .add_feature("roe", required=False, fill_strategy="median")
        .add_feature("roa", required=False, fill_strategy="median")
        .param_grid(
            n_estimators=[100, 200],
            max_depth=[3, 5],
            learning_rate=[0.05, 0.1],
            subsample=[0.8],
        )
        .cv_config(n_experiments=100)
        .build()
    )
