"""Backtest module for validating index overpriciness signals."""

from .constants import (
    SIGNAL_FORMATION_LAG_DAYS,
    WINSORIZE_LOWER_PCT,
    WINSORIZE_UPPER_PCT,
)
from .models import BacktestPoint, BacktestResult, CorrelationMetrics, BacktestConfig
from .service import BacktestService
from .index_prices import get_index_price, get_index_return, INDEX_SYMBOLS

__all__ = [
    "BacktestPoint",
    "BacktestResult",
    "CorrelationMetrics",
    "BacktestConfig",
    "BacktestService",
    "get_index_price",
    "get_index_return",
    "INDEX_SYMBOLS",
    "SIGNAL_FORMATION_LAG_DAYS",
    "WINSORIZE_LOWER_PCT",
    "WINSORIZE_UPPER_PCT",
]
