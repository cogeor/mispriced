"""Shared constants for the backtest module."""

from typing import Final

SIGNAL_FORMATION_LAG_DAYS: Final[int] = 45
"""Days between fiscal period end (snapshot_timestamp) and the assumed
public-information date. The SEC requires 10-Q within 40 days for large
accelerated filers and 45 days for accelerated filers; 45 is the
conservative upper bound and is used as the forward-return start offset
so the backtest does not implicitly trade on pre-release fundamentals.
"""

WINSORIZE_LOWER_PCT: Final[float] = 1.0
"""Lower percentile cutoff for per-quarter tail winsorization."""

WINSORIZE_UPPER_PCT: Final[float] = 99.0
"""Upper percentile cutoff for per-quarter tail winsorization."""
