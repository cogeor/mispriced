"""Unit tests for signal_metrics module."""

import numpy as np

from src.backtest.signal_metrics import compute_hit_rate


class TestComputeHitRate:
    """Tests for the directional hit-rate calculation."""

    def test_perfect_agreement_returns_one(self) -> None:
        """All signal/return sign pairs agree -> hit rate of 1.0."""
        signal = np.array([1.0, 2.0, -1.0, -2.0, 3.0])
        returns = np.array([0.1, 0.2, -0.1, -0.2, 0.3])

        hit_rate, _ = compute_hit_rate(signal, returns)

        assert hit_rate == 1.0

    def test_perfect_disagreement_returns_zero(self) -> None:
        """All signal/return sign pairs disagree -> hit rate of 0.0."""
        signal = np.array([1.0, 2.0, -1.0, -2.0, 3.0])
        returns = np.array([-0.1, -0.2, 0.1, 0.2, -0.3])

        hit_rate, _ = compute_hit_rate(signal, returns)

        assert hit_rate == 0.0

    def test_random_signal_near_half(self) -> None:
        """Random signal/return pairs should hover around 0.5."""
        rng = np.random.default_rng(seed=0)
        signal = rng.standard_normal(200)
        returns = rng.standard_normal(200)

        hit_rate, _ = compute_hit_rate(signal, returns)

        assert 0.4 <= hit_rate <= 0.6

    def test_nan_pairs_excluded(self) -> None:
        """NaN entries (in either input) are dropped before counting.

        Uses 7 observations with 2 NaN-bearing pairs so that 5 valid pairs
        remain after masking (satisfying the n>=5 guard). The NaN rows pair
        a positive signal with a negative return and vice-versa; if they
        weren't dropped they'd produce hits == 0 / 7 (0.0). After dropping
        them all 5 remaining pairs agree -> hit rate of 1.0.
        """
        signal = np.array([1.0, np.nan, -1.0, 2.0, -2.0, 3.0, -3.0])
        returns = np.array([0.1, -0.2, -0.1, 0.2, -0.2, 0.3, np.nan])

        hit_rate, _ = compute_hit_rate(signal, returns)

        assert hit_rate == 1.0

    def test_below_min_observations_returns_nan(self) -> None:
        """Fewer than 5 valid observations returns (nan, 1.0)."""
        signal = np.array([1.0, -1.0, 2.0, -2.0])
        returns = np.array([0.1, -0.1, 0.2, -0.2])

        hit_rate, pvalue = compute_hit_rate(signal, returns)

        assert np.isnan(hit_rate)
        assert pvalue == 1.0

    def test_zero_signal_or_zero_return_is_miss(self) -> None:
        """Strict inequality: zero on either side counts as a miss."""
        signal = np.array([0.0, 1.0, -1.0, 2.0, -2.0])
        returns = np.array([0.1, 0.1, -0.1, 0.0, -0.1])

        # Hits: (1, 0.1), (-1, -0.1), (-2, -0.1) -> 3 of 5
        # Misses: (0, 0.1) and (2, 0.0) both fail strict-sign agreement
        hit_rate, _ = compute_hit_rate(signal, returns)

        assert hit_rate == 0.6
