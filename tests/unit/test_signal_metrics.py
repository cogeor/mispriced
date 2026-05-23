"""Unit tests for signal_metrics module."""

from datetime import date, timedelta
from typing import Dict, List

import numpy as np
import pytest

from src.backtest.signal_metrics import (
    compute_hit_rate,
    compute_ic_decay,
    compute_ic_tstat,
    winsorize,
)


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


class TestWinsorize:
    """Tests for the percentile-clip helper."""

    def test_clips_upper_tail_to_99th_percentile(self) -> None:
        """An extreme positive outlier is clipped down to the 99th percentile."""
        arr = np.array([*range(1, 100), 10000.0])
        clipped = winsorize(arr)

        assert clipped.max() < 10000.0
        assert clipped.max() == pytest.approx(np.nanpercentile(arr, 99))

    def test_clips_lower_tail_to_1st_percentile(self) -> None:
        """An extreme negative outlier is clipped up to the 1st percentile."""
        arr = np.array([-10000.0, *range(1, 100)], dtype=float)
        clipped = winsorize(arr)

        assert clipped.min() > -10000.0
        assert clipped.min() == pytest.approx(np.nanpercentile(arr, 1))

    def test_small_cohort_returns_unchanged(self) -> None:
        """Input with fewer than 5 finite values is returned unchanged."""
        arr = np.array([1.0, 2.0, 3.0])
        clipped = winsorize(arr)

        assert np.array_equal(clipped, arr)
        # Must be a copy, not the same object (defensive contract).
        assert clipped is not arr

    def test_preserves_nans(self) -> None:
        """NaNs in the input remain in the same positions in the output."""
        arr = np.array([1.0, np.nan, 2.0, 3.0, np.nan, 4.0, 5.0, 6.0, 7.0, 100.0])
        clipped = winsorize(arr)

        # NaN positions preserved
        assert np.isnan(clipped[1])
        assert np.isnan(clipped[4])
        # Finite positions remain finite
        assert np.isfinite(clipped[0])
        assert np.isfinite(clipped[-1])

    def test_synthetic_ret_10_clipped_to_99th(self) -> None:
        """A cohort of 99 normal returns plus one 10.0 returns 100 values
        with the 10.0 clipped to <= the 99th percentile (not dropped)."""
        rng = np.random.default_rng(seed=42)
        normals = rng.uniform(-0.5, 0.5, size=99)
        arr = np.concatenate([normals, [10.0]])
        clipped = winsorize(arr)

        # Same length (nothing dropped)
        assert clipped.shape == arr.shape
        # The 10.0 has been pulled down to the 99th percentile of the cohort
        assert clipped[-1] <= np.nanpercentile(arr, 99) + 1e-9
        assert clipped[-1] < 10.0

    def test_all_identical_returns_unchanged(self) -> None:
        """Degenerate cohort (all identical values, lo == hi) returns a copy."""
        arr = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
        clipped = winsorize(arr)

        assert np.array_equal(clipped, arr)


def _make_prices(
    tickers: List[str],
    base_date: date,
    horizon: int,
    returns: List[float],
) -> Dict[str, Dict[date, float]]:
    """Build a {ticker: {date: price}} dict so each ticker's
    ``(base_date -> base_date+horizon)`` return equals the requested value."""
    start = 100.0
    end_date = base_date + timedelta(days=horizon)
    return {
        t: {base_date: start, end_date: start * (1.0 + r)}
        for t, r in zip(tickers, returns)
    }


class TestComputeICDecayWinsorization:
    """Lock in the post-fix behavior of ``compute_ic_decay``.

    Old behavior: rows where ``abs(ret) >= 2.0`` were silently dropped, which
    biased the IC toward zero by removing the high-information tails. New
    behavior: extreme rows are kept but winsorized to the cohort's [1, 99]
    percentile band before the rank correlation is computed.
    """

    def test_extreme_return_no_longer_dropped(self) -> None:
        """A row with ``ret = 1.5`` survives instead of being filtered out."""
        base_date = date(2024, 1, 1)
        horizon = 10
        tickers = [f"T{i}" for i in range(12)]
        # 11 normal returns + one extreme (+150%)
        normal_returns = [
            -0.05, -0.04, -0.03, -0.02, -0.01,
            0.0, 0.01, 0.02, 0.03, 0.04, 0.05,
        ]
        returns = normal_returns + [1.5]
        signal = np.array(
            list(range(11)) + [11], dtype=float
        )  # monotonically increasing -> high IC

        prices = _make_prices(tickers, base_date, horizon, returns)
        decay = compute_ic_decay(
            signal=signal,
            prices=prices,
            tickers=tickers,
            base_date=base_date,
            horizons=[horizon],
        )

        # The extreme row was kept (n_obs >= 12 satisfied), so IC is computed.
        assert len(decay.ic_values) == 1
        assert not np.isnan(decay.ic_values[0])

    def test_extreme_return_clipped_not_dropped(self) -> None:
        """A ``ret = 10.0`` row is winsorized into the cohort rather than dropped."""
        base_date = date(2024, 1, 1)
        horizon = 10
        tickers = [f"T{i}" for i in range(12)]
        normal_returns = [
            -0.05, -0.04, -0.03, -0.02, -0.01,
            0.0, 0.01, 0.02, 0.03, 0.04, 0.05,
        ]
        returns = normal_returns + [10.0]
        signal = np.array(list(range(11)) + [11], dtype=float)

        prices = _make_prices(tickers, base_date, horizon, returns)
        decay = compute_ic_decay(
            signal=signal,
            prices=prices,
            tickers=tickers,
            base_date=base_date,
            horizons=[horizon],
        )

        # IC is finite (row survived, not dropped to <10 obs).
        assert len(decay.ic_values) == 1
        assert np.isfinite(decay.ic_values[0])
        # Because Spearman is rank-based and the extreme value is the largest
        # whether it's 10.0 or its clipped value, the IC is high and positive
        # (perfectly monotonic signal vs. returns).
        assert decay.ic_values[0] > 0.9


class TestComputeICTstat:
    """Tests for the Grinold-Kahn-style cross-quarter IC t-statistic."""

    def test_stable_ic_yields_large_tstat(self) -> None:
        """A stable IC of 0.05 across 9 quarters yields |t| >> 2.

        With near-zero variance the SE is tiny, so even a 0.05 mean is
        highly significant. We tolerate floating noise by demanding |t|>5.
        """
        ic_values = [0.05 + 1e-6 * i for i in range(9)]  # near-flat
        tstat = compute_ic_tstat(ic_values)

        assert np.isfinite(tstat)
        assert abs(tstat) > 5.0

    def test_alternating_sign_yields_near_zero(self) -> None:
        """An IC series symmetric about zero produces |t| ~ 0."""
        ic_values = [0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05]
        tstat = compute_ic_tstat(ic_values)

        assert np.isfinite(tstat)
        assert abs(tstat) < 0.5

    def test_single_quarter_returns_nan(self) -> None:
        """One observation has no SE -> nan."""
        tstat = compute_ic_tstat([0.1])

        assert np.isnan(tstat)

    def test_nan_values_filtered_before_count(self) -> None:
        """NaNs in the input do not count toward n; if only one finite
        value remains, result is nan (matches single-quarter contract)."""
        tstat = compute_ic_tstat([0.1, np.nan, np.nan])

        assert np.isnan(tstat)

    def test_zero_variance_returns_nan(self) -> None:
        """Identical IC values across quarters -> SE = 0 -> nan."""
        tstat = compute_ic_tstat([0.05, 0.05, 0.05, 0.05])

        assert np.isnan(tstat)
