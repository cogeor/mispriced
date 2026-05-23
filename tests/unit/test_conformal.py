"""Unit tests for the Conformalized Quantile Regression (CV+) module.

Synthetic-data tests that pin down:

- Marginal-coverage guarantee on homoscedastic and heteroscedastic noise.
- Width monotonicity under heteroscedasticity (intervals widen with noise).
- Sensible behavior on edge cases (constant target, small samples, degenerate
  no-signal input).
- Input validation (alpha range, quantile ordering, non-finite entries, too
  few samples).
- Reproducibility under fixed seeds and sensitivity to seed changes.

All tests use ``np.random.default_rng(seed=...)`` to keep them deterministic and
``_fast_params()`` to keep the suite wall-clock under ~30s.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest
from scipy.stats import spearmanr

from src.valuation.conformal import ConformalQuantileRegressor


def _fast_params() -> Dict[str, Any]:
    """Return XGBoost params tuned for fast unit-test execution.

    50 trees and depth 4 are plenty to recover the synthetic signals used in
    this suite while keeping the full file under ~30 seconds.
    """
    return {
        "n_estimators": 50,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
    }


class TestCoverage:
    """Marginal coverage behavior on synthetic data."""

    def test_coverage_on_homoscedastic_gaussian(self) -> None:
        """90% target intervals cover roughly 90% on clean Gaussian noise."""
        rng = np.random.default_rng(0)
        n, p = 2000, 3
        X = rng.standard_normal((n, p))
        y = X @ np.array([1.0, -0.5, 0.3]) + rng.normal(0.0, 0.5, n)

        out = ConformalQuantileRegressor(
            alpha=0.1, n_folds=5, model_params=_fast_params()
        ).fit_predict(X, y)

        assert out["empirical_coverage"] >= 0.85
        assert 0.80 <= out["empirical_coverage"] <= 0.98
        assert np.all(out["upper"] > out["lower"])
        assert np.isfinite(out["q"])
        assert out["q"] > 0

    def test_coverage_on_heteroscedastic(self) -> None:
        """Coverage holds and interval width tracks the true noise level."""
        rng = np.random.default_rng(1)
        n = 2000
        X = rng.uniform(-3.0, 3.0, (n, 1))
        sigma = 0.2 + 0.5 * np.abs(X[:, 0])
        y = (X[:, 0] ** 2) + rng.normal(0.0, sigma)

        out = ConformalQuantileRegressor(
            alpha=0.1, n_folds=5, model_params=_fast_params()
        ).fit_predict(X, y)

        assert out["empirical_coverage"] >= 0.85

        widths = out["upper"] - out["lower"]
        rho, _ = spearmanr(widths, sigma)
        assert rho > 0.3, f"Expected width-sigma correlation > 0.3; got {rho:.3f}"

    def test_marginal_coverage_holds_under_noise_only(self) -> None:
        """Even with no signal, intervals still hit the marginal coverage target."""
        rng = np.random.default_rng(2)
        n, p = 1000, 3
        X = rng.standard_normal((n, p))
        y = rng.standard_normal(n)

        out = ConformalQuantileRegressor(
            alpha=0.1, n_folds=5, model_params=_fast_params()
        ).fit_predict(X, y)

        assert out["empirical_coverage"] >= 0.85


class TestEdgeCases:
    """Behavior on degenerate, small-sample, and invalid inputs."""

    def test_intervals_sane_on_constant_y(self) -> None:
        """Constant target collapses intervals to near-zero width with full coverage."""
        rng = np.random.default_rng(3)
        n, p = 500, 3
        X = rng.standard_normal((n, p))
        y = np.full(n, 7.0)

        out = ConformalQuantileRegressor(
            alpha=0.1, n_folds=5, model_params=_fast_params()
        ).fit_predict(X, y)

        widths = out["upper"] - out["lower"]
        assert widths.max() < 0.5
        assert out["empirical_coverage"] == 1.0

    def test_handles_small_sample(self) -> None:
        """Algorithm runs and returns finite outputs on n=30 (no coverage claim)."""
        rng = np.random.default_rng(4)
        n, p = 30, 3
        X = rng.standard_normal((n, p))
        y = X.sum(axis=1) + rng.normal(0.0, 0.5, n)

        out = ConformalQuantileRegressor(
            alpha=0.1, n_folds=5, model_params=_fast_params()
        ).fit_predict(X, y)

        assert np.all(np.isfinite(out["lower"]))
        assert np.all(np.isfinite(out["upper"]))
        assert np.isfinite(out["q"])

    def test_raises_on_too_few_samples(self) -> None:
        """n_samples < n_folds raises a ValueError mentioning the fold count."""
        X = np.zeros((3, 2))
        y = np.zeros(3)
        with pytest.raises(ValueError, match="n_folds"):
            ConformalQuantileRegressor(n_folds=5).fit_predict(X, y)

    def test_raises_on_nan_input(self) -> None:
        """A single NaN in X is caught and reported."""
        rng = np.random.default_rng(5)
        n, p = 50, 3
        X = rng.standard_normal((n, p))
        X[0, 0] = np.nan
        y = rng.standard_normal(n)
        with pytest.raises(ValueError, match="finite"):
            ConformalQuantileRegressor().fit_predict(X, y)

    def test_raises_on_bad_alpha(self) -> None:
        """alpha must be strictly inside (0, 1)."""
        with pytest.raises(ValueError):
            ConformalQuantileRegressor(alpha=0.0)
        with pytest.raises(ValueError):
            ConformalQuantileRegressor(alpha=1.0)

    def test_raises_on_inverted_quantiles(self) -> None:
        """quantile_low must be strictly less than quantile_high."""
        with pytest.raises(ValueError):
            ConformalQuantileRegressor(quantile_low=0.9, quantile_high=0.1)


class TestAlphaParameter:
    """alpha controls the coverage / width trade-off in the expected direction."""

    def test_smaller_alpha_gives_wider_intervals(self) -> None:
        """Tighter coverage target (alpha=0.05) -> wider intervals + higher coverage."""
        rng = np.random.default_rng(6)
        n = 1000
        X = rng.uniform(-2.0, 2.0, (n, 1))
        sigma = 0.3 + 0.2 * np.abs(X[:, 0])
        y = X[:, 0] + rng.normal(0.0, sigma)

        out_05 = ConformalQuantileRegressor(
            alpha=0.05, n_folds=5, random_state=42, model_params=_fast_params()
        ).fit_predict(X, y)
        out_20 = ConformalQuantileRegressor(
            alpha=0.20, n_folds=5, random_state=42, model_params=_fast_params()
        ).fit_predict(X, y)

        width_05 = float(np.mean(out_05["upper"] - out_05["lower"]))
        width_20 = float(np.mean(out_20["upper"] - out_20["lower"]))

        assert width_05 > width_20
        assert out_05["empirical_coverage"] > out_20["empirical_coverage"]


class TestReproducibility:
    """Identical seeds produce identical output; different seeds diverge."""

    def test_same_seed_same_result(self) -> None:
        """Two fits with the same random_state are bit-identical."""
        rng = np.random.default_rng(7)
        n, p = 500, 3
        X = rng.standard_normal((n, p))
        y = X.sum(axis=1) + rng.normal(0.0, 0.5, n)

        out_a = ConformalQuantileRegressor(
            alpha=0.1, n_folds=5, random_state=42, model_params=_fast_params()
        ).fit_predict(X, y)
        out_b = ConformalQuantileRegressor(
            alpha=0.1, n_folds=5, random_state=42, model_params=_fast_params()
        ).fit_predict(X, y)

        assert np.array_equal(out_a["lower"], out_b["lower"])
        assert np.array_equal(out_a["upper"], out_b["upper"])
        assert out_a["q"] == out_b["q"]

    def test_different_seed_different_q(self) -> None:
        """Changing the seed perturbs the calibration offset."""
        rng = np.random.default_rng(8)
        n, p = 500, 3
        X = rng.standard_normal((n, p))
        y = X.sum(axis=1) + rng.normal(0.0, 0.5, n)

        out_a = ConformalQuantileRegressor(
            alpha=0.1, n_folds=5, random_state=42, model_params=_fast_params()
        ).fit_predict(X, y)
        out_b = ConformalQuantileRegressor(
            alpha=0.1, n_folds=5, random_state=7, model_params=_fast_params()
        ).fit_predict(X, y)

        assert out_a["q"] != out_b["q"]
