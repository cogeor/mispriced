"""Conformalized Quantile Regression (CQR) with CV+ calibration.

This module implements the CV+ variant of Conformalized Quantile Regression for
producing distribution-free prediction intervals with a finite-sample marginal
coverage guarantee on exchangeable data.

The method combines:

- **Conformalized Quantile Regression** (Romano, Patterson, Candes 2019,
  "Conformalized Quantile Regression"): wraps a pair of quantile regressors with
  a conformal calibration step so the resulting intervals are guaranteed to cover
  the true target with probability >= 1 - alpha, marginally.
- **CV+** (Barber, Candes, Ramdas, Tibshirani 2021, "Predictive inference with
  the jackknife+"): replaces the split-conformal hold-out set with K-fold
  out-of-fold (OOF) residuals so every training sample contributes to both the
  fitted models and the calibration distribution.

The module is intentionally pure: no database access, no pipeline coupling, no
currency or feature-engineering logic. It takes a feature matrix and a target
vector (in log-mcap space) and returns per-sample lower/upper bounds plus
calibration diagnostics. Downstream pipeline code (a later loop) is responsible
for converting bounds back to raw market-cap units and persisting them.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


# Default XGBoost hyperparameters for the quantile regressors. Mirrors the
# defaults documented in web/public/methodology.html section 5 (Mispricing
# Calculation) so CQR intervals are comparable to the existing point model.
DEFAULT_MODEL_PARAMS: Dict[str, Any] = {
    "max_depth": 5,
    "n_estimators": 200,
    "learning_rate": 0.1,
    "subsample": 0.8,
}


class ConformalQuantileRegressor:
    """CV+ Conformalized Quantile Regression for log-space targets.

    Trains two XGBoost quantile regressors (at ``quantile_low`` and
    ``quantile_high``) under K-fold cross-validation, pools the out-of-fold
    nonconformity scores into a single calibration distribution, and applies the
    finite-sample conservative ``(1 - alpha)`` quantile correction to widen the
    raw quantile predictions into intervals with a marginal coverage guarantee.

    The marginal coverage guarantee
        P(y in [lower, upper]) >= 1 - alpha
    holds under the exchangeability of the training samples. Within a single
    quarter (the intended use site) this is reasonable; across quarters the
    assumption breaks and intervals must be re-fit per quarter.

    Parameters
    ----------
    alpha : float, default=0.1
        Miscoverage target. Intervals aim for ``1 - alpha`` empirical coverage.
        Must satisfy ``0 < alpha < 1``.
    n_folds : int, default=5
        Number of folds for the CV+ split. Must be at least 2.
    random_state : int, default=42
        Seed for both the K-fold shuffle and the underlying XGBoost regressors.
    quantile_low : float, default=0.05
        Target quantile for the lower-bound XGBoost regressor.
    quantile_high : float, default=0.95
        Target quantile for the upper-bound XGBoost regressor. Must be strictly
        greater than ``quantile_low``.
    model_params : dict, optional
        Extra XGBoost hyperparameters; merged on top of ``DEFAULT_MODEL_PARAMS``
        with caller-provided keys winning. The objective, ``quantile_alpha``,
        and ``random_state`` are always injected by the class and cannot be
        overridden through this argument (they are algorithmic, not tuning,
        choices).

    Raises
    ------
    ValueError
        If ``alpha`` is outside ``(0, 1)``, ``n_folds < 2``, or the quantile
        ordering is invalid.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        n_folds: int = 5,
        random_state: int = 42,
        quantile_low: float = 0.05,
        quantile_high: float = 0.95,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(
                f"alpha must be in (0, 1); got {alpha!r}"
            )
        if n_folds < 2:
            raise ValueError(
                f"n_folds must be >= 2; got {n_folds!r}"
            )
        if not (0.0 < quantile_low < quantile_high < 1.0):
            raise ValueError(
                "quantile_low and quantile_high must satisfy "
                f"0 < quantile_low < quantile_high < 1; got "
                f"quantile_low={quantile_low!r}, quantile_high={quantile_high!r}"
            )

        self.alpha = alpha
        self.n_folds = n_folds
        self.random_state = random_state
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high
        # Copy-then-merge so the module-level DEFAULT_MODEL_PARAMS is never
        # mutated and caller-provided keys override the defaults.
        self._model_params: Dict[str, Any] = DEFAULT_MODEL_PARAMS | (model_params or {})

    def _make_regressor(self, quantile_alpha: float) -> XGBRegressor:
        """Instantiate an XGBoost quantile regressor for a given target quantile.

        Centralises construction so the objective and per-fit ``quantile_alpha``
        cannot be overridden via the public ``model_params`` argument.

        Parameters
        ----------
        quantile_alpha : float
            The target quantile (e.g. 0.05 for the lower band, 0.95 for the
            upper band).

        Returns
        -------
        XGBRegressor
            A fresh, unfitted XGBoost regressor configured for the requested
            quantile.
        """
        return XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=quantile_alpha,
            random_state=self.random_state,
            **self._model_params,
        )

    def fit_predict(self, X: np.ndarray, y_log: np.ndarray) -> Dict[str, Any]:
        """Fit the CV+ CQR procedure and return per-sample intervals.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix. Must be 2-D and finite.
        y_log : np.ndarray, shape (n_samples,)
            Target vector in log-mcap space. Must be 1-D, finite, and share its
            leading dimension with ``X``.

        Returns
        -------
        dict
            Dictionary with the following keys:

            - ``lower`` : np.ndarray, shape (n_samples,)
                Calibrated lower bound per training sample in log space.
            - ``upper`` : np.ndarray, shape (n_samples,)
                Calibrated upper bound per training sample in log space.
            - ``y_hat_lo`` : np.ndarray, shape (n_samples,)
                Raw OOF lower-quantile predictions (pre-calibration).
            - ``y_hat_hi`` : np.ndarray, shape (n_samples,)
                Raw OOF upper-quantile predictions (pre-calibration).
            - ``q`` : float
                Conformal calibration offset applied in log space.
            - ``empirical_coverage`` : float
                Fraction of ``y_log`` falling inside ``[lower, upper]``.
            - ``alpha`` : float
                The miscoverage target echoed from the constructor.
            - ``n_calibration`` : int
                Size of the pooled nonconformity-score set. Equal to
                ``n_samples`` for CV+.

        Raises
        ------
        ValueError
            If shapes are inconsistent, ``n_samples < n_folds``, or any entry
            of ``X`` / ``y_log`` is non-finite.
        """
        X = np.asarray(X)
        y_log = np.asarray(y_log)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D; got shape {X.shape}")
        if y_log.ndim != 1:
            raise ValueError(f"y_log must be 1-D; got shape {y_log.shape}")
        if X.shape[0] != y_log.shape[0]:
            raise ValueError(
                f"X and y_log must share the leading dimension; got "
                f"X.shape={X.shape}, y_log.shape={y_log.shape}"
            )

        n_samples = X.shape[0]
        if n_samples < self.n_folds:
            raise ValueError(
                f"n_samples ({n_samples}) must be >= n_folds ({self.n_folds})"
            )
        if not np.all(np.isfinite(y_log)) or not np.all(np.isfinite(X)):
            raise ValueError("X and y_log must be finite (no NaN/Inf)")

        y_hat_lo = np.empty(n_samples, dtype=float)
        y_hat_hi = np.empty(n_samples, dtype=float)

        kf = KFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        for train_idx, test_idx in kf.split(X):
            lo_model = self._make_regressor(self.quantile_low)
            hi_model = self._make_regressor(self.quantile_high)

            lo_model.fit(X[train_idx], y_log[train_idx])
            hi_model.fit(X[train_idx], y_log[train_idx])

            y_hat_lo[test_idx] = lo_model.predict(X[test_idx])
            y_hat_hi[test_idx] = hi_model.predict(X[test_idx])

        # Pooled nonconformity scores across all folds (CV+: every sample
        # contributes exactly once as held-out).
        s = np.maximum(y_hat_lo - y_log, y_log - y_hat_hi)

        # The "higher" method is the small-sample conservative choice: when the
        # requested percentile falls between two scores, snap to the upper one.
        # This is the finite-sample correction that gives the marginal coverage
        # guarantee for CV+.
        q = float(np.quantile(s, 1.0 - self.alpha, method="higher"))

        lower = y_hat_lo - q
        upper = y_hat_hi + q
        empirical_coverage = float(np.mean((y_log >= lower) & (y_log <= upper)))

        logger.info(
            "CQR fit_predict: n=%d, alpha=%.3f, q=%.4f, empirical_coverage=%.4f",
            n_samples,
            self.alpha,
            q,
            empirical_coverage,
        )

        return {
            "lower": lower,
            "upper": upper,
            "y_hat_lo": y_hat_lo,
            "y_hat_hi": y_hat_hi,
            "q": q,
            "empirical_coverage": empirical_coverage,
            "alpha": float(self.alpha),
            "n_calibration": int(s.shape[0]),
        }
