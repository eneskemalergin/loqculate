"""Delta-method analytical LOQ helpers for piecewise linear hosts.

Builds a prediction-variance CV profile on the linear segment and finds LOQ
with :func:`~loqculate.utils.threshold.find_loq_threshold`.  Does not change
``fit()``, ``predict()``, or the default bootstrap ``loq()`` path.

The host must expose ``predict``, ``covariance``, ``lod``, ``weights_``,
``x_``, ``y_``, ``params_`` (slope, intercept_linear, intercept_noise, knot_x),
and ``is_fitted_``.

When ``3 <= n_L < 5``, emits ``UserWarning`` and still computes.  Missing
weights raise; unavailable covariance or undefined MSE yield infinite LOQ.
"""

from __future__ import annotations

import warnings

import numpy as np

from loqculate.config import (
    DEFAULT_CV_THRESH,
    DEFAULT_DELTA_GRID_POINTS,
    DEFAULT_KINK_GUARD_FACTOR,
    DEFAULT_SLIDING_WINDOW,
)
from loqculate.utils.threshold import find_loq_threshold
from loqculate.utils.weights import inverse_sqrt_weights


def kink_concentration(host: object) -> float:
    """Return the piecewise join ``(c - b) / a``, or ``nan`` if undefined."""
    a = float(host.params_["slope"])
    if a <= 0.0:
        return float("nan")
    b = float(host.params_["intercept_linear"])
    c = float(host.params_["intercept_noise"])
    return (c - b) / a


def min_concentration_spacing(x: np.ndarray) -> float:
    """Return the smallest gap between adjacent unique concentrations."""
    ux = np.unique(np.asarray(x, dtype=float))
    if ux.size < 2:
        return 0.0
    return float(np.min(np.diff(ux)))


def linear_segment_mse(host: object) -> float | None:
    """Return weighted linear-segment MSE, or ``None`` when undefined.

    Uses observations with ``x > knot_x`` and denominator ``n_L - 2``.
    """
    _require_fitted_weights(host)
    x = np.asarray(host.x_, dtype=float)
    y = np.asarray(host.y_, dtype=float)
    W = np.asarray(host.weights_, dtype=float) ** 2
    knot_x = float(host.params_["knot_x"])
    a = float(host.params_["slope"])
    b = float(host.params_["intercept_linear"])

    lin_mask = x > knot_x
    n_lin = int(np.sum(lin_mask))
    if n_lin < 3:
        return None

    residuals = y[lin_mask] - (a * x[lin_mask] + b)
    wrss = float(np.sum(W[lin_mask] * residuals**2))
    return wrss / (n_lin - 2)


def prediction_variance(x0: float, mse: float, cov: np.ndarray) -> float:
    """Return delta-method prediction variance at concentration ``x0``.

    Parameters
    ----------
    x0:
        Concentration for a new measurement.
    mse:
        Linear-segment weighted residual MSE.
    cov:
        2x2 parameter covariance with MSE already included.
    """
    w0 = float(inverse_sqrt_weights(np.asarray([x0], dtype=float))[0])
    W0 = w0**2
    x_vec = np.array([x0, 1.0], dtype=float)
    var = mse / W0 + float(x_vec @ cov @ x_vec)
    return max(var, 0.0)


def delta_cv_profile(
    host: object,
    *,
    n_grid: int = DEFAULT_DELTA_GRID_POINTS,
    kink_guard_factor: float = DEFAULT_KINK_GUARD_FACTOR,
    lod: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x_grid, cv)`` from LOD to ``max(x)`` for the delta method.

    Points inside the kink band receive ``cv = inf``.  When the analytical
    path is unavailable, both arrays are empty.  Warns when
    ``3 <= n_L < 5``.
    """
    _require_fitted_weights(host)

    lod_val = float(host.lod()) if lod is None else float(lod)
    x_max = float(np.max(host.x_))
    if not np.isfinite(lod_val) or lod_val >= x_max:
        return np.array([], dtype=float), np.array([], dtype=float)

    mse = linear_segment_mse(host)
    cov = host.covariance()
    if mse is None or cov is None:
        return np.array([], dtype=float), np.array([], dtype=float)

    n_lin = int(np.sum(np.asarray(host.x_, dtype=float) > float(host.params_["knot_x"])))
    if 3 <= n_lin < 5:
        warnings.warn(
            f"Delta-method LOQ is based on only {n_lin} linear-segment points; "
            "residual degrees of freedom are low (n_L - 2).",
            UserWarning,
            stacklevel=2,
        )

    x_grid = np.linspace(lod_val, x_max, num=int(n_grid))
    y_hat = np.asarray(host.predict(x_grid), dtype=float)

    x_star = kink_concentration(host)
    spacing = min_concentration_spacing(host.x_)
    half_band = kink_guard_factor * spacing

    w0 = inverse_sqrt_weights(x_grid)
    W0 = w0**2
    cov = np.asarray(cov, dtype=float)
    quad = cov[0, 0] * x_grid**2 + 2.0 * cov[0, 1] * x_grid + cov[1, 1]
    var = np.maximum(float(mse) / W0 + quad, 0.0)

    cv = np.full(x_grid.shape, np.inf, dtype=float)
    usable = np.isfinite(y_hat) & (y_hat != 0.0)
    if np.isfinite(x_star):
        usable &= np.abs(x_grid - x_star) >= half_band
    cv[usable] = np.sqrt(var[usable]) / y_hat[usable]

    return x_grid, cv


def delta_loq(
    host: object,
    cv_thresh: float = DEFAULT_CV_THRESH,
    n_grid: int = DEFAULT_DELTA_GRID_POINTS,
    *,
    kink_guard_factor: float = DEFAULT_KINK_GUARD_FACTOR,
    window: int = DEFAULT_SLIDING_WINDOW,
) -> float:
    """Return analytical LOQ from the delta-method CV profile, or ``inf``.

    Raises
    ------
    RuntimeError
        If the host is not fitted or ``weights_`` is missing.
    """
    x_grid, cv = delta_cv_profile(host, n_grid=n_grid, kink_guard_factor=kink_guard_factor)
    if x_grid.size == 0:
        return float(np.inf)

    loq_val = find_loq_threshold(x_grid, cv, cv_thresh=cv_thresh, window=window)
    if not np.isfinite(loq_val) or loq_val <= 0.0 or loq_val >= float(np.max(host.x_)):
        return float(np.inf)
    return float(loq_val)


def _require_fitted_weights(host: object) -> None:
    if not getattr(host, "is_fitted_", False):
        raise RuntimeError("Host must be fitted before delta-method LOQ.")
    if getattr(host, "weights_", None) is None:
        raise RuntimeError("Host weights_ are missing; refuse to invent weights.")
