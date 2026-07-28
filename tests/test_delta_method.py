"""Tests for delta-method analytical LOQ helpers."""

from __future__ import annotations

import numpy as np
import pytest

from loqculate import PiecewiseCF
from loqculate.config import DEFAULT_KINK_GUARD_FACTOR
from loqculate.models.delta_method import (
    delta_cv_profile,
    delta_loq,
    kink_concentration,
    linear_segment_mse,
    min_concentration_spacing,
    prediction_variance,
)
from loqculate.utils.threshold import find_loq_threshold
from loqculate.utils.weights import inverse_sqrt_weights


def _fit_piecewise_curve(*, noise_sd: float = 0.0, seed: int = 0) -> PiecewiseCF:
    """Return a fitted CF model with a clear noise floor and linear rise."""
    x = np.concatenate([np.repeat([0.1, 0.5, 1.0], 4), np.repeat([2.0, 5.0, 10.0], 4)])
    y = np.where(x <= 1.0, 10.0, 10.0 + 5.0 * (x - 1.0))
    if noise_sd > 0.0:
        y = y + np.random.default_rng(seed).normal(0.0, noise_sd, size=y.shape)
    return PiecewiseCF(n_boot_reps=0, seed=42).fit(x, y)


def test_prediction_variance_matches_manual_formula() -> None:
    """Prediction variance away from the kink matches the hand formula."""
    cf = _fit_piecewise_curve(noise_sd=0.5, seed=1)
    assert cf.params_["slope"] > 0.0

    mse = linear_segment_mse(cf)
    cov = cf.covariance()
    assert mse is not None and mse > 0.0
    assert cov is not None

    x_star = kink_concentration(cf)
    x0 = float(np.max(cf.x_))
    assert abs(x0 - x_star) > 1.0

    w0 = float(inverse_sqrt_weights(np.asarray([x0]))[0])
    x_vec = np.array([x0, 1.0])
    noise_term = mse / (w0**2)
    mean_term = float(x_vec @ cov @ x_vec)
    expected = noise_term + mean_term
    assert noise_term > 0.0

    got = prediction_variance(x0, mse, cov)
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_linear_jacobian_matches_finite_difference() -> None:
    """On the linear branch, predict Jacobian matches [x, 1] within 1e-6."""
    cf = _fit_piecewise_curve()
    x_star = kink_concentration(cf)
    x0 = float(np.max(cf.x_))
    assert abs(x0 - x_star) > 1.0

    a0 = float(cf.params_["slope"])
    b0 = float(cf.params_["intercept_linear"])
    c0 = float(cf.params_["intercept_noise"])
    # Stay well above the noise floor so FD does not hit the max() kink.
    assert a0 * x0 + b0 - c0 > 1.0

    eps = 1e-6
    try:
        cf.params_["slope"] = a0 + eps
        y_a_plus = float(cf.predict(np.asarray([x0]))[0])
        cf.params_["slope"] = a0 - eps
        y_a_minus = float(cf.predict(np.asarray([x0]))[0])
        cf.params_["slope"] = a0

        cf.params_["intercept_linear"] = b0 + eps
        y_b_plus = float(cf.predict(np.asarray([x0]))[0])
        cf.params_["intercept_linear"] = b0 - eps
        y_b_minus = float(cf.predict(np.asarray([x0]))[0])
    finally:
        cf.params_["slope"] = a0
        cf.params_["intercept_linear"] = b0

    jac_fd = np.array([(y_a_plus - y_a_minus) / (2.0 * eps), (y_b_plus - y_b_minus) / (2.0 * eps)])
    jac_an = np.array([x0, 1.0])
    np.testing.assert_allclose(jac_fd, jac_an, rtol=1e-6)


def test_kink_band_sets_cv_inf_far_point_finite() -> None:
    """CV is inf inside the kink band and finite far above the join."""
    cf = _fit_piecewise_curve()
    x_star = kink_concentration(cf)
    assert np.isfinite(x_star)

    # Production grids often start at LOD above the join; force LOD below x*
    # so this unit test actually covers the kink band.
    lod = max(float(np.min(cf.x_)), x_star - 1.0)
    x_grid, cv = delta_cv_profile(cf, n_grid=501, lod=lod)
    assert x_grid.size > 0

    half_band = DEFAULT_KINK_GUARD_FACTOR * min_concentration_spacing(cf.x_)
    in_band = np.abs(x_grid - x_star) < half_band
    assert np.any(in_band)
    assert np.all(np.isinf(cv[in_band]))

    far = np.argmax(x_grid)
    assert not in_band[far]
    assert np.isfinite(cv[far])
    assert cv[far] > 0.0


def test_unfitted_host_raises() -> None:
    """Unfitted host raises RuntimeError."""
    cf = PiecewiseCF(n_boot_reps=0, seed=42)
    with pytest.raises(RuntimeError, match="fitted"):
        delta_loq(cf)


def test_missing_weights_raises() -> None:
    """Missing weights_ raises RuntimeError and does not invent weights."""
    cf = _fit_piecewise_curve()
    cf.weights_ = None
    with pytest.raises(RuntimeError, match="weights_"):
        delta_loq(cf)


def test_fewer_than_three_linear_points_returns_inf() -> None:
    """Natural fit with n_L == 2 returns infinite delta LOQ."""
    x = np.concatenate([np.repeat([1.0, 2.0, 3.0], 8), np.array([4.0, 5.0])])
    y = np.where(x <= 3.0, 10.0, 10.0 + 5.0 * (x - 3.0))
    cf = PiecewiseCF(n_boot_reps=0, seed=42).fit(x, y)

    assert cf.params_["slope"] > 0.0
    assert int(np.sum(cf.x_ > cf.params_["knot_x"])) == 2
    assert cf.covariance() is None
    assert np.isinf(delta_loq(cf))


def test_delta_loq_uses_find_loq_threshold_non_strict() -> None:
    """Delta LOQ matches find_loq_threshold on the same CV grid (<= rule)."""
    cf = _fit_piecewise_curve(noise_sd=0.5, seed=2)
    x_grid, cv = delta_cv_profile(cf, n_grid=200)
    assert x_grid.size > 0

    expected = find_loq_threshold(x_grid, cv, cv_thresh=0.2, window=3)
    got = delta_loq(cf, cv_thresh=0.2, n_grid=200)
    assert np.isfinite(expected), "pre-condition: test curve must yield a finite LOQ"
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=0.0)
