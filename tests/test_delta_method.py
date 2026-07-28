"""Tests for delta-method analytical LOQ helpers."""

from __future__ import annotations

import warnings
from unittest.mock import patch

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


def test_loq_delta_twice_identical() -> None:
    """Two loq_delta() calls on the same fit return identical floats (no seed)."""
    cf = _fit_piecewise_curve(noise_sd=0.5, seed=2)
    a = cf.loq_delta()
    b = cf.loq_delta()
    assert np.isfinite(a), "pre-condition: synthetic curve must yield finite delta LOQ"
    assert a == b


def test_loq_method_delta_matches_loq_delta() -> None:
    """loq(method='delta') equals loq_delta() on the same fit."""
    cf = _fit_piecewise_curve(noise_sd=0.5, seed=2)
    via_method = cf.loq(method="delta")
    via_helper = cf.loq_delta()
    assert np.isfinite(via_method), "pre-condition: synthetic curve must yield finite delta LOQ"
    assert via_method == via_helper


def test_loq_delta_does_not_run_bootstrap() -> None:
    """Delta path leaves bootstrap state untouched."""
    cf = _fit_piecewise_curve(noise_sd=0.5, seed=2)

    cf.loq_delta()
    assert cf._boot_summary is None
    assert cf._x_grid is None

    cf.loq(method="delta")
    assert cf._boot_summary is None
    assert cf._x_grid is None


def test_loq_cache_isolates_bootstrap_and_delta() -> None:
    """Bootstrap and delta LOQ cache entries do not overwrite each other."""
    cf = _fit_piecewise_curve(noise_sd=0.5, seed=2)
    delta = cf.loq(method="delta")
    boot = cf.loq()  # n_boot_reps=0 -> inf

    assert np.isfinite(delta), "pre-condition: synthetic curve must yield finite delta LOQ"
    assert np.isinf(boot)
    assert ("delta", 0.2) in cf._loq_cache
    assert ("bootstrap", 0.2) in cf._loq_cache
    assert cf._loq_cache[("delta", 0.2)] == delta
    assert cf._loq_cache[("bootstrap", 0.2)] == boot
    assert cf.loq(method="delta") == delta
    assert cf.loq() == boot

    # Reverse order on a fresh fit: bootstrap first must not poison delta.
    cf2 = _fit_piecewise_curve(noise_sd=0.5, seed=2)
    boot2 = cf2.loq()
    delta2 = cf2.loq(method="delta")
    assert np.isinf(boot2)
    assert delta2 == delta
    assert cf2.loq() == boot2


def test_summary_loq_stays_bootstrap() -> None:
    """summary()['loq'] uses default bootstrap loq(), not loq_delta()."""
    cf = _fit_piecewise_curve(noise_sd=0.5, seed=2)
    delta = cf.loq_delta()
    assert np.isfinite(delta), "pre-condition: synthetic curve must yield finite delta LOQ"

    summary_loq = cf.summary()["loq"]
    assert summary_loq == cf.loq()
    assert np.isinf(summary_loq)
    assert summary_loq != delta


def test_loq_unknown_method_raises() -> None:
    """Unknown loq method raises ValueError."""
    cf = _fit_piecewise_curve()
    with pytest.raises(ValueError, match="bootstrap.*delta"):
        cf.loq(method="jackknife")


def _fit_with_n_lin(n_lin: int) -> PiecewiseCF:
    """Fit a curve whose linear segment has exactly ``n_lin`` observations.

    Noise-floor and linear levels are spaced so knot FP roundoff cannot pull
    floor concentrations into ``x > knot_x``.
    """
    if n_lin < 1:
        raise ValueError("n_lin must be >= 1")
    noise_x = np.repeat([0.1, 1.0], 12)
    lin_x = 10.0 + 10.0 * np.arange(n_lin, dtype=float)
    x = np.concatenate([noise_x, lin_x])
    y = np.where(x <= 1.0, 5.0, 5.0 + (x - 1.0))
    cf = PiecewiseCF(n_boot_reps=0, seed=42).fit(x, y)
    assert int(np.sum(cf.x_ > cf.params_["knot_x"])) == n_lin
    return cf


def test_small_n_lin_warns_and_still_computes() -> None:
    """3 <= n_L < 5 emits UserWarning and still computes a usable LOQ."""
    for n_lin in (3, 4):
        cf = _fit_with_n_lin(n_lin)
        with pytest.warns(UserWarning, match="linear-segment points"):
            loq = delta_loq(cf)
        assert np.isfinite(loq), "thin linear segment must still compute, not refuse"


def test_n_lin_at_least_five_does_not_warn() -> None:
    """n_L >= 5 computes without the small-n UserWarning."""
    cf = _fit_with_n_lin(5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        loq = delta_loq(cf)
    small_n = [w for w in caught if "linear-segment points" in str(w.message)]
    assert not small_n
    assert np.isfinite(loq)


def test_n_lin_below_three_does_not_warn() -> None:
    """n_L < 3 returns inf without the small-n warning (path refuses to compute)."""
    x = np.concatenate([np.repeat([1.0, 2.0, 3.0], 8), np.array([4.0, 5.0])])
    y = np.where(x <= 3.0, 10.0, 10.0 + 5.0 * (x - 3.0))
    cf = PiecewiseCF(n_boot_reps=0, seed=42).fit(x, y)
    assert int(np.sum(cf.x_ > cf.params_["knot_x"])) == 2

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        loq = delta_loq(cf)
    small_n = [w for w in caught if "linear-segment points" in str(w.message)]
    assert not small_n
    assert np.isinf(loq)


def test_missing_covariance_returns_inf_even_when_mse_defined() -> None:
    """None from covariance() yields inf LOQ even when MSE is defined."""
    cf = _fit_piecewise_curve(noise_sd=0.5, seed=2)
    assert linear_segment_mse(cf) is not None
    assert np.isfinite(delta_loq(cf)), "pre-condition: healthy fit must yield finite delta LOQ"

    with patch.object(cf, "covariance", return_value=None):
        assert linear_segment_mse(cf) is not None
        assert np.isinf(delta_loq(cf))


def test_infinite_lod_returns_inf_delta_loq() -> None:
    """Flat curve with infinite LOD yields infinite delta LOQ."""
    x = np.repeat([0.1, 1.0, 10.0], 6)
    y = np.full_like(x, 10.0)
    cf = PiecewiseCF(n_boot_reps=0, seed=42).fit(x, y)
    assert np.isinf(cf.lod())
    assert np.isinf(delta_loq(cf))


def test_zero_prediction_sets_cv_inf() -> None:
    """Zero predicted signal sets CV to inf without divide-by-zero."""
    cf = _fit_piecewise_curve(noise_sd=0.5, seed=2)
    # Drive max(c, a*x+b) to zero on the grid while keeping a fitted cov/MSE path.
    cf.params_["intercept_noise"] = 0.0
    cf.params_["intercept_linear"] = -1.0e9

    x_grid, cv = delta_cv_profile(cf, n_grid=50, lod=float(np.min(cf.x_)))
    assert x_grid.size > 0
    assert np.all(cf.predict(x_grid) == 0.0)
    assert np.all(np.isinf(cv))


def test_prediction_variance_clamps_negative() -> None:
    """Negative raw prediction variance is clamped to zero before sqrt."""
    cov = np.array([[-10.0, 0.0], [0.0, -10.0]])
    assert prediction_variance(1.0, mse=0.0, cov=cov) == 0.0
