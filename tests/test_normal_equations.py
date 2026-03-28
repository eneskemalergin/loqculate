"""Tests for loqculate.utils.normal_equations."""

from __future__ import annotations

import numpy as np
import pytest

from loqculate.utils.normal_equations import (
    solve_2x2_wls,
    solve_2x2_wls_batch,
    weighted_mean,
)

# ---------------------------------------------------------------------------
# weighted_mean
# ---------------------------------------------------------------------------


class TestWeightedMean:
    def test_hand_computed(self):
        # mean = (1*2 + 2*4 + 1*6) / (1+2+1) = 16/4 = 4
        # rss  = 1*(2-4)^2 + 2*(4-4)^2 + 1*(6-4)^2 = 4 + 0 + 4 = 8
        y = np.array([2.0, 4.0, 6.0])
        W = np.array([1.0, 2.0, 1.0])
        mean, rss = weighted_mean(y, W)
        assert mean == pytest.approx(4.0)
        assert rss == pytest.approx(8.0)

    def test_uniform_weights(self):
        y = np.array([1.0, 3.0, 5.0])
        W = np.ones(3)
        mean, rss = weighted_mean(y, W)
        assert mean == pytest.approx(3.0)
        assert rss == pytest.approx(8.0)  # 1^2 + 0^2 + 2^2 = 1 + 0 + 4 = ... wait
        # rss = 1*(1-3)^2 + 1*(3-3)^2 + 1*(5-3)^2 = 4 + 0 + 4 = 8
        assert rss == pytest.approx(8.0)

    def test_zero_weights_fallback(self):
        y = np.array([1.0, 2.0, 3.0])
        W = np.zeros(3)
        mean, rss = weighted_mean(y, W)
        assert mean == pytest.approx(2.0)
        assert rss == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# solve_2x2_wls
# ---------------------------------------------------------------------------


class TestSolve2x2Wls:
    def test_hand_computed(self):
        # Exact line y = 2x + 1 with equal weights. Residuals are zero.
        x = np.array([1.0, 2.0, 3.0])
        y = 2.0 * x + 1.0
        W = np.ones(3)
        slope, intercept, rss = solve_2x2_wls(x, y, W)
        assert slope == pytest.approx(2.0, abs=1e-12)
        assert intercept == pytest.approx(1.0, abs=1e-12)
        assert rss == pytest.approx(0.0, abs=1e-12)

    def test_matches_lstsq(self):
        """H1: normal equations must match np.linalg.lstsq within 1e-12."""
        rng = np.random.default_rng(0)
        x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        w_obs = 1.0 / (np.sqrt(x) + 1e-12)  # observation weights
        W = w_obs**2  # precision weights
        y = 3.5 * x + 0.8 + rng.normal(scale=0.1, size=len(x))

        slope, intercept, _ = solve_2x2_wls(x, y, W)

        # Reference: weighted lstsq via design matrix scaling by sqrt(W).
        A = np.column_stack([x, np.ones_like(x)])
        sqrt_W = np.sqrt(W)
        popt, _, _, _ = np.linalg.lstsq(sqrt_W[:, None] * A, sqrt_W * y, rcond=None)
        ref_slope, ref_intercept = popt

        assert slope == pytest.approx(ref_slope, abs=1e-12)
        assert intercept == pytest.approx(ref_intercept, abs=1e-12)

    def test_singular_fallback(self):
        """All x identical: returns slope=0, intercept=weighted mean, no exception."""
        x = np.array([5.0, 5.0, 5.0])
        y = np.array([1.0, 2.0, 3.0])
        W = np.ones(3)
        slope, intercept, rss = solve_2x2_wls(x, y, W)
        assert slope == 0.0
        assert intercept == pytest.approx(2.0)  # unweighted mean = 2
        # rss = 1*(1-2)^2 + 1*(2-2)^2 + 1*(3-2)^2 = 2
        assert rss == pytest.approx(2.0)

    def test_single_point_fallback(self):
        """n=1: det=0, falls back to weighted mean; slope=0."""
        x = np.array([3.0])
        y = np.array([7.0])
        W = np.array([2.0])
        slope, intercept, rss = solve_2x2_wls(x, y, W)
        assert slope == 0.0
        assert intercept == pytest.approx(7.0)
        assert rss == pytest.approx(0.0)

    def test_weighted_vs_unweighted_differ(self):
        """Confirm weighting actually changes the solution."""
        x = np.array([1.0, 2.0, 10.0])
        y = np.array([1.0, 3.0, 4.0])
        W_uniform = np.ones(3)
        W_heavy_first = np.array([100.0, 1.0, 1.0])

        slope_u, intercept_u, _ = solve_2x2_wls(x, y, W_uniform)
        slope_w, intercept_w, _ = solve_2x2_wls(x, y, W_heavy_first)

        # Heavily weighting the first point should pull the line toward (1, 1).
        assert slope_u != pytest.approx(slope_w, abs=1e-6)
        assert intercept_u != pytest.approx(intercept_w, abs=1e-6)


# ---------------------------------------------------------------------------
# solve_2x2_wls_batch
# ---------------------------------------------------------------------------


class TestSolve2x2WlsBatch:
    def test_matches_scalar_loop(self):
        """Batch variant must match scalar solve_2x2_wls called in a loop."""
        rng = np.random.default_rng(42)
        n = 8
        n_reps = 50
        x = np.sort(rng.uniform(1.0, 20.0, size=n))
        w_obs = 1.0 / (np.sqrt(x) + 1e-12)
        W = w_obs**2
        Y_matrix = rng.uniform(0.5, 10.0, size=(n_reps, n))

        slopes_b, intercepts_b, rss_b = solve_2x2_wls_batch(x, Y_matrix, W)

        for i in range(n_reps):
            s, ic, r = solve_2x2_wls(x, Y_matrix[i], W)
            assert slopes_b[i] == pytest.approx(s, abs=1e-12)
            assert intercepts_b[i] == pytest.approx(ic, abs=1e-12)
            assert rss_b[i] == pytest.approx(r, abs=1e-12)

    def test_output_shapes(self):
        n, n_reps = 6, 20
        x = np.arange(1.0, n + 1)
        W = np.ones(n)
        Y_matrix = np.ones((n_reps, n))
        slopes, intercepts, rss_array = solve_2x2_wls_batch(x, Y_matrix, W)
        assert slopes.shape == (n_reps,)
        assert intercepts.shape == (n_reps,)
        assert rss_array.shape == (n_reps,)

    def test_singular_batch_fallback(self):
        """All x identical: every row falls back to slope=0, intercept=row mean."""
        n_reps = 5
        x = np.array([3.0, 3.0, 3.0, 3.0])
        W = np.ones(4)
        rng = np.random.default_rng(1)
        Y_matrix = rng.uniform(1.0, 5.0, size=(n_reps, 4))

        slopes, intercepts, rss_array = solve_2x2_wls_batch(x, Y_matrix, W)

        assert np.all(slopes == 0.0)
        for i in range(n_reps):
            assert intercepts[i] == pytest.approx(float(np.mean(Y_matrix[i])))
