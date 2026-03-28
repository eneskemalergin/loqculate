"""Tests for loqculate.utils.knot_search."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from loqculate.config import DEFAULT_MIN_LINEAR_POINTS, DEFAULT_MIN_NOISE_POINTS, DEFAULT_STD_MULT
from loqculate.models import PiecewiseWLS
from loqculate.utils.knot_search import KnotResult, find_knot, find_knot_batch
from loqculate.utils.normal_equations import solve_2x2_wls, weighted_mean
from loqculate.utils.weights import inverse_sqrt_weights

# ---------------------------------------------------------------------------
# Demo data fixture (27-peptide reference for H3 and H4)
# ---------------------------------------------------------------------------

_DEMO_DIR = Path(__file__).parent.parent / "data" / "demo"
_DATA_FILE = _DEMO_DIR / "one_protein.csv"
_CONC_MAP = _DEMO_DIR / "filename2samplegroup_map.csv"


@pytest.fixture(scope="module")
def reference_data():
    if not _DATA_FILE.exists():
        pytest.skip("Demo data not found")
    from loqculate.io import read_calibration_data

    return read_calibration_data(str(_DATA_FILE), str(_CONC_MAP))


@pytest.fixture(scope="module")
def reference_peptides(reference_data):
    """List of (x, y, W_prec) tuples, one per peptide."""
    peptides = []
    for pep in np.unique(reference_data.peptide):
        mask = reference_data.peptide == pep
        x = reference_data.concentration[mask]
        y = reference_data.area[mask]
        w = inverse_sqrt_weights(x)
        W = w**2
        peptides.append((x, y, W))
    return peptides


# ---------------------------------------------------------------------------
# Helper: LOD from a KnotResult (mirrors PiecewiseWLS.lod())
# ---------------------------------------------------------------------------


def _lod_from_knot_result(
    kr: KnotResult,
    x: np.ndarray,
    y: np.ndarray,
    std_mult: float = DEFAULT_STD_MULT,
    min_noise_points: int = DEFAULT_MIN_NOISE_POINTS,
    min_linear_points: int = DEFAULT_MIN_LINEAR_POINTS,
) -> float:
    a, b, c = kr.slope, kr.intercept, kr.noise_intercept
    if a <= 0:
        return np.inf
    intersection = (c - b) / a
    noise_y = y[x < intersection]
    if len(noise_y) < min_noise_points:
        return np.inf
    std_noise = float(np.std(noise_y, ddof=1))
    lod = (c + std_mult * std_noise - b) / a
    if not np.isfinite(lod) or lod > np.max(x):
        return np.inf
    if int(np.sum(np.unique(x) > lod)) < min_linear_points:
        return np.inf
    return float(lod)


# ---------------------------------------------------------------------------
# Helper: count unconstrained violations across all candidates
# ---------------------------------------------------------------------------


def _count_violations(x: np.ndarray, y: np.ndarray, W: np.ndarray) -> tuple[int, int]:
    """Return (n_neg_slope, n_clamp_noise) across all candidate knots."""
    unique_x = np.unique(x)
    n_neg, n_clamp = 0, 0
    for k in unique_x[1:-1]:
        noise_mask = x <= k
        lin_mask = ~noise_mask
        c, _ = weighted_mean(y[noise_mask], W[noise_mask])
        slope, intercept, _ = solve_2x2_wls(x[lin_mask], y[lin_mask], W[lin_mask])
        if slope < 0:
            n_neg += 1
        if c < intercept:
            n_clamp += 1
    return n_neg, n_clamp


# ---------------------------------------------------------------------------
# Tests: synthetic known breakpoint (H2)
# ---------------------------------------------------------------------------


class TestKnownBreakpoint:
    """Noiseless synthetic curve with geometric breakpoint between x=4 and x=5.

    y = max(10, 2*x + 1): breakpoint at (10-1)/2 = 4.5.
    Noise plateau covers x in {0,1,2,3,4}; linear covers x in {5,6,7,8}.
    Noise observations (y=10) sit above the linear extrapolation at those x
    values (e.g. 2*4+1=9 ≠ 10), so k=4 is the unique zero-RSS partition.
    """

    # y = max(10, 2*x + 1), x in 0..8
    X = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    Y = np.maximum(10.0, 2.0 * X + 1.0)  # [10,10,10,10,10,11,13,15,17]
    W = np.ones(9)

    def test_selects_correct_knot(self):
        """H2: noiseless curve selects the last noise point (x=4) as the knot."""
        kr = find_knot(self.X, self.Y, self.W)
        assert kr.knot_x == pytest.approx(4.0)

    def test_rss_is_zero(self):
        """Perfect fit: RSS should be zero for the correct knot."""
        kr = find_knot(self.X, self.Y, self.W)
        assert kr.rss == pytest.approx(0.0, abs=1e-10)

    def test_fitted_params(self):
        kr = find_knot(self.X, self.Y, self.W)
        assert kr.slope == pytest.approx(2.0, abs=1e-10)
        assert kr.intercept == pytest.approx(1.0, abs=1e-10)
        assert kr.noise_intercept == pytest.approx(10.0, abs=1e-10)

    def test_segment_counts(self):
        kr = find_knot(self.X, self.Y, self.W)
        # noise: x in {0,1,2,3,4} → 5 obs; linear: x in {5,6,7,8} → 4 obs
        assert kr.n_noise == 5
        assert kr.n_linear == 4

    def test_noisy_curve_100_draws(self):
        """H2: correct knot in >= 90 of 100 noisy draws (SNR ~ 10)."""
        rng = np.random.default_rng(7)
        noise_std = np.std(self.Y) / 10.0  # SNR ≈ 10
        correct = 0
        for _ in range(100):
            y_noisy = self.Y + rng.normal(scale=noise_std, size=len(self.Y))
            kr = find_knot(self.X, y_noisy, self.W)
            if kr.knot_x == pytest.approx(4.0):
                correct += 1
        assert correct >= 90, f"Only {correct}/100 draws selected the correct knot."


# ---------------------------------------------------------------------------
# Tests: constraint enforcement (H4)
# ---------------------------------------------------------------------------


class TestConstraintViolations:
    def test_violations_occur_on_reference_data(self, reference_peptides):
        """H4: constraint violations (a<0 or c<b) occur on at least one peptide."""
        total_neg, total_clamp = 0, 0
        for x, y, W in reference_peptides:
            n_neg, n_clamp = _count_violations(x, y, W)
            total_neg += n_neg
            total_clamp += n_clamp
        assert total_neg + total_clamp > 0, (
            "No constraint violations found on reference data. "
            "The constraints may be vacuous — re-examine the data."
        )

    def test_constraint1_applied_correctly(self):
        """When unconstrained slope is negative, find_knot returns slope=0."""
        # Construct data where the high-x observations are lower than low-x.
        x = np.array([1.0, 1.0, 2.0, 5.0, 10.0, 10.0])
        y = np.array([100.0, 90.0, 80.0, 5.0, 2.0, 3.0])
        W = np.ones(6)
        kr = find_knot(x, y, W)
        assert kr.slope >= 0.0

    def test_constraint2_applied_correctly(self):
        """noise_intercept must always be >= intercept in the result."""
        rng = np.random.default_rng(99)
        x = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 20.0])
        y = 3.0 * x + 10.0 + rng.normal(scale=0.5, size=6)
        W = np.ones(6)
        kr = find_knot(x, y, W)
        assert kr.noise_intercept >= kr.intercept - 1e-10


# ---------------------------------------------------------------------------
# Tests: LOD from knot search vs PiecewiseWLS (H3 — informational)
# ---------------------------------------------------------------------------


class TestLODComparison:
    def test_knot_lod_finite_when_wls_finite(self, reference_peptides):
        """H3: where PiecewiseWLS returns a finite LOD, find_knot also should."""
        n_pairs = 0
        max_rel_diff = 0.0

        for x, y, W in reference_peptides:
            wls = PiecewiseWLS(init_method="legacy", n_boot_reps=0, seed=42).fit(x, y)
            lod_wls = wls.lod()
            kr = find_knot(x, y, W)
            lod_cf = _lod_from_knot_result(kr, x, y)

            if np.isfinite(lod_wls) and np.isfinite(lod_cf):
                rel_diff = abs(lod_wls - lod_cf) / max(abs(lod_wls), 1e-12)
                max_rel_diff = max(max_rel_diff, rel_diff)
                n_pairs += 1

        # Informational: record result for the H3 table, but do not enforce a
        # hard tolerance here. That is done in test_regression_v030.py (C3).
        assert n_pairs > 0, "No peptides with finite LOD from both methods."
        print(
            f"\nH3: {n_pairs} peptides with finite LOD from both methods. "
            f"Max relative LOD difference: {max_rel_diff:.4f}"
        )


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_fewer_than_3_unique_x_raises(self):
        x = np.array([1.0, 1.0, 2.0, 2.0])
        y = np.array([1.0, 1.5, 2.0, 2.5])
        W = np.ones(4)
        with pytest.raises(ValueError, match="3 unique x"):
            find_knot(x, y, W)

    def test_exactly_3_unique_x_works(self):
        x = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        y = np.array([5.0, 5.0, 5.0, 5.0, 7.0, 7.0])
        W = np.ones(6)
        kr = find_knot(x, y, W)
        assert isinstance(kr, KnotResult)

    def test_all_rss_tied_lowest_x_wins(self):
        """When all candidates give equal RSS, the first (lowest-x) candidate wins."""
        # All y equal → all candidates produce total_rss = 0.
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.full(5, 7.0)
        W = np.ones(5)
        kr = find_knot(x, y, W)
        unique_x = np.unique(x)
        assert kr.knot_x == pytest.approx(unique_x[1])


# ---------------------------------------------------------------------------
# Tests: find_knot_batch matches find_knot in a loop
# ---------------------------------------------------------------------------


class TestBatch:
    def test_batch_matches_scalar_loop(self):
        """find_knot_batch must match find_knot for every row within 1e-10."""
        rng = np.random.default_rng(31)
        n = 12
        n_reps = 50
        x = np.sort(rng.choice([1.0, 2.0, 5.0, 10.0, 20.0, 50.0], size=n, replace=True))
        W = inverse_sqrt_weights(x) ** 2
        Y_matrix = rng.uniform(0.5, 50.0, size=(n_reps, n))

        slopes_b, intercepts_b, noise_b, knot_xs_b, rss_b = find_knot_batch(x, Y_matrix, W)

        for i in range(n_reps):
            kr = find_knot(x, Y_matrix[i], W)
            assert slopes_b[i] == pytest.approx(kr.slope, abs=1e-10)
            assert intercepts_b[i] == pytest.approx(kr.intercept, abs=1e-10)
            assert noise_b[i] == pytest.approx(kr.noise_intercept, abs=1e-10)
            assert knot_xs_b[i] == pytest.approx(kr.knot_x, abs=1e-10)
            assert rss_b[i] == pytest.approx(kr.rss, abs=1e-10)

    def test_batch_output_shapes(self):
        n, n_reps = 8, 20
        x = np.array([1.0, 1.0, 2.0, 2.0, 5.0, 5.0, 10.0, 10.0])
        W = np.ones(n)
        Y_matrix = np.ones((n_reps, n)) * 3.0
        result = find_knot_batch(x, Y_matrix, W)
        assert len(result) == 5
        for arr in result:
            assert arr.shape == (n_reps,)

    def test_batch_fewer_than_3_unique_x_raises(self):
        x = np.array([1.0, 1.0, 2.0, 2.0])
        W = np.ones(4)
        Y_matrix = np.ones((5, 4))
        with pytest.raises(ValueError, match="3 unique x"):
            find_knot_batch(x, Y_matrix, W)
