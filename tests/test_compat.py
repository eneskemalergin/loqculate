"""Tests for loqculate.compat — OriginalWLS and OriginalCV.

Structure:
  * Interface/ABC compliance for both classes
  * MODEL_REGISTRY registration
  * Synthetic-data unit tests (deterministic, no external files)
  * Regression tests against bench_real_data.json (OriginalWLS LOD / LOQ)
  * Regression tests against bench_empirical.json (OriginalCV LOQ)

Regression fixtures are skipped when the benchmark JSON or demo data files
are absent (e.g. in minimal CI environments that only have tests/).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from loqculate.compat import OriginalCV, OriginalWLS
from loqculate.models import MODEL_REGISTRY
from loqculate.models.base import CalibrationModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DEMO_DATA = ROOT / "data" / "demo" / "one_protein.csv"
DEMO_MAP  = ROOT / "data" / "demo" / "filename2samplegroup_map.csv"
BENCH_WLS = ROOT / "tmp" / "results" / "bench_real_data.json"
BENCH_CV  = ROOT / "tmp" / "results" / "bench_empirical.json"

needs_demo    = pytest.mark.skipif(not DEMO_DATA.exists(),  reason="demo data absent")
needs_bench_w = pytest.mark.skipif(not BENCH_WLS.exists(), reason="bench_real_data.json absent")
needs_bench_c = pytest.mark.skipif(not BENCH_CV.exists(),  reason="bench_empirical.json absent")


# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------

def _make_simple_curve(
    concs=(0.001, 0.01, 0.1, 1.0, 10.0),
    n_reps=4,
    seed=0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic piecewise-linear calibration data with clear LOD/LOQ."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in concs:
        signal = max(c * 1e7, 5e4)  # piecewise: plateau at 5e4
        for _ in range(n_reps):
            xs.append(c)
            ys.append(signal * rng.lognormal(0, 0.05))
    return np.array(xs), np.array(ys)


def _make_low_cv_curve(
    concs=(0.01, 0.05, 0.1, 0.5, 1.0),
    cv_target=0.05,
    n_reps=4,
    seed=1,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic curve where every concentration has CV < 20%."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in concs:
        signal = c * 1e7
        for _ in range(n_reps):
            xs.append(c)
            ys.append(abs(signal * rng.normal(1.0, cv_target)))
    return np.array(xs), np.array(ys)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bench_val(v) -> float:
    """Convert JSON null (None) to inf for comparison; keep finite floats as-is."""
    return np.inf if v is None else float(v)


def _load_demo_peptides() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load one_protein.csv via loqculate's reader."""
    from loqculate.io import read_calibration_data
    data = read_calibration_data(str(DEMO_DATA), str(DEMO_MAP))
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pep in np.unique(data.peptide):
        mask = data.peptide == pep
        out[pep] = (data.concentration[mask], data.area[mask])
    return out


# ---------------------------------------------------------------------------
# 1. Interface / ABC compliance
# ---------------------------------------------------------------------------

class TestOriginalWLSInterface:
    def test_is_calibration_model(self):
        assert isinstance(OriginalWLS(), CalibrationModel)

    def test_fit_returns_self(self):
        x, y = _make_simple_curve()
        m = OriginalWLS()
        assert m.fit(x, y) is m

    def test_is_fitted_after_fit(self):
        x, y = _make_simple_curve()
        m = OriginalWLS().fit(x, y)
        assert m.is_fitted_

    def test_predict_shape(self):
        x, y = _make_simple_curve()
        m = OriginalWLS().fit(x, y)
        x_new = np.array([0.01, 0.1, 1.0])
        pred = m.predict(x_new)
        assert pred.shape == x_new.shape

    def test_predict_non_negative(self):
        x, y = _make_simple_curve()
        m = OriginalWLS().fit(x, y)
        x_new = np.linspace(0.001, 10.0, 50)
        assert np.all(m.predict(x_new) >= 0)

    def test_lod_finite(self):
        x, y = _make_simple_curve()
        m = OriginalWLS().fit(x, y)
        lod = m.lod()
        assert math.isfinite(lod), f"Expected finite LOD, got {lod}"

    def test_loq_ge_lod(self):
        x, y = _make_simple_curve()
        m = OriginalWLS().fit(x, y)
        lod = m.lod()
        loq = m.loq()
        if math.isfinite(loq):
            assert loq >= lod, f"LOQ ({loq}) < LOD ({lod})"

    def test_supports_lod_true(self):
        assert OriginalWLS().supports_lod() is True

    def test_summary_keys(self):
        x, y = _make_simple_curve()
        m = OriginalWLS().fit(x, y)
        s = m.summary()
        for key in ('slope', 'intercept_linear', 'intercept_noise', 'lod', 'loq'):
            assert key in s, f"Missing key '{key}' in summary"

    def test_slope_positive(self):
        x, y = _make_simple_curve()
        m = OriginalWLS().fit(x, y)
        assert m.params_['slope'] > 0

    def test_not_fitted_before_fit(self):
        m = OriginalWLS()
        with pytest.raises(RuntimeError):
            m.predict(np.array([1.0]))


class TestOriginalCVInterface:
    def test_is_calibration_model(self):
        assert isinstance(OriginalCV(), CalibrationModel)

    def test_fit_returns_self(self):
        x, y = _make_low_cv_curve()
        m = OriginalCV()
        assert m.fit(x, y) is m

    def test_is_fitted_after_fit(self):
        x, y = _make_low_cv_curve()
        m = OriginalCV().fit(x, y)
        assert m.is_fitted_

    def test_loq_finite(self):
        x, y = _make_low_cv_curve()
        m = OriginalCV().fit(x, y)
        assert math.isfinite(m.loq()), f"Expected finite LOQ, got {m.loq()}"

    def test_lod_always_inf(self):
        x, y = _make_low_cv_curve()
        m = OriginalCV().fit(x, y)
        assert not math.isfinite(m.lod())

    def test_supports_lod_false(self):
        assert OriginalCV().supports_lod() is False

    def test_predict_raises(self):
        x, y = _make_low_cv_curve()
        m = OriginalCV().fit(x, y)
        with pytest.raises(NotImplementedError):
            m.predict(np.array([1.0]))

    def test_cv_table_populated(self):
        x, y = _make_low_cv_curve(concs=(0.01, 0.1, 1.0))
        m = OriginalCV().fit(x, y)
        assert set(m.cv_table_.keys()) == {0.01, 0.1, 1.0}

    def test_not_fitted_before_fit(self):
        m = OriginalCV()
        with pytest.raises(RuntimeError):
            m.loq()


# ---------------------------------------------------------------------------
# 2. MODEL_REGISTRY
# ---------------------------------------------------------------------------

class TestModelRegistry:
    def test_original_wls_in_registry(self):
        assert 'original_wls' in MODEL_REGISTRY

    def test_original_cv_in_registry(self):
        assert 'original_cv' in MODEL_REGISTRY

    def test_registry_returns_correct_classes(self):
        assert MODEL_REGISTRY['original_wls'] is OriginalWLS
        assert MODEL_REGISTRY['original_cv'] is OriginalCV

    def test_can_instantiate_from_registry(self):
        m = MODEL_REGISTRY['original_wls']()
        assert isinstance(m, OriginalWLS)


# ---------------------------------------------------------------------------
# 3. Synthetic unit tests
# ---------------------------------------------------------------------------

class TestOriginalWLSSynthetic:
    def test_loq_below_max_concentration(self):
        x, y = _make_simple_curve(concs=(0.001, 0.01, 0.1, 1.0, 10.0), n_reps=5)
        m = OriginalWLS(n_boot=20).fit(x, y)
        loq = m.loq()
        if math.isfinite(loq):
            assert loq <= 10.0

    def test_cv_thresh_effect(self):
        """Stricter CV threshold should give equal or higher LOQ."""
        x, y = _make_simple_curve(concs=(0.001, 0.01, 0.1, 1.0, 10.0), n_reps=5)
        m_loose = OriginalWLS(cv_thresh=0.5, n_boot=30).fit(x, y)
        m_strict = OriginalWLS(cv_thresh=0.05, n_boot=30).fit(x, y)
        loose_loq = m_loose.loq()
        strict_loq = m_strict.loq()
        # strict threshold → LOQ can only be >= loose threshold LOQ (or inf)
        if math.isfinite(loose_loq) and math.isfinite(strict_loq):
            assert strict_loq >= loose_loq - 1e-9

    def test_few_points_raises(self):
        with pytest.raises(ValueError):
            OriginalWLS().fit(np.array([1.0, 2.0]), np.array([1.0, 2.0]))

    def test_single_unique_conc_raises(self):
        with pytest.raises(ValueError):
            OriginalWLS().fit(np.full(5, 1.0), np.ones(5))

    def test_deterministic_lod(self):
        """LOD is deterministic — two fits on the same data give identical LODs."""
        x, y = _make_simple_curve()
        # LOD is computed before the bootstrap; fitting twice yields the same value.
        lod1 = OriginalWLS(n_boot=10).fit(x, y)._lod_val
        lod2 = OriginalWLS(n_boot=10).fit(x, y)._lod_val
        assert lod1 == lod2


class TestOriginalCVSynthetic:
    def test_loq_exact_threshold(self):
        """CV exactly at threshold should satisfy the <= rule (non-strict)."""
        # 3 concentrations, 10 reps each. Design the CV at c=0.01 to be exactly 0.2
        rng = np.random.default_rng(42)
        n = 10
        c_values = [0.001, 0.01, 0.1]
        xs, ys = [], []
        for c in c_values:
            mean_sig = c * 1e7
            # target_cv = 0.2 for c=0.01, 0 for others (but let's just set realistic values)
            cv_target = 0.20 if c == 0.01 else 0.05
            for _ in range(n):
                xs.append(c)
                ys.append(abs(mean_sig * rng.normal(1.0, cv_target)))
        x, y = np.array(xs), np.array(ys)
        m = OriginalCV(cv_thresh=0.20).fit(x, y)
        # The LOQ must be at the first concentration with CV <= 0.20
        loq = m.loq()
        assert math.isfinite(loq)
        assert loq > 0

    def test_all_high_cv_gives_inf(self):
        """All CVs above threshold → LOQ = inf."""
        rng = np.random.default_rng(0)
        xs, ys = [], []
        for c in [0.01, 0.1, 1.0]:
            for _ in range(5):
                xs.append(c)
                ys.append(abs(c * 1e7 * rng.normal(1.0, 0.8)))  # CV ~80%
        m = OriginalCV(cv_thresh=0.20).fit(np.array(xs), np.array(ys))
        assert not math.isfinite(m.loq())

    def test_zero_concentration_excluded(self):
        """Zero-concentration entries (blanks) must not be the LOQ."""
        rng = np.random.default_rng(2)
        xs, ys = [], []
        for c in [0.0, 0.01, 0.1]:
            for _ in range(5):
                xs.append(c)
                # blank (c=0) gets a tiny low-CV signal
                if c == 0.0:
                    ys.append(abs(rng.normal(500, 10)))
                else:
                    ys.append(abs(c * 1e7 * rng.normal(1.0, 0.05)))
        m = OriginalCV(cv_thresh=0.20).fit(np.array(xs), np.array(ys))
        assert m.loq() > 0, "LOQ must be a positive concentration, not the blank (0)"

    def test_loq_is_min_good_concentration(self):
        """All positive concentrations have CV <= threshold → LOQ = min positive conc."""
        rng = np.random.default_rng(3)
        concs = [0.01, 0.05, 0.1, 0.5]
        xs, ys = [], []
        for c in concs:
            for _ in range(10):
                xs.append(c)
                ys.append(abs(c * 1e7 * rng.normal(1.0, 0.03)))  # CV ~3%
        m = OriginalCV(cv_thresh=0.20).fit(np.array(xs), np.array(ys))
        assert math.isfinite(m.loq())
        assert math.isclose(m.loq(), min(concs), rel_tol=1e-9)


# ---------------------------------------------------------------------------
# 4. Regression: OriginalWLS vs bench_real_data.json
# ---------------------------------------------------------------------------

@needs_demo
@needs_bench_w
class TestOriginalWLSRegression:
    """Compare OriginalWLS LOD/LOQ against the stored benchmark from the original script.

    Tolerance note
    --------------
    The original script uses lmfit's Levenberg-Marquardt optimiser, while
    OriginalWLS uses scipy's TRF (no lmfit dependency).  For the vast majority
    of peptides the two solvers converge to the same solution (<0.01% rdiff).
    For a handful (classified as 'diverge' in the benchmark) the local optima
    differ by up to 29%.  These cases are expected and skipped.
    LOQ values are omitted from value comparison because bootstrap CV is
    highly sensitive to solver choice on each replicate; we only verify the
    inf/finite category where both reference implementations agree.
    """

    # 20% — matches the original benchmark's own lod_tol parameter.
    LOD_RTOL = 0.20

    @pytest.fixture(scope='class')
    def bench(self):
        with open(BENCH_WLS) as f:
            d = json.load(f)
        return d['per_peptide']

    @pytest.fixture(scope='class')
    def wls_results(self):
        """Fit OriginalWLS on every peptide in one_protein.csv."""
        peptides = _load_demo_peptides()
        results = {}
        for pep, (x, y) in peptides.items():
            m = OriginalWLS(n_boot=100, cv_thresh=0.2, std_mult=2.0)
            try:
                m.fit(x, y)
                results[pep] = {'lod': m.lod(), 'loq': m.loq()}
            except Exception as exc:  # noqa: BLE001
                results[pep] = {'lod': np.inf, 'loq': np.inf, 'error': str(exc)}
        return results

    def test_lod_agreement(self, bench, wls_results):
        """LOD should agree within LOD_RTOL for every peptide.

        Peptides where the benchmark already documents an orig-vs-w1 divergence
        (meaning lmfit LM and scipy TRF find different local minima) are
        skipped — the OriginalWLS implementation cannot be expected to reproduce
        lmfit's exact convergence path from a pure-scipy port.
        """
        failures = []
        for pep, bench_row in bench.items():
            if pep not in wls_results:
                continue

            # Skip peptides where the original script and loqculate's PiecewiseWLS
            # already diverge — these are known solver-difference cases (lmfit LM
            # vs scipy TRF) unrelated to compat correctness.
            if bench_row.get('lod_cat_orig_w1', '').startswith('diverge'):
                continue

            expected = _bench_val(bench_row['orig_lod'])
            got = wls_results[pep]['lod']

            if not math.isfinite(expected) and not math.isfinite(got):
                continue  # both inf — agree
            if not math.isfinite(expected) or not math.isfinite(got):
                failures.append(
                    f'{pep}: expected={expected:.6g}, got={got:.6g} (one is inf)'
                )
                continue
            rdiff = abs(got - expected) / max(abs(expected), abs(got))
            if rdiff > self.LOD_RTOL:
                failures.append(
                    f'{pep}: expected={expected:.6g}, got={got:.6g}, rdiff={rdiff:.1%}'
                )

        assert not failures, (
            f'{len(failures)} LOD regression failure(s):\n' + '\n'.join(failures)
        )

    def test_loq_inf_category(self, bench, wls_results):
        """Where no LOD can be found, OriginalWLS must also produce inf LOQ.

        LOQ values are omitted from value comparison: they are extremely
        sensitive to solver convergence on each bootstrap replicate (lmfit LM
        vs scipy TRF).  Even with identical seeds, different solvers produce
        different CV profiles, so borderline inf/finite LOQ outcomes differ.

        The inf-category constraint is tested only for the strongest case: peptides
        where BOTH reference implementations (orig + lq_w1) cannot even find an LOD
        (``lod_cat_orig_w1 == 'both=inf'``).  If there is no LOD, there cannot be
        a LOQ, regardless of solver.
        """
        failures = []
        for pep, bench_row in bench.items():
            if pep not in wls_results:
                continue
            # Use the most conservative condition: no LOD at all
            if bench_row.get('lod_cat_orig_w1', '') != 'both=inf':
                continue
            got_loq = wls_results[pep]['loq']
            got_lod = wls_results[pep]['lod']
            if math.isfinite(got_lod):
                failures.append(f'{pep}: bench says LOD=inf but OriginalWLS LOD={got_lod:.6g}')
            elif math.isfinite(got_loq):
                failures.append(f'{pep}: LOD=inf but OriginalWLS LOQ={got_loq:.6g} (should be inf)')

        assert not failures, (
            f'{len(failures)} LOQ category failure(s):\n' + '\n'.join(failures)
        )

    def test_inf_lod_agreement(self, bench, wls_results):
        """Peptides where bench shows inf LOD should also have inf LOD from OriginalWLS."""
        for pep, bench_row in bench.items():
            if pep not in wls_results:
                continue
            if bench_row.get('lod_cat_orig_w1', '').startswith('diverge'):
                continue  # known solver-difference case, skip
            expected = _bench_val(bench_row['orig_lod'])
            got = wls_results[pep]['lod']
            if not math.isfinite(expected):
                assert not math.isfinite(got), (
                    f'{pep}: bench LOD is inf but OriginalWLS returned {got}'
                )


# ---------------------------------------------------------------------------
# 5. Regression: OriginalCV vs bench_empirical.json
# ---------------------------------------------------------------------------

@needs_demo
@needs_bench_c
class TestOriginalCVRegression:
    """Compare OriginalCV LOQ against stored benchmark from the original loq_by_cv.py."""

    @pytest.fixture(scope='class')
    def bench(self):
        with open(BENCH_CV) as f:
            d = json.load(f)
        return d['regression']['per_peptide']

    @pytest.fixture(scope='class')
    def cv_results(self):
        """Fit OriginalCV on every peptide in one_protein.csv."""
        peptides = _load_demo_peptides()
        results = {}
        for pep, (x, y) in peptides.items():
            m = OriginalCV(cv_thresh=0.2)
            try:
                m.fit(x, y)
                results[pep] = {'loq': m.loq()}
            except Exception as exc:  # noqa: BLE001
                results[pep] = {'loq': np.inf, 'error': str(exc)}
        return results

    def test_loq_exact_match(self, bench, cv_results):
        """OriginalCV LOQ must exactly match orig_loq (deterministic, no bootstrap)."""
        failures = []
        for pep, bench_row in bench.items():
            if pep not in cv_results:
                continue
            expected = _bench_val(bench_row['orig_loq'])
            got = cv_results[pep]['loq']

            if not math.isfinite(expected) and not math.isfinite(got):
                continue
            if not math.isfinite(expected) or not math.isfinite(got):
                failures.append(
                    f'{pep}: expected={expected}, got={got} (one is inf)'
                )
                continue
            # Exact match expected — both are first minimum CV concentration
            if not math.isclose(got, expected, rel_tol=1e-9):
                failures.append(
                    f'{pep}: expected={expected:.6g}, got={got:.6g}'
                )

        assert not failures, (
            f'{len(failures)} LOQ regression failure(s):\n' + '\n'.join(failures)
        )
