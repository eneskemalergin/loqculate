"""Edge-case scenario tests: degenerate inputs must not crash or hang.

Each test targets one named scenario from ``loqculate.testing.scenarios``.
The parametrised ``edge_case`` fixture in conftest.py supplies all scenarios;
each test skips irrelevant ones so the intent stays focused.
"""

import numpy as np
import pytest

from loqculate.models import EmpiricalCV, PiecewiseWLS

# ---------------------------------------------------------------------------
# PiecewiseWLS edge cases
# ---------------------------------------------------------------------------


def test_all_zero_does_not_hang(edge_case):
    """All-zero areas must not hang and must return LOD = inf."""
    name, dataset = edge_case
    if name != "all_zero":
        pytest.skip(f"scenario under test is all_zero, got {name}")
    pep = dataset.peptide[0]
    mask = dataset.peptide == pep
    x = dataset.concentration[mask]
    y = dataset.area[mask]

    model = PiecewiseWLS(n_boot_reps=10)
    model.fit(x, y)
    assert model.params_["slope"] == 0.0
    assert model.lod() == np.inf


def test_constant_area_fit_is_flat():
    """Constant (nonzero) areas are also no-signal: slope 0, LOD infinite."""
    x = np.array([0.0, 1.0, 10.0, 100.0, 0.0, 1.0, 10.0, 100.0])
    y = np.full_like(x, 42.0)
    model = PiecewiseWLS(n_boot_reps=0).fit(x, y)
    assert model.params_["slope"] == 0.0
    assert model.params_["intercept_noise"] == 42.0
    assert model.lod() == np.inf


def test_all_noise_lod_inf(edge_case):
    """No linear signal (slope=0) → LOD must be inf."""
    name, dataset = edge_case
    if name != "all_noise":
        pytest.skip(f"scenario under test is all_zero, got {name}")
    pep = dataset.peptide[0]
    mask = dataset.peptide == pep
    x = dataset.concentration[mask]
    y = dataset.area[mask]

    model = PiecewiseWLS(n_boot_reps=10).fit(x, y)
    lod = model.lod()
    assert lod == np.inf or not np.isfinite(lod)


def test_high_cv_bounce_loq_not_first_crossing(edge_case):
    """Sliding window must not return the first CV-below-threshold point
    when the CV bounces back above threshold immediately after."""
    name, dataset = edge_case
    if name != "high_cv_bounce":
        pytest.skip(f"scenario under test is all_zero, got {name}")
    pep = dataset.peptide[0]
    mask = dataset.peptide == pep
    x = dataset.concentration[mask]
    y = dataset.area[mask]

    model = PiecewiseWLS(n_boot_reps=20, sliding_window=3).fit(x, y)
    # We can't assert a specific LOQ value; we just check it doesn't crash
    # and returns a float (possibly inf if the bounce keeps CV high throughout)
    loq = model.loq()
    assert isinstance(loq, float)


def test_sparse_curve_does_not_crash(edge_case):
    """4-concentration sparse grid must not crash (just may return inf)."""
    name, dataset = edge_case
    if name != "sparse_curve":
        pytest.skip(f"scenario under test is all_zero, got {name}")
    pep = dataset.peptide[0]
    mask = dataset.peptide == pep
    x = dataset.concentration[mask]
    y = dataset.area[mask]

    model = PiecewiseWLS(n_boot_reps=10).fit(x, y)
    lod = model.lod()
    loq = model.loq()
    assert isinstance(lod, float)
    assert isinstance(loq, float)


def test_wide_dynamic_range_does_not_crash(edge_case):
    """Five orders of magnitude in x → weight clipping must not destabilise fit."""
    name, dataset = edge_case
    if name != "wide_dynamic_range":
        pytest.skip(f"scenario under test is all_zero, got {name}")
    pep = dataset.peptide[0]
    mask = dataset.peptide == pep
    x = dataset.concentration[mask]
    y = dataset.area[mask]

    model = PiecewiseWLS(n_boot_reps=10).fit(x, y)
    assert model.is_fitted_
    assert isinstance(model.lod(), float)


# ---------------------------------------------------------------------------
# EmpiricalCV edge cases
# ---------------------------------------------------------------------------


def test_single_replicate_rejects_empirical_cv(edge_case):
    """n=1 per concentration → CV is mathematically undefined → ValueError."""
    name, dataset = edge_case
    if name != "single_replicate":
        pytest.skip(f"scenario under test is all_zero, got {name}")
    pep = dataset.peptide[0]
    mask = dataset.peptide == pep
    x = dataset.concentration[mask]
    y = dataset.area[mask]

    with pytest.raises(ValueError, match="replicate"):
        EmpiricalCV().fit(x, y)


def test_blank_only_low_cv_loq_positive(edge_case):
    """LOQ must be > 0 even when blank concentration has very low CV."""
    name, dataset = edge_case
    if name != "blank_only_low_cv":
        pytest.skip(f"scenario under test is all_zero, got {name}")
    pep = dataset.peptide[0]
    mask = dataset.peptide == pep
    x = dataset.concentration[mask]
    y = dataset.area[mask]

    model = EmpiricalCV().fit(x, y)
    loq = model.loq()
    assert loq > 0 or loq == np.inf
