"""Base class contract tests: every model must satisfy these."""
import numpy as np
import pytest

from loqculate.models import PiecewiseWLS, EmpiricalCV
from loqculate.models.base import CalibrationModel


# Concrete models under test
_MODEL_CLASSES = [PiecewiseWLS, EmpiricalCV]


@pytest.fixture(params=_MODEL_CLASSES, ids=[m.__name__ for m in _MODEL_CLASSES])
def empty_model(request):
    return request.param()


@pytest.fixture
def simple_xy():
    """Minimal valid dataset: 3 concentration levels, 3 reps each."""
    concs = np.repeat([1.0, 10.0, 100.0], 3)
    areas = np.array([
        550, 560, 540,      # noise plateau
        5200, 5100, 5300,   # linear
        50200, 49800, 50500,
    ], dtype=float)
    return concs, areas


class TestBaseContract:
    def test_is_calibration_model(self, empty_model):
        assert isinstance(empty_model, CalibrationModel)

    def test_not_fitted_before_fit(self, empty_model):
        assert not empty_model.is_fitted_

    def test_predict_before_fit_raises(self, empty_model):
        with pytest.raises(RuntimeError, match=r'fit\(\)'):
            empty_model.predict(np.array([1.0]))

    def test_fit_returns_self(self, empty_model, simple_xy):
        x, y = simple_xy
        result = empty_model.fit(x, y)
        assert result is empty_model

    def test_is_fitted_after_fit(self, empty_model, simple_xy):
        x, y = simple_xy
        empty_model.fit(x, y)
        assert empty_model.is_fitted_

    def test_predict_returns_array(self, empty_model, simple_xy):
        x, y = simple_xy
        empty_model.fit(x, y)
        out = empty_model.predict(np.array([5.0, 50.0]))
        assert isinstance(out, np.ndarray)
        assert out.shape == (2,)

    def test_lod_is_float(self, empty_model, simple_xy):
        x, y = simple_xy
        empty_model.fit(x, y)
        lod = empty_model.lod()
        assert isinstance(lod, float)

    def test_loq_is_float(self, empty_model, simple_xy):
        x, y = simple_xy
        empty_model.fit(x, y)
        loq = empty_model.loq()
        assert isinstance(loq, float)

    def test_summary_is_dict(self, empty_model, simple_xy):
        x, y = simple_xy
        empty_model.fit(x, y)
        s = empty_model.summary()
        assert isinstance(s, dict)
        assert 'loq' in s

    def test_supports_uloq_false_by_default(self, empty_model):
        assert not empty_model.supports_uloq()

    def test_uloq_returns_inf(self, empty_model):
        assert empty_model.uloq() == np.inf

    def test_stored_training_data(self, empty_model, simple_xy):
        x, y = simple_xy
        empty_model.fit(x, y)
        np.testing.assert_array_equal(empty_model.x_, x)
        np.testing.assert_array_equal(empty_model.y_, y)


class TestPiecewiseWLS:
    def test_lod_finite_on_clear_curve(self, ideal_single_peptide):
        x, y = ideal_single_peptide
        model = PiecewiseWLS(n_boot_reps=10).fit(x, y)
        assert np.isfinite(model.lod())

    def test_params_keys(self, simple_xy):
        x, y = simple_xy
        model = PiecewiseWLS().fit(x, y)
        assert 'slope' in model.params_
        assert 'intercept_linear' in model.params_
        assert 'intercept_noise' in model.params_

    def test_fit_too_few_points_raises(self):
        with pytest.raises(ValueError, match='3 data points'):
            PiecewiseWLS().fit(np.array([1.0, 2.0]), np.array([1.0, 2.0]))

    def test_slope_positive(self, simple_xy):
        x, y = simple_xy
        model = PiecewiseWLS().fit(x, y)
        assert model.params_['slope'] > 0


class TestEmpiricalCV:
    def test_lod_inf(self, simple_xy):
        x, y = simple_xy
        model = EmpiricalCV().fit(x, y)
        assert model.lod() == np.inf

    def test_supports_lod_false(self, simple_xy):
        x, y = simple_xy
        model = EmpiricalCV().fit(x, y)
        assert not model.supports_lod()

    def test_cv_table_populated(self, simple_xy):
        x, y = simple_xy
        model = EmpiricalCV().fit(x, y)
        assert len(model.cv_table_) == 3

    def test_loq_excludes_zero_conc(self):
        """LOQ must not be assigned to concentration 0."""
        concs = np.repeat([0.0, 1.0, 10.0, 100.0], 5)
        areas = np.tile([100, 600, 5500, 51000], 5).astype(float)
        # Add small noise
        rng = np.random.default_rng(0)
        areas += rng.normal(0, 10, size=len(areas))
        model = EmpiricalCV().fit(concs, areas)
        loq = model.loq(cv_thresh=0.2)
        assert loq > 0 or loq == np.inf

    def test_single_replicate_raises(self):
        """Fitting with n=1 at every concentration must raise ValueError (CV undefined)."""
        concs = np.array([1.0, 10.0, 100.0])
        areas = np.array([500.0, 5000.0, 50000.0])
        with pytest.raises(ValueError, match='replicate'):
            EmpiricalCV().fit(concs, areas)

    def test_two_reps_warns_not_raises(self):
        """With 2 reps per concentration CV is computable — only a warning."""
        concs = np.repeat([1.0, 10.0, 100.0], 2)
        areas = np.array([495.0, 505.0, 4950.0, 5050.0, 49500.0, 50500.0])
        with pytest.warns(UserWarning, match='replicate'):
            EmpiricalCV(min_replicates=3).fit(concs, areas)
