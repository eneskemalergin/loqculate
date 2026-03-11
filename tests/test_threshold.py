"""Sliding-window threshold algorithm edge-case tests."""
import numpy as np
import pytest

from loqculate.utils.threshold import find_loq_threshold


class TestFindLoqThreshold:
    def test_happy_path(self):
        """Monotonically decreasing CVs → LOQ at the first point below threshold."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cv = np.array([0.30, 0.25, 0.18, 0.12, 0.10])
        loq = find_loq_threshold(x, cv, cv_thresh=0.2, window=3)
        assert loq == pytest.approx(3.0)

    def test_non_monotonic_bounce(self):
        """V1 naive min returns 2.0 (wrong); V2 sliding window must return 3.0."""
        x = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        cv = np.array([0.45, 0.32, 0.18, 0.25, 0.15, 0.12, 0.10, 0.09, 0.08])
        loq = find_loq_threshold(x, cv, cv_thresh=0.2, window=3)
        assert loq == pytest.approx(3.0)

    def test_all_above_threshold(self):
        x = np.array([1.0, 2.0, 3.0])
        cv = np.array([0.5, 0.4, 0.3])
        assert find_loq_threshold(x, cv, cv_thresh=0.2, window=2) == np.inf

    def test_all_below_threshold(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        cv = np.array([0.10, 0.08, 0.07, 0.06])
        loq = find_loq_threshold(x, cv, cv_thresh=0.2, window=2)
        assert loq == pytest.approx(1.0)

    def test_excludes_zero_concentration(self):
        """LOQ must skip concentration=0 regardless of its CV."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        cv = np.array([0.01, 0.30, 0.15, 0.10])
        loq = find_loq_threshold(x, cv, cv_thresh=0.2, window=2)
        assert loq > 0

    def test_empty_grid(self):
        assert find_loq_threshold(np.array([]), np.array([]), cv_thresh=0.2, window=3) == np.inf

    def test_all_zero_grid(self):
        """Grid only has concentration=0; no valid positive point."""
        x = np.array([0.0, 0.0, 0.0])
        cv = np.array([0.01, 0.01, 0.01])
        assert find_loq_threshold(x, cv, cv_thresh=0.2, window=2) == np.inf

    def test_sparse_window_scaling(self):
        """Window larger than grid → dynamically capped to grid length."""
        x = np.array([1.0, 2.0])
        cv = np.array([0.10, 0.10])
        loq = find_loq_threshold(x, cv, cv_thresh=0.2, window=10)
        assert np.isfinite(loq)

    def test_window_1(self):
        """Window=1 is equivalent to simple min-above-threshold."""
        x = np.array([1.0, 2.0, 3.0, 2.5, 4.0])
        cv = np.array([0.3, 0.19, 0.3, 0.19, 0.15])
        # With window=1, first point where cv <= 0.2 is x=2.0
        loq = find_loq_threshold(x, cv, cv_thresh=0.2, window=1)
        # x is not sorted here, but x[0]=1>threshold? no. x[1]=2 has cv=0.19
        # Actually threshold checks x[nonzero] as-is order, so first hit is x[1]=2.0
        assert loq == pytest.approx(2.0)
