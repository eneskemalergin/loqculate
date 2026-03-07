from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from loqculate.config import DEFAULT_CV_THRESH, DEFAULT_SLIDING_WINDOW
from loqculate.models.base import CalibrationModel
from loqculate.utils.threshold import find_loq_threshold


class EmpiricalCV(CalibrationModel):
    """Empirical CV model: derives LOQ directly from replicate CVs.

    This model has **no regression**.  It groups replicate measurements at
    each concentration level, computes CV = std/mean, and applies the same
    sliding-window threshold search used by :class:`PiecewiseWLS`.

    It is a v2 port of ``v1/loq_by_cv.py`` into the base-class contract.

    Parameters
    ----------
    min_replicates:
        Warn (not error) if any concentration has fewer replicates than this.
        CV is unreliable with fewer than 3 replicates; default 3.
    sliding_window:
        Consecutive calibration points that must stay below ``cv_thresh``.
        Dynamically capped at the number of available distinct concentrations.
    """

    def __init__(
        self,
        min_replicates: int = 3,
        sliding_window: int = DEFAULT_SLIDING_WINDOW,
    ) -> None:
        super().__init__()
        self.min_replicates = min_replicates
        self.sliding_window = sliding_window

        self.cv_table_: dict[float, float] = {}  # {concentration: cv}
        self.mean_table_: dict[float, float] = {}  # {concentration: mean}
        self.replicate_counts_: dict[float, int] = {}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> "EmpiricalCV":
        """Group by concentration and compute per-group CV.

        Parameters
        ----------
        x:
            Concentration values (repetitions allowed — one entry per
            measurement, mirroring the melted input format).
        y:
            Signal areas, same length as *x*.
        weights:
            Ignored — this model does not use weights.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if len(x) < 1:
            raise ValueError("EmpiricalCV.fit() requires at least one data point.")

        self.x_ = x
        self.y_ = y
        self.weights_ = None  # not used

        unique_concs = np.unique(x)

        # Pre-check: if NO concentration has ≥2 replicates, CV is entirely
        # uncomputable.  Raise before entering the loop (and before emitting
        # any per-concentration warnings) so callers see a single clear error.
        pre_counts = {float(c): int(np.sum(x == c)) for c in unique_concs}
        if all(n < 2 for n in pre_counts.values()):
            raise ValueError(
                f"EmpiricalCV requires ≥2 observations per concentration to compute CV. "
                f"All {len(pre_counts)} concentration level(s) have exactly 1 replicate. "
                f"Provide data with ≥2 replicates per concentration."
            )

        cv_table: dict[float, float] = {}
        mean_table: dict[float, float] = {}
        count_table: dict[float, int] = {}

        for c in unique_concs:
            group = y[x == c]
            n = len(group)
            count_table[float(c)] = n

            if n < self.min_replicates:
                warnings.warn(
                    f"EmpiricalCV: concentration {c} has only {n} replicate(s); "
                    f"CV is unreliable with fewer than {self.min_replicates} replicates.",
                    UserWarning,
                    stacklevel=2,
                )

            # A single replicate makes CV undefined — store NaN
            if n < 2:
                cv_table[float(c)] = float('nan')
                mean_table[float(c)] = float(np.mean(group))
                continue

            mean_g = float(np.mean(group))
            std_g = float(np.std(group, ddof=1))
            cv_table[float(c)] = float(std_g / mean_g) if mean_g != 0 else float('nan')
            mean_table[float(c)] = mean_g

        self.cv_table_ = cv_table
        self.mean_table_ = mean_table
        self.replicate_counts_ = count_table
        self.params_ = {}
        self.is_fitted_ = True

        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, x_new: np.ndarray) -> np.ndarray:
        """Linearly interpolate group means to *x_new*.

        Outside the calibration range the nearest edge value is used
        (constant extrapolation).
        """
        self._check_is_fitted()
        x_new = np.asarray(x_new, dtype=float)
        concs = np.array(sorted(self.mean_table_.keys()))
        means = np.array([self.mean_table_[c] for c in concs])
        return np.interp(x_new, concs, means)

    # ------------------------------------------------------------------
    # lod / loq
    # ------------------------------------------------------------------

    def lod(self, std_mult: float = 2.0) -> float:
        """EmpiricalCV cannot determine LOD — always returns ``np.inf``."""
        return np.inf

    def supports_lod(self) -> bool:
        return False

    def loq(self, cv_thresh: float = DEFAULT_CV_THRESH) -> float:
        """Return LOQ using a sliding-window CV threshold search.

        Blanks (concentration = 0) are excluded per v1 convention.
        """
        self._check_is_fitted()

        # Filter out zero-concentration blanks
        valid = {c: cv for c, cv in self.cv_table_.items() if c > 0}
        if not valid:
            return np.inf

        concs = np.array(sorted(valid.keys()))
        cvs = np.array([valid[c] for c in concs])

        return find_loq_threshold(
            concs,
            cvs,
            cv_thresh=cv_thresh,
            window=self.sliding_window,
        )

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        self._check_is_fitted()
        return {
            'loq': self.loq(),
            'cv_table': dict(self.cv_table_),
            'mean_table': dict(self.mean_table_),
            'n_replicates': dict(self.replicate_counts_),
        }

