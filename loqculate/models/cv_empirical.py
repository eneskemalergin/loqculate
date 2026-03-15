from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np

from loqculate.config import DEFAULT_CV_THRESH, DEFAULT_SLIDING_WINDOW
from loqculate.models.base import CalibrationModel
from loqculate.utils.cv import vectorized_cv_stats as _vectorized_cv_stats
from loqculate.utils.threshold import find_loq_threshold


def _sliding_window_loq(
    concs: np.ndarray, cvs: np.ndarray,
    cv_thresh: float, window: int,
) -> float:
    """Same logic as find_loq_threshold but inlined for zero-overhead bulk use."""
    pos = concs > 0
    x_pos = concs[pos]
    cv_pos = cvs[pos]
    n = len(x_pos)
    if n == 0:
        return np.inf
    ew = min(window, n)
    below = cv_pos <= cv_thresh
    for i in range(n - ew + 1):
        if below[i: i + ew].all():
            return float(x_pos[i])
    remaining = n - (n - ew + 1)
    if remaining > 0 and below[n - ew:].all():
        return float(x_pos[n - ew])
    return np.inf


class EmpiricalCV(CalibrationModel):
    """Empirical CV model: derives LOQ directly from replicate CVs.

    This model has **no regression**.  It groups replicate measurements at
    each concentration level, computes CV = std/mean, and applies the same
    sliding-window threshold search used by :class:`PiecewiseWLS`.

    It is a reimplementation of the original ``loq_by_cv.py`` in the base-class contract.

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
    # fit (vectorized — no Python per-concentration loop)
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

        # Vectorized groupby via bincount — replaces Python for-loop
        unique_concs, counts, means, cvs = _vectorized_cv_stats(x, y)

        # Pre-check: if NO concentration has ≥2 replicates, CV is entirely
        # uncomputable.
        if np.all(counts < 2):
            raise ValueError(
                f"EmpiricalCV requires ≥2 observations per concentration to compute CV. "
                f"All {len(unique_concs)} concentration level(s) have exactly 1 replicate. "
                f"Provide data with ≥2 replicates per concentration."
            )

        # Emit warnings for low-replicate concentrations
        low_rep_mask = counts < self.min_replicates
        if np.any(low_rep_mask):
            for idx in np.where(low_rep_mask)[0]:
                c = unique_concs[idx]
                n = int(counts[idx])
                warnings.warn(
                    f"EmpiricalCV: concentration {c} has only {n} replicate(s); "
                    f"CV is unreliable with fewer than {self.min_replicates} replicates.",
                    UserWarning,
                    stacklevel=2,
                )

        # Build dicts (required by predict/loq/summary API)
        conc_f = unique_concs.astype(float)
        self.cv_table_ = dict(zip(conc_f, cvs.astype(float)))
        self.mean_table_ = dict(zip(conc_f, means.astype(float)))
        self.replicate_counts_ = {float(c): int(n) for c, n in zip(unique_concs, counts)}
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

        Blanks (concentration = 0) are excluded (LOQ physically cannot be zero concentration).
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

    # ------------------------------------------------------------------
    # Bulk batch classmethod — processes ALL peptides in one pass
    # ------------------------------------------------------------------

    @classmethod
    def compute_loqs_bulk(
        cls,
        peptides: np.ndarray,
        concentrations: np.ndarray,
        areas: np.ndarray,
        *,
        sliding_window: int = DEFAULT_SLIDING_WINDOW,
        cv_thresh: float = DEFAULT_CV_THRESH,
    ) -> Dict[str, float]:
        """Compute LOQs for many peptides in a single vectorized pass.

        This is the fast path for batch processing — avoids per-peptide
        object creation, multiprocessing pickle overhead, and Python loops.
        Uses numpy ``bincount`` for all groupby operations, matching the
        speed of the pandas-groupby approach in ``loq_by_cv.py``.

        Parameters
        ----------
        peptides:
            Peptide identifiers (one per measurement row), same length
            as *concentrations* and *areas*.
        concentrations:
            Concentration values (repetitions allowed).
        areas:
            Signal areas, same length as *concentrations*.
        sliding_window:
            Consecutive calibration points below *cv_thresh* required.
        cv_thresh:
            CV threshold for LOQ determination.

        Returns
        -------
        dict
            ``{peptide_name: loq_value}`` where loq is ``np.inf`` if
            undetermined.
        """
        peptides = np.asarray(peptides)
        concentrations = np.asarray(concentrations, dtype=float)
        areas = np.asarray(areas, dtype=float)

        # Step 1: Encode (peptide, concentration) pairs as integer labels
        # for fast bincount groupby.
        unique_peps, pep_inv = np.unique(peptides, return_inverse=True)
        unique_concs, conc_inv = np.unique(concentrations, return_inverse=True)
        n_peps = len(unique_peps)
        n_concs = len(unique_concs)

        # Combined group index: pep_idx * n_concs + conc_idx
        group_idx = pep_inv * n_concs + conc_inv
        n_groups = n_peps * n_concs

        counts = np.bincount(group_idx, minlength=n_groups).reshape(n_peps, n_concs).astype(float)
        sums = np.bincount(group_idx, weights=areas, minlength=n_groups).reshape(n_peps, n_concs)
        sum_sq = np.bincount(group_idx, weights=areas * areas, minlength=n_groups).reshape(n_peps, n_concs)

        with np.errstate(invalid='ignore', divide='ignore'):
            means = np.where(counts > 0, sums / counts, 0.0)
            var = np.where(
                counts > 1,
                (sum_sq / counts - means ** 2) * counts / (counts - 1.0),
                0.0,
            )
        var = np.maximum(var, 0.0)
        stds = np.sqrt(var)

        with np.errstate(invalid='ignore', divide='ignore'):
            cvs = np.where((counts >= 2) & (means != 0), stds / means, np.nan)

        # Step 2: For each peptide, apply sliding-window LOQ search on
        #         the concentration axis (excluding conc=0).
        results: Dict[str, float] = {}
        for pi in range(n_peps):
            loq = _sliding_window_loq(unique_concs, cvs[pi], cv_thresh, sliding_window)
            results[str(unique_peps[pi])] = loq

        return results

