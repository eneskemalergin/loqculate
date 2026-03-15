"""OriginalCV — verbatim port of old/loq_by_cv.py calculate_LOQ_byCV().

Reproduces the exact Pino 2020 empirical-CV logic:
  - Per-concentration CV using ddof=1 (Bessel-corrected)
  - LOQ = lowest concentration > 0 with CV ≤ cv_thresh (single-point, ≤ not <)
  - No LOD support
  - No sliding window
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from loqculate.config import DEFAULT_CV_THRESH
from loqculate.models.base import CalibrationModel
from loqculate.utils.cv import vectorized_cv_stats


class OriginalCV(CalibrationModel):
    """Verbatim port of loq_by_cv.py ``calculate_LOQ_byCV()``.

    Reproduces the original empirical-CV LOQ rule exactly:

    * Groups observations by concentration
    * Computes CV = std(ddof=1) / mean per group
    * LOQ = lowest concentration > 0 with CV **≤** ``cv_thresh``
      (original uses ``<= 0.2``, not strict ``<``)
    * **No LOD** (the original script does not compute one)
    * **No sliding window** — single-point rule

    Parameters
    ----------
    cv_thresh:
        CV threshold; original uses ``<= cv_thresh`` (non-strict).
    """

    def __init__(self, cv_thresh: float = DEFAULT_CV_THRESH) -> None:
        super().__init__()
        self.cv_thresh = cv_thresh

        self._loq_val: float = np.inf
        self.cv_table_: Dict[float, float] = {}
        self.mean_table_: Dict[float, float] = {}
        self.replicate_counts_: Dict[float, int] = {}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> 'OriginalCV':
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        self.x_ = x
        self.y_ = y
        self.weights_ = None

        unique_concs, counts, means, cvs = vectorized_cv_stats(x, y)

        self.cv_table_ = dict(zip(unique_concs.tolist(), cvs.tolist()))
        self.mean_table_ = dict(zip(unique_concs.tolist(), means.tolist()))
        self.replicate_counts_ = dict(zip(unique_concs.tolist(), counts.astype(int).tolist()))

        # Single-point rule: lowest concentration > 0 with CV <= cv_thresh
        # Original uses <=, not strict < (see loq_by_cv.py line:
        #   good_cv_rows = peptideCVs[(peptideCVs['%CV'] <= 0.2) & ...]
        pos_mask = unique_concs > 0
        good_mask = pos_mask & (cvs <= self.cv_thresh)

        if np.any(good_mask):
            self._loq_val = float(unique_concs[good_mask].min())
        else:
            self._loq_val = np.inf

        self.params_ = {}
        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, x_new: np.ndarray) -> np.ndarray:
        """Not applicable for OriginalCV (no regression model)."""
        raise NotImplementedError('OriginalCV has no regression model; predict() is not supported.')

    # ------------------------------------------------------------------
    # lod / loq
    # ------------------------------------------------------------------

    def lod(self, std_mult: float = 2.0) -> float:
        """OriginalCV does not compute an LOD; always returns inf."""
        return np.inf

    def loq(self, cv_thresh: float = DEFAULT_CV_THRESH) -> float:
        """LOQ computed during fit().  ``cv_thresh`` argument ignored after fit."""
        self._check_is_fitted()
        return self._loq_val

    def supports_lod(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        self._check_is_fitted()
        return {
            'lod': np.inf,
            'loq': self._loq_val,
            'n_points': len(self.x_),
            'n_concentrations': len(self.cv_table_),
            'compat_model': 'OriginalCV',
        }
