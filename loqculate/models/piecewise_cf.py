"""Closed-form piecewise WLS model using discrete knot search.

Implements the same statistical model as :class:`PiecewiseWLS`
(y = max(c, a*x + b), weights 1/sqrt(x)) but replaces the TRF optimizer
with an exhaustive discrete knot search followed by analytical refinement.
The solver is deterministic and cannot be trapped in local minima.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from loqculate.config import (
    DEFAULT_BOOT_REPS,
    DEFAULT_CV_THRESH,
    DEFAULT_LOQ_GRID_POINTS,
    DEFAULT_MIN_LINEAR_POINTS,
    DEFAULT_MIN_NOISE_POINTS,
    DEFAULT_STD_MULT,
    KNOT_SEARCH_SINGULAR_THRESHOLD,
)
from loqculate.models.base import CalibrationModel
from loqculate.utils.knot_search import find_knot
from loqculate.utils.threshold import find_loq_threshold
from loqculate.utils.weights import inverse_sqrt_weights

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class PiecewiseCF(CalibrationModel):
    """Closed-form piecewise WLS model using discrete knot search.

    Same statistical model as :class:`PiecewiseWLS`: ``y = max(c, a*x + b)``
    with precision weights ``W_i = 1/x_i``.  No optimizer, no initial-guess
    sensitivity, fully deterministic.

    Parameters
    ----------
    n_boot_reps:
        Bootstrap replicates for LOQ / CV profile calculation.
    seed:
        RNG seed forwarded to the bootstrap.
    min_noise_points:
        Minimum observations with x < intersection required to compute LOD.
    min_linear_points:
        Minimum unique concentration levels above LOD required to trust it.
    sliding_window:
        Consecutive grid points that must stay below the CV threshold.
    grid_points:
        Number of bins from LOD to max(x) for the bootstrap CV grid.
    """

    def __init__(
        self,
        n_boot_reps: int = DEFAULT_BOOT_REPS,
        seed: int = 42,
        min_noise_points: int = DEFAULT_MIN_NOISE_POINTS,
        min_linear_points: int = DEFAULT_MIN_LINEAR_POINTS,
        sliding_window: int = 3,
        grid_points: int = DEFAULT_LOQ_GRID_POINTS,
    ) -> None:
        super().__init__()
        self.n_boot_reps = n_boot_reps
        self.seed = seed
        self.min_noise_points = min_noise_points
        self.min_linear_points = min_linear_points
        self.sliding_window = sliding_window
        self.grid_points = grid_points

        self._lod_cache: dict = {}
        self._loq_cache: dict = {}
        self._boot_summary: Optional[dict] = None
        self._x_grid: Optional[np.ndarray] = None
        self._gram_inv: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> "PiecewiseCF":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if len(x) < 3:
            raise ValueError(f"PiecewiseCF.fit() requires at least 3 data points, got {len(x)}.")

        self.x_ = x
        self.y_ = y

        if weights is None:
            w = inverse_sqrt_weights(x)
        else:
            w = np.asarray(weights, dtype=float)
        self.weights_ = w
        W = w**2

        kr = find_knot(x, y, W)

        self.params_ = {
            "slope": kr.slope,
            "intercept_linear": kr.intercept,
            "intercept_noise": kr.noise_intercept,
            "knot_x": kr.knot_x,
        }
        self.is_fitted_ = True

        # Compute and cache the 2x2 Gram matrix inverse for the linear segment.
        # This is the inverse of [[sum(W*x^2), sum(W*x)], [sum(W*x), sum(W)]].
        # Stored now at zero marginal cost so covariance() (C4) needs no refit.
        lin_mask = x > kr.knot_x
        x_lin = x[lin_mask]
        W_lin = W[lin_mask]
        self._gram_inv = _gram_inverse(x_lin, W_lin)

        # Invalidate caches
        self._lod_cache = {}
        self._loq_cache = {}
        self._boot_summary = None
        self._x_grid = None

        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, x_new: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        x_new = np.asarray(x_new, dtype=float)
        a = self.params_["slope"]
        b = self.params_["intercept_linear"]
        c = self.params_["intercept_noise"]
        return np.maximum(c, a * x_new + b)

    # ------------------------------------------------------------------
    # lod
    # ------------------------------------------------------------------

    def lod(self, std_mult: float = DEFAULT_STD_MULT) -> float:
        """Limit of detection using the same formula as :class:`PiecewiseWLS`."""
        self._check_is_fitted()

        if std_mult in self._lod_cache:
            return self._lod_cache[std_mult]

        a = self.params_["slope"]
        b = self.params_["intercept_linear"]
        c = self.params_["intercept_noise"]
        x = self.x_
        y = self.y_

        if a <= 0:
            self._lod_cache[std_mult] = np.inf
            return np.inf

        # Intersection of noise plateau and linear segment: x_int = (c - b) / a
        intersection = (c - b) / a

        noise_y = y[x < intersection]
        if len(noise_y) < self.min_noise_points:
            self._lod_cache[std_mult] = np.inf
            return np.inf

        std_noise = float(np.std(noise_y, ddof=1))
        lod_val = (c + std_mult * std_noise - b) / a

        if not np.isfinite(lod_val) or lod_val > np.max(x):
            self._lod_cache[std_mult] = np.inf
            return np.inf

        n_above = int(np.sum(np.unique(x) > lod_val))
        if n_above < self.min_linear_points:
            self._lod_cache[std_mult] = np.inf
            return np.inf

        self._lod_cache[std_mult] = float(lod_val)
        return float(lod_val)

    # ------------------------------------------------------------------
    # loq
    # ------------------------------------------------------------------

    def loq(self, cv_thresh: float = DEFAULT_CV_THRESH) -> float:
        """Limit of quantitation via bootstrap CV + sliding-window search."""
        self._check_is_fitted()

        if cv_thresh in self._loq_cache:
            return self._loq_cache[cv_thresh]

        lod_val = self.lod()
        if not np.isfinite(lod_val):
            self._loq_cache[cv_thresh] = np.inf
            return np.inf

        self._ensure_boot_summary(lod_val)

        if self._x_grid is None or self._boot_summary is None:
            self._loq_cache[cv_thresh] = np.inf
            return np.inf

        loq_val = find_loq_threshold(
            self._x_grid,
            self._boot_summary["cv"],
            cv_thresh=cv_thresh,
            window=self.sliding_window,
        )

        if not np.isfinite(loq_val) or loq_val >= np.max(self.x_) or loq_val <= 0:
            loq_val = np.inf

        self._loq_cache[cv_thresh] = loq_val
        return loq_val

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        self._check_is_fitted()
        return {
            "slope": self.params_["slope"],
            "intercept_linear": self.params_["intercept_linear"],
            "intercept_noise": self.params_["intercept_noise"],
            "knot_x": self.params_["knot_x"],
            "lod": self.lod(),
            "loq": self.loq(),
            "n_points": len(self.x_),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_boot_summary(self, lod_val: float) -> None:
        if self._boot_summary is not None:
            return

        x_grid = np.linspace(lod_val, float(np.max(self.x_)), num=self.grid_points)
        W = self.weights_**2

        _, summary = _bootstrap_loop_cf(
            self.x_,
            self.y_,
            W,
            x_grid,
            n_reps=self.n_boot_reps,
            seed=self.seed,
        )
        self._x_grid = x_grid
        self._boot_summary = summary


# ---------------------------------------------------------------------------
# Gram matrix inverse (stored in fit, used by covariance() in C4)
# ---------------------------------------------------------------------------


def _gram_inverse(x_lin: np.ndarray, W_lin: np.ndarray) -> Optional[np.ndarray]:
    """Return the 2x2 inverse of the WLS Gram matrix for the linear segment.

    G = [[sum(W*x^2), sum(W*x)], [sum(W*x), sum(W)]]
    Returns None when det(G) < KNOT_SEARCH_SINGULAR_THRESHOLD (degenerate).
    """
    sum_W = float(np.sum(W_lin))
    sum_Wx = float(np.sum(W_lin * x_lin))
    sum_Wxx = float(np.sum(W_lin * x_lin * x_lin))
    det = sum_Wxx * sum_W - sum_Wx * sum_Wx
    if abs(det) < KNOT_SEARCH_SINGULAR_THRESHOLD:
        return None
    inv = np.array([[sum_W, -sum_Wx], [-sum_Wx, sum_Wxx]]) / det
    return inv


# ---------------------------------------------------------------------------
# Loop bootstrap (vectorized path added in C5)
# ---------------------------------------------------------------------------


def _bootstrap_loop_cf(
    x: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    x_grid: np.ndarray,
    n_reps: int,
    seed: int,
) -> tuple:
    """Bootstrap the CF model with one find_knot call per replicate.

    Mirrors :func:`loqculate.models.piecewise_wls._bootstrap_lean_piecewise`
    in structure and seeding convention.  Uses ``SeedSequence(seed).spawn``
    so replicate ``i`` always draws the same subsample regardless of whether
    the loop runs in full or resuming from a checkpoint.

    Parameters
    ----------
    x, y, W:
        Full dataset (precision weights W = w^2).
    x_grid:
        Concentration grid at which bootstrap predictions are evaluated.
    n_reps:
        Number of bootstrap replicates.
    seed:
        Base seed for :class:`numpy.random.SeedSequence`.

    Returns
    -------
    predictions : ndarray, shape (n_reps, len(x_grid))
    summary : dict — keys ``mean``, ``std``, ``cv``, ``pct_5``, ``pct_95``.
    """
    n = len(x)
    n_grid = len(x_grid)

    # Degenerate shortcut: constant signal, no useful bootstrap.
    if np.unique(y).size <= 1:
        const = float(y[0]) if n else np.nan
        summary = {
            "mean": np.full(n_grid, const),
            "std": np.zeros(n_grid),
            "cv": np.full(n_grid, np.inf),
            "pct_5": np.full(n_grid, const),
            "pct_95": np.full(n_grid, const),
        }
        return np.full((n_reps, n_grid), np.nan), summary

    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(n_reps)
    predictions = np.empty((n_reps, n_grid))

    for i, cs in enumerate(child_seeds):
        rng = np.random.default_rng(cs)
        idx = rng.choice(n, size=n, replace=True)
        xb = x[idx]
        yb = y[idx]
        Wb = W[idx]

        # Retry if all resampled y are identical (rare but possible).
        retries = 0
        while np.unique(yb).size <= 1:
            idx = rng.choice(n, size=n, replace=True)
            xb, yb, Wb = x[idx], y[idx], W[idx]
            retries += 1
            if retries > 100:
                predictions[i] = np.nan
                break
        else:
            try:
                kr = find_knot(xb, yb, Wb)
                a, b, c = kr.slope, kr.intercept, kr.noise_intercept
                predictions[i] = np.maximum(c, a * x_grid + b)
            except (ValueError, Exception):
                predictions[i] = np.nan

    with np.errstate(invalid="ignore"):
        mean_pred = np.nanmean(predictions, axis=0)
        std_pred = np.nanstd(predictions, axis=0, ddof=1)
        cv_pred = np.where(mean_pred != 0, std_pred / mean_pred, np.inf)

    summary = {
        "mean": mean_pred,
        "std": std_pred,
        "cv": cv_pred,
        "pct_5": np.nanpercentile(predictions, 5, axis=0),
        "pct_95": np.nanpercentile(predictions, 95, axis=0),
    }
    return predictions, summary
