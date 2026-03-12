from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit

from loqculate.config import (
    DEFAULT_BOOT_REPS,
    DEFAULT_CV_THRESH,
    DEFAULT_LOQ_GRID_POINTS,
    DEFAULT_MIN_LINEAR_POINTS,
    DEFAULT_MIN_NOISE_POINTS,
    DEFAULT_STD_MULT,
)
from loqculate.models.base import CalibrationModel
from loqculate.utils.bootstrap import bootstrap_predictions
from loqculate.utils.threshold import find_loq_threshold
from loqculate.utils.weights import inverse_sqrt_weights

# ---------------------------------------------------------------------------
# Piecewise function
# ---------------------------------------------------------------------------

def _piecewise(x: np.ndarray, a: float, b: float, c_minus_b: float) -> np.ndarray:
    """Piecewise linear / noise plateau model.

    y = max(c, a*x + b)  where  c = b + c_minus_b

    Parameterized as (a, b, c_minus_b) so that ``c_minus_b >= 0`` can be
    enforced as a lower bound without introducing a dependent variable.
    """
    c = b + c_minus_b
    return np.maximum(c, a * x + b)


# ---------------------------------------------------------------------------
# Initial-guess helpers (ported from original Pino implementation)
# ---------------------------------------------------------------------------

def _initialize_params_legacy(x: np.ndarray, y: np.ndarray):
    """2019 Pino model: slope from two highest points, intercept at lowest."""
    conc_list = sorted(set(x))
    top = conc_list[-1]
    second_top = conc_list[-2]
    bottom = conc_list[0]

    # Use means per concentration level
    def mean_at(c):
        return float(np.mean(y[x == c]))

    slope = (mean_at(second_top) - mean_at(top)) / (second_top - top)
    noise_intercept = mean_at(bottom)
    linear_intercept = mean_at(top) - slope * top

    if noise_intercept <= linear_intercept:
        noise_intercept = linear_intercept * 1.05

    return slope, linear_intercept, noise_intercept


def _initialize_params(x: np.ndarray, y: np.ndarray, weights: np.ndarray):
    """Improved init: WLS linear fit above the two lowest concentrations."""
    conc_list = sorted(set(x))
    noise_mask = x == conc_list[0]
    if len(conc_list) > 1:
        noise_mask = noise_mask | (x == conc_list[1])

    noise_intercept = float(np.mean(y[noise_mask]))

    reg_x = x[~noise_mask]
    reg_y = y[~noise_mask]
    reg_w = weights[~noise_mask]

    if len(reg_x) < 2:
        # fallback
        slope, linear_intercept = 1.0, noise_intercept
    else:
        # weighted least-squares linear fit
        W = np.diag(reg_w)
        A = np.column_stack([reg_x, np.ones_like(reg_x)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(W @ A, W @ reg_y, rcond=None)
            slope, linear_intercept = coeffs[0], coeffs[1]
        except Exception:
            slope, linear_intercept = 1.0, noise_intercept

    if noise_intercept <= linear_intercept:
        noise_intercept = linear_intercept * 1.05

    return slope, linear_intercept, noise_intercept


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PiecewiseWLS(CalibrationModel):
    """Pino 2020 piecewise WLS model, reimplemented with scipy TRF optimization.

    Key improvements over the original implementation
    --------------------------------------------------
    * ``scipy.optimize.curve_fit`` (TRF) replaces ``lmfit`` Levenberg-Marquardt.
      Single-fit latency is comparable (~1.8 ms, 40 observations, 3 parameters);
      TRF is preferred for its numerical robustness with parameter bounds.
    * Parameter initialisation uses pure numpy instead of a pandas ``GroupBy``
      (~10× faster per replicate: 0.03 ms vs 0.3 ms), but both are dwarfed by
      the ~1.8 ms optimizer call so total per-peptide latency is comparable.
    * Observation sigma is pre-computed once from the primary fit's weights and
      indexed per replicate (avoids repeated ``inverse_sqrt_weights`` allocation).
    * Bootstrap correctly cold-starts each replicate from its own resampled data,
      preserving the full sampling distribution of parameter estimates.
    * LOQ search uses a vectorized sliding-window (prevents false LOQs from
      non-monotonic CV bounces).
    * Multiprocessing: the CLI dispatches peptides across CPU cores; at scale
      (23 k peptides, 32 cores) the full dataset runs in ~1 min.

    Parameters
    ----------
    init_method:
        ``'legacy'`` uses the 2019 Pino slope-from-top-two-points init.
        ``'auto'`` uses the improved WLS-based init.
    n_boot_reps:
        Bootstrap replicates for LOQ / prediction interval calculation.
    seed:
        Seed forwarded to the bootstrap engine.
    min_noise_points:
        Minimum unique concentration levels below the intersection needed to
        trust the LOD.
    min_linear_points:
        Minimum unique concentration levels above the LOD needed to trust it.
    sliding_window:
        Consecutive points that must stay below the CV threshold.
    grid_points:
        Number of bins from LOD to max(x) for the bootstrap CV grid.
    """

    def __init__(
        self,
        init_method: str = 'legacy',
        n_boot_reps: int = DEFAULT_BOOT_REPS,
        seed: int = 42,
        min_noise_points: int = DEFAULT_MIN_NOISE_POINTS,
        min_linear_points: int = DEFAULT_MIN_LINEAR_POINTS,
        sliding_window: int = 3,
        grid_points: int = DEFAULT_LOQ_GRID_POINTS,
    ) -> None:
        super().__init__()
        self.init_method = init_method
        self.n_boot_reps = n_boot_reps
        self.seed = seed
        self.min_noise_points = min_noise_points
        self.min_linear_points = min_linear_points
        self.sliding_window = sliding_window
        self.grid_points = grid_points

        # Cache for LOD/LOQ so they are computed at most once per fit
        self._lod_cache: dict = {}
        self._loq_cache: dict = {}
        self._boot_summary: Optional[dict] = None
        self._x_grid: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> "PiecewiseWLS":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if len(x) < 3:
            raise ValueError(
                f"PiecewiseWLS.fit() requires at least 3 data points, got {len(x)}."
            )

        self.x_ = x
        self.y_ = y

        # Compute weights if not provided
        if weights is None:
            w = inverse_sqrt_weights(x)
        else:
            w = np.asarray(weights, dtype=float)
        self.weights_ = w

        # Initial parameter guesses
        if self.init_method == 'legacy':
            a0, b0, c0 = _initialize_params_legacy(x, y)
        else:
            a0, b0, c0 = _initialize_params(x, y, w)

        cmb0 = c0 - b0
        # Nudge initial guesses off lower boundaries to improve TRF stability.
        # TRF finite-difference Jacobian becomes one-sided when p0 sits exactly
        # on a bound, which can cause ValueError or slow convergence.
        if a0 <= 0:
            a0 = 1e-6
        if cmb0 <= 0:
            cmb0 = 1e-5

        # sigma = 1/weight  (curve_fit interprets sigma as observation std dev)
        sigma = 1.0 / np.clip(w, 1e-12, None)

        try:
            popt, _ = curve_fit(
                _piecewise,
                x,
                y,
                p0=[a0, b0, cmb0],
                sigma=sigma,
                bounds=([0.0, -np.inf, 0.0], [np.inf, np.inf, np.inf]),
                maxfev=5000,
            )
        except (RuntimeError, ValueError):
            # RuntimeError: max function evaluations exceeded.
            # ValueError: initial guess outside bounds (e.g. degenerate data).
            # Fall back to the (already boundary-clamped) initial guess.
            popt = np.array([a0, b0, cmb0])

        a, b, cmb = popt
        c = b + cmb

        self.params_ = {
            'slope': float(a),
            'intercept_linear': float(b),
            'intercept_noise': float(c),
        }
        self.is_fitted_ = True

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
        a = self.params_['slope']
        b = self.params_['intercept_linear']
        c = self.params_['intercept_noise']
        return np.maximum(c, a * x_new + b)

    # ------------------------------------------------------------------
    # lod
    # ------------------------------------------------------------------

    def lod(self, std_mult: float = DEFAULT_STD_MULT) -> float:
        """Limit of detection using the same math as the original ``calculate_lod()``."""
        self._check_is_fitted()

        if std_mult in self._lod_cache:
            return self._lod_cache[std_mult]

        a = self.params_['slope']
        b = self.params_['intercept_linear']
        c = self.params_['intercept_noise']
        x = self.x_
        y = self.y_

        if a <= 0:
            self._lod_cache[std_mult] = np.inf
            return np.inf

        # Intersection of noise plateau (slope=0, intercept=c) and linear
        # segment (slope=a, intercept=b):  c = a*x_int + b  →  x_int = (c-b)/a
        intersection = (b - c) / (0.0 - a)  # noise slope is 0

        noise_y = y[x < intersection]
        if len(noise_y) < self.min_noise_points:
            self._lod_cache[std_mult] = np.inf
            return np.inf

        std_noise = float(np.std(noise_y, ddof=1))
        lod_val = (c + std_mult * std_noise - b) / a

        # Edge-case guards
        if not np.isfinite(lod_val) or lod_val > np.max(x):
            self._lod_cache[std_mult] = np.inf
            return np.inf

        # Require at least min_linear_points unique concentrations ABOVE the LOD
        # so the linear segment is supported by data.
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
            self._boot_summary['cv'],
            cv_thresh=cv_thresh,
            window=self.sliding_window,
        )

        if not np.isfinite(loq_val) or loq_val >= np.max(self.x_) or loq_val <= 0:
            loq_val = np.inf

        self._loq_cache[cv_thresh] = loq_val
        return loq_val

    # ------------------------------------------------------------------
    # prediction_interval
    # ------------------------------------------------------------------

    def prediction_interval(
        self, x_new: np.ndarray, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bootstrap (1-alpha) prediction interval at *x_new* concentrations."""
        self._check_is_fitted()
        x_new = np.asarray(x_new, dtype=float)
        preds, _ = bootstrap_predictions(
            self.x_,
            self.y_,
            PiecewiseWLS,
            {'init_method': self.init_method, 'n_boot_reps': 0},
            x_new,
            n_reps=self.n_boot_reps,
            seed=self.seed,
        )
        lower = np.nanpercentile(preds, 100 * alpha / 2, axis=0)
        upper = np.nanpercentile(preds, 100 * (1 - alpha / 2), axis=0)
        return lower, upper

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        self._check_is_fitted()
        return {
            'slope': self.params_['slope'],
            'intercept_linear': self.params_['intercept_linear'],
            'intercept_noise': self.params_['intercept_noise'],
            'lod': self.lod(),
            'loq': self.loq(),
            'n_points': len(self.x_),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_boot_summary(self, lod_val: float) -> None:
        """Run bootstrap once and cache the summary.

        Uses :func:`_bootstrap_lean_piecewise` which calls ``curve_fit``
        directly (avoiding per-replicate :class:`PiecewiseWLS` construction)
        and reuses pre-computed ``sigma_all`` to skip redundant weight
        recomputation.  Each replicate still cold-starts from its own data so
        the bootstrap correctly samples the full parameter distribution.
        """
        if self._boot_summary is not None:
            return

        x_grid = np.linspace(lod_val, float(np.max(self.x_)), num=self.grid_points)

        # p0_warm passed for API completeness; cold-start is used per replicate.
        a = self.params_['slope']
        b = self.params_['intercept_linear']
        cmb = self.params_['intercept_noise'] - b
        p0_warm = np.array([a, b, max(cmb, 1e-5)])

        # Pre-compute sigma once; bootstrap reps reuse sigma[idx].
        sigma_all = 1.0 / np.clip(self.weights_, 1e-12, None)

        _, summary = _bootstrap_lean_piecewise(
            self.x_,
            self.y_,
            p0_warm=p0_warm,
            sigma_all=sigma_all,
            x_grid=x_grid,
            n_reps=self.n_boot_reps,
            seed=self.seed,
        )
        self._x_grid = x_grid
        self._boot_summary = summary


# ---------------------------------------------------------------------------
# Lean warm-start bootstrap (module-level, no circular import)
# ---------------------------------------------------------------------------

def _bootstrap_lean_piecewise(
    x: np.ndarray,
    y: np.ndarray,
    p0_warm: np.ndarray,
    sigma_all: np.ndarray,
    x_grid: np.ndarray,
    n_reps: int,
    seed: int,
) -> tuple:
    """Bootstrap the piecewise model with cold-start initialisation per replicate.

    Compared to the generic :func:`loqculate.utils.bootstrap.bootstrap_predictions`
    this avoids per-replicate :class:`PiecewiseWLS` object construction by
    calling ``curve_fit`` directly.  Each replicate still runs
    :func:`_initialize_params_legacy` on its own resampled data (cold start),
    so the bootstrap correctly samples the full parameter variability — using
    ``p0_warm`` here would bias estimates toward the primary fit.

    Pre-indexed ``sigma_all[idx]`` eliminates one ``inverse_sqrt_weights``
    call per replicate (trivial individually, eliminates repeated allocation).

    Parameters
    ----------
    p0_warm : array [a, b, cmb]
        Reserved for future use (kept in signature for API stability).
        Currently unused — each replicate cold-starts from its own data.
    sigma_all : array, length n
        ``1 / weights`` for all original observations; indexed per replicate.
    x_grid : array
        Concentration grid at which bootstrap predictions are evaluated.
    n_reps : int
        Number of bootstrap replicates.
    seed : int
        Base seed for :class:`numpy.random.SeedSequence`.

    Returns
    -------
    predictions : ndarray, shape (n_reps, len(x_grid))
    summary : dict
        Keys: ``mean``, ``std``, ``cv``, ``pct_5``, ``pct_95``.
    """
    n = len(x)
    n_grid = len(x_grid)

    # Shortcut for degenerate (constant) peptides.
    if np.unique(y).size <= 1:
        const = float(y[0]) if n else np.nan
        summary = {
            'mean': np.full(n_grid, const),
            'std': np.zeros(n_grid),
            'cv': np.full(n_grid, np.inf),
            'pct_5': np.full(n_grid, const),
            'pct_95': np.full(n_grid, const),
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
        sb = sigma_all[idx]

        # Retry if all resampled y values are identical (pathological but
        # possible for peptides with many zero-area observations).
        retries = 0
        while np.unique(yb).size <= 1:
            idx = rng.choice(n, size=n, replace=True)
            xb, yb, sb = x[idx], y[idx], sigma_all[idx]
            retries += 1
            if retries > 100:
                predictions[i] = np.nan
                break
        else:
            # Cold-start: compute initialisation from this replicate's data so
            # the bootstrap correctly samples the full parameter distribution.
            try:
                a0, b0, c0 = _initialize_params_legacy(xb, yb)
                cmb0 = c0 - b0
                if a0 <= 0:
                    a0 = 1e-6
                if cmb0 <= 0:
                    cmb0 = 1e-5
                popt, _ = curve_fit(
                    _piecewise,
                    xb,
                    yb,
                    p0=[a0, b0, cmb0],
                    sigma=sb,
                    bounds=([0.0, -np.inf, 0.0], [np.inf, np.inf, np.inf]),
                    maxfev=5000,
                )
                a, b, cmb = popt
            except (RuntimeError, ValueError):
                a, b, cmb = a0, b0, cmb0
            c = b + cmb
            predictions[i] = np.maximum(c, a * x_grid + b)

    with np.errstate(invalid='ignore'):
        mean_pred = np.nanmean(predictions, axis=0)
        std_pred = np.nanstd(predictions, axis=0, ddof=1)
        cv_pred = np.where(mean_pred != 0, std_pred / mean_pred, np.inf)

    summary = {
        'mean': mean_pred,
        'std': std_pred,
        'cv': cv_pred,
        'pct_5': np.nanpercentile(predictions, 5, axis=0),
        'pct_95': np.nanpercentile(predictions, 95, axis=0),
    }
    return predictions, summary

