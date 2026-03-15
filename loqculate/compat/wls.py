"""OriginalWLS — verbatim port of old/calculate-loq.py process_peptide().

Reproduces the exact numerical logic of the Pino 2020 WLS implementation:
  - Parameter initialisation: slope from top-two concentration points
  - Solver: scipy.optimize.least_squares (TRF), same parameterisation as lmfit
  - LOD: intersection of flat noise floor and linear segment + std_mult * σ_noise
  - LOQ: bootstrap CV, single-point rule (window=1, no sliding window)
  - Bootstrap seeding: per-replicate SeedSequence(i) for i in range(n_boot),
    identical to _bootstrap_once() in the original script
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from loqculate.config import (
    DEFAULT_BOOT_REPS,
    DEFAULT_CV_THRESH,
    DEFAULT_LOQ_GRID_POINTS,
    DEFAULT_MIN_LINEAR_POINTS,
    DEFAULT_MIN_NOISE_POINTS,
    DEFAULT_STD_MULT,
    DEFAULT_WEIGHT_CAP,
)
from loqculate.models.base import CalibrationModel


# ---------------------------------------------------------------------------
# Module-level helpers (verbatim ports from calculate-loq.py)
# ---------------------------------------------------------------------------

def _initialize_params_legacy(
    x: np.ndarray, y: np.ndarray
) -> Tuple[float, float, float]:
    """2019 Pino init: slope from top-two points, noise from bottom point.

    Verbatim port of ``initialize_params_legacy()`` in calculate-loq.py.
    """
    conc_list = sorted(set(x))
    top = conc_list[-1]
    second_top = conc_list[-2]
    bottom = conc_list[0]

    def mean_at(c: float) -> float:
        return float(np.mean(y[x == c]))

    slope = (mean_at(second_top) - mean_at(top)) / (second_top - top)
    noise_intercept = mean_at(bottom)
    linear_intercept = mean_at(top) - slope * top

    if noise_intercept <= linear_intercept:
        noise_intercept = linear_intercept * 1.05

    return slope, linear_intercept, noise_intercept


def _orig_weights(x: np.ndarray) -> np.ndarray:
    """Verbatim port of weight calculation in fit_by_lmfit_yang().

    w_i = min(1 / (sqrt(x_i) + eps), DEFAULT_WEIGHT_CAP)
    """
    return np.minimum(
        1.0 / (np.sqrt(x) + np.finfo(float).eps),
        DEFAULT_WEIGHT_CAP,
    )


def _residuals(
    params: np.ndarray, x: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Weighted residuals for the piecewise model.

    params = [a, b, c_minus_b]  where  c = b + c_minus_b.
    Verbatim port of fcn2min() in fit_by_lmfit_yang().
    """
    a, b, cmb = params
    c = b + cmb
    model_vals = np.maximum(c, a * x + b)
    return (model_vals - y) * weights


def _fit_piecewise_legacy(
    x: np.ndarray, y: np.ndarray
) -> Tuple[float, float, float]:
    """Fit the piecewise model using the legacy initialisation.

    Returns (slope a, linear_intercept b, noise_intercept c).
    """
    weights = _orig_weights(x)
    a0, b0, c0 = _initialize_params_legacy(x, y)
    cmb0 = max(c0 - b0, 1e-5)
    if a0 <= 0:
        a0 = 1e-6

    result = least_squares(
        _residuals,
        x0=[a0, b0, cmb0],
        args=(x, y, weights),
        bounds=([0.0, -np.inf, 0.0], [np.inf, np.inf, np.inf]),
        method='trf',
        max_nfev=5000,
    )
    a, b, cmb = result.x
    c = b + cmb
    return float(a), float(b), float(c)


def _calculate_lod_legacy(
    a: float,
    b: float,
    c: float,
    x: np.ndarray,
    y: np.ndarray,
    std_mult: float,
    min_noise_points: int,
    min_linear_points: int,
) -> Tuple[float, float]:
    """LOD and noise std using the original piecewise formula.

    Verbatim port of calculate_lod(model='piecewise') in calculate-loq.py.

    Returns (lod, std_noise).  Returns (inf, inf) on any guard condition.
    """
    if a <= 0:
        return np.inf, np.inf

    # Intersection of noise plateau (slope=0, intercept=c) and linear line
    # (slope=a, intercept=b): c = a*x + b  ⟹  x = (c - b) / a
    intersection = (b - c) / (0.0 - a)  # = (c - b) / a

    noise_y = y[x.astype(float) < intersection]
    std_noise = float(np.std(noise_y, ddof=1)) if len(noise_y) >= 1 else np.inf

    lod = (c + std_mult * std_noise - b) / a

    # Guard 1: LOD above top concentration
    if lod > float(np.max(x)):
        return np.inf, np.inf

    # Guard 2: require at least one unique concentration below LOD that is
    # neither the global minimum nor the global maximum (original removes both
    # endpoints from the set before comparing).
    curve_points = sorted(set(x.tolist()))
    if len(curve_points) >= 2:
        inner_points = curve_points[1:-1]  # remove min and max
    else:
        inner_points = []

    if not inner_points or lod < float(min(inner_points)):
        return np.inf, np.inf

    return float(lod), float(std_noise)


def _bootstrap_once_legacy(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    seed: int,
) -> np.ndarray:
    """One bootstrap replicate: resample, refit, predict on x_grid.

    Verbatim port of _bootstrap_once(df, new_x, seed, model='piecewise').
    Uses per-replicate SeedSequence(seed) with pandas .sample() to match
    the original script's RNG stream exactly.

    IMPORTANT: the original script receives a DataFrame that is already sorted
    by curvepoint (concentration).  The resulting DataFrame index 0…n-1 maps
    to sorted rows, so ``df.sample()`` with a given RNG seed picks the same
    positional rows as the original.  We replicate this by sorting x/y before
    constructing the DataFrame.
    """
    rng = np.random.default_rng(np.random.SeedSequence(seed))

    # Sort by concentration to reproduce the original script's row ordering.
    sort_idx = np.argsort(x, kind='stable')
    df = pd.DataFrame({'curvepoint': x[sort_idx], 'area': y[sort_idx]})

    # Resample until we have more than one unique area value (matches original guard)
    while True:
        resampled = df.sample(n=len(df), replace=True, random_state=rng)
        if resampled['area'].nunique() > 1:
            break

    bx = resampled['curvepoint'].to_numpy(dtype=float)
    by = resampled['area'].to_numpy(dtype=float)

    a, b, c = _fit_piecewise_legacy(bx, by)
    return np.maximum(x_grid * a + b, c)


def _bootstrap_many_legacy(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    n_boot: int,
) -> dict:
    """Bootstrap the model and compute CV on x_grid.

    Verbatim port of bootstrap_many() in calculate-loq.py.
    Seeds: 0, 1, …, n_boot-1 (matches _bootstrap_once(df, new_x, i)).
    """
    rows = [_bootstrap_once_legacy(x, y, x_grid, i) for i in range(n_boot)]
    mat = np.vstack(rows)  # (n_boot, len(x_grid))

    mean = mat.mean(axis=0)
    std = mat.std(axis=0, ddof=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        cv = np.where(mean != 0, std / mean, np.nan)

    return {
        'boot_x': x_grid,
        'mean': mean,
        'std': std,
        'cv': cv,
        'pct_5': np.percentile(mat, 5, axis=0),
        'pct_95': np.percentile(mat, 95, axis=0),
    }


def _calculate_loq_legacy(
    boot: dict,
    lod: float,
    cv_thresh: float,
) -> float:
    """Single-point LOQ rule from the original calculate_loq().

    LOQ = min(boot_x[boot_x > LOD and boot_cv < cv_thresh]).
    No sliding window — this is the original single-point rule.
    """
    boot_x = boot['boot_x']
    boot_cv = boot['cv']

    above_lod = boot_x > lod
    good_cv = boot_cv < cv_thresh
    mask = above_lod & good_cv

    if not np.any(mask):
        return np.inf

    loq = float(boot_x[mask].min())

    if loq >= float(boot_x.max()) or loq <= 0:
        return np.inf

    return loq


# ---------------------------------------------------------------------------
# OriginalWLS model class
# ---------------------------------------------------------------------------

class OriginalWLS(CalibrationModel):
    """Verbatim port of calculate-loq.py ``process_peptide(model='piecewise')``.

    Reproduces the exact Pino 2020 WLS logic:

    * Legacy initialisation (slope from top-two concentration points)
    * scipy TRF solver with same bounds as lmfit (a ≥ 0, c_minus_b ≥ 0)
    * LOD from intersection of flat noise floor and linear segment
    * LOQ from bootstrap CV with **single-point rule** (no sliding window)
    * Bootstrap seeding: ``SeedSequence(i)`` for ``i in range(n_boot)``,
      identical to ``_bootstrap_once()`` in the original script

    Parameters
    ----------
    std_mult:
        Noise-floor multiplier for LOD (LOD = (c + std_mult·σ - b) / a).
    cv_thresh:
        CV threshold for LOQ (strict inequality: CV < cv_thresh).
    n_boot:
        Number of bootstrap replicates.
    min_noise_points:
        Minimum concentration levels below the LOD intersection.
    min_linear_points:
        Minimum concentration levels above the LOD (not used in original guard,
        kept for API compatibility).
    grid_points:
        Number of grid points from LOD to max(x) for the bootstrap CV sweep.
    """

    def __init__(
        self,
        std_mult: float = DEFAULT_STD_MULT,
        cv_thresh: float = DEFAULT_CV_THRESH,
        n_boot: int = DEFAULT_BOOT_REPS,
        min_noise_points: int = DEFAULT_MIN_NOISE_POINTS,
        min_linear_points: int = DEFAULT_MIN_LINEAR_POINTS,
        grid_points: int = DEFAULT_LOQ_GRID_POINTS,
    ) -> None:
        super().__init__()
        self.std_mult = std_mult
        self.cv_thresh = cv_thresh
        self.n_boot = n_boot
        self.min_noise_points = min_noise_points
        self.min_linear_points = min_linear_points
        self.grid_points = grid_points

        self._lod_val: float = np.inf
        self._std_noise: float = np.inf
        self._loq_val: float = np.inf
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
    ) -> 'OriginalWLS':
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if len(x) < 3:
            raise ValueError(
                f'OriginalWLS.fit() requires at least 3 data points, got {len(x)}.'
            )
        if len(set(x)) < 2:
            raise ValueError('OriginalWLS.fit() requires at least 2 unique concentrations.')

        self.x_ = x
        self.y_ = y
        self.weights_ = _orig_weights(x)

        # Fit the piecewise model with legacy initialisation
        a, b, c = _fit_piecewise_legacy(x, y)
        self.params_ = {
            'slope': a,
            'intercept_linear': b,
            'intercept_noise': c,
        }

        # LOD (deterministic)
        self._lod_val, self._std_noise = _calculate_lod_legacy(
            a, b, c, x, y,
            self.std_mult, self.min_noise_points, self.min_linear_points,
        )

        # Bootstrap + LOQ (only if LOD is finite)
        if np.isfinite(self._lod_val):
            self._x_grid = np.linspace(
                self._lod_val, float(np.max(x)), num=self.grid_points
            )
            self._boot_summary = _bootstrap_many_legacy(
                x, y, self._x_grid, self.n_boot
            )
            self._loq_val = _calculate_loq_legacy(
                self._boot_summary, self._lod_val, self.cv_thresh
            )
        else:
            self._loq_val = np.inf
            self._boot_summary = None
            self._x_grid = None

        self.is_fitted_ = True
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
    # lod / loq
    # ------------------------------------------------------------------

    def lod(self, std_mult: float = DEFAULT_STD_MULT) -> float:
        """LOD computed during fit().  ``std_mult`` argument ignored after fit."""
        self._check_is_fitted()
        return self._lod_val

    def loq(self, cv_thresh: float = DEFAULT_CV_THRESH) -> float:
        """LOQ computed during fit().  ``cv_thresh`` argument ignored after fit."""
        self._check_is_fitted()
        return self._loq_val

    def supports_lod(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        self._check_is_fitted()
        return {
            'slope': self.params_['slope'],
            'intercept_linear': self.params_['intercept_linear'],
            'intercept_noise': self.params_['intercept_noise'],
            'std_noise': self._std_noise,
            'lod': self._lod_val,
            'loq': self._loq_val,
            'n_points': len(self.x_),
            'compat_model': 'OriginalWLS',
        }
