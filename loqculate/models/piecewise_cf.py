"""Closed-form piecewise WLS model using discrete knot search.

Implements the same statistical model as :class:`PiecewiseWLS`
(y = max(c, a*x + b), weights 1/sqrt(x)) but replaces the TRF optimizer
with an exhaustive discrete knot search followed by analytical refinement.
The solver is deterministic and cannot be trapped in local minima.
"""

from __future__ import annotations

import warnings
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
    VECTORIZED_BOOTSTRAP_MEMORY_LIMIT_MB,
)
from loqculate.models.base import CalibrationModel
from loqculate.utils.knot_search import _fit_and_constrain, find_knot
from loqculate.utils.threshold import find_loq_threshold
from loqculate.utils.weights import inverse_sqrt_weights

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class PiecewiseCF(CalibrationModel):
    """Closed-form piecewise WLS model using discrete knot search.

    Default model as of v0.3.0.  Implements ``y = max(c, a*x + b)`` with
    precision weights ``W_i = 1/x_i``.  The partition boundary (knot) is
    selected by exhaustive search over all interior unique x values, picking
    the one that minimises total weighted RSS under the constraint set.  There
    is no optimizer, no initial-guess sensitivity, and no convergence tolerance.

    Parameters
    ----------
    n_boot_reps:
        Bootstrap replicates for LOQ / CV profile calculation.
    seed:
        RNG seed forwarded to the bootstrap.
    min_noise_points:
        Minimum observations with x < LOD intersection required to compute LOD.
    min_linear_points:
        Minimum unique concentration levels above LOD required before LOD is
        accepted.  Prevents LOD from being placed at the top of the curve.
    sliding_window:
        Consecutive CV-grid points that must all stay below ``cv_thresh`` for
        the LOQ to be declared at the first of those points.
    grid_points:
        Number of evenly spaced evaluation points from LOD to max(x) for the
        bootstrap CV profile.

    Public methods
    --------------
    fit(x, y, weights=None) -> PiecewiseCF
        Fit the model.  Returns self for chaining.
        Stores: ``params_``, ``x_``, ``y_``, ``weights_``, ``is_fitted_``.
    predict(x_new) -> ndarray
        Model prediction at new concentrations.
    lod(std_mult=2) -> float
        Limit of detection.  Returns ``np.inf`` when the curve cannot support
        a reliable LOD (slope <= 0, too few noise points, LOD above max(x)).
    loq(cv_thresh=0.20) -> float
        Limit of quantitation via bootstrap CV profile.  Returns ``np.inf``
        when LOD is infinite or the CV never drops below threshold.
    covariance() -> ndarray or None
        2x2 parameter covariance matrix for the linear segment (slope,
        intercept).  Returns ``None`` when slope == 0 (degenerate fit).
    summary() -> dict
        Flat dict with slope, intercept_linear, intercept_noise, knot_x,
        lod, loq, n_points.

    Fitted attributes
    -----------------
    params_ : dict
        Keys: ``slope``, ``intercept_linear``, ``intercept_noise``, ``knot_x``.
    x_, y_, weights_ : ndarray
        Training data and precision weights (``w_i``, not ``w_i^2``).
    is_fitted_ : bool
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
        #
        # Guard: when constraint 1 clamped slope to 0, the linear segment was
        # fit as a weighted mean (1-parameter horizontal line).  Storing a 2x2
        # Gram inverse would correspond to a 2-parameter model that was never
        # actually fit, producing wrong covariance estimates in C4.  Use None
        # to signal "covariance undefined" for this degenerate case.
        if kr.slope > 0:
            lin_mask = x > kr.knot_x
            x_lin = x[lin_mask]
            W_lin = W[lin_mask]
            self._gram_inv = _gram_inverse(x_lin, W_lin)
        else:
            self._gram_inv = None

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
    # covariance
    # ------------------------------------------------------------------

    def covariance(self) -> Optional[np.ndarray]:
        """Return the 2x2 parameter covariance matrix for the linear segment.

        Shape ``(2, 2)``::

            [[var(slope),          cov(slope, intercept)],
             [cov(slope, intercept), var(intercept)     ]]

        Derived from the stored Gram matrix inverse scaled by the residual
        mean squared error of the *linear* segment.  The MSE denominator is
        ``n_lin - 2`` (one degree of freedom lost per fitted parameter:
        slope and intercept).  This is the standard unbiased regression MSE,
        distinct from ``np.std(ddof=1)`` used in :meth:`lod` which estimates
        a scalar standard deviation from noise observations.

        Returns ``None`` when:

        - the model has not been fitted yet, or
        - ``slope == 0`` (constraint 1 clamped the linear segment to a
          horizontal weighted mean — the 2-parameter model was never fit
          and the Gram inverse is undefined).
        """
        if not self.is_fitted_:
            return None
        if self._gram_inv is None:
            return None

        # Residuals on the linear segment only (x > knot_x).
        x = self.x_
        y = self.y_
        W = self.weights_**2
        knot_x = self.params_["knot_x"]
        a = self.params_["slope"]
        b = self.params_["intercept_linear"]

        lin_mask = x > knot_x
        x_lin = x[lin_mask]
        y_lin = y[lin_mask]
        W_lin = W[lin_mask]
        n_lin = int(np.sum(lin_mask))

        if n_lin < 2:
            # Cannot compute ddof=1 MSE with fewer than 2 linear observations.
            return None

        residuals = y_lin - (a * x_lin + b)
        # Weighted RSS for the linear segment.
        wrss = float(np.sum(W_lin * residuals**2))
        # Unbiased MSE: divide by (n_lin - 2) because we estimated 2 parameters
        # (slope and intercept).  Analogous to dividing by (n-1) for a scalar
        # std, but here the degrees-of-freedom penalty is the number of parameters.
        mse = wrss / (n_lin - 2)

        return mse * self._gram_inv

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_boot_summary(self, lod_val: float) -> None:
        if self._boot_summary is not None:
            return

        x_grid = np.linspace(lod_val, float(np.max(self.x_)), num=self.grid_points)
        W = self.weights_**2

        # The vectorized path is the production default.  Both paths use the
        # same W*C full-array masking arithmetic and the same per-rep candidate
        # eligibility rules (unique(xb)[1:-1] interior gate), so knot selection
        # is identical.  Residual differences are sub-ULP FP accumulation noise
        # from numpy's pairwise axis=1 reductions (<1e-14 relative on reference
        # data).  The loop path (_bootstrap_loop_cf) is the memory-guard
        # fallback and the reference implementation for benchmarking.
        _, summary = _bootstrap_vectorized_cf(
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
# Vectorized bootstrap (C5)
# ---------------------------------------------------------------------------


def _bootstrap_vectorized_cf(
    x: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    x_grid: np.ndarray,
    n_reps: int,
    seed: int,
) -> tuple:
    """Bootstrap the CF model with a fully-vectorized candidate inner loop.

    Replaces the per-replicate :func:`find_knot` call in
    :func:`_bootstrap_loop_cf` with a vectorized computation that processes
    all ``n_reps`` replicates simultaneously for each candidate knot.

    **Numerical equivalence with the loop path**

    Both paths draw identical index arrays via::

        SeedSequence(seed).spawn(n_reps)[i] → default_rng(cs).choice(n, n, replace=True)

    Each replicate resamples ``(x, y, W)`` jointly (standard nonparametric
    bootstrap — same observation-level pairs). Candidate ``k`` is only
    evaluated for replicate ``i`` when ``k ∈ unique(X_mat[i])`` and ``k`` is
    an interior point of that resample (not the per-rep min or max), exactly
    mirroring ``find_knot(xb)``'s ``unique(xb)[1:-1]`` candidate set.

    The only numerical difference from the loop path is sub-ULP accumulation
    noise from numpy's pairwise ``axis=1`` reductions vs the loop's sequential
    1-D sums (observed: max ~2e-7 absolute, <1e-14 relative on the 27-peptide
    reference dataset). No knot-selection disagreements occur.

    **Memory guard**

    The working set is proportional to ``n_reps × n``. When a single
    ``float64`` replica matrix exceeds ``VECTORIZED_BOOTSTRAP_MEMORY_LIMIT_MB``
    the function falls back to the loop path and emits a :pyexc:`ResourceWarning`.

    Parameters
    ----------
    x, y, W:
        Full dataset (precision weights ``W = w²``).
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

    # ── n_reps == 0: identical early-return to the loop path ──────────────
    if n_reps == 0:
        summary = {
            "mean": np.full(n_grid, np.nan),
            "std": np.full(n_grid, np.nan),
            "cv": np.full(n_grid, np.inf),
            "pct_5": np.full(n_grid, np.nan),
            "pct_95": np.full(n_grid, np.nan),
        }
        return np.empty((0, n_grid)), summary

    # ── Memory guard ───────────────────────────────────────────────────────
    # Estimated peak working set: ~12 matrices of shape (n_reps, n).
    # The guard fires when ONE such matrix exceeds the configured limit,
    # ensuring total working set stays within ~12× the limit.
    limit_bytes = VECTORIZED_BOOTSTRAP_MEMORY_LIMIT_MB * 1024 * 1024
    if n_reps * n * 8 > limit_bytes:
        warnings.warn(
            f"Vectorized bootstrap working set (~{n_reps * n * 8 * 12 / 1024**2:.0f} MB) "
            f"would exceed memory limit ({VECTORIZED_BOOTSTRAP_MEMORY_LIMIT_MB} MB). "
            "Falling back to loop bootstrap.",
            ResourceWarning,
            stacklevel=2,
        )
        return _bootstrap_loop_cf(x, y, W, x_grid, n_reps, seed)

    # ── Degenerate shortcut: constant signal ───────────────────────────────
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

    # ── Build idx_matrix using identical seeding to the loop path ─────────
    # SeedSequence(seed).spawn(n_reps)[i] → default_rng(cs).choice(n, n)
    # mirrors _bootstrap_loop_cf exactly, so X_matrix[i] = x[idx_i],
    # Y_matrix[i] = y[idx_i], W_matrix[i] = W[idx_i] — full paired resample.
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(n_reps)
    idx_matrix = np.array(
        [np.random.default_rng(cs).choice(n, size=n, replace=True) for cs in child_seeds],
        dtype=np.intp,
    )  # shape (n_reps, n)

    X_mat = x[idx_matrix]  # (n_reps, n)
    Y_mat = y[idx_matrix]  # (n_reps, n)
    W_mat = W[idx_matrix]  # (n_reps, n)
    X2_mat = X_mat**2  # (n_reps, n) — precomputed once

    # ── Candidate loop: vectorized over all replicates ─────────────────────
    # Candidates are drawn from the ORIGINAL unique(x)[1:-1], but a candidate
    # k is only eligible for replicate i if k ∈ unique(X_mat[i]) — matching
    # the loop path's find_knot(xb) which uses unique(xb)[1:-1].  When a
    # bootstrap resample drops a concentration level, the missing k would
    # partition the resampled data differently from any present candidate,
    # producing genuinely different (not equivalent) results.  The gate
    # `rep_has_k` enforces consistency with the loop path.
    unique_x = np.unique(x)
    candidates = unique_x[1:-1]

    best_rss = np.full(n_reps, np.inf)
    best_a = np.zeros(n_reps)
    best_b = np.zeros(n_reps)
    best_c = np.zeros(n_reps)
    best_kx = np.full(n_reps, candidates[0])

    # Precompute per-rep min/max once; used in the candidate loop to match
    # find_knot(xb)'s unique(xb)[1:-1] which excludes the per-rep min and max.
    rep_x_min = X_mat.min(axis=1)  # (n_reps,)
    rep_x_max = X_mat.max(axis=1)  # (n_reps,)

    for k in candidates:
        # k is a valid interior candidate for rep i iff:
        #   1. k ∈ unique(xb_i)  — k appears in the resample
        #   2. k ≠ min(xb_i)     — k is not the per-rep minimum
        #   3. k ≠ max(xb_i)     — k is not the per-rep maximum
        # Conditions 2 & 3 match unique(xb)[1:-1] which strips first/last.
        # X_mat / x values are exact float64 copies so == is bit-exact.
        rep_has_k = np.any(X_mat == k, axis=1)  # (n_reps,) bool
        rep_k_interior = rep_has_k & (rep_x_min != k) & (rep_x_max != k)
        if not rep_k_interior.any():
            continue

        # Per-rep binary partition at k — mask varies because X_mat varies.
        C = (X_mat <= k).astype(np.float64)  # (n_reps, n)
        L = 1.0 - C

        # ── Noise segment: weighted mean ───────────────────────────────────
        sum_W_n = np.sum(W_mat * C, axis=1)  # (n_reps,)
        # Divide only where sum_W_n > 0; fall back to unweighted mean otherwise
        # (mirrors _fit_and_constrain's explicit if/elif guard).
        has_noise_weight = sum_W_n > 0
        c_arr = np.where(
            has_noise_weight,
            np.sum(W_mat * Y_mat * C, axis=1) / np.where(has_noise_weight, sum_W_n, 1.0),
            np.sum(Y_mat * C, axis=1) / np.maximum(np.sum(C, axis=1), 1.0),
        )
        rss_n = np.sum(W_mat * C * (Y_mat - c_arr[:, None]) ** 2, axis=1)

        # ── Linear segment: normal equations ──────────────────────────────
        sum_WXX = np.sum(W_mat * X2_mat * L, axis=1)
        sum_WX = np.sum(W_mat * X_mat * L, axis=1)
        sum_WYX = np.sum(W_mat * Y_mat * X_mat * L, axis=1)
        sum_WY = np.sum(W_mat * Y_mat * L, axis=1)
        sum_W_l = np.sum(W_mat * L, axis=1)

        det = sum_WXX * sum_W_l - sum_WX**2
        valid = np.abs(det) > KNOT_SEARCH_SINGULAR_THRESHOLD
        safe_det = np.where(valid, det, 1.0)
        a_arr = np.where(valid, (sum_WYX * sum_W_l - sum_WY * sum_WX) / safe_det, 0.0)
        # Singular fallback: weighted mean of linear observations (like _fit_and_constrain).
        has_lin_weight = sum_W_l > 0
        lin_wmean = np.where(
            has_lin_weight,
            sum_WY / np.where(has_lin_weight, sum_W_l, 1.0),
            np.sum(Y_mat * L, axis=1) / np.maximum(np.sum(L, axis=1), 1.0),
        )
        b_arr = np.where(valid, (sum_WXX * sum_WY - sum_WX * sum_WYX) / safe_det, lin_wmean)

        # ── Constraint 1: slope < 0 → weighted mean of linear observations ─
        neg = a_arr < 0
        if neg.any():
            a_arr[neg] = 0.0
            b_arr[neg] = lin_wmean[neg]

        # ── Constraint 2: noise floor < linear intercept → clamp ──────────
        clamp = c_arr < b_arr
        if clamp.any():
            c_arr[clamp] = b_arr[clamp]
            rss_n[clamp] = np.sum(
                W_mat[clamp] * C[clamp] * (Y_mat[clamp] - c_arr[clamp, None]) ** 2,
                axis=1,
            )

        # RSS for linear segment (recomputed after all constraint fixes).
        rss_l = np.sum(
            W_mat * L * (Y_mat - (a_arr[:, None] * X_mat + b_arr[:, None])) ** 2,
            axis=1,
        )

        total_rss = rss_n + rss_l
        improve = (total_rss < best_rss) & rep_k_interior
        if improve.any():
            best_rss[improve] = total_rss[improve]
            best_a[improve] = a_arr[improve]
            best_b[improve] = b_arr[improve]
            best_c[improve] = c_arr[improve]
            best_kx[improve] = float(k)

    # ── Phase 2: analytical refinement (scalar loop, same rule as find_knot) ─
    # Each replicate may have a different x_join, preventing vectorization.
    # One _fit_and_constrain call per replicate — same cost as the corresponding
    # call inside find_knot.
    for i in range(n_reps):
        ai = best_a[i]
        if ai <= 0:
            continue
        xi = X_mat[i]
        x_join_i = (best_c[i] - best_b[i]) / ai
        # Use bootstrap-resample bounds, matching how find_knot() checks in the loop path.
        if not (xi.min() < x_join_i < xi.max()):
            continue
        sl_r, int_r, c_r, _ = _fit_and_constrain(xi, Y_mat[i], W_mat[i], x_join_i)
        best_a[i] = sl_r
        best_b[i] = int_r
        best_c[i] = c_r

    # ── Grid predictions and summary ──────────────────────────────────────
    # Reps where best_rss is still inf had fewer than 3 unique x values in
    # their resample (no interior candidates) — set their predictions to NaN
    # so they are excluded from nanmean/nanstd, matching the loop path's
    # try/except → predictions[i] = NaN handling.
    no_winner = ~np.isfinite(best_rss)  # (n_reps,)

    linear = best_a[:, None] * x_grid[None, :] + best_b[:, None]
    predictions = np.maximum(best_c[:, None], linear)
    predictions[no_winner] = np.nan

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


# ---------------------------------------------------------------------------
# Loop bootstrap (fallback for memory-guard and benchmarking)
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

    # No replicates requested — return an empty predictions array and a summary
    # filled with NaN/inf.  Avoids calling nanmean/nanstd on a (0, n_grid) array,
    # which would emit "Mean of empty slice" RuntimeWarnings.
    if n_reps == 0:
        summary = {
            "mean": np.full(n_grid, np.nan),
            "std": np.full(n_grid, np.nan),
            "cv": np.full(n_grid, np.inf),
            "pct_5": np.full(n_grid, np.nan),
            "pct_95": np.full(n_grid, np.nan),
        }
        return np.empty((0, n_grid)), summary

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
