"""Discrete knot search for the piecewise WLS model.

Searches over all interior unique x values as candidate partition boundaries
and selects the one that minimises the total weighted RSS under constrained
parameters.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from loqculate.config import KNOT_SEARCH_SINGULAR_THRESHOLD
from loqculate.utils.normal_equations import (
    solve_2x2_wls_batch,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fit_and_constrain(
    x: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    k: float,
) -> tuple[float, float, float, float]:
    """Fit both segments at partition boundary k and apply constraint rules.

    Uses full-array masking arithmetic (``W * C`` style sums over all *n*
    observations) so that every floating-point sum touches elements in the
    same order as the vectorised bootstrap Phase 1.  This guarantees
    bit-exact agreement between the loop and vectorised paths even for
    bootstrap resamples where two candidate RSS values are within 1 ULP.

    Returns (slope, intercept, noise_intercept, total_rss).
    """
    C = (x <= k).astype(np.float64)  # 1 for noise, 0 for linear
    L = 1.0 - C

    # ── Noise segment ──────────────────────────────────────────────────────
    sum_WC = float(np.sum(W * C))
    if sum_WC > 0:
        c = float(np.sum(W * y * C) / sum_WC)
    elif C.any():
        c = float(np.mean(y[C > 0]))  # all-zero weights edge case
    else:
        c = -np.inf  # empty noise segment; clamp c ≥ intercept fires below
    rss_noise = float(np.sum(W * C * (y - c) ** 2)) if np.isfinite(c) else 0.0

    # ── Linear segment ─────────────────────────────────────────────────────
    sum_WL = float(np.sum(W * L))
    if not L.any():
        # All observations are in the noise segment; degenerate partition.
        return 0.0, c, c, rss_noise

    sum_WXL = float(np.sum(W * x * L))
    sum_WXXL = float(np.sum(W * x * x * L))
    sum_WYL = float(np.sum(W * y * L))
    sum_WXYL = float(np.sum(W * x * y * L))

    det = sum_WXXL * sum_WL - sum_WXL * sum_WXL
    if abs(det) < KNOT_SEARCH_SINGULAR_THRESHOLD:
        intercept = sum_WYL / sum_WL if sum_WL > 0 else float(np.mean(y[L > 0]))
        slope, rss_lin = 0.0, float(np.sum(W * L * (y - intercept) ** 2))
    else:
        slope = float((sum_WXYL * sum_WL - sum_WYL * sum_WXL) / det)
        intercept = float((sum_WXXL * sum_WYL - sum_WXL * sum_WXYL) / det)
        rss_lin = float(np.sum(W * L * (y - slope * x - intercept) ** 2))

    if slope < 0:
        slope = 0.0
        intercept = sum_WYL / sum_WL if sum_WL > 0 else float(np.mean(y[L > 0]))
        rss_lin = float(np.sum(W * L * (y - intercept) ** 2))

    if c < intercept:
        c = intercept
        rss_noise = float(np.sum(W * C * (y - c) ** 2))

    return slope, intercept, c, rss_noise + rss_lin


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class KnotResult(NamedTuple):
    slope: float
    intercept: float  # linear segment intercept (b)
    noise_intercept: float  # noise plateau (c)
    knot_x: float  # partition boundary used during search
    rss: float  # total weighted RSS under constrained parameters
    n_noise: int  # observations with x <= knot_x
    n_linear: int  # observations with x > knot_x


# ---------------------------------------------------------------------------
# Scalar search
# ---------------------------------------------------------------------------


def find_knot(
    x: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
) -> KnotResult:
    """Find the optimal partition knot by discrete search with analytical refinement.

    Phase 1 — discrete search: evaluates ``unique(x)[1:-1]`` as candidate
    partition boundaries. Applies constraint enforcement after each fit.
    Selects the candidate with the lowest total weighted RSS. Ties go to the
    lowest-x candidate.

    Phase 2 — analytical refinement: using the best candidate parameters
    ``(a, b, c)``, compute the analytical join point ``x_join = (c - b) / a``
    and refit both segments with the full observation set partitioned at
    ``x_join`` instead of at the discrete candidate boundary. This collapses
    the LOD/LOQ gap that arises when the true join falls between two
    concentration levels. Same constraint rules apply.

    The refinement is skipped when ``slope == 0`` (horizontal model) or when
    ``x_join`` falls outside ``(min(x), max(x))``, i.e. when the discrete
    result already assigns all observations to one segment.

    Parameters
    ----------
    x:
        Concentration values.
    y:
        Signal values.
    W:
        Precision weights (w_i^2). Same length as x and y.

    Returns
    -------
    KnotResult

    Raises
    ------
    ValueError
        When fewer than 3 unique x values are present.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    W = np.asarray(W, dtype=float)

    unique_x = np.unique(x)
    if len(unique_x) < 3:
        raise ValueError(f"find_knot requires at least 3 unique x values, got {len(unique_x)}.")

    candidates = unique_x[1:-1]
    best: KnotResult | None = None

    for k in candidates:
        slope, intercept, c, total_rss = _fit_and_constrain(x, y, W, k)

        if best is None or total_rss < best.rss:
            noise_mask = x <= k
            best = KnotResult(
                slope=slope,
                intercept=intercept,
                noise_intercept=c,
                knot_x=float(k),
                rss=total_rss,
                n_noise=int(np.sum(noise_mask)),
                n_linear=int(np.sum(~noise_mask)),
            )

    # Phase 2: analytical refinement at x_join = (c - b) / a.
    # Refit using all observations partitioned at x_join instead of the
    # discrete candidate boundary. Skipped when slope == 0 or x_join is
    # outside the data range (no change of partition possible).
    a, b, c = best.slope, best.intercept, best.noise_intercept  # type: ignore[union-attr]
    if a > 0:
        x_join = (c - b) / a
        if np.min(x) < x_join < np.max(x):
            slope_r, intercept_r, c_r, rss_r = _fit_and_constrain(x, y, W, x_join)
            noise_mask_r = x <= x_join
            best = KnotResult(
                slope=slope_r,
                intercept=intercept_r,
                noise_intercept=c_r,
                knot_x=x_join,
                rss=rss_r,
                n_noise=int(np.sum(noise_mask_r)),
                n_linear=int(np.sum(~noise_mask_r)),
            )

    return best  # type: ignore[return-value]  # candidates always non-empty


# ---------------------------------------------------------------------------
# Vectorized batch search (used by the bootstrap)
# ---------------------------------------------------------------------------


def find_knot_batch(
    x: np.ndarray,
    Y_matrix: np.ndarray,
    W: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find the optimal knot for each row of Y_matrix simultaneously.

    Parameters
    ----------
    x:
        Concentration values. Shape (n,). Fixed across all replicates.
    Y_matrix:
        Signal matrix. Shape (n_reps, n). Each row is one bootstrap replicate.
    W:
        Precision weights (w_i^2). Shape (n,). Fixed across all replicates.

    Returns
    -------
    slopes : ndarray, shape (n_reps,)
    intercepts : ndarray, shape (n_reps,)
    noise_intercepts : ndarray, shape (n_reps,)
    knot_xs : ndarray, shape (n_reps,)
    rss_values : ndarray, shape (n_reps,)

    Raises
    ------
    ValueError
        When fewer than 3 unique x values are present.
    """
    x = np.asarray(x, dtype=float)
    Y_matrix = np.asarray(Y_matrix, dtype=float)
    W = np.asarray(W, dtype=float)

    unique_x = np.unique(x)
    if len(unique_x) < 3:
        raise ValueError(
            f"find_knot_batch requires at least 3 unique x values, got {len(unique_x)}."
        )

    candidates = unique_x[1:-1]
    n_reps = Y_matrix.shape[0]

    best_rss = np.full(n_reps, np.inf)
    best_slopes = np.zeros(n_reps)
    best_intercepts = np.zeros(n_reps)
    best_noise_intercepts = np.zeros(n_reps)
    best_knot_xs = np.full(n_reps, candidates[0])

    for k in candidates:
        noise_mask = x <= k
        lin_mask = ~noise_mask

        W_noise = W[noise_mask]
        W_lin = W[lin_mask]
        x_lin = x[lin_mask]
        Y_noise = Y_matrix[:, noise_mask]  # (n_reps, n_noise)
        Y_lin = Y_matrix[:, lin_mask]  # (n_reps, n_lin)

        # Noise: weighted mean per rep
        sum_W_noise = np.sum(W_noise)
        if sum_W_noise > 0:
            c_arr = Y_noise @ W_noise / sum_W_noise  # (n_reps,)
        else:
            c_arr = np.mean(Y_noise, axis=1)
        noise_resid = Y_noise - c_arr[:, None]
        rss_noise = np.sum(W_noise * noise_resid**2, axis=1)  # (n_reps,)

        # Linear: batch solver
        slopes, intercepts, rss_lin = solve_2x2_wls_batch(x_lin, Y_lin, W_lin)

        # Constraint 1: negative slope → horizontal line at the weighted mean
        neg_mask = slopes < 0
        if np.any(neg_mask):
            sum_W_lin = np.sum(W_lin)
            if sum_W_lin > 0:
                lin_means = (Y_lin[neg_mask] @ W_lin) / sum_W_lin
            else:
                lin_means = np.mean(Y_lin[neg_mask], axis=1)
            slopes[neg_mask] = 0.0
            intercepts[neg_mask] = lin_means
            lin_resid = Y_lin[neg_mask] - lin_means[:, None]
            rss_lin[neg_mask] = np.sum(W_lin * lin_resid**2, axis=1)

        # Constraint 2: noise floor below linear intercept → clamp, recompute
        clamp_mask = c_arr < intercepts
        if np.any(clamp_mask):
            c_arr[clamp_mask] = intercepts[clamp_mask]
            noise_resid_c = Y_noise[clamp_mask] - c_arr[clamp_mask, None]
            rss_noise[clamp_mask] = np.sum(W_noise * noise_resid_c**2, axis=1)

        total_rss = rss_noise + rss_lin

        improve = total_rss < best_rss
        best_rss[improve] = total_rss[improve]
        best_slopes[improve] = slopes[improve]
        best_intercepts[improve] = intercepts[improve]
        best_noise_intercepts[improve] = c_arr[improve]
        best_knot_xs[improve] = float(k)

    # Phase 2: analytical refinement for each replicate.
    # For reps where slope > 0, compute x_join = (c - b) / a and refit
    # at that boundary. The refinement is applied in a scalar loop because
    # each rep may have a different x_join, making a fully-vectorized
    # implementation complex for marginal gain (one extra solve per rep).
    for i in range(n_reps):
        a_i = best_slopes[i]
        if a_i <= 0:
            continue
        x_join_i = (best_noise_intercepts[i] - best_intercepts[i]) / a_i
        if not (np.min(x) < x_join_i < np.max(x)):
            continue
        slope_r, intercept_r, c_r, _ = _fit_and_constrain(x, Y_matrix[i], W, x_join_i)
        best_slopes[i] = slope_r
        best_intercepts[i] = intercept_r
        best_noise_intercepts[i] = c_r
        best_knot_xs[i] = x_join_i
        # rss_values not updated — used only for candidate selection, already done

    return best_slopes, best_intercepts, best_noise_intercepts, best_knot_xs, best_rss
