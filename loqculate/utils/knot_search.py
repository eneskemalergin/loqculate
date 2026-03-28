"""Discrete knot search for the piecewise WLS model.

Searches over all interior unique x values as candidate partition boundaries
and selects the one that minimises the total weighted RSS under constrained
parameters.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from loqculate.utils.normal_equations import (
    solve_2x2_wls,
    solve_2x2_wls_batch,
    weighted_mean,
)

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
    """Find the optimal partition knot by discrete search.

    Searches over ``unique(x)[1:-1]`` as candidate partition boundaries.
    For each candidate ``k``: the noise segment is ``x <= k``, the linear
    segment is ``x > k``. Applies constraint enforcement after each fit.
    Selects the candidate with the lowest total weighted RSS. Ties go to the
    lowest-x candidate.

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
        noise_mask = x <= k
        lin_mask = ~noise_mask

        y_noise = y[noise_mask]
        W_noise = W[noise_mask]
        y_lin = y[lin_mask]
        x_lin = x[lin_mask]
        W_lin = W[lin_mask]

        # Fit noise segment: weighted mean
        c, rss_noise = weighted_mean(y_noise, W_noise)

        # Fit linear segment: 2×2 normal equations
        slope, intercept, rss_lin = solve_2x2_wls(x_lin, y_lin, W_lin)

        # Constraint 1: negative slope → horizontal line at weighted mean
        if slope < 0:
            intercept, rss_lin = weighted_mean(y_lin, W_lin)
            slope = 0.0

        # Constraint 2: noise floor below linear intercept → clamp and recompute
        if c < intercept:
            c = intercept
            rss_noise = float(np.sum(W_noise * (y_noise - c) ** 2))

        total_rss = rss_noise + rss_lin

        if best is None or total_rss < best.rss:
            best = KnotResult(
                slope=slope,
                intercept=intercept,
                noise_intercept=c,
                knot_x=float(k),
                rss=total_rss,
                n_noise=int(np.sum(noise_mask)),
                n_linear=int(np.sum(lin_mask)),
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

    return best_slopes, best_intercepts, best_noise_intercepts, best_knot_xs, best_rss
