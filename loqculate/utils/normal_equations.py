"""2×2 WLS normal equations solver.

Three pure functions with no state. All operate on precision weights
W_i = w_i^2, where w_i are the observation weights (e.g. 1/sqrt(x_i)).
"""

from __future__ import annotations

import numpy as np

from loqculate.config import KNOT_SEARCH_SINGULAR_THRESHOLD

# ---------------------------------------------------------------------------
# Noise-segment fit: weighted mean
# ---------------------------------------------------------------------------


def weighted_mean(
    y: np.ndarray,
    W: np.ndarray,
) -> tuple[float, float]:
    """Compute the precision-weighted mean and weighted RSS.

    Parameters
    ----------
    y:
        Observed values.
    W:
        Precision weights (w_i^2). Same length as y.

    Returns
    -------
    mean : float
        Weighted mean Σ(W_i * y_i) / Σ W_i.
    rss : float
        Weighted residual sum of squares Σ W_i * (y_i - mean)^2.
    """
    if len(y) == 0:
        return np.nan, 0.0
    sum_W = float(np.sum(W))
    if sum_W == 0.0:
        return float(np.mean(y)), 0.0
    mean = float(np.sum(W * y) / sum_W)
    rss = float(np.sum(W * (y - mean) ** 2))
    return mean, rss


# ---------------------------------------------------------------------------
# Linear-segment fit: 2×2 normal equations
# ---------------------------------------------------------------------------


def solve_2x2_wls(
    x: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
) -> tuple[float, float, float]:
    """Fit y = a*x + b by WLS via 2×2 normal equations.

    Parameters
    ----------
    x:
        Predictor values.
    y:
        Response values.
    W:
        Precision weights (w_i^2). Same length as x and y.

    Returns
    -------
    slope : float
    intercept : float
    rss : float
        Weighted residual sum of squares Σ W_i * (y_i - slope*x_i - intercept)^2.

    Notes
    -----
    Falls back to a horizontal line (slope=0, intercept=weighted mean) when
    |det(G)| < KNOT_SEARCH_SINGULAR_THRESHOLD. This happens when all x values
    in the segment are identical.

    W must be the precision weight array (w_i^2), not the observation weight w_i.
    """
    sum_W = np.sum(W)
    sum_Wx = np.sum(W * x)
    sum_Wxx = np.sum(W * x * x)
    sum_Wy = np.sum(W * y)
    sum_Wxy = np.sum(W * x * y)

    det = sum_Wxx * sum_W - sum_Wx * sum_Wx

    if abs(det) < KNOT_SEARCH_SINGULAR_THRESHOLD:
        mean, rss = weighted_mean(y, W)
        return 0.0, mean, rss

    slope = float((sum_W * sum_Wxy - sum_Wx * sum_Wy) / det)
    intercept = float((sum_Wxx * sum_Wy - sum_Wx * sum_Wxy) / det)
    residuals = y - slope * x - intercept
    rss = float(np.sum(W * residuals**2))
    return slope, intercept, rss


# ---------------------------------------------------------------------------
# Vectorized batch variant
# ---------------------------------------------------------------------------


def solve_2x2_wls_batch(
    x: np.ndarray,
    Y_matrix: np.ndarray,
    W: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit y = a*x + b for multiple y-vectors simultaneously.

    Parameters
    ----------
    x:
        Predictor values. Shape (n,).
    Y_matrix:
        Bootstrap replicate matrix. Shape (n_reps, n). Each row is one y-vector.
    W:
        Precision weights (w_i^2). Shape (n,). Applied to every row.

    Returns
    -------
    slopes : ndarray, shape (n_reps,)
    intercepts : ndarray, shape (n_reps,)
    rss_array : ndarray, shape (n_reps,)

    Notes
    -----
    When |det(G)| < KNOT_SEARCH_SINGULAR_THRESHOLD every row falls back to
    slope=0 and intercept=weighted mean of that row. Because x and W are fixed
    across rows, det is a scalar: either all rows are singular or none are.
    """
    # Gram matrix entries are the same for every row (x and W are fixed).
    sum_W = np.sum(W)
    sum_Wx = np.sum(W * x)
    sum_Wxx = np.sum(W * x * x)
    det = sum_Wxx * sum_W - sum_Wx * sum_Wx

    # RHS varies per row.
    sum_Wy = Y_matrix @ W  # shape (n_reps,)
    sum_Wxy = Y_matrix @ (W * x)  # shape (n_reps,)

    if abs(det) < KNOT_SEARCH_SINGULAR_THRESHOLD:
        if sum_W == 0.0:
            means = np.mean(Y_matrix, axis=1)
        else:
            means = sum_Wy / sum_W
        residuals = Y_matrix - means[:, None]
        rss_array = np.sum(W * residuals**2, axis=1)
        return np.zeros(len(Y_matrix)), means, rss_array

    slopes = (sum_W * sum_Wxy - sum_Wx * sum_Wy) / det
    intercepts = (sum_Wxx * sum_Wy - sum_Wx * sum_Wxy) / det
    residuals = Y_matrix - slopes[:, None] * x[None, :] - intercepts[:, None]
    rss_array = np.sum(W * residuals**2, axis=1)
    return slopes, intercepts, rss_array
