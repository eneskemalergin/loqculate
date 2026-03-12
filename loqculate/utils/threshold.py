import numpy as np

from loqculate.config import DEFAULT_CV_THRESH, DEFAULT_SLIDING_WINDOW


def find_loq_threshold(
    x_grid: np.ndarray,
    cv_array: np.ndarray,
    cv_thresh: float = DEFAULT_CV_THRESH,
    window: int = DEFAULT_SLIDING_WINDOW,
) -> float:
    """Vectorized LOQ search with a sliding-window stability check.

    Returns the lowest concentration on *x_grid* (excluding zero) at which
    *cv_array* is at or below *cv_thresh* **and** stays below threshold for
    the next ``effective_window - 1`` consecutive points as well.

    This prevents false LOQs caused by non-monotonic CV bounces, which the original
    naive ``min()`` approach is susceptible to.

    Parameters
    ----------
    x_grid:
        Sorted concentration grid (ascending). Must be the same length as
        *cv_array*.
    cv_array:
        Bootstrap CV at each grid point.
    cv_thresh:
        Upper CV threshold for quantitation (e.g. 0.2 for 20 %).
    window:
        Minimum number of *consecutive* points that must remain below
        *cv_thresh*.  Dynamically capped at ``len(x_grid)`` so sparse grids
        are handled gracefully.

    Returns
    -------
    float
        Lowest qualifying concentration, or ``np.inf`` when none is found.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    cv_array = np.asarray(cv_array, dtype=float)

    # Filter out blank (zero-concentration) points.
    nonzero = x_grid > 0
    x_pos = x_grid[nonzero]
    cv_pos = cv_array[nonzero]

    if len(x_pos) == 0:
        return np.inf

    effective_window = min(window, len(x_pos))
    below = cv_pos <= cv_thresh

    for i in range(len(x_pos) - effective_window + 1):
        if below[i : i + effective_window].all():
            return float(x_pos[i])

    # If we couldn't fit a full window but the tail is all below threshold,
    # accept the remaining points as sufficient (handles end-of-grid case).
    remaining = len(x_pos) - (len(x_pos) - effective_window + 1)
    if remaining > 0 and below[len(x_pos) - effective_window :].all():
        return float(x_pos[len(x_pos) - effective_window])

    return np.inf
