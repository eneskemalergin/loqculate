"""Sliding-window LOQ threshold search on a CV-vs-concentration grid."""

from __future__ import annotations

import numpy as np

from loqculate.config import DEFAULT_CV_THRESH, DEFAULT_SLIDING_WINDOW


def find_loq_threshold(
    x_grid: np.ndarray,
    cv_array: np.ndarray,
    cv_thresh: float = DEFAULT_CV_THRESH,
    window: int = DEFAULT_SLIDING_WINDOW,
) -> float:
    """Return the lowest concentration where CV stays at or below threshold.

    Finds the first grid point (excluding zero concentration) at which
    ``cv_array`` is ``<= cv_thresh`` for ``effective_window`` consecutive
    points.  ``effective_window`` is ``min(window, n_positive)``.

    Parameters
    ----------
    x_grid:
        Concentration grid (same length as ``cv_array``).
    cv_array:
        CV at each grid point.
    cv_thresh:
        Upper CV threshold for quantitation (e.g. 0.2 for 20%).
    window:
        Minimum number of consecutive points that must remain at or below
        ``cv_thresh``.  Capped at the number of positive-concentration points.
        Values less than 1 yield ``inf`` (no valid window).

    Returns
    -------
    float
        Lowest qualifying concentration, or ``np.inf`` when none is found.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    cv_array = np.asarray(cv_array, dtype=float)

    nonzero = x_grid > 0
    x_pos = x_grid[nonzero]
    cv_pos = cv_array[nonzero]

    if x_pos.size == 0:
        return float(np.inf)

    effective_window = min(int(window), int(x_pos.size))
    if effective_window < 1:
        return float(np.inf)

    below = cv_pos <= cv_thresh
    run = np.convolve(below.astype(np.int8), np.ones(effective_window, dtype=np.int8), mode="valid")
    hits = np.flatnonzero(run == effective_window)
    if hits.size:
        return float(x_pos[int(hits[0])])
    return float(np.inf)
