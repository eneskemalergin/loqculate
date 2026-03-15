"""Shared CV computation utilities used by EmpiricalCV and OriginalCV."""
from __future__ import annotations

import numpy as np
from typing import Tuple


def vectorized_cv_stats(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-group count, mean, std (ddof=1), and CV using bincount.

    Parameters
    ----------
    x:
        Concentration values (one per observation).
    y:
        Signal/area values (one per observation).

    Returns
    -------
    unique_concs : ndarray
        Sorted unique concentrations.
    counts : ndarray
        Number of observations at each concentration.
    means : ndarray
        Mean signal at each concentration.
    cvs : ndarray
        CV = std/mean at each concentration. NaN where count < 2 or mean == 0.
    """
    unique_concs, inverse = np.unique(x, return_inverse=True)
    n_groups = len(unique_concs)

    counts = np.bincount(inverse, minlength=n_groups).astype(float)
    sums = np.bincount(inverse, weights=y, minlength=n_groups)
    means = sums / counts

    # Bessel-corrected variance: (Σxi²/n - mean²) * n/(n-1)
    sum_sq = np.bincount(inverse, weights=y * y, minlength=n_groups)
    with np.errstate(invalid='ignore', divide='ignore'):
        var = (sum_sq / counts - means ** 2) * counts / (counts - 1.0)
    var = np.maximum(var, 0.0)
    stds = np.sqrt(var)

    with np.errstate(invalid='ignore', divide='ignore'):
        cvs = np.where(means != 0, stds / means, np.nan)

    cvs[counts < 2] = np.nan

    return unique_concs, counts, means, cvs
