from __future__ import annotations

from typing import TYPE_CHECKING, Type

import numpy as np

if TYPE_CHECKING:
    from loqculate.models.base import CalibrationModel


def bootstrap_predictions(
    x: np.ndarray,
    y: np.ndarray,
    model_class: Type["CalibrationModel"],
    model_kwargs: dict,
    x_grid: np.ndarray,
    n_reps: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Bootstrap the calibration model and evaluate predictions on *x_grid*.

    Parameters
    ----------
    x, y:
        Raw numpy arrays (all observations).
    model_class:
        Any :class:`~loqculate.models.base.CalibrationModel` subclass.
    model_kwargs:
        Keyword arguments forwarded to ``model_class(...)``.
    x_grid:
        Concentration points at which predictions are evaluated.
    n_reps:
        Number of bootstrap replicates.
    seed:
        Base seed for :class:`numpy.random.SeedSequence`.

    Returns
    -------
    predictions : ndarray, shape (n_reps, len(x_grid))
    summary : dict
        Keys: ``mean``, ``std``, ``cv``, ``pct_5``, ``pct_95``.
        Each value is a 1-D array over *x_grid*.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)

    n = len(x)
    n_grid = len(x_grid)
    nan_predictions = np.full((n_reps, n_grid), np.nan)

    # --- Pre-validation: catch all-identical y before entering the loop.
    # All-zero (or constant) peptides are common in proteomics.  The v1 code
    # has a latent infinite-loop bug here; v2 catches this explicitly.
    if np.unique(y).size <= 1:
        const = float(y[0]) if len(y) else np.nan
        summary = {
            'mean': np.full(n_grid, const),
            'std': np.zeros(n_grid),
            'cv': np.full(n_grid, np.inf),
            'pct_5': np.full(n_grid, const),
            'pct_95': np.full(n_grid, const),
        }
        return nan_predictions, summary

    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(n_reps)
    predictions = np.empty((n_reps, n_grid))

    for i, cs in enumerate(child_seeds):
        rng = np.random.default_rng(cs)
        idx = rng.choice(n, size=n, replace=True)

        # Defense-in-depth retry (original data has >1 unique y so this loop
        # almost never executes, but guards against pathological arrays).
        retries = 0
        while np.unique(y[idx]).size <= 1:
            idx = rng.choice(n, size=n, replace=True)
            retries += 1
            if retries > 100:
                predictions[i] = np.nan
                break
        else:
            model = model_class(**model_kwargs)
            model.fit(x[idx], y[idx])
            predictions[i] = model.predict(x_grid)

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
