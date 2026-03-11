import numpy as np
from loqculate.config import DEFAULT_WEIGHT_CAP


def inverse_sqrt_weights(x: np.ndarray, cap: float = DEFAULT_WEIGHT_CAP) -> np.ndarray:
    """Return WLS weights w_i = min(1 / (sqrt(x_i) + eps), cap).

    This is the same scheme used in the original implementation.  Centralising it here allows the cap
    to be tuned from the CLI without touching model code.

    Parameters
    ----------
    x:
        Concentration values (must be non-negative).
    cap:
        Maximum allowed weight.  Prevents numerical instability at x ≈ 0.
    """
    x = np.asarray(x, dtype=float)
    return np.minimum(1.0 / (np.sqrt(x) + np.finfo(float).eps), cap)
