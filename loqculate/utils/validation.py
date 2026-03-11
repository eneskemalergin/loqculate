import sys
import numpy as np
import pandas as pd
from typing import Optional


def require_numpy_arrays(*arrays, names=None):
    """Raise TypeError if any argument is not a numpy ndarray."""
    for i, arr in enumerate(arrays):
        label = names[i] if names else f'arg[{i}]'
        if not isinstance(arr, np.ndarray):
            raise TypeError(f'{label} must be a numpy ndarray, got {type(arr).__name__}')


def validate_concentration_map(conc_map_path: str) -> Optional[pd.DataFrame]:
    """Load and sanity-check the concentration map file.

    Writes warnings to stderr for common problems but does *not* raise so that
    the main reader can produce a more context-rich error if needed.
    """
    try:
        df = pd.read_csv(conc_map_path)
    except Exception:
        return None

    if df.empty or 'concentration' not in df.columns:
        sys.stderr.write(
            'WARNING: the concentration map appears to be blank or is missing '
            'a "concentration" column. No curve points will be mapped.\n'
        )
        return df

    blank = df['concentration'].isna() | (df['concentration'].astype(str).str.strip() == '')
    n_blank = int(blank.sum())
    if n_blank:
        sys.stderr.write(
            f'WARNING: {n_blank} row(s) in the concentration map have a '
            f'blank/unannotated concentration value and will be skipped:\n'
        )
        fname_col = 'filename' if 'filename' in df.columns else df.columns[0]
        for fname in df.loc[blank, fname_col].tolist():
            sys.stderr.write(f'  {fname}\n')

    return df


def check_enough_points(
    x: np.ndarray,
    intersection: float,
    min_noise_points: int,
    min_linear_points: int,
) -> bool:
    """Return True when there are enough unique concentration levels on each
    side of *intersection* to trust the LOD estimate."""
    below = np.unique(x[x < intersection])
    above = np.unique(x[x >= intersection])
    return len(below) >= min_noise_points and len(above) >= min_linear_points
