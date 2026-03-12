"""Single-point calibration multiplier."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from loqculate.io.readers import CalibrationData


def apply_multiplier(
    data: CalibrationData,
    multiplier_path: Union[str, Path],
) -> CalibrationData:
    """Scale concentrations by a per-peptide multiplier.

    Reads a CSV with columns ``peptide`` and ``multiplier``.  Inner-joins on
    peptide so that only peptides present in both the data and the multiplier
    file are retained.

    Parameters
    ----------
    data:
        Input :class:`~loqculate.io.readers.CalibrationData`.
    multiplier_path:
        Path to the multiplier CSV.

    Returns
    -------
    CalibrationData
        New object with ``concentration`` scaled by the multiplier.
    """
    mult_df = pd.read_csv(multiplier_path)
    if 'peptide' not in mult_df.columns or 'multiplier' not in mult_df.columns:
        raise ValueError(
            "Multiplier file must have columns 'peptide' and 'multiplier'."
        )

    df = pd.DataFrame({
        'peptide': data.peptide,
        'concentration': data.concentration,
        'area': data.area,
    })
    merged = pd.merge(df, mult_df[['peptide', 'multiplier']], on='peptide', how='inner')
    merged['concentration'] = merged['concentration'] * merged['multiplier']

    return CalibrationData(
        peptide=merged['peptide'].to_numpy(dtype=str),
        concentration=merged['concentration'].to_numpy(dtype=float),
        area=merged['area'].to_numpy(dtype=float),
        metadata={**data.metadata, 'multiplier_applied': True},
    )
