"""Auto-detect and parse all supported calibration curve formats."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from loqculate.utils.validation import validate_concentration_map


@dataclass
class CalibrationData:
    """Standardised container returned by all format readers."""

    peptide: np.ndarray        # string array of peptide IDs
    concentration: np.ndarray  # float array
    area: np.ndarray           # float array
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def read_calibration_data(
    filepath: Union[str, Path],
    conc_map_filepath: Union[str, Path],
    fmt: str = 'auto',
) -> CalibrationData:
    """Read any supported calibration curve file.

    Parameters
    ----------
    filepath:
        Path to the quantitative data file.
    conc_map_filepath:
        Path to the concentration map CSV (columns ``filename``,
        ``concentration``).
    fmt:
        Force a specific format instead of auto-detecting.  One of:
        ``'auto'``, ``'encyclopedia'``, ``'skyline'``, ``'diann_report'``,
        ``'diann_matrix'``, ``'spectronaut'``, ``'generic'``.

    Returns
    -------
    CalibrationData
    """
    filepath = Path(filepath)
    conc_map_filepath = Path(conc_map_filepath)

    validate_concentration_map(str(conc_map_filepath))

    with open(filepath, 'r', errors='replace') as fh:
        header = fh.readline()

    if fmt == 'auto':
        fmt = _detect_format(header)

    dispatch = {
        'encyclopedia': _read_encyclopedia,
        'skyline': _read_skyline,
        'diann_report': _read_diann_report,
        'diann_matrix': _read_diann_matrix,
        'spectronaut': _read_spectronaut,
        'generic': _read_generic,
    }

    if fmt not in dispatch:
        raise ValueError(
            f"Unknown format '{fmt}'.  Supported: {list(dispatch.keys())}"
        )

    df = dispatch[fmt](filepath, conc_map_filepath)

    n_peptides = int(df['peptide'].nunique())
    n_points = len(df)
    sys.stdout.write(
        f'Read {n_peptides} peptides, {n_points} total measurements '
        f'(format: {fmt}).\n'
    )

    return CalibrationData(
        peptide=df['peptide'].to_numpy(dtype=str),
        concentration=df['concentration'].to_numpy(dtype=float),
        area=df['area'].to_numpy(dtype=float),
        metadata={'format': fmt, 'n_peptides': n_peptides, 'n_measurements': n_points},
    )


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _detect_format(header: str) -> str:
    if 'numFragments' in header:
        sys.stdout.write('Input assumed to be EncyclopeDIA *.elib.peptides.txt\n')
        return 'encyclopedia'
    if all(c in header for c in ('Total Area Fragment', 'Peptide Sequence', 'File Name')):
        sys.stdout.write('Input assumed to be Skyline export\n')
        return 'skyline'
    if 'Stripped.Sequence' in header and 'Precursor.Quantity' in header:
        sys.stdout.write('Input assumed to be DIA-NN diann_report.tsv\n')
        return 'diann_report'
    if 'Stripped.Sequence' in header:
        sys.stdout.write('Input assumed to be DIA-NN *.pr_matrix.tsv\n')
        sys.stderr.write(
            'WARNING: Use DIA-NN diann_report.tsv instead of pr_matrix!\n'
        )
        return 'diann_matrix'
    if 'PEP.StrippedSequence' in header:
        sys.stdout.write('Input assumed to be Spectronaut output\n')
        return 'spectronaut'
    # Generic fallback
    sys.stdout.write('Input assumed to be generic CSV (peptide/concentration/area)\n')
    return 'generic'


# ---------------------------------------------------------------------------
# Per-format private readers
# All return a DataFrame with columns: peptide, concentration, area
# ---------------------------------------------------------------------------

def _read_encyclopedia(filepath: Path, conc_map_path: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep=None, engine='python')
    df.drop(['numFragments', 'Protein'], axis='columns', inplace=True)
    conc_map = pd.read_csv(conc_map_path, index_col='filename')
    df.rename(columns={**dict(conc_map['concentration']), 'Peptide': 'peptide'}, inplace=True)
    melted = pd.melt(df, id_vars=['peptide'])
    melted.columns = ['peptide', 'concentration', 'area']
    melted = melted[melted['concentration'].isin(conc_map['concentration'])]
    melted['concentration'] = pd.to_numeric(melted['concentration'])
    melted.fillna({'area': 0}, inplace=True)
    return melted


def _read_skyline(filepath: Path, conc_map_path: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.rename(columns={'File Name': 'filename'}, inplace=True)
    conc_map = pd.read_csv(conc_map_path)
    df = df[df['filename'].isin(conc_map['filename'])]
    df = pd.merge(df, conc_map, on='filename', how='outer')
    df.rename(columns={
        'Total Area Fragment': 'area',
        'Peptide Sequence': 'peptide',
        'concentration': 'concentration',
    }, inplace=True)
    df['concentration'] = pd.to_numeric(df['concentration'], errors='coerce')
    df.dropna(subset=['concentration'], inplace=True)
    df.fillna({'area': 0}, inplace=True)
    return df[['peptide', 'concentration', 'area']]


def _read_diann_report(filepath: Path, conc_map_path: Path) -> pd.DataFrame:
    df = pd.read_table(filepath, sep=None, engine='python')
    cols = ['Precursor.Id', 'File.Name', 'Precursor.Quantity']
    df = df[cols].rename(columns={
        'File.Name': 'filename',
        'Precursor.Id': 'peptide',
        'Precursor.Quantity': 'area',
    })
    conc_map = pd.read_csv(conc_map_path)
    df = pd.merge(df, conc_map[['filename', 'concentration']], on='filename', how='inner')
    df['peptide'] = df['peptide'].str.replace(':', '', regex=False)
    df.fillna({'area': 0}, inplace=True)
    return df[['peptide', 'concentration', 'area']]


def _read_diann_matrix(filepath: Path, conc_map_path: Path) -> pd.DataFrame:
    df = pd.read_table(filepath, sep=None, engine='python')
    df['Precursor.Charge'] = df['Precursor.Charge'].astype(str)
    df['Modified.Sequence'] = df['Modified.Sequence'].astype(str)
    df['peptide'] = df['Modified.Sequence'] + '_' + df['Precursor.Charge']
    drop_cols = [
        'Protein.Group', 'Modified.Sequence', 'Protein.Ids', 'Protein.Names',
        'Genes', 'First.Protein.Description', 'Proteotypic', 'Stripped.Sequence',
        'Precursor.Charge', 'Precursor.Id',
    ]
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)
    conc_map = pd.read_csv(conc_map_path)
    df = df.rename(columns=conc_map.set_index('filename')['concentration'])
    melted = pd.melt(df, id_vars=['peptide'])
    melted.columns = ['peptide', 'concentration', 'area']
    melted = melted[melted['concentration'].isin(conc_map['concentration'])]
    melted['concentration'] = pd.to_numeric(melted['concentration'])
    melted['peptide'] = melted['peptide'].str.replace(':', '', regex=False)
    melted.fillna({'area': 0}, inplace=True)
    return melted


def _read_spectronaut(filepath: Path, conc_map_path: Path) -> pd.DataFrame:
    df = pd.read_table(filepath, sep=None, engine='python')
    df['Precursor.Charge'] = df['EG.PrecursorId'].str.split('.', expand=True)[1]
    df['Modified.Sequence'] = (
        df['EG.PrecursorId'].str.split('.', expand=True)[0].str.strip('_')
    )
    df['peptide'] = df['Modified.Sequence'] + '_' + df['Precursor.Charge']
    drop_cols = [
        'PG.ProteinGroups', 'PG.Organisms', 'PG.ProteinNames', 'PEP.StrippedSequence',
        'PEP.PeptidePosition', 'EG.PrecursorId', 'EG.ModifiedSequence',
        'Modified.Sequence', 'Precursor.Charge',
    ]
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)
    conc_map = pd.read_csv(conc_map_path)
    df = df.rename(columns=conc_map.set_index('filename')['concentration'])
    melted = pd.melt(df, id_vars=['peptide'])
    melted.columns = ['peptide', 'concentration', 'area']
    melted = melted[melted['concentration'].isin(conc_map['concentration'])]
    melted['concentration'] = pd.to_numeric(melted['concentration'])
    melted['peptide'] = melted['peptide'].str.replace(':', '', regex=False)
    melted.fillna({'area': 0}, inplace=True)
    return melted


def _read_generic(filepath: Path, conc_map_path: Path) -> pd.DataFrame:
    """Read a generic CSV with columns peptide / concentration / area."""
    df = pd.read_csv(filepath, sep=None, engine='python')
    # Case-insensitive rename
    rename = {c: c.lower() for c in df.columns}
    df.rename(columns=rename, inplace=True)
    required = {'peptide', 'concentration', 'area'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Generic format requires columns {required}; missing: {missing}.  "
            f"Columns found: {list(df.columns)}"
        )
    df['concentration'] = pd.to_numeric(df['concentration'], errors='coerce')
    df.dropna(subset=['concentration'], inplace=True)
    df.fillna({'area': 0}, inplace=True)
    return df[['peptide', 'concentration', 'area']]
