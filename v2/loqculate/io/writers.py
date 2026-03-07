"""CSV / TSV output helpers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Union

import pandas as pd


def write_figures_of_merit(
    rows: Iterable[dict],
    output_path: Union[str, Path],
    filename: str = 'figuresofmerit.csv',
) -> Path:
    """Write the LOD/LOQ table to a CSV file.

    Parameters
    ----------
    rows:
        Iterable of dicts, each mapping column name → value.
    output_path:
        Directory in which to write the file.
    filename:
        Output filename (default ``figuresofmerit.csv``).

    Returns
    -------
    Path to the file that was written.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / filename

    df = pd.DataFrame(list(rows))
    df.to_csv(out_file, index=False)
    return out_file


def stream_csv_writer(output_file: Union[str, Path], columns: list[str]):
    """Context manager that yields an append-writer for streaming results.

    Usage
    -----
    >>> with stream_csv_writer('fom.csv', ['peptide', 'LOD', 'LOQ']) as write:
    ...     for row in results:
    ...         write(row)
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    class _Writer:
        def __init__(self, path: Path, cols: list) -> None:
            self._path = path
            self._cols = cols
            self._fh = open(path, 'w', newline='')
            self._fh.write(','.join(cols) + '\n')
            self._fh.flush()

        def __call__(self, row: dict) -> None:
            values = [str(row.get(c, '')) for c in self._cols]
            self._fh.write(','.join(values) + '\n')
            self._fh.flush()
            os.fsync(self._fh.fileno())

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self._fh.close()

    return _Writer(output_file, columns)
