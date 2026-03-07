"""Reader round-trip tests against hand-crafted fixture files."""
from pathlib import Path

import numpy as np
import pytest

from loqculate.io import read_calibration_data

FIXTURE_DIR = Path(__file__).parent / 'fixtures'


@pytest.mark.parametrize('data_file', [
    'encyclopedia_sample.elib.peptides.txt',
    'skyline_sample.csv',
    'diann_report_sample.tsv',
    'diann_matrix_sample.pr_matrix.tsv',
    'spectronaut_sample.tsv',
    'generic_sample.csv',
])
def test_reader_parses_fixture(data_file):
    """Each fixture file must parse without error and yield the expected schema."""
    data_path = FIXTURE_DIR / data_file
    conc_map_path = FIXTURE_DIR / 'conc_map.csv'

    if not data_path.exists():
        pytest.skip(f'Fixture not found: {data_file}')

    data = read_calibration_data(str(data_path), str(conc_map_path))

    assert len(data.peptide) > 0, 'No rows parsed'
    assert data.concentration.dtype == float
    assert data.area.dtype == float
    assert len(data.peptide) == len(data.concentration) == len(data.area)


def test_generic_fallback_requires_correct_columns(tmp_path):
    """Generic reader raises a clear error when mandatory columns are missing."""
    import pandas as pd
    bad_file = tmp_path / 'bad.csv'
    pd.DataFrame({'sample': ['a'], 'value': [1]}).to_csv(bad_file, index=False)

    conc_map = tmp_path / 'map.csv'
    pd.DataFrame({'filename': ['a'], 'concentration': [1.0]}).to_csv(conc_map, index=False)

    with pytest.raises(ValueError, match='concentration'):
        read_calibration_data(str(bad_file), str(conc_map), fmt='generic')
