"""Shared pytest fixtures for loqculate v2 tests."""
import pytest
import numpy as np

from loqculate.testing.simulator import CurveSimulator, SimulatedDataset
from loqculate.testing.scenarios import get_scenario


@pytest.fixture(scope='session')
def ideal_dataset() -> SimulatedDataset:
    """Standard happy-path dataset: clear LOD and LOQ."""
    return get_scenario('ideal_curve').generate()


@pytest.fixture(scope='session')
def ideal_single_peptide():
    """Single-peptide arrays from the ideal scenario (first peptide)."""
    ds = get_scenario('ideal_curve').generate()
    mask = ds.peptide == ds.peptide[0]
    return ds.concentration[mask], ds.area[mask]


@pytest.fixture(
    params=[
        'all_zero', 'all_noise', 'high_cv_bounce', 'sparse_curve',
        'single_replicate', 'blank_only_low_cv', 'wide_dynamic_range',
    ]
)
def edge_case(request):
    """Parametrized: yields (name, SimulatedDataset) for every edge-case scenario."""
    name = request.param
    return name, get_scenario(name).generate()
