"""Named edge-case presets for :class:`~loqculate.testing.simulator.CurveSimulator`."""
from __future__ import annotations

import numpy as np

from loqculate.testing.simulator import CurveSimulator, SimulatedDataset

_DEFAULT_CONCS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
_SPARSE_CONCS = [0.0, 1.0, 10.0, 100.0]  # only 4 levels


class _AllZeroSimulator(CurveSimulator):
    """Override generate() so every area is exactly 0.0."""
    def generate(self) -> SimulatedDataset:
        ds = super().generate()
        return SimulatedDataset(
            peptide=ds.peptide,
            concentration=ds.concentration,
            area=np.zeros_like(ds.area),
            ground_truth=ds.ground_truth,
        )


class _HighCVBounceSimulator(CurveSimulator):
    """Inject a high-CV spike at mid-range to stress-test the sliding window."""
    def generate(self) -> SimulatedDataset:
        ds = super().generate()
        rng = np.random.default_rng(self.seed + 999)
        areas = ds.area.copy()

        # Find indices where concentration == 10.0 (mid-range) and multiply by random
        spike_mask = ds.concentration == 10.0
        if spike_mask.any():
            # apply massive noise to spike concentration
            areas[spike_mask] = areas[spike_mask] * rng.uniform(0.01, 5.0, size=spike_mask.sum())

        return SimulatedDataset(
            peptide=ds.peptide,
            concentration=ds.concentration,
            area=areas,
            ground_truth=ds.ground_truth,
        )


SCENARIOS: dict[str, CurveSimulator] = {
    # Happy path — clear LOD, clear LOQ
    'ideal_curve': CurveSimulator(
        slope=5000.0,
        intercept_linear=200.0,
        intercept_noise=500.0,
        concentrations=_DEFAULT_CONCS,
        n_replicates=3,
        n_peptides=5,
        cv=0.08,
        seed=42,
    ),
    # All areas exactly zero → no bootstrap variance → must not hang
    'all_zero': _AllZeroSimulator(
        slope=0.0,
        intercept_linear=0.0,
        intercept_noise=0.0,
        concentrations=_DEFAULT_CONCS,
        n_replicates=3,
        n_peptides=1,
        cv=0.0,
        seed=0,
    ),
    # No signal at any concentration (slope = 0)
    'all_noise': CurveSimulator(
        slope=0.0,
        intercept_linear=500.0,
        intercept_noise=500.0,
        concentrations=_DEFAULT_CONCS,
        n_replicates=3,
        n_peptides=1,
        cv=0.15,
        seed=1,
    ),
    # Non-monotonic CV with a spike at mid-range → sliding window must catch it
    'high_cv_bounce': _HighCVBounceSimulator(
        slope=5000.0,
        intercept_linear=200.0,
        intercept_noise=500.0,
        concentrations=_DEFAULT_CONCS,
        n_replicates=3,
        n_peptides=1,
        cv=0.08,
        seed=2,
    ),
    # Only 4 concentration levels — test min-points validation
    'sparse_curve': CurveSimulator(
        slope=5000.0,
        intercept_linear=200.0,
        intercept_noise=500.0,
        concentrations=_SPARSE_CONCS,
        n_replicates=3,
        n_peptides=1,
        cv=0.1,
        seed=3,
    ),
    # n_replicates=1 — EmpiricalCV should warn and CV will be NaN
    'single_replicate': CurveSimulator(
        slope=5000.0,
        intercept_linear=200.0,
        intercept_noise=500.0,
        concentrations=_DEFAULT_CONCS,
        n_replicates=1,
        n_peptides=1,
        cv=0.1,
        seed=4,
    ),
    # Blank (conc=0) has very low CV — LOQ must NOT be 0
    'blank_only_low_cv': CurveSimulator(
        slope=5000.0,
        intercept_linear=200.0,
        intercept_noise=500.0,
        concentrations=_DEFAULT_CONCS,
        n_replicates=5,  # many reps so blank CV is low
        n_peptides=1,
        cv=0.02,  # very low noise → blank will have low CV too
        seed=5,
    ),
    # Five orders of magnitude in x — weight numerical stability
    'wide_dynamic_range': CurveSimulator(
        slope=5000.0,
        intercept_linear=200.0,
        intercept_noise=500.0,
        concentrations=[0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0],
        n_replicates=3,
        n_peptides=1,
        cv=0.1,
        seed=6,
    ),
}


def get_scenario(name: str) -> CurveSimulator:
    """Return a pre-configured :class:`CurveSimulator` by name.

    Available names
    ---------------
    ``ideal_curve``, ``all_zero``, ``all_noise``, ``high_cv_bounce``,
    ``sparse_curve``, ``single_replicate``, ``blank_only_low_cv``,
    ``wide_dynamic_range``.
    """
    if name not in SCENARIOS:
        raise KeyError(
            f"Unknown scenario '{name}'. Available: {list(SCENARIOS.keys())}"
        )
    return SCENARIOS[name]
