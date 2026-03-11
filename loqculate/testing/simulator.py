"""Minimal synthetic signal generator for structural testing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np


class SimulatedDataset(NamedTuple):
    """Output of :meth:`CurveSimulator.generate`."""

    peptide: np.ndarray     # shape (n_peptides * n_concentrations * n_replicates,)
    concentration: np.ndarray
    area: np.ndarray
    ground_truth: dict      # {'slope', 'intercept_linear', 'intercept_noise', 'cv'}


@dataclass
class CurveSimulator:
    """Generate ``y = max(c, a*x + b) + heteroscedastic_noise`` data.

    Noise model
    -----------
    Heteroscedastic Gaussian: ``sigma_i = signal_i * cv``.
    This is structurally simple and sufficient for crash/interface testing.
    It is **not** statistically calibrated to real instrument data — see the
    plan for a note on this honest limitation.

    Parameters
    ----------
    slope:
        Linear segment slope *a*.
    intercept_linear:
        Linear segment intercept *b*.
    intercept_noise:
        Noise plateau level *c*.  Must satisfy ``c >= b`` (otherwise the
        piecewise definition is degenerate).
    concentrations:
        Concentration grid.  The default covers four orders of magnitude,
        matching a typical proteomics calibration curve.
    n_replicates:
        Replicates per concentration level.
    n_peptides:
        Number of independent peptide curves to generate.
    cv:
        Coefficient of variation for the heteroscedastic noise.
    seed:
        RNG seed for reproducibility.
    """

    slope: float = 5000.0
    intercept_linear: float = 200.0
    intercept_noise: float = 500.0
    concentrations: list = field(
        default_factory=lambda: [
            0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0,
            100.0, 200.0, 500.0, 1000.0, 2000.0,
        ]
    )
    n_replicates: int = 3
    n_peptides: int = 10
    cv: float = 0.12
    seed: int = 42

    def generate(self) -> SimulatedDataset:
        """Return a :class:`SimulatedDataset` with (peptide, concentration, area) triples."""
        rng = np.random.default_rng(self.seed)
        concs = np.asarray(self.concentrations, dtype=float)

        peptide_list: list[str] = []
        conc_list: list[float] = []
        area_list: list[float] = []

        for pep_idx in range(self.n_peptides):
            pep_name = f'peptide_{pep_idx:04d}'
            for c in concs:
                # True signal: y = max(intercept_noise, slope * c + intercept_linear)
                true_signal = max(self.intercept_noise, self.slope * c + self.intercept_linear)
                for _ in range(self.n_replicates):
                    sigma = true_signal * self.cv
                    # Clip at 0 to avoid negative areas
                    area = float(max(0.0, rng.normal(true_signal, sigma)))
                    peptide_list.append(pep_name)
                    conc_list.append(float(c))
                    area_list.append(area)

        ground_truth = {
            'slope': self.slope,
            'intercept_linear': self.intercept_linear,
            'intercept_noise': self.intercept_noise,
            'cv': self.cv,
        }

        return SimulatedDataset(
            peptide=np.array(peptide_list, dtype=str),
            concentration=np.array(conc_list, dtype=float),
            area=np.array(area_list, dtype=float),
            ground_truth=ground_truth,
        )
