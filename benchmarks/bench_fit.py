"""bench_fit.py — per-fit timing: v1 (lmfit LM) vs v2 (scipy TRF).

Extracts REAL peptides from the demo dataset and times the core fitting
step in isolation (no bootstrap, no LOD/LOQ calculation).

Run from the repository root::

    python benchmarks/bench_fit.py
    python benchmarks/bench_fit.py --n_reps 50
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# --- path setup ---------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))   # allow _helpers import
from _helpers import DEMO_DATA, DEMO_MAP, load_v1_calc

from loqculate.models.piecewise_wls import PiecewiseWLS
from loqculate.io import read_calibration_data


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description='Per-fit timing: v1 vs v2')
    p.add_argument('--n_reps', type=int, default=30,
                   help='Number of repeated fits per peptide (default: 30)')
    return p.parse_args()


# -----------------------------------------------------------------------
# Load real peptides
# -----------------------------------------------------------------------

def _load_peptides():
    data = read_calibration_data(str(DEMO_DATA), str(DEMO_MAP))
    peptides = {}
    for pep in np.unique(data.peptide):
        mask = data.peptide == pep
        peptides[pep] = (data.concentration[mask], data.area[mask])
    return peptides


# -----------------------------------------------------------------------
# Core fitting functions (no bootstrap)
# -----------------------------------------------------------------------

def _time_v2_fit(x: np.ndarray, y: np.ndarray, n_reps: int) -> float:
    """Return median wall time (seconds) for a v2 single fit.

    Times ``PiecewiseWLS.fit()`` which includes:
      - parameter initialisation (numpy, ~0.03 ms)
      - scipy curve_fit / TRF optimisation (~1.7 ms)
    Object construction (``__init__``) is trivial (<0.001 ms) and excluded.
    """
    # warm-up
    try:
        PiecewiseWLS(init_method='legacy', n_boot_reps=0).fit(x, y)
    except Exception:
        return float('nan')
    # timed runs
    times = []
    for _ in range(n_reps):
        m = PiecewiseWLS(init_method='legacy', n_boot_reps=0)
        t0 = time.perf_counter()
        try:
            m.fit(x, y)
        except Exception:
            pass
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _time_v1_fit(v1_mod, x: np.ndarray, y: np.ndarray, n_reps: int) -> float:
    """Return median wall time (seconds) for a v1 single piecewise fit.

    Times ``fit_by_lmfit_yang(x, y, 'piecewise')`` which includes:
      - parameter initialisation via ``initialize_params_legacy`` (pandas, ~0.3 ms)
      - lmfit / Levenberg-Marquardt optimisation (~1.5 ms)
    Note: v1's ``initialize_params_legacy`` uses a pandas GroupBy; v2's numpy
    equivalent is ~10× faster (~0.03 ms) but the optimizer is comparable.
    """
    _fit = v1_mod.fit_by_lmfit_yang

    # warm-up (suppress any lmfit/scipy warnings)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            _fit(x, y, 'piecewise')
        except Exception:
            return float('nan')

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                _fit(x, y, 'piecewise')
            except Exception:
                pass
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# -----------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------

def _report(peptides, timings_v1, timings_v2):
    n = len(peptides)
    print(f'\n{"=" * 65}')
    print(f'  Per-fit timing (median over {list(timings_v1.values())[0][0]} reps)')
    print(f'  Data: {DEMO_DATA.name}  ({n} peptides)')
    print(f'{"=" * 65}')
    print(f'{"Peptide":<40} {"v1 (ms)":>10} {"v2 (ms)":>10} {"speedup":>10}')
    print('-' * 75)

    all_v1, all_v2 = [], []
    for pep in sorted(peptides):
        n_reps, v1_t = timings_v1[pep]
        _,      v2_t = timings_v2[pep]
        v1_ms = v1_t * 1000
        v2_ms = v2_t * 1000
        if np.isfinite(v1_t) and np.isfinite(v2_t) and v2_t > 0:
            spd = f'{v1_t/v2_t:.2f}x'
            all_v1.append(v1_t)
            all_v2.append(v2_t)
        else:
            spd = '  —'
        v1_str = f'{v1_ms:.3f}' if np.isfinite(v1_ms) else '  NaN'
        v2_str = f'{v2_ms:.3f}' if np.isfinite(v2_ms) else '  NaN'
        print(f'{pep:<40} {v1_str:>10} {v2_str:>10} {spd:>10}')

    if all_v1 and all_v2:
        print('-' * 75)
        print(f'  median v1 : {np.median(all_v1)*1000:.3f} ms/fit')
        print(f'  median v2 : {np.median(all_v2)*1000:.3f} ms/fit')
        print(f'  speedup   : {np.median(all_v1)/np.median(all_v2):.2f}x')
    print()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    args = _parse_args()
    print('Loading v1 module ...')
    v1 = load_v1_calc()

    print(f'Reading demo data ({DEMO_DATA.name}) ...')
    peptides = _load_peptides()
    print(f'  {len(peptides)} peptides loaded.')
    print(f'  Each will be timed over {args.n_reps} repeated fits.\n')

    timings_v1, timings_v2 = {}, {}
    for pep, (x, y) in peptides.items():
        t_v2 = _time_v2_fit(x, y, args.n_reps)
        t_v1 = _time_v1_fit(v1, x, y, args.n_reps)
        timings_v1[pep] = (args.n_reps, t_v1)
        timings_v2[pep] = (args.n_reps, t_v2)
        print(f'  {pep:<40}  v1={t_v1*1000:.2f}ms  v2={t_v2*1000:.2f}ms')

    _report(peptides, timings_v1, timings_v2)


if __name__ == '__main__':
    main()
