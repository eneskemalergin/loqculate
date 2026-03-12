"""bench_fit.py — per-fit timing: original (lmfit LM) vs loqculate (scipy TRF).

Extracts REAL peptides from the demo dataset and times the core fitting
step in isolation.  Two timing regimes are reported:

1. **Single-fit (no bootstrap)** — isolates the optimizer (TRF vs LM).
   Both methods timed with ``n_boot_reps=0`` so bootstrap is excluded.
2. **Full-fit (with bootstrap)** — the complete wall time a user experiences.
   loqculate only, timed with ``--bootreps`` reps.  The optimizer fraction
   of end-to-end time is derived from (1) / (2).

Run from the repository root::

    python benchmarks/bench_fit.py
    python benchmarks/bench_fit.py --n_reps 50 --bootreps 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# --- path setup ---------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))   # allow _helpers import
from _helpers import DEMO_DATA, DEMO_MAP, load_original_calc, _json_safe

from loqculate.models.piecewise_wls import PiecewiseWLS
from loqculate.io import read_calibration_data


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description='Per-fit timing: original vs loqculate')
    p.add_argument('--n_reps', type=int, default=30,
                   help='Number of repeated single-fits per peptide (default: 30)')
    p.add_argument('--boot_n_reps', type=int, default=10,
                   help='Repeated full-fits (with bootstrap) per peptide (default: 10)')
    p.add_argument('--bootreps', type=int, default=100,
                   help='Bootstrap replicates per fit in the full-fit section (default: 100)')
    p.add_argument('--save', type=str, default=None, metavar='PATH',
                   help='Write JSON results to PATH (e.g. tmp/results/bench_fit.json)')
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

def _time_loqculate_fit(x: np.ndarray, y: np.ndarray, n_reps: int) -> float:
    """Return median wall time (seconds) for a single loqculate fit (no bootstrap).

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


def _time_loqculate_full(x: np.ndarray, y: np.ndarray, n_reps: int, bootreps: int) -> float:
    """Return median wall time (seconds) for a full loqculate fit WITH bootstrap.

    This is the end-to-end time per peptide that a user experiences in practice.
    Comparing with ``_time_loqculate_fit`` shows the optimizer vs bootstrap split.
    """
    try:
        PiecewiseWLS(init_method='legacy', n_boot_reps=bootreps, seed=42).fit(x, y)
    except Exception:
        return float('nan')
    times = []
    for _ in range(n_reps):
        m = PiecewiseWLS(init_method='legacy', n_boot_reps=bootreps, seed=42)
        t0 = time.perf_counter()
        try:
            m.fit(x, y)
        except Exception:
            pass
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _time_original_fit(orig_mod, x: np.ndarray, y: np.ndarray, n_reps: int) -> float:
    """Return median wall time (seconds) for the original single piecewise fit.

    Times ``fit_by_lmfit_yang(x, y, 'piecewise')`` which includes:
      - parameter initialisation via ``initialize_params_legacy`` (pandas, ~0.3 ms)
      - lmfit / Levenberg-Marquardt optimisation (~1.5 ms)
    Note: the original ``initialize_params_legacy`` uses a pandas GroupBy; the
    numpy equivalent used here is ~10× faster (~0.03 ms) but the optimizer is comparable.
    """
    _fit = orig_mod.fit_by_lmfit_yang

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

def _report(peptides, timings_orig, timings_new, timings_full, bootreps):
    n = len(peptides)
    print(f'\n{"=" * 75}')
    print(f'  Part 1: Single-fit timing (median over {list(timings_orig.values())[0][0]} reps, no bootstrap)')
    print(f'  Data: {DEMO_DATA.name}  ({n} peptides)')
    print(f'{"=" * 75}')
    print(f'{"Peptide":<40} {"orig (ms)":>10} {"lq (ms)":>10} {"speedup":>10}')
    print('-' * 75)

    all_orig, all_new = [], []
    for pep in sorted(peptides):
        n_reps, t_orig = timings_orig[pep]
        _,      t_new  = timings_new[pep]
        ms_orig = t_orig * 1000
        ms_new  = t_new  * 1000
        if np.isfinite(t_orig) and np.isfinite(t_new) and t_new > 0:
            spd = f'{t_orig/t_new:.2f}x'
            all_orig.append(t_orig)
            all_new.append(t_new)
        else:
            spd = '  —'
        orig_str = f'{ms_orig:.3f}' if np.isfinite(ms_orig) else '  NaN'
        new_str  = f'{ms_new:.3f}'  if np.isfinite(ms_new)  else '  NaN'
        print(f'{pep:<40} {orig_str:>10} {new_str:>10} {spd:>10}')

    if all_orig and all_new:
        print('-' * 75)
        print(f'  median orig : {np.median(all_orig)*1000:.3f} ms/fit')
        print(f'  median lq   : {np.median(all_new)*1000:.3f} ms/fit')
        print(f'  speedup     : {np.median(all_orig)/np.median(all_new):.2f}x')

    # Part 2: full-fit with bootstrap
    if timings_full:
        n_boot_reps_used = list(timings_full.values())[0][0]
        print(f'\n{"=" * 75}')
        print(f'  Part 2: Full-fit timing (median over {n_boot_reps_used} reps, {bootreps} bootstrap reps)')
        print(f'  Shows optimizer vs bootstrap time split for loqculate.')
        print(f'{"=" * 75}')
        print(f'{"Peptide":<40} {"opt (ms)":>10} {"full (ms)":>10} {"opt%":>8}')
        print('-' * 75)
        all_full = []
        for pep in sorted(peptides):
            _, t_opt  = timings_new[pep]
            _, t_full = timings_full[pep]
            ms_opt  = t_opt  * 1000
            ms_full = t_full * 1000
            if np.isfinite(t_opt) and np.isfinite(t_full) and t_full > 0:
                opt_pct = f'{t_opt/t_full*100:.0f}%'
                all_full.append(t_full)
            else:
                opt_pct = '  —'
            opt_str  = f'{ms_opt:.3f}'  if np.isfinite(ms_opt)  else '  NaN'
            full_str = f'{ms_full:.3f}' if np.isfinite(ms_full) else '  NaN'
            print(f'{pep:<40} {opt_str:>10} {full_str:>10} {opt_pct:>8}')
        if all_new and all_full:
            med_opt   = float(np.median([t for t in all_new if np.isfinite(t)]))
            med_full  = float(np.median(all_full))
            print('-' * 75)
            print(f'  median optimizer  : {med_opt*1000:.3f} ms/fit')
            print(f'  median full       : {med_full*1000:.3f} ms/fit')
            pct = med_opt / med_full * 100 if med_full > 0 else float('nan')
            print(f'  optimizer is {pct:.0f}% of full-fit wall time  '
                f'(bootstrap = {100-pct:.0f}%)')
    print()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    args = _parse_args()
    print('Loading original module ...')
    orig = load_original_calc()

    print(f'Reading demo data ({DEMO_DATA.name}) ...')
    peptides = _load_peptides()
    print(f'  {len(peptides)} peptides loaded.')
    print(f'  Single-fit: {args.n_reps} reps/peptide (no bootstrap)')
    print(f'  Full-fit:   {args.boot_n_reps} reps/peptide ({args.bootreps} bootstrap reps)\n')

    timings_orig, timings_new = {}, {}
    for pep, (x, y) in peptides.items():
        t_new  = _time_loqculate_fit(x, y, args.n_reps)
        t_orig = _time_original_fit(orig, x, y, args.n_reps)
        timings_orig[pep] = (args.n_reps, t_orig)
        timings_new[pep]  = (args.n_reps, t_new)
        print(f'  [single] {pep:<40}  orig={t_orig*1000:.2f}ms  lq={t_new*1000:.2f}ms')

    print()
    timings_full = {}
    for pep, (x, y) in peptides.items():
        t_full = _time_loqculate_full(x, y, args.boot_n_reps, args.bootreps)
        timings_full[pep] = (args.boot_n_reps, t_full)
        t_opt = timings_new[pep][1]
        opt_pct = f'{t_opt/t_full*100:.0f}%' if np.isfinite(t_full) and t_full > 0 else '?'
        print(f'  [full]   {pep:<40}  lq={t_full*1000:.2f}ms  optimizer={opt_pct}')

    _report(peptides, timings_orig, timings_new, timings_full, args.bootreps)

    if args.save:
        import datetime
        all_valid = [
            (p, timings_orig[p][1], timings_new[p][1])
            for p in peptides
            if np.isfinite(timings_orig[p][1]) and np.isfinite(timings_new[p][1])
            and timings_new[p][1] > 0
        ]
        speedups = [o / n for _, o, n in all_valid]
        all_valid_full = [
            (p, timings_new[p][1], timings_full[p][1])
            for p in peptides
            if np.isfinite(timings_new[p][1]) and np.isfinite(timings_full[p][1])
            and timings_full[p][1] > 0
        ]
        per_pep = {}
        for pep in sorted(peptides):
            n_reps_val, t_o = timings_orig[pep]
            _,           t_n = timings_new[pep]
            _,          t_f = timings_full[pep]
            per_pep[pep] = {
                'ms_orig':       t_o * 1000 if np.isfinite(t_o) else None,
                'ms_lq_single':  t_n * 1000 if np.isfinite(t_n)  else None,
                'ms_lq_full':    t_f * 1000 if np.isfinite(t_f)  else None,
                'speedup_single': (t_o / t_n) if (np.isfinite(t_o) and np.isfinite(t_n) and t_n > 0) else None,
                'opt_frac': (t_n / t_f) if (np.isfinite(t_n) and np.isfinite(t_f) and t_f > 0)
                            else None,
            }
        all_single_new = [t for _, _, t in all_valid]
        all_full_new   = [t for _, _, t in all_valid_full]
        results = {
            'meta': {
                'n_reps':          args.n_reps,
                'boot_n_reps':     args.boot_n_reps,
                'bootreps':        args.bootreps,
                'n_peptides':      len(peptides),
                'dataset':         DEMO_DATA.name,
                'timestamp':       datetime.datetime.now().isoformat(),
            },
            'per_peptide': per_pep,
            'summary_single': {
                'median_ms_orig':  float(np.median([o for _, o, _ in all_valid]) * 1000)
                                if all_valid else None,
                'median_ms_lq':    float(np.median(all_single_new) * 1000)
                                if all_single_new else None,
                'median_speedup':  float(np.median(speedups)) if speedups else None,
                'p05_speedup':     float(np.percentile(speedups,  5)) if speedups else None,
                'p95_speedup':     float(np.percentile(speedups, 95)) if speedups else None,
                'n_valid':         len(all_valid),
            },
            'summary_full': {
                'median_ms_lq_single': float(np.median(all_single_new) * 1000)
                                    if all_single_new else None,
                'median_ms_lq_full':   float(np.median(all_full_new) * 1000)
                                    if all_full_new else None,
                'median_opt_frac':     float(np.median(
                                        [t_n / t_f for _, t_n, t_f in all_valid_full
                                            if t_f > 0]
                                        )) if all_valid_full else None,
                'n_valid':             len(all_valid_full),
            },
        }
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(_json_safe(results), f, indent=2)
        print(f'Results saved → {out}')


if __name__ == '__main__':
    main()
