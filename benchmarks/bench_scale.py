"""bench_scale.py — Head-to-head v1 vs v2 throughput, memory, and LOQ coverage
at scale on the real full-dataset.

Answers
-------
* How many peptides per second does each version process in parallel?
* Which version uses more Python heap memory per peptide?
* How do finite-LOD/LOQ yields compare at scale?

Key methodology
---------------
* **Throughput phase**: both v1 and v2 are run via multiprocessing.Pool on
  ``--n_peptides`` (default 1000) peptides with ``--bootreps`` reps each.
* **Memory phase**: a smaller serial batch of ``--mem_peptides`` (default 50)
  peptides is run inside ``tracemalloc`` to sample peak heap allocation.
  Divide peak by ``mem_peptides`` to get MB/peptide.  v1 uses pandas DataFrames
  internally (more allocations); v2 uses pure numpy (less).
* v1 is imported once per worker via ``importlib.util`` in a Pool initializer,
  avoiding repeated file-level ``exec`` in the hot loop.

Run from the repository root::

    python benchmarks/bench_scale.py                      # first 1000 peptides
    python benchmarks/bench_scale.py --n_peptides 5000    # larger batch
    python benchmarks/bench_scale.py --n_peptides 0       # all ~23k peptides

Options
-------
--n_peptides   How many peptides for the throughput phase (0 = all, default: 1000)
--mem_peptides Peptides for the memory-measurement serial phase (default: 50)
--bootreps     Bootstrap reps per peptide (default: 100)
--workers      Parallel worker processes (default: number of CPUs)
--chunk_size   Peptides per multiprocessing chunk (default: 50)
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import tracemalloc
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

# --- path setup ---------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))   # benchmarks/ on path
from _helpers import FULL_DATA, FULL_MAP, V1_DIR, load_v1_calc

# Ensure v2 package is importable
sys.path.insert(0, str(Path(__file__).parent.parent / 'v2'))
from loqculate.io import read_calibration_data
from loqculate.models import PiecewiseWLS


# -----------------------------------------------------------------------
# Module-level sentinel used by each worker for v1 (set in initializer)
# -----------------------------------------------------------------------

_V1_WORKER_MOD = None  # populated by _v1_worker_init in each Pool worker


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ncpu = os.cpu_count() or 4
    p = argparse.ArgumentParser(
        description='v1 vs v2 throughput + memory benchmark on full 23K-peptide dataset'
    )
    p.add_argument('--n_peptides', type=int, default=1000,
                   help='Peptides to process in throughput phase (0 = all, default: 1000)')
    p.add_argument('--mem_peptides', type=int, default=50,
                   help='Peptides in serial memory-measurement phase (default: 50)')
    p.add_argument('--bootreps', type=int, default=100,
                   help='Bootstrap reps per peptide (default: 100)')
    p.add_argument('--workers', type=int, default=ncpu,
                   help=f'Parallel workers (default: {ncpu})')
    p.add_argument('--chunk_size', type=int, default=50,
                   help='Peptides per chunk sent to each worker (default: 50)')
    return p.parse_args()


# -----------------------------------------------------------------------
# v2 worker (top-level so multiprocessing can pickle it)
# -----------------------------------------------------------------------

def _v2_fit_one(args_tuple: Tuple) -> Tuple:
    pep, x_list, y_list, bootreps = args_tuple
    x = np.asarray(x_list, dtype=float)
    y = np.asarray(y_list, dtype=float)
    try:
        m = PiecewiseWLS(init_method='legacy', n_boot_reps=bootreps, seed=42)
        m.fit(x, y)
        return pep, float(m.lod()), float(m.loq()), None
    except Exception as exc:
        return pep, np.inf, np.inf, str(exc)


def _run_v2_parallel(
    mp_args: List[Tuple], workers: int, chunk_size: int
) -> Tuple[List[Tuple], float]:
    import multiprocessing as mp
    t0 = time.perf_counter()
    with mp.Pool(processes=workers) as pool:
        results = pool.map(_v2_fit_one, mp_args, chunksize=chunk_size)
    return results, time.perf_counter() - t0


# -----------------------------------------------------------------------
# v1 worker pool support
# -----------------------------------------------------------------------

def _v1_worker_init(v1_path: str) -> None:
    """Load v1/calculate-loq.py once per worker process."""
    import importlib.util
    global _V1_WORKER_MOD
    warnings.filterwarnings('ignore')
    spec = importlib.util.spec_from_file_location('_v1_calc_worker', v1_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _V1_WORKER_MOD = mod


def _v1_fit_one(args_tuple: Tuple) -> Tuple:
    """v1 worker: uses the module pre-loaded by _v1_worker_init."""
    import pandas as pd
    import tempfile
    global _V1_WORKER_MOD
    pep, x_list, y_list, bootreps = args_tuple
    x = np.asarray(x_list, dtype=float)
    y = np.asarray(y_list, dtype=float)
    subset = pd.DataFrame({
        'peptide': pep,
        'curvepoint': x,
        'area': y,
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                row_df = _V1_WORKER_MOD.process_peptide(
                    bootreps, 0.2, tmpdir, pep,
                    'n',   # plot_or_not
                    2.0,   # std_mult
                    2,     # min_noise_points
                    1,     # min_linear_points
                    subset,
                    'n',   # verbose
                    'piecewise',  # model_choice
                )
                row = row_df.iloc[0].to_dict()
                lod = float(row.get('LOD', np.inf))
                loq = float(row.get('LOQ', np.inf))
                return pep, lod, loq, None
            except Exception as exc:
                return pep, np.inf, np.inf, str(exc)


def _run_v1_parallel(
    mp_args: List[Tuple], workers: int, chunk_size: int, v1_path: str
) -> Tuple[List[Tuple], float]:
    import multiprocessing as mp
    t0 = time.perf_counter()
    with mp.Pool(
        processes=workers,
        initializer=_v1_worker_init,
        initargs=(v1_path,),
    ) as pool:
        results = pool.map(_v1_fit_one, mp_args, chunksize=chunk_size)
    return results, time.perf_counter() - t0


# -----------------------------------------------------------------------
# Memory measurement (serial, inside tracemalloc)
# -----------------------------------------------------------------------

def _measure_v2_mem_mb(mem_args: List[Tuple]) -> float:
    """Return peak Python heap (MB) for processing mem_args peptides serially."""
    gc.collect()
    tracemalloc.start()
    for pep, x_list, y_list, bootreps in mem_args:
        x = np.asarray(x_list, dtype=float)
        y = np.asarray(y_list, dtype=float)
        try:
            m = PiecewiseWLS(init_method='legacy', n_boot_reps=bootreps, seed=42)
            m.fit(x, y)
        except Exception:
            pass
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6


def _measure_v1_mem_mb(v1_mod, mem_args: List[Tuple]) -> float:
    """Return peak Python heap (MB) for v1 processing mem_args peptides serially."""
    import pandas as pd
    import tempfile
    gc.collect()
    tracemalloc.start()
    with tempfile.TemporaryDirectory() as tmpdir:
        for pep, x_list, y_list, bootreps in mem_args:
            x = np.asarray(x_list, dtype=float)
            y = np.asarray(y_list, dtype=float)
            subset = pd.DataFrame({
                'peptide': pep,
                'curvepoint': x,
                'area': y,
            })
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    v1_mod.process_peptide(
                        bootreps, 0.2, tmpdir, pep, 'n', 2.0, 2, 1,
                        subset, 'n', 'piecewise',
                    )
                except Exception:
                    pass
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6


# -----------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------

def _stats(results: List[Tuple], max_x: float = np.inf):
    """Return finite LOD/LOQ lists plus a plausible (in-range) subset.

    ``max_x`` is the maximum concentration in the calibration data.  Any
    LOD/LOQ above that value is physically impossible — it means the edge-case
    guard in v1's piecewise branch was computed but silently ignored (the
    function returns the raw ``LOD`` variable instead of the updated
    ``lod_results``).  Filtering by ``(0, max_x]`` removes those artefacts
    so the coverage numbers are comparable between versions.
    """
    finite_lods = [r[1] for r in results if np.isfinite(r[1])]
    finite_loqs = [r[2] for r in results if np.isfinite(r[2])]
    errors      = [r    for r in results if r[3] is not None]
    plausible_lods = [v for v in finite_lods if 0 < v <= max_x]
    plausible_loqs = [v for v in finite_loqs if 0 < v <= max_x]
    return finite_lods, finite_loqs, errors, plausible_lods, plausible_loqs


def _report(
    n: int, n_total: int,
    v1_res: List[Tuple], v2_res: List[Tuple],
    t_v1: float, t_v2: float,
    v1_mem_mb: float, v2_mem_mb: float,
    n_mem: int,
    bootreps: int, workers: int,
    max_x: float = np.inf,
) -> None:
    v1_lods, v1_loqs, v1_errors, v1_plod, v1_ploq = _stats(v1_res, max_x)
    v2_lods, v2_loqs, v2_errors, v2_plod, v2_ploq = _stats(v2_res, max_x)

    def _tp(t): return n / t if t > 0 else 0
    def _mspep(t): return 1000 * t / n if n and t > 0 else 0

    tp1, tp2 = _tp(t_v1), _tp(t_v2)
    spd = t_v1 / t_v2 if t_v2 > 0 else float('inf')
    winner = 'v2 faster' if spd > 1.0 else 'v1 faster'
    mem_ratio = v1_mem_mb / v2_mem_mb if v2_mem_mb > 0 else float('inf')
    mem_winner = 'v2 leaner' if mem_ratio > 1.0 else 'v1 leaner'

    def _lod_range(lst):
        if lst: return f'[{min(lst):.3e} – {max(lst):.3e}]'
        return '(none finite)'
    def _loq_range(lst):
        if lst: return f'[{min(lst):.3e} – {max(lst):.3e}]'
        return '(none finite)'

    W = 40  # column width for each version
    SEP = ' | '

    def _row(label, v1_text, v2_text, note=''):
        label_col = f'  {label:<20}'
        note_col  = f'  {note}' if note else ''
        print(f'{label_col}{SEP}{v1_text:<{W}}{SEP}{v2_text:<{W}}{note_col}')

    hdr = '=' * (24 + 2 + W + 2 + W + 4)
    print(f'\n{hdr}')
    print(f'  Scale benchmark   {SEP}{"v1  (lmfit L-M)":^{W}}{SEP}{"v2  (scipy TRF)":^{W}}')
    print(hdr)

    print(f'  {"Dataset":<20}{SEP}{FULL_DATA.name}')
    print(f'  {"Peptides (scale)":<20}{SEP}{n} / {n_total} total')
    print(f'  {"Peptides (memory)":<20}{SEP}{n_mem} (serial, tracemalloc)')
    print(f'  {"Bootstrap reps":<20}{SEP}{bootreps}')
    print(f'  {"Parallel workers":<20}{SEP}{workers}')
    print()

    _row('Wall time',
         f'{t_v1:.2f} s  ({tp1:.1f} pep/s, {_mspep(t_v1):.1f} ms/pep)',
         f'{t_v2:.2f} s  ({tp2:.1f} pep/s, {_mspep(t_v2):.1f} ms/pep)',
         f'{spd:.2f}x  ({winner})')
    _row('Peak heap',
         f'{v1_mem_mb/n_mem:.4f} MB/pep  (total {v1_mem_mb:.1f} MB)',
         f'{v2_mem_mb/n_mem:.4f} MB/pep  (total {v2_mem_mb:.1f} MB)',
         f'{mem_ratio:.2f}x  ({mem_winner})')

    print()
    # Show raw finite counts AND plausible (in calibration range) counts.
    # v1 has a bug in calculate_lod's piecewise branch: edge-case guards update
    # lod_results to inf but the function returns the raw LOD variable, so
    # LODs > max(x) slip through as finite.  The '↳ in range' row applies the
    # same (0, max_x] filter to both versions for a fair comparison.
    _row(f'Finite LODs ({n})',
         f'{len(v1_lods)} / {n}  {_lod_range(v1_lods)}',
         f'{len(v2_lods)} / {n}  {_lod_range(v2_lods)}')
    if np.isfinite(max_x):
        note_lod = ''
        if len(v1_lods) != len(v1_plod):
            note_lod = f'v1 has {len(v1_lods)-len(v1_plod)} LODs > max_x (v1 edge-case bug)'
        _row(f'  ↳ in range (0,{max_x:.3g}]',
             f'{len(v1_plod)} / {n}  {_lod_range(v1_plod)}',
             f'{len(v2_plod)} / {n}  {_lod_range(v2_plod)}',
             note_lod)
    _row(f'Finite LOQs ({n})',
         f'{len(v1_loqs)} / {n}  {_loq_range(v1_loqs)}',
         f'{len(v2_loqs)} / {n}  {_loq_range(v2_loqs)}')
    if np.isfinite(max_x):
        _row(f'  ↳ in range (0,{max_x:.3g}]',
             f'{len(v1_ploq)} / {n}  {_loq_range(v1_ploq)}',
             f'{len(v2_ploq)} / {n}  {_loq_range(v2_ploq)}')

    if n_total > n:
        est_v1 = n_total / tp1 if tp1 > 0 else float('inf')
        est_v2 = n_total / tp2 if tp2 > 0 else float('inf')
        print()
        _row(f'Est full ({n_total})',
             f'~{est_v1:.0f} s  (~{est_v1/60:.1f} min)',
             f'~{est_v2:.0f} s  (~{est_v2/60:.1f} min)')

    for label, errors in [('v1 errors', v1_errors), ('v2 errors', v2_errors)]:
        if errors:
            print(f'\n  {label}: {len(errors)} peptides failed')
            for pep, _, _, err in errors[:3]:
                print(f'    {pep}: {err}')
            if len(errors) > 3:
                print(f'    … and {len(errors) - 3} more')

    print(f'\n{hdr}\n')


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # ------------------------------------------------------------------ #
    # Load data
    # ------------------------------------------------------------------ #
    print(f'Reading full dataset ({FULL_DATA.name}) ...')
    t_read = time.perf_counter()
    data = read_calibration_data(str(FULL_DATA), str(FULL_MAP))
    t_read = time.perf_counter() - t_read

    all_peptides = np.unique(data.peptide)
    n_total = len(all_peptides)
    max_x = float(np.max(data.concentration))
    print(f'  {n_total} peptides loaded in {t_read:.2f}s.  max concentration = {max_x:.4g}')

    # Select subset for throughput phase
    n_req = args.n_peptides
    if n_req > 0 and n_req < n_total:
        selected = all_peptides[:n_req]
        print(f'  Using first {n_req} peptides for throughput phase.')
    else:
        selected = all_peptides
        print(f'  Using all {n_total} peptides for throughput phase.')

    n = len(selected)

    # Build picklable arg lists
    mp_args = []
    for pep in selected:
        mask = data.peptide == pep
        mp_args.append((pep, data.concentration[mask].tolist(),
                         data.area[mask].tolist(), args.bootreps))

    n_mem = min(args.mem_peptides, n)
    mem_args = mp_args[:n_mem]

    # ------------------------------------------------------------------ #
    # Load v1 module (main process only, for memory phase)
    # ------------------------------------------------------------------ #
    v1_path = str(V1_DIR / 'calculate-loq.py')
    print(f'\nLoading v1 module from {v1_path} ...')
    v1_mod = load_v1_calc()

    # ------------------------------------------------------------------ #
    # Memory phase (serial, in main process, to keep tracemalloc isolated)
    # ------------------------------------------------------------------ #
    print(f'\nMemory phase: {n_mem} peptides, {args.bootreps} bootreps (serial) ...')
    print('  Measuring v1 peak heap ...', end='', flush=True)
    v1_mem_mb = _measure_v1_mem_mb(v1_mod, mem_args)
    print(f' {v1_mem_mb:.2f} MB total  ({v1_mem_mb/n_mem:.4f} MB/pep)')

    print('  Measuring v2 peak heap ...', end='', flush=True)
    v2_mem_mb = _measure_v2_mem_mb(mem_args)
    print(f' {v2_mem_mb:.2f} MB total  ({v2_mem_mb/n_mem:.4f} MB/pep)')

    # ------------------------------------------------------------------ #
    # Throughput phase (parallel)
    # ------------------------------------------------------------------ #
    print(f'\nThroughput phase: {n} peptides, {args.bootreps} bootreps '
          f'({args.workers} workers, chunk={args.chunk_size}) ...')
    print('  Running v1 ...', end='', flush=True)
    v1_res, t_v1 = _run_v1_parallel(mp_args, args.workers, args.chunk_size, v1_path)
    print(f' done  {t_v1:.2f}s  ({n/t_v1:.1f} pep/s)')

    print('  Running v2 ...', end='', flush=True)
    v2_res, t_v2 = _run_v2_parallel(mp_args, args.workers, args.chunk_size)
    print(f' done  {t_v2:.2f}s  ({n/t_v2:.1f} pep/s)')

    # ------------------------------------------------------------------ #
    # Report
    # ------------------------------------------------------------------ #
    _report(
        n=n, n_total=n_total,
        v1_res=v1_res, v2_res=v2_res,
        t_v1=t_v1, t_v2=t_v2,
        v1_mem_mb=v1_mem_mb, v2_mem_mb=v2_mem_mb,
        n_mem=n_mem,
        bootreps=args.bootreps,
        workers=args.workers,
        max_x=max_x,
    )


if __name__ == '__main__':
    main()
