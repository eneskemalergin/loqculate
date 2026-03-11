"""bench_scale.py — Throughput, memory, and LOQ coverage comparison at scale on the real full-dataset.

Answers
-------
* How many peptides per second does each implementation process in parallel?
* Which implementation uses more Python heap memory per peptide?
* How do finite-LOD/LOQ yields compare at scale?

Key methodology
---------------
* **Throughput phase**: both the original and loqculate are run via multiprocessing.Pool on
  ``--n_peptides`` (default 1000) peptides with ``--bootreps`` reps each.
* **Memory phase**: a smaller serial batch of ``--mem_peptides`` (default 50)
  peptides is run inside ``tracemalloc`` to sample peak heap allocation.
  Divide peak by ``mem_peptides`` to get MB/peptide.  The original uses pandas DataFrames
  internally (more allocations); loqculate uses pure numpy (less).
* The original is imported once per worker via ``importlib.util`` in a Pool initializer,
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
--n_reps       Timing repetitions per implementation for CI estimation (default: 10)
--workers      Parallel worker processes (default: number of CPUs)
--chunk_size   Peptides per multiprocessing chunk (default: 50)
"""
from __future__ import annotations

import argparse
import gc
import json
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
from _helpers import DEMO_DATA, DEMO_MAP, FULL_DATA, FULL_MAP, OLD_DIR, load_original_calc, _json_safe, _ci95

# loqculate package is at repo root — _helpers already adds it to sys.path
from loqculate.io import read_calibration_data
from loqculate.models import PiecewiseWLS


# -----------------------------------------------------------------------
# Module-level sentinel used by each worker for the original implementation
# -----------------------------------------------------------------------

_ORIGINAL_WORKER_MOD = None  # populated by _original_worker_init in each Pool worker


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ncpu = os.cpu_count() or 4
    p = argparse.ArgumentParser(
        description='original vs loqculate throughput + memory benchmark on full 23K-peptide dataset'
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
    p.add_argument('--n_reps', type=int, default=10,
                   help='Timing repetitions per implementation for CI estimation (default: 10)')
    p.add_argument('--demo', action='store_true',
                   help='Use demo dataset instead of full dataset '
                        '(auto-enabled if data/full/ is absent)')
    p.add_argument('--save', type=str, default=None, metavar='PATH',
                   help='Write JSON results to PATH (e.g. tmp/results/bench_scale.json)')
    return p.parse_args()


# -----------------------------------------------------------------------
# loqculate worker (top-level so multiprocessing can pickle it)
# -----------------------------------------------------------------------

def _loqculate_fit_one(args_tuple: Tuple) -> Tuple:
    pep, x_list, y_list, bootreps = args_tuple
    x = np.asarray(x_list, dtype=float)
    y = np.asarray(y_list, dtype=float)
    try:
        m = PiecewiseWLS(init_method='legacy', n_boot_reps=bootreps, seed=42)
        m.fit(x, y)
        return pep, float(m.lod()), float(m.loq()), None
    except Exception as exc:
        return pep, np.inf, np.inf, str(exc)


def _run_loqculate_parallel(
    mp_args: List[Tuple], workers: int, chunk_size: int
) -> Tuple[List[Tuple], float]:
    import multiprocessing as mp
    t0 = time.perf_counter()
    with mp.Pool(processes=workers) as pool:
        results = pool.map(_loqculate_fit_one, mp_args, chunksize=chunk_size)
    return results, time.perf_counter() - t0


# -----------------------------------------------------------------------
# original worker pool support
# -----------------------------------------------------------------------

def _original_worker_init(orig_path: str) -> None:
    """Load old/calculate-loq.py once per worker process."""
    import importlib.util
    global _ORIGINAL_WORKER_MOD
    warnings.filterwarnings('ignore')
    spec = importlib.util.spec_from_file_location('_original_calc_worker', orig_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _ORIGINAL_WORKER_MOD = mod


def _original_fit_one(args_tuple: Tuple) -> Tuple:
    """Original worker: uses the module pre-loaded by _original_worker_init."""
    import pandas as pd
    import tempfile
    global _ORIGINAL_WORKER_MOD
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
                row_df = _ORIGINAL_WORKER_MOD.process_peptide(
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


def _run_original_parallel(
    mp_args: List[Tuple], workers: int, chunk_size: int, orig_path: str
) -> Tuple[List[Tuple], float]:
    import multiprocessing as mp
    t0 = time.perf_counter()
    with mp.Pool(
        processes=workers,
        initializer=_original_worker_init,
        initargs=(orig_path,),
    ) as pool:
        results = pool.map(_original_fit_one, mp_args, chunksize=chunk_size)
    return results, time.perf_counter() - t0


# -----------------------------------------------------------------------
# Memory measurement (serial, inside tracemalloc)
# -----------------------------------------------------------------------

def _measure_loqculate_mem_mb(mem_args: List[Tuple]) -> float:
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


def _measure_original_mem_mb(orig_mod, mem_args: List[Tuple]) -> float:
    """Return peak Python heap (MB) for original processing mem_args peptides serially."""
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
                    orig_mod.process_peptide(
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
    guard in the original's piecewise branch was computed but silently ignored (the
    function returns the raw ``LOD`` variable instead of the updated
    ``lod_results``).  Filtering by ``(0, max_x]`` removes those artefacts
    so the coverage numbers are comparable between implementations.
    """
    finite_lods = [r[1] for r in results if np.isfinite(r[1])]
    finite_loqs = [r[2] for r in results if np.isfinite(r[2])]
    errors      = [r    for r in results if r[3] is not None]
    plausible_lods = [v for v in finite_lods if 0 < v <= max_x]
    plausible_loqs = [v for v in finite_loqs if 0 < v <= max_x]
    return finite_lods, finite_loqs, errors, plausible_lods, plausible_loqs


def _report(
    n: int, n_total: int,
    orig_res: List[Tuple], new_res: List[Tuple],
    orig_t_runs: List[float], new_t_runs: List[float],
    orig_mem_mb: float, new_mem_mb: float,
    n_mem: int,
    bootreps: int, workers: int, n_reps: int,
    max_x: float = np.inf,
    dataset_name: str = '',
) -> None:
    orig_lods, orig_loqs, orig_errors, orig_plod, orig_ploq = _stats(orig_res, max_x)
    new_lods,  new_loqs,  new_errors,  new_plod,  new_ploq  = _stats(new_res,  max_x)

    t_orig = float(np.mean(orig_t_runs))
    t_new  = float(np.mean(new_t_runs))
    t_orig_std = float(np.std(orig_t_runs, ddof=1)) if len(orig_t_runs) > 1 else 0.0
    t_new_std  = float(np.std(new_t_runs,  ddof=1)) if len(new_t_runs)  > 1 else 0.0
    orig_ci = _ci95(orig_t_runs)
    new_ci  = _ci95(new_t_runs)

    def _tp(t): return n / t if t > 0 else 0
    def _mspep(t): return 1000 * t / n if n and t > 0 else 0

    tp_orig, tp_new = _tp(t_orig), _tp(t_new)
    spd = t_orig / t_new if t_new > 0 else float('inf')
    winner = 'lq faster' if spd > 1.0 else 'orig faster'
    mem_ratio = orig_mem_mb / new_mem_mb if new_mem_mb > 0 else float('inf')
    mem_winner = 'lq leaner' if mem_ratio > 1.0 else 'orig leaner'

    def _lod_range(lst):
        if lst: return f'[{min(lst):.3e} – {max(lst):.3e}]'
        return '(none finite)'
    def _loq_range(lst):
        if lst: return f'[{min(lst):.3e} – {max(lst):.3e}]'
        return '(none finite)'

    W = 40  # column width for each version
    SEP = ' | '

    def _row(label, orig_text, new_text, note=''):
        label_col = f'  {label:<20}'
        note_col  = f'  {note}' if note else ''
        print(f'{label_col}{SEP}{orig_text:<{W}}{SEP}{new_text:<{W}}{note_col}')

    hdr = '=' * (24 + 2 + W + 2 + W + 4)
    print(f'\n{hdr}')
    print(f'  Scale benchmark   {SEP}{"original (lmfit L-M)":^{W}}{SEP}{"loqculate (scipy TRF)":^{W}}')
    print(hdr)

    print(f'  {"Dataset":<20}{SEP}{dataset_name}')
    print(f'  {"Peptides (scale)":<20}{SEP}{n} / {n_total} total')
    print(f'  {"Peptides (memory)":<20}{SEP}{n_mem} (serial, tracemalloc)')
    print(f'  {"Bootstrap reps":<20}{SEP}{bootreps}')
    print(f'  {"Parallel workers":<20}{SEP}{workers}')
    print(f'  {"Timing reps (CI)":<20}{SEP}{n_reps}')
    print()

    def _fmt_time(mean, ci, tp):
        if ci > 0:
            return f'{mean:.2f} ± {ci:.2f} s  ({_mspep(mean):.1f} ms/pep, {tp:.1f} pep/s)'
        return f'{mean:.2f} s  ({_mspep(mean):.1f} ms/pep, {tp:.1f} pep/s)'

    _row('Wall time (mean±CI)',
         _fmt_time(t_orig, orig_ci, tp_orig),
         _fmt_time(t_new,  new_ci,  tp_new),
         f'{spd:.2f}x  ({winner})')
    _row('Peak heap',
         f'{orig_mem_mb/n_mem:.4f} MB/pep  (total {orig_mem_mb:.1f} MB)',
         f'{new_mem_mb/n_mem:.4f} MB/pep  (total {new_mem_mb:.1f} MB)',
         f'{mem_ratio:.2f}x  ({mem_winner})')

    print()
    # Show raw finite counts AND plausible (in calibration range) counts.
    # The original implementation has a bug in calculate_lod's piecewise branch: edge-case
    # guards update lod_results to inf but the function returns the raw LOD variable, so
    # LODs > max(x) slip through as finite.  The '↳ in range' row applies the
    # same (0, max_x] filter to both for a fair comparison.
    _row(f'Finite LODs ({n})',
         f'{len(orig_lods)} / {n}  {_lod_range(orig_lods)}',
         f'{len(new_lods)} / {n}  {_lod_range(new_lods)}')
    if np.isfinite(max_x):
        note_lod = ''
        if len(orig_lods) != len(orig_plod):
            note_lod = f'original has {len(orig_lods)-len(orig_plod)} LODs > max_x (edge-case bug)'
        _row(f'  ↳ in range (0,{max_x:.3g}]',
             f'{len(orig_plod)} / {n}  {_lod_range(orig_plod)}',
             f'{len(new_plod)} / {n}  {_lod_range(new_plod)}',
             note_lod)
    _row(f'Finite LOQs ({n})',
         f'{len(orig_loqs)} / {n}  {_loq_range(orig_loqs)}',
         f'{len(new_loqs)} / {n}  {_loq_range(new_loqs)}')
    if np.isfinite(max_x):
        _row(f'  ↳ in range (0,{max_x:.3g}]',
             f'{len(orig_ploq)} / {n}  {_loq_range(orig_ploq)}',
             f'{len(new_ploq)} / {n}  {_loq_range(new_ploq)}')

    if n_total > n:
        est_orig = n_total / tp_orig if tp_orig > 0 else float('inf')
        est_new  = n_total / tp_new  if tp_new  > 0 else float('inf')
        print()
        _row(f'Est full ({n_total})',
             f'~{est_orig:.0f} s  (~{est_orig/60:.1f} min)',
             f'~{est_new:.0f} s  (~{est_new/60:.1f} min)')

    for label, errors in [('orig errors', orig_errors), ('lq errors', new_errors)]:
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
    # ------------------------------------------------------------------ #
    # Load data — auto-fall-back to demo dataset if full data is absent
    # ------------------------------------------------------------------ #
    use_demo = args.demo or not FULL_DATA.exists()
    if use_demo:
        data_path, map_path = DEMO_DATA, DEMO_MAP
        print(f'NOTE: Full dataset not found or --demo specified. '
              f'Using demo dataset ({DEMO_DATA.name}).')
        print(f'      Provide data/full/ and omit --demo for scale results.')
    else:
        data_path, map_path = FULL_DATA, FULL_MAP
    print(f'Reading dataset ({data_path.name}) ...')
    t_read = time.perf_counter()
    data = read_calibration_data(str(data_path), str(map_path))
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
    # Load original module (main process only, for memory phase)
    # ------------------------------------------------------------------ #
    orig_path = str(OLD_DIR / 'calculate-loq.py')
    print(f'\nLoading original module from {orig_path} ...')
    orig_mod = load_original_calc()

    # ------------------------------------------------------------------ #
    # Memory phase (serial, in main process, to keep tracemalloc isolated)
    # ------------------------------------------------------------------ #
    print(f'\nMemory phase: {n_mem} peptides, {args.bootreps} bootreps (serial) ...')
    print('  Measuring original peak heap ...', end='', flush=True)
    orig_mem_mb = _measure_original_mem_mb(orig_mod, mem_args)
    print(f' {orig_mem_mb:.2f} MB total  ({orig_mem_mb/n_mem:.4f} MB/pep)')

    print('  Measuring loqculate peak heap ...', end='', flush=True)
    new_mem_mb = _measure_loqculate_mem_mb(mem_args)
    print(f' {new_mem_mb:.2f} MB total  ({new_mem_mb/n_mem:.4f} MB/pep)')

    # ------------------------------------------------------------------ #
    # Throughput phase (parallel)
    # ------------------------------------------------------------------ #
    n_reps = args.n_reps
    print(f'\nThroughput phase: {n} peptides, {args.bootreps} bootreps, '
          f'{n_reps} timing rep(s) ({args.workers} workers, chunk={args.chunk_size}) ...')

    orig_t_runs: List[float] = []
    orig_res = []
    print('  Running original ...')
    for i in range(n_reps):
        orig_res, t = _run_original_parallel(mp_args, args.workers, args.chunk_size, orig_path)
        orig_t_runs.append(t)
        print(f'    rep {i+1:2d}/{n_reps}: {t:.2f}s  ({n/t:.1f} pep/s)', flush=True)

    new_t_runs: List[float] = []
    new_res = []
    print('  Running loqculate ...')
    for i in range(n_reps):
        new_res, t = _run_loqculate_parallel(mp_args, args.workers, args.chunk_size)
        new_t_runs.append(t)
        print(f'    rep {i+1:2d}/{n_reps}: {t:.2f}s  ({n/t:.1f} pep/s)', flush=True)

    # ------------------------------------------------------------------ #
    # Report
    # ------------------------------------------------------------------ #
    _report(
        n=n, n_total=n_total,
        orig_res=orig_res, new_res=new_res,
        orig_t_runs=orig_t_runs, new_t_runs=new_t_runs,
        orig_mem_mb=orig_mem_mb, new_mem_mb=new_mem_mb,
        n_mem=n_mem,
        bootreps=args.bootreps,
        workers=args.workers,
        n_reps=n_reps,
        max_x=max_x,
        dataset_name=data_path.name,
    )

    if args.save:
        import datetime

        def _agg(res_list):
            finite_lods   = [r[1] for r in res_list if np.isfinite(r[1])]
            finite_loqs   = [r[2] for r in res_list if np.isfinite(r[2])]
            plausible_lods = [v for v in finite_lods if 0 < v <= max_x]
            plausible_loqs = [v for v in finite_loqs if 0 < v <= max_x]
            return {
                'n_finite_lod':    len(finite_lods),
                'n_finite_loq':    len(finite_loqs),
                'n_plausible_lod': len(plausible_lods),
                'n_plausible_loq': len(plausible_loqs),
                'n_errors':        sum(1 for r in res_list if r[3] is not None),
            }

        t_orig_mean = float(np.mean(orig_t_runs))
        t_new_mean  = float(np.mean(new_t_runs))
        t_orig_std  = float(np.std(orig_t_runs, ddof=1)) if len(orig_t_runs) > 1 else 0.0
        t_new_std   = float(np.std(new_t_runs,  ddof=1)) if len(new_t_runs)  > 1 else 0.0
        orig_ci95   = _ci95(orig_t_runs)
        new_ci95    = _ci95(new_t_runs)
        speedup     = t_orig_mean / t_new_mean if t_new_mean > 0 else None
        tp_orig     = n / t_orig_mean if t_orig_mean > 0 else None
        tp_new      = n / t_new_mean  if t_new_mean  > 0 else None
        save_results = {
            'meta': {
                'dataset':    data_path.name,
                'n':          n,
                'n_total':    n_total,
                'n_mem':      n_mem,
                'bootreps':   args.bootreps,
                'workers':    args.workers,
                'n_reps':     n_reps,
                'max_x':      max_x,
                'timestamp':  datetime.datetime.now().isoformat(),
            },
            'throughput': {
                't_orig_runs_s':   orig_t_runs,
                't_new_runs_s':    new_t_runs,
                't_orig_mean_s':   t_orig_mean,
                't_new_mean_s':    t_new_mean,
                't_orig_std_s':    t_orig_std,
                't_new_std_s':     t_new_std,
                't_orig_ci95_s':   orig_ci95,
                't_new_ci95_s':    new_ci95,
                'pep_per_s_orig':  tp_orig,
                'pep_per_s_new':   tp_new,
                'speedup':         speedup,
            },
            'memory': {
                'orig_mb_total':       orig_mem_mb,
                'new_mb_total':        new_mem_mb,
                'orig_mb_per_pep':     orig_mem_mb / n_mem if n_mem > 0 else None,
                'new_mb_per_pep':      new_mem_mb  / n_mem if n_mem > 0 else None,
                'ratio_orig_over_new': orig_mem_mb / new_mem_mb if new_mem_mb > 0 else None,
            },
            'coverage_orig': _agg(orig_res),
            'coverage_new':  _agg(new_res),
        }
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(_json_safe(save_results), f, indent=2)
        print(f'Results saved \u2192 {out}')


if __name__ == '__main__':
    main()
