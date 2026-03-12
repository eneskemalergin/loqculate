"""bench_scale.py — Throughput, memory, and LOQ coverage comparison at scale on the real full-dataset.

Answers
-------
* How many peptides per second does each implementation process in parallel?
* Which implementation uses more RSS memory per peptide?
* How do finite-LOD/LOQ yields compare at scale?
* What fraction of peptides are quantifiable at each calibration concentration?

Key methodology
---------------
* **Throughput phase**: both the original and loqculate are run via multiprocessing.Pool on
  ``--n_peptides`` (default 1000) peptides with ``--bootreps`` reps each.
* **Memory phase**: a smaller serial batch of ``--mem_peptides`` (default 50)
  peptides is processed serially and VmRSS (from /proc/self/status) is sampled
  before and after.  The RSS delta captures C-extension (NumPy/scipy) allocations
  that tracemalloc misses.  Divide delta by ``mem_peptides`` to get MB/peptide.
* **LOQ yield phase**: for each calibration concentration level, the fraction of
  processed peptides with a finite LOQ ≤ that level is reported (empirical
  sensitivity curve of the assay).
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
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

# --- path setup ---------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))   # benchmarks/ on path
from _helpers import DEMO_DATA, DEMO_MAP, FULL_DATA, FULL_MAP, OLD_DIR, load_original_calc, load_original_cv, _json_safe, _ci95, _rss_mb, suppress_stdio

# loqculate package is at repo root — _helpers already adds it to sys.path
from loqculate.io import read_calibration_data
from loqculate.models import PiecewiseWLS
from loqculate.models.cv_empirical import EmpiricalCV


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
# EmpiricalCV bulk path (no multiprocessing — vectorized across all peptides)
# -----------------------------------------------------------------------

def _run_empcv_bulk(
    data_peptides: np.ndarray, data_concs: np.ndarray, data_areas: np.ndarray,
    selected_peps: np.ndarray, sliding_window: int = 3,
) -> Tuple[List[Tuple], float]:
    """Run EmpiricalCV.compute_loqs_bulk on selected peptides. Returns (results, elapsed)."""
    # Subset data to selected peptides
    mask = np.isin(data_peptides, selected_peps)
    peps_sub = data_peptides[mask]
    concs_sub = data_concs[mask]
    areas_sub = data_areas[mask]

    t0 = time.perf_counter()
    loq_dict = EmpiricalCV.compute_loqs_bulk(
        peps_sub, concs_sub, areas_sub,
        sliding_window=sliding_window,
    )
    elapsed = time.perf_counter() - t0

    results = [(pep, np.inf, loq, None) for pep, loq in loq_dict.items()]
    return results, elapsed


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
    """Return RSS delta (MB) while loqculate processes mem_args peptides serially.

    Uses /proc/self/status VmRSS to capture C-extension (NumPy/scipy) allocations
    that tracemalloc misses.  Returns max(0, rss_after - rss_before).
    """
    gc.collect()
    rss_before = _rss_mb()
    for pep, x_list, y_list, bootreps in mem_args:
        x = np.asarray(x_list, dtype=float)
        y = np.asarray(y_list, dtype=float)
        try:
            m = PiecewiseWLS(init_method='legacy', n_boot_reps=bootreps, seed=42)
            m.fit(x, y)
        except Exception:
            pass
    rss_after = _rss_mb()
    return max(0.0, rss_after - rss_before)


def _measure_original_mem_mb(orig_mod, mem_args: List[Tuple]) -> float:
    """Return RSS delta (MB) while the original processes mem_args peptides serially."""
    import pandas as pd
    import tempfile
    gc.collect()
    rss_before = _rss_mb()
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
    rss_after = _rss_mb()
    return max(0.0, rss_after - rss_before)


def _measure_empcv_mem_mb(
    data_peptides: np.ndarray, data_concs: np.ndarray, data_areas: np.ndarray,
    selected_peps: np.ndarray, n_mem: int,
) -> float:
    """Return RSS delta (MB) while EmpiricalCV.compute_loqs_bulk processes n_mem peptides."""
    peps_sub = selected_peps[:n_mem]
    mask = np.isin(data_peptides, peps_sub)
    gc.collect()
    rss_before = _rss_mb()
    EmpiricalCV.compute_loqs_bulk(
        data_peptides[mask], data_concs[mask], data_areas[mask],
        sliding_window=3,
    )
    rss_after = _rss_mb()
    return max(0.0, rss_after - rss_before)


def _measure_origcv_mem_mb(origcv_mod, data_path, map_path) -> float:
    """Return RSS delta (MB) while loq_by_cv.py's compute kernel runs."""
    import tempfile
    # Pre-load the DataFrame outside the measurement window
    with suppress_stdio():
        df = origcv_mod.read_input(str(data_path), str(map_path))
    gc.collect()
    rss_before = _rss_mb()
    with tempfile.TemporaryDirectory() as tmpdir:
        origcv_mod.output_dir = tmpdir
        with suppress_stdio():
            origcv_mod.calculate_LOQ_byCV(df)
    rss_after = _rss_mb()
    return max(0.0, rss_after - rss_before)


def _run_origcv_bulk(origcv_mod, data_path, map_path) -> Tuple[List[Tuple], float]:
    """Run loq_by_cv.py on the entire dataset; return (results_list, elapsed).

    loq_by_cv.py processes ALL peptides at once via pandas groupby.
    Returns list of (pep, inf, loq, error) tuples matching the per-peptide format.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        origcv_mod.output_dir = tmpdir
        with suppress_stdio():
            df = origcv_mod.read_input(str(data_path), str(map_path))
            t0 = time.perf_counter()
            result_df = origcv_mod.calculate_LOQ_byCV(df)
            elapsed = time.perf_counter() - t0
    results = []
    for pep, grp in result_df.groupby('peptide'):
        val = grp['loq'].iloc[0]
        loq = np.inf if (isinstance(val, float) and np.isnan(val)) else float(val)
        results.append((str(pep), np.inf, loq, None))
    return results, elapsed


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
    loq_yield: dict = None,    # {conc_level: {'n_quant': int, 'frac': float}}
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
    print(f'  {"Peptides (memory)":<20}{SEP}{n_mem} (serial, VmRSS delta)')
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
    _row('RSS delta (VmRSS)',
         f'{orig_mem_mb/n_mem:.4f} MB/pep  (total Δ {orig_mem_mb:.1f} MB)',
         f'{new_mem_mb/n_mem:.4f} MB/pep  (total Δ {new_mem_mb:.1f} MB)',
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

    # LOQ yield by concentration band (loqculate only — empirical sensitivity curve)
    if loq_yield:
        print()
        print(f'  LOQ yield by concentration (loqculate) — fraction of {n} peptides quantifiable')
        print(f'  at-or-below each calibration level:')
        print(f'  {"conc":>12}  {"n_quant":>8}  {"fraction":>9}')
        print(f'  {"-"*34}')
        for c, info in sorted(loq_yield.items()):
            bar = '#' * int(info['frac'] * 20)
            print(f'  {c:>12.4g}  {info["n_quant"]:>8d}  {info["frac"]:>8.1%}  {bar}')

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
    # Load original modules (main process, for memory phase + CV bulk)
    # ------------------------------------------------------------------ #
    orig_path = str(OLD_DIR / 'calculate-loq.py')
    print(f'\nLoading original modules ...')
    orig_mod = load_original_calc()
    origcv_mod = load_original_cv()

    # ------------------------------------------------------------------ #
    # Memory phase (serial, in main process, to keep RSS measurement clean)
    # All 4 distinct code paths measured.
    # ------------------------------------------------------------------ #
    print(f'\nMemory phase: {n_mem} peptides, {args.bootreps} bootreps (serial, VmRSS) ...')
    print('  original_wls  ...', end='', flush=True)
    orig_mem_mb = _measure_original_mem_mb(orig_mod, mem_args)
    print(f' \u0394{orig_mem_mb:.2f} MB  ({orig_mem_mb/n_mem:.4f} MB/pep)')

    print('  loqculate_wls ...', end='', flush=True)
    new_mem_mb = _measure_loqculate_mem_mb(mem_args)
    print(f' \u0394{new_mem_mb:.2f} MB  ({new_mem_mb/n_mem:.4f} MB/pep)')

    print('  original_cv   ...', end='', flush=True)
    origcv_mem_mb = _measure_origcv_mem_mb(origcv_mod, data_path, map_path)
    print(f' \u0394{origcv_mem_mb:.2f} MB')

    print('  EmpiricalCV   ...', end='', flush=True)
    empcv_mem_mb = _measure_empcv_mem_mb(data.peptide, data.concentration, data.area, selected, n_mem)
    print(f' \u0394{empcv_mem_mb:.2f} MB  ({empcv_mem_mb/n_mem:.4f} MB/pep)')

    # ------------------------------------------------------------------ #
    # Throughput phase — 5 implementations
    # ------------------------------------------------------------------ #
    n_reps = args.n_reps
    print(f'\nThroughput phase: {n} peptides, {args.bootreps} bootreps, '
          f'{n_reps} timing rep(s) ({args.workers} workers, chunk={args.chunk_size}) ...')

    # --- Bootstrap / WLS methods (per-peptide, parallel) ----------------
    orig_t_runs: List[float] = []
    orig_res = []
    print('  Running original_wls (calculate-loq.py) ...')
    for i in range(n_reps):
        orig_res, t = _run_original_parallel(mp_args, args.workers, args.chunk_size, orig_path)
        orig_t_runs.append(t)
        print(f'    rep {i+1:2d}/{n_reps}: {t:.2f}s  ({n/t:.1f} pep/s)', flush=True)

    new_t_runs: List[float] = []
    new_res = []
    print('  Running loqculate_wls (PiecewiseWLS, w=3) ...')
    for i in range(n_reps):
        new_res, t = _run_loqculate_parallel(mp_args, args.workers, args.chunk_size)
        new_t_runs.append(t)
        print(f'    rep {i+1:2d}/{n_reps}: {t:.2f}s  ({n/t:.1f} pep/s)', flush=True)

    # --- CV methods (bulk, no bootstrap) --------------------------------
    empcv_w1_t_runs: List[float] = []
    empcv_w1_res = []
    print('  Running EmpiricalCV(w=1) (bulk) ...')
    for i in range(n_reps):
        empcv_w1_res, t = _run_empcv_bulk(
            data.peptide, data.concentration, data.area,
            selected, sliding_window=1,
        )
        empcv_w1_t_runs.append(t)
        print(f'    rep {i+1:2d}/{n_reps}: {t:.4f}s  ({len(empcv_w1_res)/t:.0f} pep/s)', flush=True)

    empcv_w3_t_runs: List[float] = []
    empcv_w3_res = []
    print('  Running EmpiricalCV(w=3) (bulk) ...')
    for i in range(n_reps):
        empcv_w3_res, t = _run_empcv_bulk(
            data.peptide, data.concentration, data.area,
            selected, sliding_window=3,
        )
        empcv_w3_t_runs.append(t)
        print(f'    rep {i+1:2d}/{n_reps}: {t:.4f}s  ({len(empcv_w3_res)/t:.0f} pep/s)', flush=True)

    origcv_t_runs: List[float] = []
    origcv_res = []
    print('  Running original_cv (loq_by_cv.py, bulk) ...')
    for i in range(n_reps):
        origcv_res, t = _run_origcv_bulk(origcv_mod, data_path, map_path)
        origcv_t_runs.append(t)
        print(f'    rep {i+1:2d}/{n_reps}: {t:.4f}s  ({len(origcv_res)/t:.0f} pep/s)', flush=True)

    # ------------------------------------------------------------------ #
    # LOQ yield by concentration band — all implementations
    # ------------------------------------------------------------------ #
    conc_levels = sorted(c for c in np.unique(data.concentration) if c > 0)

    def _compute_yield(res_list: List[Tuple]) -> dict:
        n_res = len(res_list)
        yld: dict = {}
        for c in conc_levels:
            n_quant = sum(1 for r in res_list if np.isfinite(r[2]) and 0 < r[2] <= c)
            yld[float(c)] = {'n_quant': n_quant, 'frac': n_quant / n_res if n_res > 0 else 0.0}
        return yld

    yields = {
        'original_wls':     _compute_yield(orig_res),
        'loqculate_wls':    _compute_yield(new_res),
        'original_cv':      _compute_yield(origcv_res),
        'EmpiricalCV(w=1)': _compute_yield(empcv_w1_res),
        'EmpiricalCV(w=3)': _compute_yield(empcv_w3_res),
    }

    # ------------------------------------------------------------------ #
    # LOQ agreement — EmpiricalCV(w=1) vs original_cv
    # ------------------------------------------------------------------ #
    empcv_w1_dict = {r[0]: r[2] for r in empcv_w1_res}
    origcv_dict   = {r[0]: r[2] for r in origcv_res}
    common_peps = sorted(set(empcv_w1_dict.keys()) & set(origcv_dict.keys()))
    agree_count = 0
    disagree_peps: List[Tuple] = []
    for pep in common_peps:
        v1, v2 = empcv_w1_dict[pep], origcv_dict[pep]
        if (v1 == v2) or (np.isinf(v1) and np.isinf(v2)):
            agree_count += 1
        else:
            disagree_peps.append((pep, v1, v2))

    # ------------------------------------------------------------------ #
    # Report — unified summary
    # ------------------------------------------------------------------ #
    def _n_fin(res): return sum(1 for r in res if np.isfinite(r[2]))
    def _fmt_t(runs):
        m = float(np.mean(runs))
        ci = _ci95(runs)
        if m < 0.01:
            return f'{m*1000:.2f} \u00b1 {ci*1000:.2f} ms'
        return f'{m:.2f} \u00b1 {ci:.2f} s'
    def _fmt_pps(runs, n_pep):
        m = float(np.mean(runs))
        return f'{n_pep/m:,.0f}' if m > 0 else '\u2014'
    def _fmt_mem(mb, n_pep):
        if mb is None: return '\u2014'
        if mb == 0: return '~0'
        return f'{mb/n_pep:.4f}'

    hdr = '=' * 105
    print(f'\n{hdr}')
    print(f'  SCALE BENCHMARK \u2014 {data_path.name} \u2014 {n}/{n_total} peptides, '
          f'{args.bootreps} bootreps, {n_reps} reps, {args.workers} workers')
    print(hdr)

    # --- WLS section ---
    print()
    print(f'  \u2500\u2500\u2500 Bootstrap / WLS methods (per-peptide, parallel, {args.workers} workers) \u2500\u2500\u2500')
    print(f'  {"Implementation":<28} {"Wall time":>20}  {"pep/s":>10}  {"MB/pep":>10}  {"LOQs":>10}')
    print(f'  {"\u2500"*82}')
    for label, runs, res, mem_mb in [
        ('original_wls',   orig_t_runs, orig_res, orig_mem_mb),
        ('loqculate_wls',  new_t_runs,  new_res,  new_mem_mb),
    ]:
        print(f'  {label:<28} {_fmt_t(runs):>20}  {_fmt_pps(runs, n):>10}  '
              f'{_fmt_mem(mem_mb, n_mem):>10}  {_n_fin(res):>4}/{len(res)}')
    t_o = float(np.mean(orig_t_runs))
    t_n = float(np.mean(new_t_runs))
    if t_n > 0:
        print(f'  \u2192 loqculate_wls speedup: {t_o/t_n:.2f}x')

    # --- CV section ---
    print()
    print(f'  \u2500\u2500\u2500 CV methods (bulk vectorized, no bootstrap) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500')
    print(f'  {"Implementation":<28} {"Wall time":>20}  {"pep/s":>10}  {"MB/pep":>10}  {"LOQs":>10}')
    print(f'  {"\u2500"*82}')
    for label, runs, res, mem_mb in [
        ('original_cv (loq_by_cv)',  origcv_t_runs,   origcv_res,   origcv_mem_mb),
        ('EmpiricalCV(w=1)',         empcv_w1_t_runs, empcv_w1_res, empcv_mem_mb),
        ('EmpiricalCV(w=3)',         empcv_w3_t_runs, empcv_w3_res, empcv_mem_mb),
    ]:
        n_pep = len(res)
        print(f'  {label:<28} {_fmt_t(runs):>20}  {_fmt_pps(runs, n_pep):>10}  '
              f'{_fmt_mem(mem_mb, n_mem):>10}  {_n_fin(res):>4}/{n_pep}')
    t_ocv = float(np.mean(origcv_t_runs))
    t_ew1 = float(np.mean(empcv_w1_t_runs))
    if t_ew1 > 0:
        print(f'  \u2192 EmpiricalCV speedup over original_cv: {t_ocv/t_ew1:.1f}x')

    # --- LOQ agreement ---
    print()
    print(f'  \u2500\u2500\u2500 LOQ agreement: EmpiricalCV(w=1) vs original_cv \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500')
    print(f'  {len(common_peps)} common peptides: {agree_count} agree, '
          f'{len(disagree_peps)} disagree')
    if disagree_peps:
        for pep, v1, v2 in disagree_peps[:5]:
            print(f'    {pep}: EmpCV(w=1)={v1:.4g}  origcv={v2:.4g}')
        if len(disagree_peps) > 5:
            print(f'    \u2026 and {len(disagree_peps) - 5} more')
    else:
        print(f'  \u2713 Perfect match \u2014 EmpiricalCV(w=1) reproduces loq_by_cv.py exactly.')

    # --- Window effect ---
    n_w1 = _n_fin(empcv_w1_res)
    n_w3 = _n_fin(empcv_w3_res)
    diff = n_w1 - n_w3
    print()
    print(f'  \u2500\u2500\u2500 Window effect: EmpiricalCV(w=1) \u2192 EmpiricalCV(w=3) \u2500\u2500\u2500\u2500\u2500')
    if n_w1 > 0:
        print(f'  LOQs: {n_w1} \u2192 {n_w3}  '
              f'(window=3 filters {diff} peptides, {100*diff/n_w1:.0f}% reduction)')
    else:
        print(f'  LOQs: {n_w1} \u2192 {n_w3}')

    # --- LOQ yield ---
    print()
    print(f'  \u2500\u2500\u2500 LOQ yield by concentration \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500')
    impl_names = list(yields.keys())
    hdr_row = f'  {"conc":>10}'
    for nm in impl_names:
        hdr_row += f'  {nm:>17}'
    print(hdr_row)
    print(f'  {"\u2500" * (12 + 19 * len(impl_names))}')
    for c in conc_levels:
        row = f'  {c:>10.4g}'
        for nm in impl_names:
            info = yields[nm][float(c)]
            row += f'  {info["frac"]:>16.1%}'
        print(row)

    # --- WLS LOD/LOQ detail ---
    orig_lods_f, orig_loqs_f, orig_errors, _, orig_ploqs = _stats(orig_res, max_x)
    new_lods_f,  new_loqs_f,  new_errors,  _, new_ploqs  = _stats(new_res,  max_x)
    orig_plods = [v for v in orig_lods_f if 0 < v <= max_x]
    new_plods  = [v for v in new_lods_f  if 0 < v <= max_x]

    print()
    print(f'  \u2500\u2500\u2500 WLS LOD/LOQ detail \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500')
    print(f'  {"":>22}  {"original_wls":>16}  {"loqculate_wls":>16}')
    print(f'  {"Finite LODs":<22}  {len(orig_lods_f):>8}/{n:>5}  {len(new_lods_f):>8}/{n:>5}')
    if np.isfinite(max_x):
        bug_count = len(orig_lods_f) - len(orig_plods)
        note = f'  ({bug_count} orig LODs > max_x)' if bug_count > 0 else ''
        print(f'  {"  \u21b3 in range":<22}  {len(orig_plods):>8}/{n:>5}  {len(new_plods):>8}/{n:>5}{note}')
    print(f'  {"Finite LOQs":<22}  {len(orig_loqs_f):>8}/{n:>5}  {len(new_loqs_f):>8}/{n:>5}')
    if np.isfinite(max_x):
        print(f'  {"  \u21b3 in range":<22}  {len(orig_ploqs):>8}/{n:>5}  {len(new_ploqs):>8}/{n:>5}')

    for label, errors in [('original_wls', orig_errors), ('loqculate_wls', new_errors)]:
        if errors:
            print(f'\n  {label} errors: {len(errors)} peptides')
            for pep, _, _, err in errors[:3]:
                print(f'    {pep}: {err}')

    print(f'\n{hdr}\n')

    # ------------------------------------------------------------------ #
    # Save JSON
    # ------------------------------------------------------------------ #
    if args.save:
        import datetime

        def _impl_block(label, t_runs, res, mem_mb, n_peps):
            t_m = float(np.mean(t_runs))
            finite_lods = [r[1] for r in res if np.isfinite(r[1])]
            finite_loqs = [r[2] for r in res if np.isfinite(r[2])]
            plausible_lods = [v for v in finite_lods if 0 < v <= max_x]
            plausible_loqs = [v for v in finite_loqs if 0 < v <= max_x]
            return {
                't_runs_s':        t_runs,
                't_mean_s':        t_m,
                't_ci95_s':        _ci95(t_runs),
                'pep_per_s':       n_peps / t_m if t_m > 0 else None,
                'mem_mb_total':    mem_mb,
                'mem_mb_per_pep':  mem_mb / n_mem if (mem_mb is not None and n_mem > 0) else None,
                'n_finite_lod':    len(finite_lods),
                'n_finite_loq':    len(finite_loqs),
                'n_plausible_lod': len(plausible_lods),
                'n_plausible_loq': len(plausible_loqs),
                'n_errors':        sum(1 for r in res if r[3] is not None),
                'n_total':         len(res),
            }

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
            'implementations': {
                'original_wls':  _impl_block('original_wls',  orig_t_runs, orig_res, orig_mem_mb, n),
                'loqculate_wls': _impl_block('loqculate_wls', new_t_runs,  new_res,  new_mem_mb,  n),
                'original_cv':   _impl_block('original_cv',   origcv_t_runs, origcv_res, origcv_mem_mb, len(origcv_res)),
                'empcv_w1':      _impl_block('empcv_w1',      empcv_w1_t_runs, empcv_w1_res, empcv_mem_mb, len(empcv_w1_res)),
                'empcv_w3':      _impl_block('empcv_w3',      empcv_w3_t_runs, empcv_w3_res, empcv_mem_mb, len(empcv_w3_res)),
            },
            'speedup_wls': t_o / t_n if t_n > 0 else None,
            'speedup_cv':  t_ocv / t_ew1 if t_ew1 > 0 else None,
            'loq_agreement_empcv_w1_vs_origcv': {
                'n_common':   len(common_peps),
                'n_agree':    agree_count,
                'n_disagree': len(disagree_peps),
                'disagree_peps': [
                    {'peptide': p, 'empcv_w1_loq': v1, 'origcv_loq': v2}
                    for p, v1, v2 in disagree_peps
                ],
            },
            'loq_yield': {
                impl: {
                    str(c): info
                    for c, info in sorted(yld.items())
                }
                for impl, yld in yields.items()
            },
            'memory_method': 'VmRSS_delta',
        }
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(_json_safe(save_results), f, indent=2)
        print(f'Results saved \u2192 {out}')


if __name__ == '__main__':
    main()
