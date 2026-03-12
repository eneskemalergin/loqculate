"""bench_n_concentrations.py — Effect of # concentration levels on LOQ detection.

Answers
-------
  How does the number of matrix-matched fractions (concentration levels)
  affect sensitivity / specificity / FDR of LOQ determination for both
  EmpiricalCV and PiecewiseWLS methods?

  At what point does reducing the number of fractions cause LOQ detection
  to degrade, and which method is more robust to sparse grids?

What this script tests (one concern per experiment)
----------------------------------------------------
  Exp 1 — FDR vs n_conc (null case, both methods)
      No true LOQ (flat signal).  Any finite LOQ call = false positive.
      Sweeps the number of concentration levels to show how grid density
      affects false-discovery rate.

  Exp 2 — TPR vs n_conc (signal case, both methods)
      True LOQ exists.  Detection rate as a function of grid density.
      Key expected result: EmpiricalCV degrades faster than PiecewiseWLS
      because EmpiricalCV LOQ is restricted to measured concentrations
      and the sliding window spans a larger fraction of the grid.
      PiecewiseWLS evaluates bootstrap CV on a continuous grid.

  Exp 3 — LOQ accuracy vs n_conc
      For detected LOQs, how close are they to the analytic true LOQ?
      Shows grid-resolution bias for EmpiricalCV vs smooth-grid advantage
      of PiecewiseWLS.

  Exp 4 — CV precision vs n_conc
      Per-concentration CV stability (std of sample CV across repeated
      experiments).  Decoupled from grid effects — shows that per-point
      precision is an n_reps property, but the window decision depends on
      how many points are available.

Noise model
-----------
  Signal:  y_true(c) = max(noise_floor, slope × c + intercept_linear)
  Noise:   σ(c) = √(σ_bg² + (cv_mult × y_true(c))²)

  This gives a concentration-dependent CV that decreases with signal
  strength, matching the characteristic CV profile in proteomics
  calibration data.  σ_bg captures additive instrument / chemical
  background noise; cv_mult captures ion-statistics (multiplicative).

Run from the repository root::

    python benchmarks/bench_n_concentrations.py             # defaults (~5 min)
    python benchmarks/bench_n_concentrations.py --quick     # smoke test (~1 min)
    python benchmarks/bench_n_concentrations.py --save tmp/results/bench_n_concentrations.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')          # non-interactive backend before any pyplot import

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from _helpers import RULES, _json_safe, load_original_calc, load_original_cv, suppress_stdio

_REPO = Path(__file__).parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from loqculate.models import PiecewiseWLS, EmpiricalCV
from loqculate.utils.threshold import find_loq_threshold


# ---------------------------------------------------------------------------
# Demo-derived concentration grids at different densities
# ---------------------------------------------------------------------------
# The demo dataset has 14 levels (incl. blank).  Subsampled grids are created
# by keeping endpoints and logarithmically spacing interiors.

_CONC_GRIDS: Dict[int, List[float]] = {
    4:  [0.0, 0.01, 0.1, 1.0],
    5:  [0.0, 0.007, 0.05, 0.3, 1.0],
    6:  [0.0, 0.005, 0.01, 0.1, 0.5, 1.0],
    8:  [0.0, 0.003, 0.007, 0.03, 0.1, 0.3, 0.7, 1.0],
    10: [0.0, 0.003, 0.005, 0.01, 0.03, 0.07, 0.1, 0.3, 0.7, 1.0],
    14: [0.0, 0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07,
         0.1, 0.3, 0.5, 0.7, 1.0],
}

# ---------------------------------------------------------------------------
# Simulation parameters (realistic proteomics calibration curve)
# ---------------------------------------------------------------------------

# --- Signal case (true LOQ exists) ---
# Chosen so the true LOQ falls around c ≈ 0.17, placing the transition
# between c = 0.1 (above threshold) and c = 0.3 (below) on the demo grid.
_SIG_SLOPE     = 3e5
_SIG_LIN_INT   = 1e4
_SIG_NF        = 4e4    # noise-floor signal level
_SIG_SIGMA_BG  = 1.2e4  # additive background noise
_SIG_CV_MULT   = 0.06   # multiplicative CV component

# --- Null case (no true LOQ — CV stays above threshold everywhere) ---
_NULL_SLOPE    = 0.0     # flat signal
_NULL_LIN_INT  = 5e4
_NULL_NF       = 5e4
_NULL_SIGMA_BG = 1.5e4
_NULL_CV_MULT  = 0.15    # keeps CV ≈ 0.34 >> 0.20 everywhere


# ---------------------------------------------------------------------------
# Simulation primitives
# ---------------------------------------------------------------------------

def _true_signal(c: float, slope: float, nf: float, lin_int: float) -> float:
    return max(nf, slope * c + lin_int)


def _true_cv(c: float, slope: float, nf: float, lin_int: float,
             sigma_bg: float, cv_mult: float) -> float:
    y = _true_signal(c, slope, nf, lin_int)
    sigma = np.sqrt(sigma_bg ** 2 + (cv_mult * y) ** 2)
    return sigma / y


def _analytic_true_loq(concs: np.ndarray, cv_thresh: float, window: int,
                       slope: float, nf: float, lin_int: float,
                       sigma_bg: float, cv_mult: float) -> float:
    """Grid-resolved true LOQ: smallest non-blank concentration where
    *window* consecutive true CVs are below cv_thresh."""
    nonblank = concs[concs > 0]
    cvs = np.array([_true_cv(c, slope, nf, lin_int, sigma_bg, cv_mult)
                    for c in nonblank])
    val = find_loq_threshold(nonblank, cvs, cv_thresh=cv_thresh, window=window)
    return float(val)


def _continuous_true_loq(cv_thresh: float, slope: float, nf: float,
                         lin_int: float, sigma_bg: float,
                         cv_mult: float) -> float:
    """Analytic true LOQ on a continuous axis (no grid restriction)."""
    # Solve: cv_thresh = sqrt(sigma_bg² + (cv_mult * y)²) / y
    # => cv_thresh² * y² = sigma_bg² + cv_mult² * y²
    # => y² (cv_thresh² - cv_mult²) = sigma_bg²
    # => y_loq = sigma_bg / sqrt(cv_thresh² - cv_mult²)
    denom = cv_thresh ** 2 - cv_mult ** 2
    if denom <= 0:
        return np.inf  # multiplicative CV alone exceeds threshold
    y_loq = sigma_bg / np.sqrt(denom)
    # Invert the signal model: y = max(nf, slope * c + lin_int)
    if y_loq <= nf:
        return 0.0  # LOQ is in the noise-floor region (always below thresh)
    if slope <= 0:
        return np.inf
    c_loq = (y_loq - lin_int) / slope
    return max(0.0, c_loq)


def _generate_data(concs: np.ndarray, n_reps: int, rng: np.random.Generator,
                   slope: float, nf: float, lin_int: float,
                   sigma_bg: float, cv_mult: float
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate realistic calibration data with concentration-dependent CV."""
    x_all, y_all = [], []
    for c in concs:
        y_true = _true_signal(c, slope, nf, lin_int)
        sigma = np.sqrt(sigma_bg ** 2 + (cv_mult * y_true) ** 2)
        for _ in range(n_reps):
            y = max(0.0, rng.normal(y_true, sigma))
            x_all.append(c)
            y_all.append(y)
    return np.array(x_all), np.array(y_all)


def _fit_empirical_all_windows(x: np.ndarray, y: np.ndarray,
                              cv_thresh: float) -> Dict[str, float]:
    """Fit EmpiricalCV once and return LOQ for every window rule."""
    m = EmpiricalCV(sliding_window=3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            m.fit(x, y)
        except Exception:
            return {name: np.inf for name in RULES}
    valid = {c: cv for c, cv in m.cv_table_.items() if c > 0}
    if not valid:
        return {name: np.inf for name in RULES}
    concs = np.array(sorted(valid.keys()))
    cvs = np.array([valid[c] for c in concs])
    return {name: float(find_loq_threshold(concs, cvs, cv_thresh, win))
            for name, win in RULES.items()}


def _fit_piecewise_all_windows(x: np.ndarray, y: np.ndarray,
                               cv_thresh: float, n_boot: int,
                               seed: int) -> Dict[str, float]:
    """Fit PiecewiseWLS once and return LOQ for every window rule."""
    m = PiecewiseWLS(init_method='legacy', n_boot_reps=n_boot, seed=seed,
                     sliding_window=3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            m.fit(x, y)
            lod = m.lod()
            if not np.isfinite(lod):
                return {name: np.inf for name in RULES}
            m._ensure_boot_summary(lod)
            if m._x_grid is None or m._boot_summary is None:
                return {name: np.inf for name in RULES}
        except Exception:
            return {name: np.inf for name in RULES}
    out: Dict[str, float] = {}
    for name, win in RULES.items():
        loq = find_loq_threshold(m._x_grid, m._boot_summary['cv'],
                                 cv_thresh=cv_thresh, window=win)
        if not np.isfinite(loq) or loq >= np.max(m.x_) or loq <= 0:
            loq = np.inf
        out[name] = float(loq)
    return out


# ---------------------------------------------------------------------------
# Original method wrappers (window=1 by design)
# ---------------------------------------------------------------------------

# Per-process caches – populated lazily so forked workers inherit parent state.
_orig_calc_mod = None
_orig_cv_mod = None


def _get_orig_calc():
    global _orig_calc_mod
    if _orig_calc_mod is None:
        _orig_calc_mod = load_original_calc()
    return _orig_calc_mod


def _get_orig_cv():
    global _orig_cv_mod
    if _orig_cv_mod is None:
        _orig_cv_mod = load_original_cv()
    return _orig_cv_mod


def _fit_original_wls(x: np.ndarray, y: np.ndarray,
                      cv_thresh: float, n_boot: int) -> float:
    """Call original WLS (process_peptide). Returns LOQ (window=1)."""
    orig = _get_orig_calc()
    subset = pd.DataFrame({
        'peptide': 'sim',
        'curvepoint': x.astype(float),
        'area': y.astype(float),
    }).sort_values('curvepoint').reset_index(drop=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        with suppress_stdio():
            try:
                row_df = orig.process_peptide(
                    n_boot, cv_thresh, tmpdir, 'sim',
                    'n', 2.0, 2, 1, subset, 'n', 'piecewise')
                return float(row_df.iloc[0]['LOQ'])
            except Exception:
                return np.inf


def _fit_original_cv(x: np.ndarray, y: np.ndarray,
                     cv_thresh: float) -> float:
    """Call original CV (calculate_LOQ_byCV). Returns LOQ (window=1, cv≤0.2 hardcoded)."""
    orig = _get_orig_cv()
    df = pd.DataFrame({
        'peptide': 'sim',
        'curvepoint': x.astype(float),
        'area': y.astype(float),
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        orig.output_dir = tmpdir
        with suppress_stdio():
            try:
                result_df = orig.calculate_LOQ_byCV(df)
                loq_vals = result_df['loq'].dropna().unique()
                if len(loq_vals) > 0 and np.isfinite(loq_vals[0]):
                    return float(loq_vals[0])
                return np.inf
            except Exception:
                return np.inf


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

def _worker_profile(args):
    """Evaluate all methods on one simulated profile."""
    (seed_i, concs_list, n_reps, slope, nf, lin_int,
     sigma_bg, cv_mult, cv_thresh, n_boot) = args
    concs = np.asarray(concs_list)
    rng = np.random.default_rng(seed_i)
    x, y = _generate_data(concs, n_reps, rng, slope, nf, lin_int,
                           sigma_bg, cv_mult)
    return {
        'emp': _fit_empirical_all_windows(x, y, cv_thresh),
        'wls': _fit_piecewise_all_windows(x, y, cv_thresh, n_boot, seed_i),
        'orig_wls': _fit_original_wls(x, y, cv_thresh, n_boot),
        'orig_cv': _fit_original_cv(x, y, cv_thresh),
    }


def _run_profiles(n_profiles, concs, n_reps, slope, nf, lin_int,
                  sigma_bg, cv_mult, cv_thresh, n_boot,
                  base_seed, n_workers):
    """Run *n_profiles* simulations, optionally in parallel."""
    concs_list = concs.tolist()          # picklable
    tasks = [
        (base_seed + i, concs_list, n_reps, slope, nf, lin_int,
         sigma_bg, cv_mult, cv_thresh, n_boot)
        for i in range(n_profiles)
    ]
    if n_workers > 1:
        chunksize = max(1, n_profiles // (n_workers * 4))
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_worker_profile, tasks,
                                    chunksize=chunksize))
    else:
        results = [_worker_profile(t) for t in tasks]
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Effect of # concentration levels on LOQ detection')
    p.add_argument('--n_profiles', type=int, default=300,
                   help='Profiles per condition (default: 300)')
    p.add_argument('--n_reps', type=int, default=3,
                   help='Replicates per concentration (default: 3)')
    p.add_argument('--cv_thresh', type=float, default=0.20)
    p.add_argument('--n_boot', type=int, default=50,
                   help='Bootstrap reps for PiecewiseWLS (default: 50)')
    p.add_argument('--n_conc_list', type=str, default='4,5,6,8,10,14',
                   help='Comma-separated grid sizes to sweep')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--save', type=str, default=None, metavar='PATH')
    _default_workers = min(24, max(1, os.cpu_count() or 1))
    p.add_argument('--n_workers', type=int, default=_default_workers,
                   help=f'Parallel workers (default: {_default_workers})')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Exp 1 — FDR vs n_conc (null, both methods)
# ---------------------------------------------------------------------------

def experiment_fdr(
    n_conc_list: List[int],
    n_profiles: int,
    n_reps: int,
    cv_thresh: float,
    n_boot: int,
    seed: int,
    n_workers: int = 1,
) -> Dict:
    """FDR under the null (no true LOQ) as a function of concentration levels.

    Returns {n_conc: {method: {rule: fdr} or fdr_float}}.
    """
    results: Dict = {}

    for n_conc in n_conc_list:
        concs = np.array(_CONC_GRIDS[n_conc])
        print(f'    n_conc={n_conc}', end='', flush=True)

        profiles = _run_profiles(
            n_profiles, concs, n_reps,
            _NULL_SLOPE, _NULL_NF, _NULL_LIN_INT,
            _NULL_SIGMA_BG, _NULL_CV_MULT,
            cv_thresh, n_boot,
            base_seed=seed + n_conc * 10000,
            n_workers=n_workers)

        emp_hits: Dict[str, int] = {name: 0 for name in RULES}
        wls_hits: Dict[str, int] = {name: 0 for name in RULES}
        orig_wls_hits = 0
        orig_cv_hits = 0

        for r in profiles:
            for name in RULES:
                if np.isfinite(r['emp'][name]):
                    emp_hits[name] += 1
                if np.isfinite(r['wls'][name]):
                    wls_hits[name] += 1
            if np.isfinite(r['orig_wls']):
                orig_wls_hits += 1
            if np.isfinite(r['orig_cv']):
                orig_cv_hits += 1

        results[n_conc] = {
            'empirical_cv': {name: emp_hits[name] / n_profiles for name in RULES},
            'piecewise_wls': {name: wls_hits[name] / n_profiles for name in RULES},
            'original_wls': orig_wls_hits / n_profiles,
            'original_cv': orig_cv_hits / n_profiles,
        }
        print(' ✓', flush=True)

    return results


# ---------------------------------------------------------------------------
# Exp 2 — TPR vs n_conc (signal, both methods)
# ---------------------------------------------------------------------------

def experiment_tpr(
    n_conc_list: List[int],
    n_profiles: int,
    n_reps: int,
    cv_thresh: float,
    n_boot: int,
    seed: int,
    n_workers: int = 1,
) -> Dict:
    """TPR when a true LOQ exists, as a function of concentration levels.

    Returns ({n_conc: {method: ...}}, true_loqs_dict).
    """
    results: Dict = {}
    true_loqs: Dict[int, Dict[str, float]] = {}

    for n_conc in n_conc_list:
        concs = np.array(_CONC_GRIDS[n_conc])
        print(f'    n_conc={n_conc}', end='', flush=True)

        # True LOQ for this grid (per window rule)
        grid_loqs = {}
        for name, win in RULES.items():
            grid_loqs[name] = _analytic_true_loq(
                concs, cv_thresh, win,
                _SIG_SLOPE, _SIG_NF, _SIG_LIN_INT, _SIG_SIGMA_BG, _SIG_CV_MULT)
        true_loqs[n_conc] = grid_loqs

        profiles = _run_profiles(
            n_profiles, concs, n_reps,
            _SIG_SLOPE, _SIG_NF, _SIG_LIN_INT,
            _SIG_SIGMA_BG, _SIG_CV_MULT,
            cv_thresh, n_boot,
            base_seed=seed + 1000 + n_conc * 10000,
            n_workers=n_workers)

        emp_hits: Dict[str, int] = {name: 0 for name in RULES}
        wls_hits: Dict[str, int] = {name: 0 for name in RULES}
        orig_wls_hits = 0
        orig_cv_hits = 0

        for r in profiles:
            for name in RULES:
                if np.isfinite(r['emp'][name]):
                    emp_hits[name] += 1
                if np.isfinite(r['wls'][name]):
                    wls_hits[name] += 1
            if np.isfinite(r['orig_wls']):
                orig_wls_hits += 1
            if np.isfinite(r['orig_cv']):
                orig_cv_hits += 1

        results[n_conc] = {
            'empirical_cv': {name: emp_hits[name] / n_profiles for name in RULES},
            'piecewise_wls': {name: wls_hits[name] / n_profiles for name in RULES},
            'original_wls': orig_wls_hits / n_profiles,
            'original_cv': orig_cv_hits / n_profiles,
        }
        print(' ✓', flush=True)

    return results, true_loqs


# ---------------------------------------------------------------------------
# Exp 3 — LOQ accuracy vs n_conc
# ---------------------------------------------------------------------------

def experiment_accuracy(
    n_conc_list: List[int],
    n_profiles: int,
    n_reps: int,
    cv_thresh: float,
    n_boot: int,
    seed: int,
    n_workers: int = 1,
) -> Dict:
    """LOQ accuracy (signed relative error vs analytic true LOQ).

    Returns {n_conc: {method: {mean_signed_error, mean_abs_error, n_finite}}}.
    """
    continuous_loq = _continuous_true_loq(
        cv_thresh, _SIG_SLOPE, _SIG_NF, _SIG_LIN_INT,
        _SIG_SIGMA_BG, _SIG_CV_MULT)
    results: Dict = {}

    for n_conc in n_conc_list:
        concs = np.array(_CONC_GRIDS[n_conc])
        print(f'    n_conc={n_conc}', end='', flush=True)

        profiles = _run_profiles(
            n_profiles, concs, n_reps,
            _SIG_SLOPE, _SIG_NF, _SIG_LIN_INT,
            _SIG_SIGMA_BG, _SIG_CV_MULT,
            cv_thresh, n_boot,
            base_seed=seed + 2000 + n_conc * 10000,
            n_workers=n_workers)

        emp_loqs: List[float] = []
        wls_loqs: List[float] = []
        orig_wls_loqs: List[float] = []
        orig_cv_loqs: List[float] = []

        for r in profiles:
            loq_emp = r['emp'].get('window=3 (default)', np.inf)
            if np.isfinite(loq_emp):
                emp_loqs.append(loq_emp)
            loq_wls = r['wls'].get('window=3 (default)', np.inf)
            if np.isfinite(loq_wls):
                wls_loqs.append(loq_wls)
            if np.isfinite(r['orig_wls']):
                orig_wls_loqs.append(r['orig_wls'])
            if np.isfinite(r['orig_cv']):
                orig_cv_loqs.append(r['orig_cv'])

        def _accuracy_stats(estimates: List[float], ref: float) -> Dict:
            if not estimates or ref <= 0:
                return {'n_finite': 0, 'mean_signed_error': None,
                        'mean_abs_error': None, 'median_estimate': None}
            arr = np.array(estimates)
            rel = (arr - ref) / ref
            return {
                'n_finite': len(arr),
                'mean_signed_error': float(np.mean(rel)),
                'mean_abs_error': float(np.mean(np.abs(rel))),
                'median_estimate': float(np.median(arr)),
            }

        results[n_conc] = {
            'continuous_true_loq': continuous_loq,
            'empirical_cv': _accuracy_stats(emp_loqs, continuous_loq),
            'piecewise_wls': _accuracy_stats(wls_loqs, continuous_loq),
            'original_wls': _accuracy_stats(orig_wls_loqs, continuous_loq),
            'original_cv': _accuracy_stats(orig_cv_loqs, continuous_loq),
        }
        print(' ✓', flush=True)

    return results


# ---------------------------------------------------------------------------
# Exp 4 — CV precision vs n_conc
# ---------------------------------------------------------------------------

def experiment_cv_precision(
    n_conc_list: List[int],
    n_profiles: int,
    n_reps: int,
    cv_thresh: float,
    seed: int,
) -> Dict:
    """Per-concentration CV stability across repeated experiments.

    For each (n_conc, concentration), compute the sample CV n_profiles times
    and report the std of that sample CV across runs.  This is the
    per-point "measurement precision" that feeds the window decision.
    """
    rng = np.random.default_rng(seed + 3000)
    results: Dict = {}

    for n_conc in n_conc_list:
        concs = np.array(_CONC_GRIDS[n_conc])
        nonblank = concs[concs > 0]
        print(f'    n_conc={n_conc}', end='', flush=True)

        # cv_samples[conc_idx][trial_idx] = sample CV
        cv_samples = {float(c): [] for c in nonblank}

        for _ in range(n_profiles):
            x, y = _generate_data(concs, n_reps, rng,
                                  _SIG_SLOPE, _SIG_NF, _SIG_LIN_INT,
                                  _SIG_SIGMA_BG, _SIG_CV_MULT)
            for c in nonblank:
                mask = x == c
                vals = y[mask]
                if len(vals) >= 2:
                    cv = float(np.std(vals, ddof=1) / np.mean(vals))
                else:
                    cv = np.nan
                cv_samples[float(c)].append(cv)

        per_conc: Dict = {}
        for c in nonblank:
            arr = np.array(cv_samples[float(c)])
            valid = arr[np.isfinite(arr)]
            true_cv = _true_cv(c, _SIG_SLOPE, _SIG_NF, _SIG_LIN_INT,
                               _SIG_SIGMA_BG, _SIG_CV_MULT)
            per_conc[float(c)] = {
                'true_cv': true_cv,
                'mean_sample_cv': float(np.mean(valid)) if len(valid) else None,
                'std_sample_cv': float(np.std(valid, ddof=1)) if len(valid) > 1 else None,
                'n_valid': int(len(valid)),
            }

        results[n_conc] = per_conc
        print(' ✓', flush=True)

    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _print_exp1(results: Dict, cv_thresh: float) -> None:
    print(f'\n{"=" * 80}')
    print('  EXP 1 — FDR vs n_conc  (null: flat signal, no true LOQ)')
    print(f'  cv_thresh={cv_thresh:.2f}.  Any finite LOQ call = false positive.')
    print('=' * 80)
    rule_names = list(RULES)

    for method_key, method_label in [('empirical_cv', 'EmpiricalCV'),
                                      ('piecewise_wls', 'PiecewiseWLS')]:
        print(f'\n  --- {method_label} ---')
        hdr = f'  {"n_conc":>6}'
        for name in rule_names:
            hdr += f'  {name:>20}'
        print(hdr)
        print('  ' + '-' * (len(hdr) - 2))
        for n_conc in sorted(results):
            row = f'  {n_conc:>6}'
            for name in rule_names:
                row += f'  {results[n_conc][method_key][name]:>19.1%}'
            print(row)

    # Original methods (window=1 by design)
    print(f'\n  --- Original Methods (window=1) ---')
    print(f'  {"n_conc":>6}  {"OrigWLS":>10}  {"OrigCV":>10}')
    print('  ' + '-' * 30)
    for n_conc in sorted(results):
        ow = results[n_conc]['original_wls']
        oc = results[n_conc]['original_cv']
        print(f'  {n_conc:>6}  {ow:>9.1%}  {oc:>9.1%}')
    print()


def _print_exp2(results: Dict, true_loqs: Dict, cv_thresh: float) -> None:
    print(f'\n{"=" * 80}')
    print('  EXP 2 — TPR vs n_conc  (signal present, true LOQ exists)')
    print(f'  cv_thresh={cv_thresh:.2f}.  Continuous true LOQ ≈ '
          f'{_continuous_true_loq(cv_thresh, _SIG_SLOPE, _SIG_NF, _SIG_LIN_INT, _SIG_SIGMA_BG, _SIG_CV_MULT):.4f}')
    print('=' * 80)
    rule_names = list(RULES)

    print(f'\n  Grid-resolved true LOQs (window=3):')
    for n_conc in sorted(true_loqs):
        loq_w3 = true_loqs[n_conc].get('window=3 (default)', np.inf)
        loq_str = f'{loq_w3:.4g}' if np.isfinite(loq_w3) else 'inf'
        print(f'    n_conc={n_conc:>2}:  true_loq(w=3) = {loq_str}')

    for method_key, method_label in [('empirical_cv', 'EmpiricalCV'),
                                      ('piecewise_wls', 'PiecewiseWLS')]:
        print(f'\n  --- {method_label} ---')
        hdr = f'  {"n_conc":>6}'
        for name in rule_names:
            hdr += f'  {name:>20}'
        print(hdr)
        print('  ' + '-' * (len(hdr) - 2))
        for n_conc in sorted(results):
            row = f'  {n_conc:>6}'
            for name in rule_names:
                row += f'  {results[n_conc][method_key][name]:>19.1%}'
            print(row)

    # Original methods (window=1 by design)
    print(f'\n  --- Original Methods (window=1) ---')
    print(f'  {"n_conc":>6}  {"OrigWLS":>10}  {"OrigCV":>10}')
    print('  ' + '-' * 30)
    for n_conc in sorted(results):
        ow = results[n_conc]['original_wls']
        oc = results[n_conc]['original_cv']
        print(f'  {n_conc:>6}  {ow:>9.1%}  {oc:>9.1%}')
    print()


def _print_exp3(results: Dict) -> None:
    first_key = next(iter(results))
    cont_loq = results[first_key]['continuous_true_loq']
    print(f'\n{"=" * 80}')
    print(f'  EXP 3 — LOQ Accuracy vs n_conc  (window=3, signal case)')
    print(f'  Continuous true LOQ = {cont_loq:.4f}')
    print(f'  rel_error = (estimated − true) / true')
    print('=' * 80)
    print(f'  {"n_conc":>6}  {"method":<14}  {"n_finite":>9}  '
          f'{"mean_rel_err":>12}  {"mean|err|":>10}  {"median_LOQ":>11}')
    print('  ' + '-' * 72)
    for n_conc in sorted(results):
        for mk, ml in [('empirical_cv', 'EmpiricalCV'),
                        ('piecewise_wls', 'PiecewiseWLS'),
                        ('original_wls', 'OrigWLS'),
                        ('original_cv', 'OrigCV')]:
            r = results[n_conc][mk]
            n_fin = r['n_finite']
            if n_fin == 0:
                print(f'  {n_conc:>6}  {ml:<14}  {0:>9}  {"—":>12}  {"—":>10}  {"—":>11}')
            else:
                print(f'  {n_conc:>6}  {ml:<14}  {n_fin:>9}  '
                      f'{r["mean_signed_error"]:>+11.1%}  '
                      f'{r["mean_abs_error"]:>9.1%}  '
                      f'{r["median_estimate"]:>11.4g}')
    print()


def _print_exp4(results: Dict, cv_thresh: float) -> None:
    print(f'\n{"=" * 80}')
    print(f'  EXP 4 — CV Precision vs n_conc  (signal case, n_reps=fixed)')
    print(f'  std of sample CV across runs for concentrations near the LOQ transition.')
    print(f'  cv_thresh = {cv_thresh:.2f}.  Large std → noisy window decision.')
    print('=' * 80)
    # Show only concentrations near the LOQ transition (true CV in [0.05, 0.50])
    for n_conc in sorted(results):
        print(f'\n  n_conc = {n_conc}:')
        print(f'    {"conc":>8}  {"true_cv":>8}  {"mean_cv":>8}  {"std_cv":>8}  {"status":>10}')
        print('    ' + '-' * 48)
        for c_str, stats in sorted(results[n_conc].items(), key=lambda kv: float(kv[0])):
            c = float(c_str)
            tcv = stats['true_cv']
            if tcv > 0.50 or tcv < 0.03:
                continue  # skip far-from-threshold concentrations
            mcv = stats['mean_sample_cv']
            scv = stats['std_sample_cv']
            status = '< thresh' if tcv < cv_thresh else '> thresh'
            mcv_s = f'{mcv:.4f}' if mcv is not None else '—'
            scv_s = f'{scv:.4f}' if scv is not None else '—'
            print(f'    {c:>8.4g}  {tcv:>8.4f}  {mcv_s:>8}  {scv_s:>8}  {status:>10}')
    print()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(fdr: Dict, tpr: Dict, accuracy: Dict, cv_thresh: float) -> None:
    print(f'\n{"=" * 80}')
    print('  SUMMARY — Effect of # Concentration Levels')
    print('=' * 80)

    cont_loq = _continuous_true_loq(
        cv_thresh, _SIG_SLOPE, _SIG_NF, _SIG_LIN_INT,
        _SIG_SIGMA_BG, _SIG_CV_MULT)
    print(f'\n  Continuous true LOQ = {cont_loq:.4f}')

    n_conc_list = sorted(fdr.keys())
    print(f'\n  Window=3 (default) at a glance — loqculate methods:')
    print(f'  {"n_conc":>6}  {"FDR_emp":>8}  {"FDR_wls":>8}  '
          f'{"TPR_emp":>8}  {"TPR_wls":>8}  '
          f'{"|err|_emp":>10}  {"|err|_wls":>10}')
    print('  ' + '-' * 68)
    w3 = 'window=3 (default)'
    for nc in n_conc_list:
        fdr_e = fdr[nc]['empirical_cv'].get(w3, float('nan'))
        fdr_w = fdr[nc]['piecewise_wls'].get(w3, float('nan'))
        tpr_e = tpr[nc]['empirical_cv'].get(w3, float('nan'))
        tpr_w = tpr[nc]['piecewise_wls'].get(w3, float('nan'))
        ae = accuracy[nc]['empirical_cv']['mean_abs_error']
        aw = accuracy[nc]['piecewise_wls']['mean_abs_error']
        ae_s = f'{ae:.1%}' if ae is not None else '—'
        aw_s = f'{aw:.1%}' if aw is not None else '—'
        print(f'  {nc:>6}  {fdr_e:>7.1%}  {fdr_w:>7.1%}  '
              f'{tpr_e:>7.1%}  {tpr_w:>7.1%}  '
              f'{ae_s:>10}  {aw_s:>10}')

    print(f'\n  Original methods (window=1 by design):')
    print(f'  {"n_conc":>6}  {"FDR_owls":>9}  {"FDR_ocv":>9}  '
          f'{"TPR_owls":>9}  {"TPR_ocv":>9}  '
          f'{"|err|_owls":>11}  {"|err|_ocv":>11}')
    print('  ' + '-' * 72)
    for nc in n_conc_list:
        fdr_ow = fdr[nc]['original_wls']
        fdr_oc = fdr[nc]['original_cv']
        tpr_ow = tpr[nc]['original_wls']
        tpr_oc = tpr[nc]['original_cv']
        aow = accuracy[nc]['original_wls']['mean_abs_error']
        aoc = accuracy[nc]['original_cv']['mean_abs_error']
        aow_s = f'{aow:.1%}' if aow is not None else '—'
        aoc_s = f'{aoc:.1%}' if aoc is not None else '—'
        print(f'  {nc:>6}  {fdr_ow:>8.1%}  {fdr_oc:>8.1%}  '
              f'{tpr_ow:>8.1%}  {tpr_oc:>8.1%}  '
              f'{aow_s:>11}  {aoc_s:>11}')

    print(f'\n  Key findings:')
    # Check if EmpiricalCV TPR drops significantly at low n_conc
    tpr_14_emp = tpr[14]['empirical_cv'].get(w3, 0) if 14 in tpr else 0
    tpr_4_emp = tpr[min(n_conc_list)]['empirical_cv'].get(w3, 0)
    tpr_14_wls = tpr[14]['piecewise_wls'].get(w3, 0) if 14 in tpr else 0
    tpr_4_wls = tpr[min(n_conc_list)]['piecewise_wls'].get(w3, 0)
    tpr_14_ow = tpr[14]['original_wls'] if 14 in tpr else 0
    tpr_14_oc = tpr[14]['original_cv'] if 14 in tpr else 0
    print(f'    • EmpiricalCV TPR:  n=14 → {tpr_14_emp:.0%},  '
          f'n={min(n_conc_list)} → {tpr_4_emp:.0%}')
    print(f'    • PiecewiseWLS TPR: n=14 → {tpr_14_wls:.0%},  '
          f'n={min(n_conc_list)} → {tpr_4_wls:.0%}')
    print(f'    • OrigWLS TPR:     n=14 → {tpr_14_ow:.0%}')
    print(f'    • OrigCV TPR:      n=14 → {tpr_14_oc:.0%}')
    if tpr_14_emp - tpr_4_emp > tpr_14_wls - tpr_4_wls + 0.05:
        print(f'    → EmpiricalCV degrades more sharply with fewer concentrations.')
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if args.quick:
        n_profiles = 50
        n_boot = 20
    else:
        n_profiles = args.n_profiles
        n_boot = args.n_boot

    n_reps = args.n_reps
    cv_thresh = args.cv_thresh
    n_conc_list = [int(x) for x in args.n_conc_list.split(',')]
    seed = args.seed
    n_workers = args.n_workers

    # Validate n_conc values against available grids
    for nc in n_conc_list:
        if nc not in _CONC_GRIDS:
            print(f'ERROR: n_conc={nc} not in predefined grids. '
                  f'Available: {sorted(_CONC_GRIDS.keys())}')
            sys.exit(1)

    print('bench_n_concentrations.py — Effect of # concentration levels')
    print(f'  n_conc_list   : {n_conc_list}')
    print(f'  n_reps/conc   : {n_reps}')
    print(f'  n_profiles    : {n_profiles}')
    print(f'  cv_thresh     : {cv_thresh}')
    print(f'  n_boot (WLS)  : {n_boot}')
    print(f'  Rules         : {", ".join(RULES)}')
    print(f'  n_workers     : {n_workers}')
    cont_loq = _continuous_true_loq(cv_thresh, _SIG_SLOPE, _SIG_NF,
                                    _SIG_LIN_INT, _SIG_SIGMA_BG, _SIG_CV_MULT)
    print(f'  True LOQ (cont): {cont_loq:.4f}')
    print()

    print('Exp 1: FDR vs n_conc (null) ...')
    fdr_results = experiment_fdr(n_conc_list, n_profiles, n_reps,
                                 cv_thresh, n_boot, seed, n_workers)
    print('  done\n')

    print('Exp 2: TPR vs n_conc (signal) ...')
    tpr_results, true_loqs = experiment_tpr(n_conc_list, n_profiles, n_reps,
                                            cv_thresh, n_boot, seed, n_workers)
    print('  done\n')

    print('Exp 3: LOQ accuracy vs n_conc ...')
    acc_results = experiment_accuracy(n_conc_list, n_profiles, n_reps,
                                      cv_thresh, n_boot, seed, n_workers)
    print('  done\n')

    print('Exp 4: CV precision vs n_conc ...')
    prec_results = experiment_cv_precision(n_conc_list, n_profiles, n_reps,
                                           cv_thresh, seed)
    print('  done\n')

    _print_exp1(fdr_results, cv_thresh)
    _print_exp2(tpr_results, true_loqs, cv_thresh)
    _print_exp3(acc_results)
    _print_exp4(prec_results, cv_thresh)
    _print_summary(fdr_results, tpr_results, acc_results, cv_thresh)

    if args.save:
        import datetime
        payload = {
            'meta': {
                'description': 'Effect of # concentration levels on LOQ detection',
                'n_conc_list': n_conc_list,
                'n_reps': n_reps,
                'n_profiles': n_profiles,
                'cv_thresh': cv_thresh,
                'n_boot': n_boot,
                'seed': seed,
                'rules': list(RULES.keys()),
                'continuous_true_loq': cont_loq,
                'conc_grids': {str(k): v for k, v in _CONC_GRIDS.items()
                               if k in n_conc_list},
                'signal_params': {
                    'slope': _SIG_SLOPE, 'lin_intercept': _SIG_LIN_INT,
                    'noise_floor': _SIG_NF, 'sigma_bg': _SIG_SIGMA_BG,
                    'cv_mult': _SIG_CV_MULT,
                },
                'null_params': {
                    'slope': _NULL_SLOPE, 'lin_intercept': _NULL_LIN_INT,
                    'noise_floor': _NULL_NF, 'sigma_bg': _NULL_SIGMA_BG,
                    'cv_mult': _NULL_CV_MULT,
                },
                'timestamp': datetime.datetime.now().isoformat(),
            },
            'exp1_fdr': {
                'description': 'FDR vs n_conc (null case, all 4 methods)',
                'results': {str(k): v for k, v in fdr_results.items()},
            },
            'exp2_tpr': {
                'description': 'TPR vs n_conc (signal case, all 4 methods)',
                'true_loqs_per_grid': {
                    str(k): v for k, v in true_loqs.items()
                },
                'results': {str(k): v for k, v in tpr_results.items()},
            },
            'exp3_accuracy': {
                'description': 'LOQ accuracy vs n_conc (window=3, signal)',
                'results': {str(k): v for k, v in acc_results.items()},
            },
            'exp4_cv_precision': {
                'description': 'Per-concentration CV stability vs n_conc',
                'results': {
                    str(nc): {
                        str(c): s for c, s in per_conc.items()
                    } for nc, per_conc in prec_results.items()
                },
            },
        }
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(_json_safe(payload), f, indent=2)
        print(f'Results saved → {out}')


if __name__ == '__main__':
    main()
