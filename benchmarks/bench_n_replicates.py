"""bench_n_replicates.py — Effect of # replicates on LOQ detection (CV vs WLS).

Answers
-------
  How does the number of replicates per concentration affect LOQ detection
  for EmpiricalCV vs PiecewiseWLS?

  EmpiricalCV derives CV directly from replicates (chi-distributed, df = n−1),
  so it benefits strongly from more replicates.  PiecewiseWLS fits a model and
  bootstraps — it also improves with more data but is less replicate-starved
  because bootstrap resampling effectively amplifies information.

What this script tests (one concern per experiment)
----------------------------------------------------
  Exp 1 — FDR vs n_reps (null case, both methods)
      No true LOQ.  Sweeps n_reps to show how replicate count affects
      false-discovery rate for EmpiricalCV vs PiecewiseWLS.

  Exp 2 — TPR vs n_reps (signal case, both methods)
      True LOQ exists.  Detection rate as a function of replicates.
      EmpiricalCV with n_reps=2 should have poor TPR (sample CV is very
      noisy with df=1); PiecewiseWLS may still recover the LOQ from the
      model fit.

  Exp 3 — LOQ accuracy vs n_reps
      For detected LOQs: signed relative error vs the analytic true LOQ.
      With more replicates, CV estimates tighten → more precise LOQ.

  Exp 4 — CV precision vs n_reps
      Sample-to-sample variability of per-concentration CV.
      Directly shows the chi-distribution variance floor:
      sd(CV̂) ≈ cv_true / √(2(n−1))

Noise model
-----------
  Same as bench_n_concentrations.py:
  Signal:  y_true(c) = max(noise_floor, slope × c + intercept_linear)
  Noise:   σ(c) = √(σ_bg² + (cv_mult × y_true(c))²)

Run from the repository root::

    python benchmarks/bench_n_replicates.py                 # defaults (~5 min)
    python benchmarks/bench_n_replicates.py --quick         # smoke test (~1 min)
    python benchmarks/bench_n_replicates.py --save tmp/results/bench_n_replicates.json
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
matplotlib.use('Agg')

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
# Fixed concentration grid — full demo resolution (14 levels incl. blank)
# ---------------------------------------------------------------------------
_CONCS = np.array([0.0, 0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05,
                   0.07, 0.1, 0.3, 0.5, 0.7, 1.0])


# ---------------------------------------------------------------------------
# Simulation parameters (same as bench_n_concentrations for consistency)
# ---------------------------------------------------------------------------

# Signal case
_SIG_SLOPE     = 3e5
_SIG_LIN_INT   = 1e4
_SIG_NF        = 4e4
_SIG_SIGMA_BG  = 1.2e4
_SIG_CV_MULT   = 0.06

# Null case
_NULL_SLOPE    = 0.0
_NULL_LIN_INT  = 5e4
_NULL_NF       = 5e4
_NULL_SIGMA_BG = 1.5e4
_NULL_CV_MULT  = 0.15


# ---------------------------------------------------------------------------
# Simulation primitives (shared with bench_n_concentrations)
# ---------------------------------------------------------------------------

def _true_signal(c: float, slope: float, nf: float, lin_int: float) -> float:
    return max(nf, slope * c + lin_int)


def _true_cv(c: float, slope: float, nf: float, lin_int: float,
             sigma_bg: float, cv_mult: float) -> float:
    y = _true_signal(c, slope, nf, lin_int)
    sigma = np.sqrt(sigma_bg ** 2 + (cv_mult * y) ** 2)
    return sigma / y


def _continuous_true_loq(cv_thresh: float) -> float:
    """Analytic true LOQ on a continuous axis."""
    denom = cv_thresh ** 2 - _SIG_CV_MULT ** 2
    if denom <= 0:
        return np.inf
    y_loq = _SIG_SIGMA_BG / np.sqrt(denom)
    if y_loq <= _SIG_NF:
        return 0.0
    if _SIG_SLOPE <= 0:
        return np.inf
    return max(0.0, (y_loq - _SIG_LIN_INT) / _SIG_SLOPE)


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
    concs_list = concs.tolist()
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
        description='Effect of # replicates on LOQ detection (CV vs WLS)')
    p.add_argument('--n_profiles', type=int, default=300,
                   help='Profiles per condition (default: 300)')
    p.add_argument('--n_reps_list', type=str, default='2,3,5,8,10,15,20',
                   help='Replicate counts to sweep (default: 2,3,5,8,10,15,20)')
    p.add_argument('--cv_thresh', type=float, default=0.20)
    p.add_argument('--n_boot', type=int, default=50,
                   help='Bootstrap reps for PiecewiseWLS (default: 50)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--save', type=str, default=None, metavar='PATH')
    _default_workers = min(24, max(1, os.cpu_count() or 1))
    p.add_argument('--n_workers', type=int, default=_default_workers,
                   help=f'Parallel workers (default: {_default_workers})')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Exp 1 — FDR vs n_reps (null, both methods)
# ---------------------------------------------------------------------------

def experiment_fdr(
    n_reps_list: List[int],
    n_profiles: int,
    cv_thresh: float,
    n_boot: int,
    seed: int,
    n_workers: int = 1,
) -> Dict:
    """FDR under the null as a function of # replicates."""
    results: Dict = {}

    for n_reps in n_reps_list:
        print(f'    n_reps={n_reps}', end='', flush=True)

        profiles = _run_profiles(
            n_profiles, _CONCS, n_reps,
            _NULL_SLOPE, _NULL_NF, _NULL_LIN_INT,
            _NULL_SIGMA_BG, _NULL_CV_MULT,
            cv_thresh, n_boot,
            base_seed=seed + n_reps * 10000,
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

        results[n_reps] = {
            'empirical_cv': {name: emp_hits[name] / n_profiles for name in RULES},
            'piecewise_wls': {name: wls_hits[name] / n_profiles for name in RULES},
            'original_wls': orig_wls_hits / n_profiles,
            'original_cv': orig_cv_hits / n_profiles,
        }
        print(' ✓', flush=True)

    return results


# ---------------------------------------------------------------------------
# Exp 2 — TPR vs n_reps (signal, both methods)
# ---------------------------------------------------------------------------

def experiment_tpr(
    n_reps_list: List[int],
    n_profiles: int,
    cv_thresh: float,
    n_boot: int,
    seed: int,
    n_workers: int = 1,
) -> Dict:
    """TPR when a true LOQ exists, as a function of # replicates."""
    results: Dict = {}

    for n_reps in n_reps_list:
        print(f'    n_reps={n_reps}', end='', flush=True)

        profiles = _run_profiles(
            n_profiles, _CONCS, n_reps,
            _SIG_SLOPE, _SIG_NF, _SIG_LIN_INT,
            _SIG_SIGMA_BG, _SIG_CV_MULT,
            cv_thresh, n_boot,
            base_seed=seed + 1000 + n_reps * 10000,
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

        results[n_reps] = {
            'empirical_cv': {name: emp_hits[name] / n_profiles for name in RULES},
            'piecewise_wls': {name: wls_hits[name] / n_profiles for name in RULES},
            'original_wls': orig_wls_hits / n_profiles,
            'original_cv': orig_cv_hits / n_profiles,
        }
        print(' ✓', flush=True)

    return results


# ---------------------------------------------------------------------------
# Exp 3 — LOQ accuracy vs n_reps
# ---------------------------------------------------------------------------

def experiment_accuracy(
    n_reps_list: List[int],
    n_profiles: int,
    cv_thresh: float,
    n_boot: int,
    seed: int,
    n_workers: int = 1,
) -> Dict:
    """LOQ accuracy as a function of # replicates (window=3)."""
    cont_loq = _continuous_true_loq(cv_thresh)
    results: Dict = {}

    for n_reps in n_reps_list:
        print(f'    n_reps={n_reps}', end='', flush=True)

        profiles = _run_profiles(
            n_profiles, _CONCS, n_reps,
            _SIG_SLOPE, _SIG_NF, _SIG_LIN_INT,
            _SIG_SIGMA_BG, _SIG_CV_MULT,
            cv_thresh, n_boot,
            base_seed=seed + 2000 + n_reps * 10000,
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

        def _stats(estimates: List[float], ref: float) -> Dict:
            if not estimates or ref <= 0:
                return {'n_finite': 0, 'mean_signed_error': None,
                        'mean_abs_error': None, 'median_estimate': None,
                        'loq_cv': None}
            arr = np.array(estimates)
            rel = (arr - ref) / ref
            loq_cv = float(np.std(arr, ddof=1) / np.mean(arr)) if len(arr) > 1 else None
            return {
                'n_finite': len(arr),
                'mean_signed_error': float(np.mean(rel)),
                'mean_abs_error': float(np.mean(np.abs(rel))),
                'median_estimate': float(np.median(arr)),
                'loq_cv': loq_cv,
            }

        results[n_reps] = {
            'continuous_true_loq': cont_loq,
            'empirical_cv': _stats(emp_loqs, cont_loq),
            'piecewise_wls': _stats(wls_loqs, cont_loq),
            'original_wls': _stats(orig_wls_loqs, cont_loq),
            'original_cv': _stats(orig_cv_loqs, cont_loq),
        }
        print(' ✓', flush=True)

    return results


# ---------------------------------------------------------------------------
# Exp 4 — CV precision vs n_reps
# ---------------------------------------------------------------------------

def experiment_cv_precision(
    n_reps_list: List[int],
    n_profiles: int,
    seed: int,
) -> Dict:
    """Per-concentration sample CV variability as a function of # replicates.

    For a single reference concentration (near the LOQ transition),
    compute sample CV n_profiles times at each n_reps.  Report the std
    of the sample CV — should follow sd(CV̂) ≈ cv_true / √(2(n−1)).
    """
    rng = np.random.default_rng(seed + 3000)
    # Pick a concentration near the LOQ transition:
    # c=0.3 has true CV ≈ 0.134 (below threshold), c=0.1 has ≈ 0.306 (above)
    # Use c=0.3 (just below threshold) and c=0.1 (just above) for contrast.
    ref_concs = [0.1, 0.3]
    results: Dict = {}

    for n_reps in n_reps_list:
        per_conc: Dict = {}
        for c in ref_concs:
            y_true = _true_signal(c, _SIG_SLOPE, _SIG_NF, _SIG_LIN_INT)
            sigma = np.sqrt(_SIG_SIGMA_BG ** 2 + (_SIG_CV_MULT * y_true) ** 2)
            true_cv_val = sigma / y_true

            # Simulate n_profiles independent CV measurements
            cv_samples = []
            for _ in range(n_profiles):
                obs = np.maximum(1e-6, rng.normal(y_true, sigma, size=n_reps))
                cv_hat = float(np.std(obs, ddof=1) / np.mean(obs))
                cv_samples.append(cv_hat)

            arr = np.array(cv_samples)
            # Theoretical sd: cv_true * sqrt((1 + 0.5*cv_true²) / (n_reps-1))
            # (approximate for chi-scaled distribution)
            theory_sd = true_cv_val * np.sqrt(
                (1 + 0.5 * true_cv_val ** 2) / max(n_reps - 1, 1))

            per_conc[c] = {
                'true_cv': true_cv_val,
                'mean_sample_cv': float(np.mean(arr)),
                'std_sample_cv': float(np.std(arr, ddof=1)),
                'theory_std': float(theory_sd),
            }

        results[n_reps] = per_conc

    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _print_exp1(results: Dict, cv_thresh: float) -> None:
    print(f'\n{"=" * 80}')
    print('  EXP 1 — FDR vs n_reps  (null: flat signal, no true LOQ)')
    print(f'  cv_thresh={cv_thresh:.2f}.  Window rule comparison × replicate count.')
    print('=' * 80)
    rule_names = list(RULES)

    for mk, ml in [('empirical_cv', 'EmpiricalCV'), ('piecewise_wls', 'PiecewiseWLS')]:
        print(f'\n  --- {ml} ---')
        hdr = f'  {"n_reps":>6}'
        for name in rule_names:
            hdr += f'  {name:>20}'
        print(hdr)
        print('  ' + '-' * (len(hdr) - 2))
        for n_reps in sorted(results):
            row = f'  {n_reps:>6}'
            for name in rule_names:
                row += f'  {results[n_reps][mk][name]:>19.1%}'
            print(row)

    print(f'\n  --- Original Methods (window=1) ---')
    print(f'  {"n_reps":>6}  {"OrigWLS":>10}  {"OrigCV":>10}')
    print('  ' + '-' * 30)
    for n_reps in sorted(results):
        ow = results[n_reps]['original_wls']
        oc = results[n_reps]['original_cv']
        print(f'  {n_reps:>6}  {ow:>9.1%}  {oc:>9.1%}')
    print()


def _print_exp2(results: Dict, cv_thresh: float) -> None:
    cont_loq = _continuous_true_loq(cv_thresh)
    print(f'\n{"=" * 80}')
    print(f'  EXP 2 — TPR vs n_reps  (signal present, true LOQ ≈ {cont_loq:.4f})')
    print(f'  cv_thresh={cv_thresh:.2f}.  Detection rate vs replicate count.')
    print('=' * 80)
    rule_names = list(RULES)

    for mk, ml in [('empirical_cv', 'EmpiricalCV'), ('piecewise_wls', 'PiecewiseWLS')]:
        print(f'\n  --- {ml} ---')
        hdr = f'  {"n_reps":>6}'
        for name in rule_names:
            hdr += f'  {name:>20}'
        print(hdr)
        print('  ' + '-' * (len(hdr) - 2))
        for n_reps in sorted(results):
            row = f'  {n_reps:>6}'
            for name in rule_names:
                row += f'  {results[n_reps][mk][name]:>19.1%}'
            print(row)

    print(f'\n  --- Original Methods (window=1) ---')
    print(f'  {"n_reps":>6}  {"OrigWLS":>10}  {"OrigCV":>10}')
    print('  ' + '-' * 30)
    for n_reps in sorted(results):
        ow = results[n_reps]['original_wls']
        oc = results[n_reps]['original_cv']
        print(f'  {n_reps:>6}  {ow:>9.1%}  {oc:>9.1%}')
    print()


def _print_exp3(results: Dict, cv_thresh: float) -> None:
    cont_loq = _continuous_true_loq(cv_thresh)
    print(f'\n{"=" * 80}')
    print(f'  EXP 3 — LOQ Accuracy vs n_reps  (window=3, signal case)')
    print(f'  True LOQ = {cont_loq:.4f}')
    print('=' * 80)
    print(f'  {"n_reps":>6}  {"method":<14}  {"n_finite":>9}  '
          f'{"mean_rel_err":>12}  {"mean|err|":>10}  {"LOQ CV":>8}')
    print('  ' + '-' * 68)
    for n_reps in sorted(results):
        for mk, ml in [('empirical_cv', 'EmpiricalCV'),
                        ('piecewise_wls', 'PiecewiseWLS'),
                        ('original_wls', 'OrigWLS'),
                        ('original_cv', 'OrigCV')]:
            r = results[n_reps][mk]
            n_fin = r['n_finite']
            if n_fin == 0:
                print(f'  {n_reps:>6}  {ml:<14}  {0:>9}  {"—":>12}  {"—":>10}  {"—":>8}')
            else:
                lcv = f'{r["loq_cv"]:.1%}' if r.get('loq_cv') is not None else '—'
                print(f'  {n_reps:>6}  {ml:<14}  {n_fin:>9}  '
                      f'{r["mean_signed_error"]:>+11.1%}  '
                      f'{r["mean_abs_error"]:>9.1%}  '
                      f'{lcv:>8}')
    print()


def _print_exp4(results: Dict) -> None:
    print(f'\n{"=" * 80}')
    print('  EXP 4 — CV Precision vs n_reps')
    print('  sd(CV̂) at two reference concentrations near the LOQ transition.')
    print('  Theory: sd(CV̂) ≈ cv × √((1 + cv²/2) / (n−1))')
    print('=' * 80)
    for c in [0.1, 0.3]:
        print(f'\n  --- conc = {c} ---')
        print(f'  {"n_reps":>6}  {"true_cv":>8}  {"mean_cv̂":>8}  '
              f'{"sd(cv̂)":>8}  {"theory":>8}  {"ratio":>7}')
        print('  ' + '-' * 52)
        for n_reps in sorted(results):
            s = results[n_reps][c]
            ratio = s['std_sample_cv'] / s['theory_std'] if s['theory_std'] > 0 else 0
            print(f'  {n_reps:>6}  {s["true_cv"]:>8.4f}  {s["mean_sample_cv"]:>8.4f}  '
                  f'{s["std_sample_cv"]:>8.4f}  {s["theory_std"]:>8.4f}  {ratio:>7.2f}')
    print()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(fdr: Dict, tpr: Dict, accuracy: Dict, cv_thresh: float) -> None:
    print(f'\n{"=" * 80}')
    print('  SUMMARY — Effect of # Replicates on LOQ Detection')
    print('=' * 80)

    cont_loq = _continuous_true_loq(cv_thresh)
    print(f'\n  Continuous true LOQ = {cont_loq:.4f}')

    n_reps_list = sorted(fdr.keys())
    w3 = 'window=3 (default)'
    print(f'\n  Window=3 (default) at a glance — loqculate methods:')
    print(f'  {"n_reps":>6}  {"FDR_emp":>8}  {"FDR_wls":>8}  '
          f'{"TPR_emp":>8}  {"TPR_wls":>8}  '
          f'{"LOQcv_emp":>10}  {"LOQcv_wls":>10}')
    print('  ' + '-' * 70)
    for nr in n_reps_list:
        fdr_e = fdr[nr]['empirical_cv'].get(w3, float('nan'))
        fdr_w = fdr[nr]['piecewise_wls'].get(w3, float('nan'))
        tpr_e = tpr[nr]['empirical_cv'].get(w3, float('nan'))
        tpr_w = tpr[nr]['piecewise_wls'].get(w3, float('nan'))
        lcv_e = accuracy[nr]['empirical_cv'].get('loq_cv')
        lcv_w = accuracy[nr]['piecewise_wls'].get('loq_cv')
        lcv_e_s = f'{lcv_e:.1%}' if lcv_e is not None else '—'
        lcv_w_s = f'{lcv_w:.1%}' if lcv_w is not None else '—'
        print(f'  {nr:>6}  {fdr_e:>7.1%}  {fdr_w:>7.1%}  '
              f'{tpr_e:>7.1%}  {tpr_w:>7.1%}  '
              f'{lcv_e_s:>10}  {lcv_w_s:>10}')

    print(f'\n  Original methods (window=1 by design):')
    print(f'  {"n_reps":>6}  {"FDR_owls":>9}  {"FDR_ocv":>9}  '
          f'{"TPR_owls":>9}  {"TPR_ocv":>9}')
    print('  ' + '-' * 44)
    for nr in n_reps_list:
        fdr_ow = fdr[nr]['original_wls']
        fdr_oc = fdr[nr]['original_cv']
        tpr_ow = tpr[nr]['original_wls']
        tpr_oc = tpr[nr]['original_cv']
        print(f'  {nr:>6}  {fdr_ow:>8.1%}  {fdr_oc:>8.1%}  '
              f'{tpr_ow:>8.1%}  {tpr_oc:>8.1%}')

    print(f'\n  Key findings:')
    if 2 in fdr and 10 in fdr:
        print(f'    • EmpiricalCV:  FDR goes from '
              f'{fdr[2]["empirical_cv"].get(w3, 0):.0%} (n=2) to '
              f'{fdr[10]["empirical_cv"].get(w3, 0):.0%} (n=10)')
        print(f'    • PiecewiseWLS: FDR goes from '
              f'{fdr[2]["piecewise_wls"].get(w3, 0):.0%} (n=2) to '
              f'{fdr[10]["piecewise_wls"].get(w3, 0):.0%} (n=10)')
        print(f'    • OrigWLS:      FDR goes from '
              f'{fdr[2]["original_wls"]:.0%} (n=2) to '
              f'{fdr[10]["original_wls"]:.0%} (n=10)')
        print(f'    • OrigCV:       FDR goes from '
              f'{fdr[2]["original_cv"]:.0%} (n=2) to '
              f'{fdr[10]["original_cv"]:.0%} (n=10)')
    if 2 in tpr and 10 in tpr:
        print(f'    • EmpiricalCV:  TPR goes from '
              f'{tpr[2]["empirical_cv"].get(w3, 0):.0%} (n=2) to '
              f'{tpr[10]["empirical_cv"].get(w3, 0):.0%} (n=10)')
        print(f'    • PiecewiseWLS: TPR goes from '
              f'{tpr[2]["piecewise_wls"].get(w3, 0):.0%} (n=2) to '
              f'{tpr[10]["piecewise_wls"].get(w3, 0):.0%} (n=10)')
        print(f'    • OrigWLS:      TPR goes from '
              f'{tpr[2]["original_wls"]:.0%} (n=2) to '
              f'{tpr[10]["original_wls"]:.0%} (n=10)')
        print(f'    • OrigCV:       TPR goes from '
              f'{tpr[2]["original_cv"]:.0%} (n=2) to '
              f'{tpr[10]["original_cv"]:.0%} (n=10)')
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

    cv_thresh = args.cv_thresh
    n_reps_list = [int(x) for x in args.n_reps_list.split(',')]
    seed = args.seed
    n_workers = args.n_workers

    cont_loq = _continuous_true_loq(cv_thresh)
    print('bench_n_replicates.py — Effect of # replicates on LOQ detection')
    print(f'  n_reps_list   : {n_reps_list}')
    print(f'  n_conc (fixed): {len(_CONCS)} (demo grid)')
    print(f'  n_profiles    : {n_profiles}')
    print(f'  cv_thresh     : {cv_thresh}')
    print(f'  n_boot (WLS)  : {n_boot}')
    print(f'  Rules         : {", ".join(RULES)}')
    print(f'  n_workers     : {n_workers}')
    print(f'  True LOQ (cont): {cont_loq:.4f}')
    print()

    print('Exp 1: FDR vs n_reps (null) ...')
    fdr_results = experiment_fdr(n_reps_list, n_profiles, cv_thresh,
                                 n_boot, seed, n_workers)
    print('  done\n')

    print('Exp 2: TPR vs n_reps (signal) ...')
    tpr_results = experiment_tpr(n_reps_list, n_profiles, cv_thresh,
                                 n_boot, seed, n_workers)
    print('  done\n')

    print('Exp 3: LOQ accuracy vs n_reps ...')
    acc_results = experiment_accuracy(n_reps_list, n_profiles, cv_thresh,
                                      n_boot, seed, n_workers)
    print('  done\n')

    print('Exp 4: CV precision vs n_reps ...')
    prec_results = experiment_cv_precision(n_reps_list, n_profiles, seed)
    print('  done\n')

    _print_exp1(fdr_results, cv_thresh)
    _print_exp2(tpr_results, cv_thresh)
    _print_exp3(acc_results, cv_thresh)
    _print_exp4(prec_results)
    _print_summary(fdr_results, tpr_results, acc_results, cv_thresh)

    if args.save:
        import datetime
        payload = {
            'meta': {
                'description': 'Effect of # replicates on LOQ detection (CV vs WLS)',
                'n_reps_list': n_reps_list,
                'n_conc': len(_CONCS),
                'conc_grid': _CONCS.tolist(),
                'n_profiles': n_profiles,
                'cv_thresh': cv_thresh,
                'n_boot': n_boot,
                'seed': seed,
                'rules': list(RULES.keys()),
                'continuous_true_loq': cont_loq,
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
                'description': 'FDR vs n_reps (null, all 4 methods)',
                'results': {str(k): v for k, v in fdr_results.items()},
            },
            'exp2_tpr': {
                'description': 'TPR vs n_reps (signal, all 4 methods)',
                'results': {str(k): v for k, v in tpr_results.items()},
            },
            'exp3_accuracy': {
                'description': 'LOQ accuracy vs n_reps (window=3, signal)',
                'results': {str(k): v for k, v in acc_results.items()},
            },
            'exp4_cv_precision': {
                'description': 'Sample CV precision vs n_reps at ref concentrations',
                'results': {
                    str(nr): {
                        str(c): s for c, s in per_conc.items()
                    } for nr, per_conc in prec_results.items()
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
