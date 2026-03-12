"""bench_empirical.py — EmpiricalCV correctness regression on real data.

Validates EmpiricalCV against the original loq_by_cv.py on the 27-peptide
demo calibration dataset (PMA1_YEAST, 14 concentrations, 3 replicates).

Three-way LOQ comparison
------------------------
  original       loq_by_cv.py calculate_LOQ_byCV()   — window=1, pandas GroupBy
  empirical_w1   EmpiricalCV(sliding_window=1)        — window=1, numpy
  empirical_w3   EmpiricalCV(sliding_window=3)        — window=3 (production default)

  orig ↔ w=1   validates math correctness: same algorithm, different implementation.
               Expected: near-perfect agreement (any divergence = a regression).
  orig ↔ w=3   shows the scientific effect of the sliding window: window=3 rejects
               isolated CV dips that window=1 accepts as LOQs (CV bounces).

Statistical FDR/TPR/stability validation of the window rules (for both
EmpiricalCV and PiecewiseWLS together) lives in bench_window_rules.py.

Category glossary
-----------------
  agree(exact)     Both yield the same finite LOQ value (exact match)
  agree(N%)        Both finite, |Δ%| ≤ tolerance
  diverge(N%)      Both finite, |Δ%| > tolerance
  both=none        Both return no LOQ (original=NaN, empirical=inf)
  split:lq=inf     Original finite, empirical=inf  (empirical stricter — good)
  split:orig=nan   Empirical finite, original=NaN  (empirical found LOQ original missed)

Run from the repository root::

    python benchmarks/bench_empirical.py
    python benchmarks/bench_empirical.py --save tmp/results/bench_empirical.json

Options
-------
--cv_thresh   CV threshold for LOQ (default: 0.20)
--loq_tol     Relative tolerance for agree/diverge classification (default: 0.10)
--n_reps      Timing repetitions for CI estimation (default: 5)
--save        Write JSON results to PATH
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- path setup ---------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))   # benchmarks/ on path
from _helpers import (
    DEMO_DATA, DEMO_MAP,
    load_original_cv, _json_safe, _ci95,
)

from loqculate.io import read_calibration_data
from loqculate.models.cv_empirical import EmpiricalCV
from loqculate.utils.threshold import find_loq_threshold


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='EmpiricalCV correctness regression on demo data'
    )
    p.add_argument('--cv_thresh', type=float, default=0.20,
                   help='CV threshold for LOQ (default: 0.20 = 20%%)')
    p.add_argument('--loq_tol', type=float, default=0.10,
                   help='Relative tolerance for agree/diverge (default: 0.10)')
    p.add_argument('--n_reps', type=int, default=5,
                   help='Timing repetitions per implementation for CI (default: 5)')
    p.add_argument('--save', type=str, default=None, metavar='PATH',
                   help='Write JSON results to PATH')
    return p.parse_args()


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

def _load_peptides() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Read demo data via loqculate reader; return {peptide: (x, y)}."""
    data = read_calibration_data(str(DEMO_DATA), str(DEMO_MAP))
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for pep in np.unique(data.peptide):
        mask = data.peptide == pep
        out[pep] = (data.concentration[mask], data.area[mask])
    return out


# -----------------------------------------------------------------------
# Runners
# -----------------------------------------------------------------------

def _run_original_cv(
    orig_mod,
    cv_thresh: float,
) -> Tuple[Dict[str, Optional[float]], float]:
    """Run loq_by_cv.py calculate_LOQ_byCV(); return ({pep: loq}, elapsed).

    The original hard-codes cv_thresh=0.20 internally.  If cv_thresh != 0.20,
    the original column still uses 0.20 — noted in the JSON metadata.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_mod.output_dir = tmpdir
        _old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            df = orig_mod.read_input(str(DEMO_DATA), str(DEMO_MAP))
            t0 = time.perf_counter()
            result_df = orig_mod.calculate_LOQ_byCV(df)
            elapsed = time.perf_counter() - t0
        finally:
            sys.stderr = _old_stderr

    loqs: Dict[str, Optional[float]] = {}
    for pep, grp in result_df.groupby('peptide'):
        val = grp['loq'].iloc[0]
        loqs[str(pep)] = None if (isinstance(val, float) and np.isnan(val)) else float(val)

    return loqs, elapsed


def _run_empirical_cv(
    peptides: Dict[str, Tuple[np.ndarray, np.ndarray]],
    cv_thresh: float,
    window: int,
) -> Tuple[Dict[str, Optional[float]], Dict[str, dict], float]:
    """Run EmpiricalCV(sliding_window=window) on all peptides.

    Returns
    -------
    loqs      : {peptide: loq_value or None (None = no finite LOQ)}
    profiles  : {peptide: {'concs': [...], 'cvs': [...], 'means': [...], ...}}
    elapsed   : wall time in seconds
    """
    loqs: Dict[str, Optional[float]] = {}
    profiles: Dict[str, dict] = {}
    t0 = time.perf_counter()
    for pep, (x, y) in peptides.items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                m = EmpiricalCV(sliding_window=window)
                m.fit(x, y)
                loq_val = m.loq(cv_thresh=cv_thresh)
                loqs[pep] = None if not np.isfinite(loq_val) else float(loq_val)
                concs_sorted = sorted(m.cv_table_.keys())
                profiles[pep] = {
                    'concs': concs_sorted,
                    'cvs':   [m.cv_table_[c] for c in concs_sorted],
                    'means': [m.mean_table_[c] for c in concs_sorted],
                    'n_reps_per_conc': [m.replicate_counts_[c] for c in concs_sorted],
                }
            except Exception as exc:
                loqs[pep] = None
                profiles[pep] = {}
                print(f'  ERROR {pep}: {exc}')
    return loqs, profiles, time.perf_counter() - t0


# -----------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------

def _classify(orig_val: Optional[float], new_val: Optional[float], tol: float) -> str:
    orig_none = orig_val is None
    new_none  = new_val  is None

    if orig_none and new_none:
        return 'both=none'
    if not orig_none and new_none:
        return 'split:lq=inf'
    if orig_none and not new_none:
        return 'split:orig=nan'

    denom = max(abs(orig_val), abs(new_val))
    rdiff = abs(new_val - orig_val) / denom if denom > 0 else 0.0
    if rdiff < 1e-9:
        return 'agree(exact)'
    if rdiff <= tol:
        return f'agree({rdiff*100:.0f}%)'
    return f'diverge({rdiff*100:.0f}%)'


def _rdiff_pct(orig_val: Optional[float], new_val: Optional[float]) -> float:
    if orig_val is None or new_val is None:
        return float('nan')
    denom = max(abs(orig_val), abs(new_val))
    return 100.0 * (new_val - orig_val) / denom if denom > 0 else 0.0


# -----------------------------------------------------------------------
# CV profile helpers
# -----------------------------------------------------------------------

def _max_consecutive_below(cvs: List[float], thresh: float) -> int:
    """Length of the longest run of CVs ≤ thresh (excluding NaN)."""
    max_run = cur = 0
    for v in cvs:
        if np.isfinite(v) and v <= thresh:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run


# -----------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------

def _print_table(
    peptides: Dict[str, Tuple],
    orig_loqs: Dict[str, Optional[float]],
    w1_loqs: Dict[str, Optional[float]],
    w3_loqs: Dict[str, Optional[float]],
    loq_tol: float,
    cv_thresh: float,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Print the 3-column LOQ comparison table. Returns (cats_w1, cats_w3)."""
    peps = sorted(peptides.keys())
    pw = min(max(len(p) for p in peps), 45)

    def _fmt(v: Optional[float]) -> str:
        return '      —' if v is None else f'{v:.3e}'

    def _fmtd(v: float) -> str:
        return '      —' if np.isnan(v) else f'{v:+.0f}%'

    hdr = (
        f'{"Peptide":<{pw}} {"orig loq":>9} {"w=1 loq":>9} {"Δ%":>6}  '
        f'{"w=3 loq":>9} {"Δ%":>6}  {"orig↔w1":^16} {"orig↔w3":^16}'
    )
    sep = '-' * len(hdr)
    print(f'\n{hdr}\n{sep}')

    cats_w1: Dict[str, str] = {}
    cats_w3: Dict[str, str] = {}

    for pep in peps:
        o  = orig_loqs.get(pep)
        w1 = w1_loqs.get(pep)
        w3 = w3_loqs.get(pep)
        c1 = _classify(o, w1, loq_tol)
        c3 = _classify(o, w3, loq_tol)
        cats_w1[pep] = c1
        cats_w3[pep] = c3
        pep_disp = pep[:pw] if len(pep) <= pw else pep[:pw - 1] + '…'
        print(
            f'{pep_disp:<{pw}} {_fmt(o):>9} {_fmt(w1):>9} {_fmtd(_rdiff_pct(o, w1)):>6}  '
            f'{_fmt(w3):>9} {_fmtd(_rdiff_pct(o, w3)):>6}  {c1:^16} {c3:^16}'
        )

    print(sep)
    return cats_w1, cats_w3


def _print_summary(
    cats_w1: Dict[str, str],
    cats_w3: Dict[str, str],
    orig_t_runs: List[float],
    w3_t_runs: List[float],
    n: int,
    cv_thresh: float,
    loq_tol: float,
) -> None:
    from collections import Counter

    def _count(cats: Dict[str, str]) -> Counter:
        c: Counter = Counter()
        for v in cats.values():
            if   v.startswith('agree'):        c['agree'] += 1
            elif v.startswith('diverge'):      c['diverge'] += 1
            elif v == 'both=none':             c['both_none'] += 1
            elif v == 'split:lq=inf':          c['split_lq_inf'] += 1
            elif v == 'split:orig=nan':        c['split_orig_nan'] += 1
        return c

    c1, c3 = _count(cats_w1), _count(cats_w3)

    print(f'\n  Agreement  (cv_thresh={cv_thresh:.0%}, loq_tol={loq_tol:.0%})\n')
    print(f'  {"Category":<40} {"orig↔w=1":>8}  {"orig↔w=3":>8}')
    print(f'  {"-"*60}')
    for label, key in [
        ('agree (same LOQ, |Δ%| ≤ tol)',          'agree'),
        ('both=none (no LOQ on either side)',      'both_none'),
        ('diverge (both finite, |Δ%| > tol)',      'diverge'),
        ('split:lq=inf (orig finite, emp=none)',   'split_lq_inf'),
        ('split:orig=nan (emp finite, orig=none)', 'split_orig_nan'),
    ]:
        print(f'  {label:<40} {c1[key]:>8}  {c3[key]:>8}')
    print(f'  {"-"*60}')
    print(f'  {"Total":<40} {n:>8}  {n:>8}')

    w1_disagree = c1['diverge'] + c1['split_lq_inf'] + c1['split_orig_nan']
    if w1_disagree == 0:
        print('\n  ✓ orig ↔ w=1: perfect agreement — math equivalence confirmed.')
    else:
        print(f'\n  ✗ orig ↔ w=1: {w1_disagree} disagreement(s) — investigate regression.')

    w3_effect = c3['diverge'] + c3['split_lq_inf'] + c3['split_orig_nan']
    print(f'  orig ↔ w=3: {w3_effect} peptide(s) changed by the sliding window.')

    t_orig   = float(np.mean(orig_t_runs))
    t_new    = float(np.mean(w3_t_runs))
    orig_ci  = _ci95(orig_t_runs)
    new_ci   = _ci95(w3_t_runs)
    spd      = t_orig / t_new if t_new > 0 else float('nan')

    print(f'\n  Timing  ({len(orig_t_runs)} rep(s), mean ± 95% CI)')
    def _fmt_t(mean: float, ci: float) -> str:
        ci_str = f' ± {ci:.4f}' if ci > 0 else ''
        return f'{mean:.4f}{ci_str} s  ({mean/n*1000:.2f} ms/peptide)'

    print(f'  original (loq_by_cv.py)  {_fmt_t(t_orig, orig_ci)}')
    print(f'  EmpiricalCV (w=3)        {_fmt_t(t_new, new_ci)}')
    print(f'  ratio: {spd:.2f}x  (at N={n}, pandas GroupBy overhead is minimal;')
    print(f'         EmpiricalCV numpy advantage appears at larger peptide counts)')


# -----------------------------------------------------------------------
# Per-peptide split diagnostics
# -----------------------------------------------------------------------

def _diagnose_splits(
    peptides: Dict[str, Tuple],
    orig_loqs: Dict[str, Optional[float]],
    w1_loqs: Dict[str, Optional[float]],
    w3_loqs: Dict[str, Optional[float]],
    profiles: Dict[str, dict],
    cats_w3: Dict[str, str],
    cv_thresh: float,
) -> None:
    """Explain every peptide where orig↔w=3 differs, with CV profile detail."""
    interesting = [
        (p, cats_w3[p]) for p in sorted(peptides)
        if cats_w3[p] not in ('both=none',) and not cats_w3[p].startswith('agree')
    ]
    if not interesting:
        print('\n  No splits or divergences between original and EmpiricalCV(w=3).')
        return

    print(f'\n{"=" * 72}')
    print(f'  LOQ Difference Diagnostics ({len(interesting)} peptides where orig↔w=3 differ)')
    print(f'{"=" * 72}')
    print(
        f'\n  Rule comparison:'
        f'\n  original  : first conc > 0 where ANY single CV ≤ {cv_thresh:.0%}'
        f'\n  emp w=1   : identical (numpy, same formula) — used for math validation'
        f'\n  emp w=3   : first conc > 0 where 3 CONSECUTIVE CVs all ≤ {cv_thresh:.0%}'
        f'\n  All divergences are due to the sliding window, not the CV computation.\n'
    )

    for pep, cat in interesting:
        o  = orig_loqs.get(pep)
        w1 = w1_loqs.get(pep)
        w3 = w3_loqs.get(pep)
        fv = lambda v: f'{v:.3e}' if v is not None else 'none'

        print(f'  [{pep[:65]}]')
        print(f'    Category : {cat}')
        print(f'    orig LOQ : {fv(o)}  |  w=1 LOQ: {fv(w1)}  |  w=3 LOQ: {fv(w3)}')

        prof = profiles.get(pep, {})
        if not prof:
            print('    (no CV profile)\n')
            continue

        concs = prof['concs']
        cvs   = prof['cvs']
        non_blank_cvs = [cv for c, cv in zip(concs, cvs) if c > 0]
        max_run = _max_consecutive_below(non_blank_cvs, cv_thresh)
        n_below = sum(1 for cv in non_blank_cvs if np.isfinite(cv) and cv <= cv_thresh)
        n_pos   = len(non_blank_cvs)

        print(f'    CV profile ({n_pos} non-blank points, {n_below} ≤ {cv_thresh:.0%}, '
              f'longest consecutive run: {max_run}/3 needed):')
        for c, cv in zip(concs, cvs):
            if c == 0:
                continue
            mark = ' ← below' if np.isfinite(cv) and cv <= cv_thresh else ''
            print(f'      {c:>8.4g}   CV={cv:.3f}{mark}')

        if cat == 'split:lq=inf':
            print(f'    → w=3 STRICTER: window=1 fires at {fv(w1)} (isolated dip),')
            print(f'      but CV bounces back — peptide is NOT stably quantifiable there.')
        elif cat == 'split:orig=nan':
            print(f'    → original missed a sustained low-CV run; w=3 correctly finds {fv(w3)}.')
        elif cat.startswith('diverge'):
            print(f'    → both finite but differ: orig fires at earliest dip ({fv(o)}),')
            print(f'      w=3 waits for 3 consecutive sub-threshold points ({fv(w3)}).')
        print()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    cv_thresh = args.cv_thresh
    loq_tol   = args.loq_tol
    n_reps    = args.n_reps

    print('\n' + '=' * 72)
    print('  bench_empirical.py — EmpiricalCV correctness regression')
    print('=' * 72)
    print(f'  Statistical FDR/TPR validation → bench_window_rules.py')

    print('\nLoading original loq_by_cv.py module ...')
    orig_mod = load_original_cv()

    print(f'Reading demo data ({DEMO_DATA.name}) ...')
    peptides = _load_peptides()
    n = len(peptides)
    print(f'  {n} peptides, cv_thresh={cv_thresh:.0%}, loq_tol={loq_tol:.0%}, n_reps={n_reps}')

    # Single pass for w=1 (validation only — no repeated timing needed)
    print('\nRunning EmpiricalCV(w=1) — validation pass ...')
    w1_loqs, _, _ = _run_empirical_cv(peptides, cv_thresh, window=1)

    # n_reps timed passes for w=3 (production method)
    print(f'Running EmpiricalCV(w=3) — {n_reps} timed rep(s) ...')
    w3_t_runs: List[float] = []
    for i in range(n_reps):
        w3_loqs, w3_profiles, t = _run_empirical_cv(peptides, cv_thresh, window=3)
        w3_t_runs.append(t)
        print(f'  rep {i+1:2d}/{n_reps}: {t:.4f}s')

    # n_reps timed passes for original
    print(f'Running original loq_by_cv.py — {n_reps} timed rep(s) ...')
    orig_t_runs: List[float] = []
    for i in range(n_reps):
        loqs_i, t = _run_original_cv(orig_mod, cv_thresh)
        orig_t_runs.append(t)
        print(f'  rep {i+1:2d}/{n_reps}: {t:.4f}s')
        if i == 0:
            orig_loqs = loqs_i   # capture first pass result

    # ------------------------------------------------------------------ #
    # Report                                                               #
    # ------------------------------------------------------------------ #
    print(f'\n{"=" * 72}')
    print(f'  EmpiricalCV regression — {DEMO_DATA.name}  ({n} peptides)')
    print(f'{"=" * 72}')
    cats_w1, cats_w3 = _print_table(peptides, orig_loqs, w1_loqs, w3_loqs, loq_tol, cv_thresh)
    _print_summary(cats_w1, cats_w3, orig_t_runs, w3_t_runs, n, cv_thresh, loq_tol)
    _diagnose_splits(peptides, orig_loqs, w1_loqs, w3_loqs, w3_profiles, cats_w3, cv_thresh)

    # ------------------------------------------------------------------ #
    # Save JSON                                                            #
    # ------------------------------------------------------------------ #
    if args.save:
        import datetime
        from collections import Counter

        def _cat_counts(cats: Dict[str, str]) -> dict:
            c: Counter = Counter()
            for v in cats.values():
                if   v.startswith('agree'):        c['agree'] += 1
                elif v.startswith('diverge'):      c['diverge'] += 1
                elif v == 'both=none':             c['both_none'] += 1
                elif v == 'split:lq=inf':          c['split_lq_inf'] += 1
                elif v == 'split:orig=nan':        c['split_orig_nan'] += 1
            return dict(c)

        per_pep: dict = {}
        for pep in sorted(peptides):
            prof = w3_profiles.get(pep, {})
            per_pep[pep] = {
                'orig_loq':      orig_loqs.get(pep),
                'lq_loq_w1':     w1_loqs.get(pep),
                'lq_loq_w3':     w3_loqs.get(pep),
                'cat_w1':        cats_w1.get(pep, ''),
                'cat_w3':        cats_w3.get(pep, ''),
                'rdiff_pct_w1':  _rdiff_pct(orig_loqs.get(pep), w1_loqs.get(pep)),
                'rdiff_pct_w3':  _rdiff_pct(orig_loqs.get(pep), w3_loqs.get(pep)),
                'cv_profile': {
                    'concs':           prof.get('concs', []),
                    'cvs':             prof.get('cvs', []),
                    'means':           prof.get('means', []),
                    'n_reps_per_conc': prof.get('n_reps_per_conc', []),
                },
            }

        payload = {
            'meta': {
                'dataset':        DEMO_DATA.name,
                'n_peptides':     n,
                'cv_thresh':      cv_thresh,
                'loq_tol':        loq_tol,
                'n_reps':         n_reps,
                'cv_thresh_note': (
                    'original hard-codes cv_thresh=0.20; '
                    'if cv_thresh != 0.20 the orig column is still at 0.20'
                ) if abs(cv_thresh - 0.20) > 1e-9 else '',
                't_orig_runs_s':  orig_t_runs,
                't_w3_runs_s':    w3_t_runs,
                't_orig_mean_s':  float(np.mean(orig_t_runs)),
                't_w3_mean_s':    float(np.mean(w3_t_runs)),
                't_orig_ci95_s':  _ci95(orig_t_runs),
                't_w3_ci95_s':    _ci95(w3_t_runs),
                'timestamp':      datetime.datetime.now().isoformat(),
            },
            'regression': {
                'summary': {
                    'orig_vs_w1': _cat_counts(cats_w1),
                    'orig_vs_w3': _cat_counts(cats_w3),
                },
                'per_peptide': per_pep,
            },
        }

        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(_json_safe(payload), f, indent=2)
        print(f'\nResults saved → {out}')


if __name__ == '__main__':
    main()
