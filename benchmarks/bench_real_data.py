"""bench_real_data.py — original vs loqculate one-to-one comparison on the real demo dataset.

Data: data/demo/one_protein.csv  (27 peptides, PMA1_YEAST, 42 samples)

Measures correctness (LOD/LOQ agreement) and speed on each peptide.

Agreement glossary
------------------
  agree(N%)   both finite, |\u0394%| \u2264 tolerance
  diverge(N%) both finite, |\u0394%| > tolerance
  both=inf    both versions return inf (consistent detection failure)
  split:new=\u221e  orig finite, new inf  \u2014 loqculate is more conservative
  split:orig=\u221e  new finite, orig inf  \u2014 loqculate found LOQ that the original missed

LOQ split causes
----------------
  loqculate requires *window* consecutive grid points ALL \u2264 cv_thresh (default: 3).
  The original takes the minimum x where any SINGLE point has CV < cv_thresh.

  split:new=\u221e  The CV momentarily dips below the threshold at 1-2 isolated
                points, which the original accepts, but loqculate's sliding window
                rejects as non-sustained (likely a CV bounce, not true quantitation).

  split:orig=\u221e  Same LOD on both sides, but the original bootstrap never achieves
                sub-threshold CV while loqculate's does. Root cause: TRF (loqculate) and
                Levenberg-Marquardt (original) produce different per-replicate fits
                for resampled data, leading to different aggregate CV profiles.

Run from the repository root::

    python benchmarks/bench_real_data.py
    python benchmarks/bench_real_data.py --bootreps 10   # quick check
    python benchmarks/bench_real_data.py --lod_tol 0.30  # relax LOD agreement
    python benchmarks/bench_real_data.py --n_reps 1      # single-pass (no CI)
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
from _helpers import DEMO_DATA, DEMO_MAP, load_original_calc, _json_safe, _ci95

# loqculate path already added by _helpers import
from loqculate.models import PiecewiseWLS
from loqculate.io import read_calibration_data


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description='original vs loqculate on real demo data')
    p.add_argument('--bootreps', type=int, default=100,
                   help='Bootstrap replicates (default: 100)')
    p.add_argument('--std_mult', type=float, default=2.0)
    p.add_argument('--cv_thresh', type=float, default=0.2)
    p.add_argument('--lod_tol', type=float, default=0.20,
                   help='Relative tolerance for LOD "agree" classification (default: 0.20 = 20%%)')
    p.add_argument('--loq_tol', type=float, default=0.50,
                   help='Relative tolerance for LOQ "agree" classification (default: 0.50 = 50%%)')
    p.add_argument('--save', type=str, default=None, metavar='PATH',
                   help='Write JSON results to PATH (e.g. tmp/results/bench_real_data.json)')
    p.add_argument('--n_reps', type=int, default=5,
                   help='Timing repetitions per implementation for CI estimation (default: 5)')
    return p.parse_args()


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _load_real_peptides() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read demo data through the loqculate reader; return {peptide: (x, y)} dict."""
    data = read_calibration_data(str(DEMO_DATA), str(DEMO_MAP))
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pep in np.unique(data.peptide):
        mask = data.peptide == pep
        out[pep] = (data.concentration[mask], data.area[mask])
    return out


def _run_loqculate(peptides: dict, bootreps: int, std_mult: float, cv_thresh: float,
                   sliding_window: int = 3):
    """Return (results_dict, elapsed, models_dict). models kept for split diagnostics."""
    results, models = {}, {}
    t0 = time.perf_counter()
    for pep, (x, y) in peptides.items():
        try:
            m = PiecewiseWLS(init_method='legacy', n_boot_reps=bootreps, seed=42,
                             sliding_window=sliding_window)
            m.fit(x, y)
            results[pep] = {'lod': m.lod(std_mult), 'loq': m.loq(cv_thresh),
                            'slope': m.params_['slope'],
                            'intercept_linear': m.params_['intercept_linear'],
                            'intercept_noise': m.params_['intercept_noise']}
            models[pep] = m
        except Exception as exc:
            results[pep] = {'lod': np.inf, 'loq': np.inf, 'error': str(exc)}
    return results, time.perf_counter() - t0, models


def _run_original(orig_mod, peptides: dict, bootreps: int, std_mult: float, cv_thresh: float):
    """Run original process_peptide() on each real peptide.

    original process_peptide signature:
      (bootreps, cv_thresh, output_dir, peptide, plot_or_not, std_mult,
       min_noise_points, min_linear_points, subset_df, verbose, model_choice)
    """
    import pandas as pd
    import tempfile

    results = {}
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmpdir:
        for pep, (x, y) in peptides.items():
            subset = pd.DataFrame({
                'peptide': pep,
                'curvepoint': x.astype(float),
                'area': y.astype(float),
            }).sort_values('curvepoint').reset_index(drop=True)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    row_df = orig_mod.process_peptide(
                        bootreps, cv_thresh, tmpdir, pep,
                        'n', std_mult, 2, 1, subset, 'n', 'piecewise')
                    row = row_df.iloc[0].to_dict()
                    results[pep] = {
                        'lod': float(row.get('LOD', np.inf)),
                        'loq': float(row.get('LOQ', np.inf)),
                    }
                except Exception as exc:
                    results[pep] = {'lod': np.inf, 'loq': np.inf, 'error': str(exc)}
    return results, time.perf_counter() - t0


# -----------------------------------------------------------------------
# Agreement classification
# -----------------------------------------------------------------------

def _classify(orig_val: float, new_val: float, tol: float) -> str:
    """Short category label for an (orig, new) value pair."""
    f1, f2 = np.isfinite(orig_val), np.isfinite(new_val)
    if not f1 and not f2:
        return 'both=inf'
    if f1 and not f2:
        return 'split:new=inf'
    if not f1 and f2:
        return 'split:orig=inf'
    denom = max(abs(orig_val), abs(new_val))
    rdiff = abs(new_val - orig_val) / denom if denom > 0 else 0.0
    if rdiff <= tol:
        return f'agree({rdiff*100:.0f}%)'
    return f'diverge({rdiff*100:.0f}%)'


def _rdiff_pct(orig_val: float, new_val: float) -> float:
    if np.isfinite(orig_val) and np.isfinite(new_val):
        denom = max(abs(orig_val), abs(new_val))
        if denom > 0:
            return 100.0 * (new_val - orig_val) / denom
    return float('nan')


# -----------------------------------------------------------------------
# Main results table
# -----------------------------------------------------------------------

def _print_table(peptides, orig_res, new_res, lod_tol, loq_tol):
    peps = sorted(peptides)
    pw = max(len(p) for p in peps) + 1  # dynamic peptide column width

    def _fmt(v):  return f'{v:.3e}' if np.isfinite(v) else '    inf'
    def _fmtd(v): return f'{v:+.1f}%' if np.isfinite(v) else '     —'

    hdr = (f'{"Peptide":<{pw}} {"orig LOD":>9} {"lq LOD":>9} {"\u0394%":>7}  '
           f'{"orig LOQ":>9} {"lq LOQ":>9} {"\u0394%":>7}  {"LOD status":<15} {"LOQ status":<16}')
    sep = '-' * len(hdr)
    print(f'\n{hdr}\n{sep}')

    lod_cats, loq_cats = {}, {}
    for pep in peps:
        r1, r2 = orig_res.get(pep, {}), new_res.get(pep, {})
        l1, l2 = r1.get('lod', np.inf), r2.get('lod', np.inf)
        q1, q2 = r1.get('loq', np.inf), r2.get('loq', np.inf)
        lc = _classify(l1, l2, lod_tol)
        qc = _classify(q1, q2, loq_tol)
        lod_cats[pep] = lc
        loq_cats[pep] = qc
        print(f'{pep:<{pw}} {_fmt(l1):>9} {_fmt(l2):>9} {_fmtd(_rdiff_pct(l1, l2)):>7}  '
              f'{_fmt(q1):>9} {_fmt(q2):>9} {_fmtd(_rdiff_pct(q1, q2)):>7}  '
              f'{lc:<15} {qc:<16}')
    print(sep)
    return lod_cats, loq_cats


# -----------------------------------------------------------------------
# Summary counts
# -----------------------------------------------------------------------

def _print_summary(lod_cats, loq_cats, orig_res, new_res, lod_tol, loq_tol,
                   orig_t_runs, new_t_runs, n):
    from collections import Counter

    t_orig = float(np.mean(orig_t_runs))
    t_new  = float(np.mean(new_t_runs))
    orig_ci = _ci95(orig_t_runs)
    new_ci  = _ci95(new_t_runs)

    def _count(cats):
        c = Counter()
        for v in cats.values():
            if   v.startswith('agree'):   c['agree'] += 1
            elif v.startswith('diverge'): c['diverge'] += 1
            elif v == 'both=inf':         c['both_inf'] += 1
            else:                         c['split'] += 1
        return c

    lc, qc = _count(lod_cats), _count(loq_cats)
    print(f'\n  Agreement  (LOD tol={lod_tol*100:.0f}%,  LOQ tol={loq_tol*100:.0f}%)\n')
    print(f'  {"Category":<35} {"LOD":>5}  {"LOQ":>5}')
    print(f'  {"-"*47}')
    print(f'  {"agree  (both finite, |\u0394%| \u2264 tol)":<35} {lc["agree"]:>5}  {qc["agree"]:>5}')
    print(f'  {"both=inf  (consistent failure)":<35} {lc["both_inf"]:>5}  {qc["both_inf"]:>5}')
    print(f'  {"diverge  (both finite, |\u0394%| > tol)":<35} {lc["diverge"]:>5}  {qc["diverge"]:>5}')
    print(f'  {"split  (exactly one side is inf)":<35} {lc["split"]:>5}  {qc["split"]:>5}')
    print(f'  {"-"*47}')
    print(f'  {"Total":<35} {n:>5}  {n:>5}')

    lod_d = [abs(_rdiff_pct(orig_res.get(p, {}).get('lod', np.inf),
                             new_res.get(p, {}).get('lod', np.inf)))
              for p, c in lod_cats.items()
              if not c.startswith('both') and not c.startswith('split')]
    loq_d = [abs(_rdiff_pct(orig_res.get(p, {}).get('loq', np.inf),
                             new_res.get(p, {}).get('loq', np.inf)))
              for p, c in loq_cats.items()
              if not c.startswith('both') and not c.startswith('split')]
    if lod_d:
        print(f'\n  LOD both-finite: n={len(lod_d)},  mean|\u0394|={np.mean(lod_d):.1f}%,  '
              f'max|\u0394|={max(lod_d):.1f}%')
    if loq_d:
        print(f'  LOQ both-finite: n={len(loq_d)},  mean|\u0394|={np.mean(loq_d):.1f}%,  '
              f'max|\u0394|={max(loq_d):.1f}%')

    print(f'\n  Timing  ({len(orig_t_runs)} rep(s) each)')
    def _fmt_t(mean, ci):
        if ci > 0:
            return f'{mean:.2f} ± {ci:.2f} s  ({mean/n*1000:.1f} ms/peptide)'
        return f'{mean:.2f} s  ({mean/n*1000:.1f} ms/peptide)'
    print(f'  orig  {_fmt_t(t_orig, orig_ci)}')
    print(f'  lq    {_fmt_t(t_new, new_ci)}')
    spd = t_orig / t_new if t_new > 0 else float('nan')
    print(f'  {spd:.2f}x  ({"lq faster" if spd > 1 else "orig faster"})')


# -----------------------------------------------------------------------
# Split diagnostics
# -----------------------------------------------------------------------

def _max_consecutive_below(arr: np.ndarray, thresh: float) -> int:
    """Length of the longest run of values at-or-below thresh."""
    max_run = cur = 0
    for v in arr:
        if np.isfinite(v) and v <= thresh:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run


def _diagnose_splits(peptides, orig_res, new_res, new_models, loq_cats, cv_thresh, window):
    """Print a per-peptide explanation for every LOQ split."""
    splits = [(p, loq_cats[p]) for p in sorted(peptides)
              if loq_cats[p].startswith('split')]
    if not splits:
        print('\n  No LOQ splits — both versions agree on which peptides are quantifiable.')
        return

    print(f'\n{"=" * 72}')
    print(f'  LOQ Split Diagnostics   ({len(splits)} peptides where exactly one side is inf)')
    print(f'{"=" * 72}')
    print(f'\n  LOQ RULES COMPARED')
    print(f'  original : first x > LOD where ANY single bootstrap grid point has CV < {cv_thresh}')
    print(f'  loqculate: first x > LOD where {window} CONSECUTIVE grid points ALL have CV <= {cv_thresh}')
    print(f'  (loqculate sliding window prevents false LOQs from non-monotonic CV bounces)\n')

    for pep, cat in splits:
        r1, r2 = orig_res.get(pep, {}), new_res.get(pep, {})
        l1  = r1.get('lod', np.inf)
        l2  = r2.get('lod', np.inf)
        q1  = r1.get('loq', np.inf)
        q2  = r2.get('loq', np.inf)
        l1s = f'{l1:.3e}' if np.isfinite(l1) else 'inf'
        l2s = f'{l2:.3e}' if np.isfinite(l2) else 'inf'
        q1s = f'{q1:.3e}' if np.isfinite(q1) else 'inf'
        q2s = f'{q2:.3e}' if np.isfinite(q2) else 'inf'

        print(f'  [{pep}]')
        print(f'    LOD: orig={l1s}  lq={l2s}')
        print(f'    LOQ: orig={q1s}  lq={q2s}')

        m = new_models.get(pep)
        if m is None or m._boot_summary is None:
            if cat == 'split:new=inf' and np.isfinite(l1) and not np.isfinite(l2):
                print(f'    loqculate LOD=inf \u2192 bootstrap never ran.')
                print(f'    \u2192 Cause: TRF optimizer gave no valid noise/linear intersection here;'
                      f' original LM did.')
            else:
                print(f'    loqculate bootstrap not available (likely LOD was inf).')
            print()
            continue

        cv2 = m._boot_summary['cv']
        xg2 = m._x_grid
        min_cv   = float(np.nanmin(cv2))
        n_below  = int(np.sum(np.isfinite(cv2) & (cv2 <= cv_thresh)))
        max_run  = _max_consecutive_below(cv2, cv_thresh)

        print(f'    loqculate bootstrap CV profile:')
        print(f'      Grid: {len(cv2)} pts from {xg2[0]:.3e} to {xg2[-1]:.3e}')
        print(f'      min CV = {min_cv:.3f}   pts <= {cv_thresh}: {n_below}/{len(cv2)}   '
              f'longest run: {max_run} (need {window})')

        # Show CV profile excerpt near threshold
        near = np.where(np.isfinite(cv2) & (cv2 <= cv_thresh * 2.5))[0]
        if len(near) > 0:
            lo = max(0, near[0] - 1)
            hi = min(len(cv2), near[-1] + 2)
            print(f'      CV near threshold (excerpt):')
            run_count = 0
            for i in range(lo, hi):
                below = np.isfinite(cv2[i]) and cv2[i] <= cv_thresh
                run_count = run_count + 1 if below else 0
                mark = ' <-- window met' if run_count >= window else (' <-- below thresh' if below else '')
                print(f'        x={xg2[i]:.3e}  CV={cv2[i]:.3f}{mark}')

        if cat == 'split:new=inf':
            if min_cv > cv_thresh:
                print(f'    \u2192 Cause: loqculate TRF bootstrap produces HIGHER CV everywhere than original LM.')
                print(f'      Different per-replicate fits (TRF vs LM on resampled data) lead to')
                print(f'      a wider bootstrap band in loqculate \u2192 CV never dips below {cv_thresh}.')
                print(f'      Original accepts the first point below threshold; loqculate finds none.')
            else:
                print(f'    \u2192 Cause: loqculate SLIDING WINDOW too strict for this peptide.')
                print(f'      CV dips below {cv_thresh} at {n_below} isolated point(s)')
                print(f'      (max run = {max_run}), but never for {window} consecutive points.')
                print(f'      Original grabs the first dip; loqculate waits for sustained low CV.')
        else:  # split:orig=inf
            if not np.isfinite(l1):
                print(f'    \u2192 Cause: original LOD=inf (LM gave no valid intersection).')
                print(f'      loqculate LOD={l2s} finite \u2192 its bootstrap ran \u2192 found LOQ.')
            else:
                print(f'    \u2192 Cause: Same LOD, but original bootstrap CV stays >= {cv_thresh} everywhere.')
                print(f'      loqculate TRF produces slightly tighter per-replicate fits on resampled')
                print(f'      data \u2192 lower prediction spread \u2192 lower CV \u2192 {max_run}-point run')
                print(f'      below threshold \u2192 loqculate finds LOQ where original cannot.')
        print()



# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    args = _parse_args()
    print('Loading original module ...')
    orig = load_original_calc()

    print(f'Reading demo data ({DEMO_DATA.name}) ...')
    peptides = _load_real_peptides()
    print(f'  {len(peptides)} peptides.')

    print(f'\nRunning loqculate w=3  (bootreps={args.bootreps}) ...')
    orig_t_runs: list = []
    new_t_runs: list = []
    new_res, t_new, new_models = _run_loqculate(
        peptides, args.bootreps, args.std_mult, args.cv_thresh, sliding_window=3)
    new_t_runs.append(t_new)
    print(f'  rep  1/{args.n_reps}: {t_new:.2f}s')
    for i in range(1, args.n_reps):
        _, t, _ = _run_loqculate(peptides, args.bootreps, args.std_mult, args.cv_thresh,
                                 sliding_window=3)
        new_t_runs.append(t)
        print(f'  rep {i+1:2d}/{args.n_reps}: {t:.2f}s')

    print(f'\nRunning loqculate w=1  (bootreps={args.bootreps}) ...')
    w1_t_runs: list = []
    w1_res, t_w1, w1_models = _run_loqculate(
        peptides, args.bootreps, args.std_mult, args.cv_thresh, sliding_window=1)
    w1_t_runs.append(t_w1)
    print(f'  rep  1/{args.n_reps}: {t_w1:.2f}s')
    for i in range(1, args.n_reps):
        _, t, _ = _run_loqculate(peptides, args.bootreps, args.std_mult, args.cv_thresh,
                                 sliding_window=1)
        w1_t_runs.append(t)
        print(f'  rep {i+1:2d}/{args.n_reps}: {t:.2f}s')

    print(f'Running original  (bootreps={args.bootreps}) ...')
    orig_res, t_orig = _run_original(
        orig, peptides, args.bootreps, args.std_mult, args.cv_thresh)
    orig_t_runs.append(t_orig)
    print(f'  rep  1/{args.n_reps}: {t_orig:.2f}s')
    for i in range(1, args.n_reps):
        _, t = _run_original(orig, peptides, args.bootreps, args.std_mult, args.cv_thresh)
        orig_t_runs.append(t)
        print(f'  rep {i+1:2d}/{args.n_reps}: {t:.2f}s')

    n = len(peptides)
    print(f'\n{"=" * 72}')
    print(f'  Three-way comparison: {DEMO_DATA.name}  ({n} peptides)')
    print(f'{"=" * 72}')

    # --- orig vs w=3 (combined effect — what users see) ---
    print(f'\n--- orig vs lq(w=3) [combined: optimizer + window] ---')
    lod_cats_ow3, loq_cats_ow3 = _print_table(peptides, orig_res, new_res, args.lod_tol, args.loq_tol)
    _print_summary(lod_cats_ow3, loq_cats_ow3, orig_res, new_res,
                   args.lod_tol, args.loq_tol, orig_t_runs, new_t_runs, n)
    _diagnose_splits(peptides, orig_res, new_res, new_models, loq_cats_ow3,
                     args.cv_thresh, window=3)

    # --- orig vs w=1 (optimizer effect only — same single-point LOQ rule) ---
    print(f'\n--- orig vs lq(w=1) [optimizer effect only: LM→TRF, same LOQ rule] ---')
    lod_cats_ow1, loq_cats_ow1 = _print_table(peptides, orig_res, w1_res, args.lod_tol, args.loq_tol)
    _print_summary(lod_cats_ow1, loq_cats_ow1, orig_res, w1_res,
                   args.lod_tol, args.loq_tol, orig_t_runs, w1_t_runs, n)
    _diagnose_splits(peptides, orig_res, w1_res, w1_models, loq_cats_ow1,
                     args.cv_thresh, window=1)

    # --- w=1 vs w=3 (window effect only — same optimizer) ---
    print(f'\n--- lq(w=1) vs lq(w=3) [window effect only: same optimizer, different LOQ rule] ---')
    lod_cats_w13, loq_cats_w13 = _print_table(peptides, w1_res, new_res, args.lod_tol, args.loq_tol)
    _print_summary(lod_cats_w13, loq_cats_w13, w1_res, new_res,
                   args.lod_tol, args.loq_tol, w1_t_runs, new_t_runs, n)

    if args.save:
        import datetime
        from collections import Counter

        def _cat_counts(cats):
            c = Counter()
            for v in cats.values():
                if   v.startswith('agree'):   c['agree'] += 1
                elif v.startswith('diverge'): c['diverge'] += 1
                elif v == 'both=inf':         c['both_inf'] += 1
                else:                         c['split'] += 1
            return dict(c)

        per_pep = {}
        for pep in sorted(peptides):
            r_orig = orig_res.get(pep, {})
            r_w1   = w1_res.get(pep, {})
            r_w3   = new_res.get(pep, {})
            per_pep[pep] = {
                'orig_lod':        r_orig.get('lod', None),
                'orig_loq':        r_orig.get('loq', None),
                'lq_w1_lod':       r_w1.get('lod', None),
                'lq_w1_loq':       r_w1.get('loq', None),
                'lq_w3_lod':       r_w3.get('lod', None),
                'lq_w3_loq':       r_w3.get('loq', None),
                # orig ↔ w=3 (combined effect)
                'lod_cat_orig_w3': lod_cats_ow3.get(pep, ''),
                'loq_cat_orig_w3': loq_cats_ow3.get(pep, ''),
                # orig ↔ w=1 (optimizer effect only)
                'lod_cat_orig_w1': lod_cats_ow1.get(pep, ''),
                'loq_cat_orig_w1': loq_cats_ow1.get(pep, ''),
                # w=1 ↔ w=3 (window effect only)
                'lod_cat_w1_w3':   lod_cats_w13.get(pep, ''),
                'loq_cat_w1_w3':   loq_cats_w13.get(pep, ''),
            }
        results = {
            'meta': {
                'bootreps':   args.bootreps,
                'std_mult':   args.std_mult,
                'cv_thresh':  args.cv_thresh,
                'lod_tol':    args.lod_tol,
                'loq_tol':    args.loq_tol,
                'n_peptides': n,
                'dataset':    DEMO_DATA.name,
                't_orig_runs_s':  orig_t_runs,
                't_w1_runs_s':    w1_t_runs,
                't_w3_runs_s':    new_t_runs,
                't_orig_mean_s':  float(np.mean(orig_t_runs)),
                't_w1_mean_s':    float(np.mean(w1_t_runs)),
                't_w3_mean_s':    float(np.mean(new_t_runs)),
                't_orig_ci95_s':  _ci95(orig_t_runs),
                't_w1_ci95_s':    _ci95(w1_t_runs),
                't_w3_ci95_s':    _ci95(new_t_runs),
                'n_reps':     args.n_reps,
                'timestamp':  datetime.datetime.now().isoformat(),
            },
            'per_peptide': per_pep,
            'summary': {
                'lod_orig_vs_w3': _cat_counts(lod_cats_ow3),
                'loq_orig_vs_w3': _cat_counts(loq_cats_ow3),
                'lod_orig_vs_w1': _cat_counts(lod_cats_ow1),
                'loq_orig_vs_w1': _cat_counts(loq_cats_ow1),
                'lod_w1_vs_w3':   _cat_counts(lod_cats_w13),
                'loq_w1_vs_w3':   _cat_counts(loq_cats_w13),
            },
        }
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(_json_safe(results), f, indent=2)
        print(f'Results saved → {out}')


if __name__ == '__main__':
    main()
