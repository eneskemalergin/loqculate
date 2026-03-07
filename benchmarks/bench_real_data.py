"""bench_real_data.py — v1 vs v2 one-to-one comparison on the real demo dataset.

Data: data/demo/one_protein.csv  (27 peptides, PMA1_YEAST, 42 samples)

Measures correctness (LOD/LOQ agreement) and speed on each peptide.

Agreement glossary
------------------
  agree(N%)   both finite, |\u0394%| \u2264 tolerance
  diverge(N%) both finite, |\u0394%| > tolerance
  both=inf    both versions return inf (consistent detection failure)
  split:v2=\u221e  v1 finite, v2 inf  \u2014 v2 is more conservative
  split:v1=\u221e  v2 finite, v1 inf  \u2014 v2 found LOQ that v1 missed

LOQ split causes
----------------
  v2 requires *window* consecutive grid points ALL \u2264 cv_thresh (default: 3).
  v1 takes the minimum x where any SINGLE point has CV < cv_thresh.

  split:v2=\u221e  The CV momentarily dips below the threshold at 1-2 isolated
                points, which v1 accepts, but v2's sliding window rejects as
                non-sustained (likely a CV bounce, not true quantitation).

  split:v1=\u221e  Same LOD on both sides, but v1's bootstrap never achieves
                sub-threshold CV while v2's does. Root cause: TRF (v2) and
                Levenberg-Marquardt (v1) produce different per-replicate fits
                for resampled data, leading to different aggregate CV profiles.

Run from the repository root::

    python benchmarks/bench_real_data.py
    python benchmarks/bench_real_data.py --bootreps 10   # quick check
    python benchmarks/bench_real_data.py --lod_tol 0.30  # relax LOD agreement
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

# v2 path already added by _helpers import
from loqculate.models import PiecewiseWLS
from loqculate.io import read_calibration_data


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description='v1 vs v2 on real demo data')
    p.add_argument('--bootreps', type=int, default=100,
                   help='Bootstrap replicates (default: 100)')
    p.add_argument('--std_mult', type=float, default=2.0)
    p.add_argument('--cv_thresh', type=float, default=0.2)
    p.add_argument('--lod_tol', type=float, default=0.20,
                   help='Relative tolerance for LOD "agree" classification (default: 0.20 = 20%%)')
    p.add_argument('--loq_tol', type=float, default=0.50,
                   help='Relative tolerance for LOQ "agree" classification (default: 0.50 = 50%%)')
    return p.parse_args()


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _load_real_peptides() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read demo data through v2 reader; return {peptide: (x, y)} dict."""
    data = read_calibration_data(str(DEMO_DATA), str(DEMO_MAP))
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pep in np.unique(data.peptide):
        mask = data.peptide == pep
        out[pep] = (data.concentration[mask], data.area[mask])
    return out


def _run_v2(peptides: dict, bootreps: int, std_mult: float, cv_thresh: float):
    """Return (results_dict, elapsed, models_dict). models kept for split diagnostics."""
    results, models = {}, {}
    t0 = time.perf_counter()
    for pep, (x, y) in peptides.items():
        try:
            m = PiecewiseWLS(init_method='legacy', n_boot_reps=bootreps, seed=42)
            m.fit(x, y)
            results[pep] = {'lod': m.lod(std_mult), 'loq': m.loq(cv_thresh),
                            'slope': m.params_['slope'],
                            'intercept_linear': m.params_['intercept_linear'],
                            'intercept_noise': m.params_['intercept_noise']}
            models[pep] = m
        except Exception as exc:
            results[pep] = {'lod': np.inf, 'loq': np.inf, 'error': str(exc)}
    return results, time.perf_counter() - t0, models


def _run_v1(v1_mod, peptides: dict, bootreps: int, std_mult: float, cv_thresh: float):
    """Run v1 process_peptide() on each real peptide.

    v1 process_peptide signature:
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
                    row_df = v1_mod.process_peptide(
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

def _classify(v1_val: float, v2_val: float, tol: float) -> str:
    """Short category label for a (v1, v2) value pair."""
    f1, f2 = np.isfinite(v1_val), np.isfinite(v2_val)
    if not f1 and not f2:
        return 'both=inf'
    if f1 and not f2:
        return 'split:v2=inf'
    if not f1 and f2:
        return 'split:v1=inf'
    denom = max(abs(v1_val), abs(v2_val))
    rdiff = abs(v2_val - v1_val) / denom if denom > 0 else 0.0
    if rdiff <= tol:
        return f'agree({rdiff*100:.0f}%)'
    return f'diverge({rdiff*100:.0f}%)'


def _rdiff_pct(v1_val: float, v2_val: float) -> float:
    if np.isfinite(v1_val) and np.isfinite(v2_val):
        denom = max(abs(v1_val), abs(v2_val))
        if denom > 0:
            return 100.0 * (v2_val - v1_val) / denom
    return float('nan')


# -----------------------------------------------------------------------
# Main results table
# -----------------------------------------------------------------------

def _print_table(peptides, v1_res, v2_res, lod_tol, loq_tol):
    peps = sorted(peptides)
    pw = max(len(p) for p in peps) + 1  # dynamic peptide column width

    def _fmt(v):  return f'{v:.3e}' if np.isfinite(v) else '    inf'
    def _fmtd(v): return f'{v:+.1f}%' if np.isfinite(v) else '     —'

    hdr = (f'{"Peptide":<{pw}} {"v1 LOD":>9} {"v2 LOD":>9} {"\u0394%":>7}  '
           f'{"v1 LOQ":>9} {"v2 LOQ":>9} {"\u0394%":>7}  {"LOD status":<15} {"LOQ status":<16}')
    sep = '-' * len(hdr)
    print(f'\n{hdr}\n{sep}')

    lod_cats, loq_cats = {}, {}
    for pep in peps:
        r1, r2 = v1_res.get(pep, {}), v2_res.get(pep, {})
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

def _print_summary(lod_cats, loq_cats, v1_res, v2_res, lod_tol, loq_tol, t_v1, t_v2, n):
    from collections import Counter

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

    lod_d = [abs(_rdiff_pct(v1_res.get(p, {}).get('lod', np.inf),
                             v2_res.get(p, {}).get('lod', np.inf)))
              for p, c in lod_cats.items()
              if not c.startswith('both') and not c.startswith('split')]
    loq_d = [abs(_rdiff_pct(v1_res.get(p, {}).get('loq', np.inf),
                             v2_res.get(p, {}).get('loq', np.inf)))
              for p, c in loq_cats.items()
              if not c.startswith('both') and not c.startswith('split')]
    if lod_d:
        print(f'\n  LOD both-finite: n={len(lod_d)},  mean|\u0394|={np.mean(lod_d):.1f}%,  '
              f'max|\u0394|={max(lod_d):.1f}%')
    if loq_d:
        print(f'  LOQ both-finite: n={len(loq_d)},  mean|\u0394|={np.mean(loq_d):.1f}%,  '
              f'max|\u0394|={max(loq_d):.1f}%')

    print(f'\n  Timing')
    print(f'  v1  {t_v1:.2f}s  ({t_v1/n*1000:.1f} ms/peptide)')
    print(f'  v2  {t_v2:.2f}s  ({t_v2/n*1000:.1f} ms/peptide)')
    spd = t_v1 / t_v2 if t_v2 > 0 else float('nan')
    print(f'  {spd:.2f}x  ({"v2 faster" if spd > 1 else "v1 faster"})')


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


def _diagnose_splits(peptides, v1_res, v2_res, v2_models, loq_cats, cv_thresh, window):
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
    print(f'  v1: first x > LOD where ANY single bootstrap grid point has CV < {cv_thresh}')
    print(f'  v2: first x > LOD where {window} CONSECUTIVE grid points ALL have CV <= {cv_thresh}')
    print(f'  (v2 sliding window prevents false LOQs from non-monotonic CV bounces)\n')

    for pep, cat in splits:
        r1, r2 = v1_res.get(pep, {}), v2_res.get(pep, {})
        l1  = r1.get('lod', np.inf)
        l2  = r2.get('lod', np.inf)
        q1  = r1.get('loq', np.inf)
        q2  = r2.get('loq', np.inf)
        l1s = f'{l1:.3e}' if np.isfinite(l1) else 'inf'
        l2s = f'{l2:.3e}' if np.isfinite(l2) else 'inf'
        q1s = f'{q1:.3e}' if np.isfinite(q1) else 'inf'
        q2s = f'{q2:.3e}' if np.isfinite(q2) else 'inf'

        print(f'  [{pep}]')
        print(f'    LOD: v1={l1s}  v2={l2s}')
        print(f'    LOQ: v1={q1s}  v2={q2s}')

        m = v2_models.get(pep)
        if m is None or m._boot_summary is None:
            if cat == 'split:v2=inf' and np.isfinite(l1) and not np.isfinite(l2):
                print(f'    v2 LOD=inf → bootstrap never ran.')
                print(f'    \u2192 Cause: TRF optimizer gave no valid noise/linear intersection here;'
                      f' v1 LM did.')
            else:
                print(f'    v2 bootstrap not available (likely LOD was inf).')
            print()
            continue

        cv2 = m._boot_summary['cv']
        xg2 = m._x_grid
        min_cv   = float(np.nanmin(cv2))
        n_below  = int(np.sum(np.isfinite(cv2) & (cv2 <= cv_thresh)))
        max_run  = _max_consecutive_below(cv2, cv_thresh)

        print(f'    v2 bootstrap CV profile:')
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

        if cat == 'split:v2=inf':
            if min_cv > cv_thresh:
                print(f'    \u2192 Cause: v2 TRF bootstrap produces HIGHER CV everywhere than v1 LM.')
                print(f'      Different per-replicate fits (TRF vs LM on resampled data) lead to')
                print(f'      a wider bootstrap band in v2 \u2192 CV never dips below {cv_thresh}.')
                print(f'      v1 accepts the first point below threshold; v2 finds none.')
            else:
                print(f'    \u2192 Cause: v2 SLIDING WINDOW too strict for this peptide.')
                print(f'      CV dips below {cv_thresh} at {n_below} isolated point(s)')
                print(f'      (max run = {max_run}), but never for {window} consecutive points.')
                print(f'      v1 grabs the first dip; v2 waits for sustained low CV.')
        else:  # split:v1=inf
            if not np.isfinite(l1):
                print(f'    \u2192 Cause: v1 LOD=inf (LM gave no valid intersection).')
                print(f'      v2 LOD={l2s} finite \u2192 its bootstrap ran \u2192 found LOQ.')
            else:
                print(f'    \u2192 Cause: Same LOD, but v1 bootstrap CV stays >= {cv_thresh} everywhere.')
                print(f'      v2 TRF produces slightly tighter per-replicate fits on resampled')
                print(f'      data \u2192 lower prediction spread \u2192 lower CV \u2192 {max_run}-point run')
                print(f'      below threshold \u2192 v2 finds LOQ where v1 cannot.')
        print()



# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    args = _parse_args()
    print('Loading v1 module ...')
    v1 = load_v1_calc()

    print(f'Reading demo data ({DEMO_DATA.name}) ...')
    peptides = _load_real_peptides()
    print(f'  {len(peptides)} peptides.')

    print(f'\nRunning v2  (bootreps={args.bootreps}) ...')
    v2_res, t_v2, v2_models = _run_v2(peptides, args.bootreps, args.std_mult, args.cv_thresh)

    print(f'Running v1  (bootreps={args.bootreps}) ...')
    v1_res, t_v1 = _run_v1(v1, peptides, args.bootreps, args.std_mult, args.cv_thresh)

    n = len(peptides)
    print(f'\n{"=" * 72}')
    print(f'  One-to-one comparison: {DEMO_DATA.name}  ({n} peptides)')
    print(f'{"=" * 72}')
    lod_cats, loq_cats = _print_table(peptides, v1_res, v2_res, args.lod_tol, args.loq_tol)
    _print_summary(lod_cats, loq_cats, v1_res, v2_res,
                   args.lod_tol, args.loq_tol, t_v1, t_v2, n)
    _diagnose_splits(peptides, v1_res, v2_res, v2_models, loq_cats,
                     args.cv_thresh, window=3)


if __name__ == '__main__':
    main()
