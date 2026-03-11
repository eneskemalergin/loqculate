"""gen_comparison_plots.py — Side-by-side calibration plots: original (lmfit L-M) vs loqculate (scipy TRF).

For every peptide in the demo dataset (27 peptides from PMA1_YEAST) this
script produces three PNGs per peptide:

    tmp/comparison_plots/
    ├── DDTAQTVSEAR_orig.png          ← original build_plots (lmfit L-M, window=1)
    ├── DDTAQTVSEAR_new.png           ← loqculate plot_calibration (TRF, window=3)
    ├── DDTAQTVSEAR_comparison.png  ← loqculate overlay: PiecewiseWLS + EmpiricalCV
    ├── GEGFMVVTATGDNTFVGR_orig.png
    ...

Usage
-----
    python benchmarks/gen_comparison_plots.py
    python benchmarks/gen_comparison_plots.py --bootreps 50 --out tmp/my_plots
    python benchmarks/gen_comparison_plots.py --peptide DDTAQTVSEAR  # single pep
"""
from __future__ import annotations

import argparse
import sys
import os
import warnings
from pathlib import Path

# ── ensure repo root is on PYTHONPATH ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))   # benchmarks/
from _helpers import (
    DEMO_DATA, DEMO_MAP, ROOT,
    load_original_calc, _ensure_package_on_path,
)
_ensure_package_on_path()
# ───────────────────────────────────────────────────────────────────────────

import numpy as np

from loqculate.io.readers import read_calibration_data
from loqculate.models.piecewise_wls import PiecewiseWLS
from loqculate.models.cv_empirical import EmpiricalCV
from loqculate.plotting.calibration import plot_calibration
from loqculate.plotting.comparison import plot_model_comparison


# ── defaults ────────────────────────────────────────────────────────────────
DEFAULT_BOOTREPS   = 100
DEFAULT_STD_MULT   = 2.0
DEFAULT_CV_THRESH  = 0.20
DEFAULT_OUT        = ROOT / "tmp" / "comparison_plots"


# ── helpers ──────────────────────────────────────────────────────────────────

def _original_plots(orig, peptides, df, bootreps, std_mult, cv_thresh, out_orig):
    """Run original process_peptide (with plot='y') for each peptide."""
    results = {}
    for pep in peptides:
        subset = df[df['peptide'] == pep].copy()
        try:
            row = orig.process_peptide(
                bootreps, cv_thresh, str(out_orig),
                pep, 'y', std_mult,
                2, 1,          # min_noise_points, min_linear_points
                subset, 'n', 'piecewise',
            )
            lod = float(row['LOD'].iloc[0])
            loq = float(row['LOQ'].iloc[0])
            results[pep] = {'lod': lod, 'loq': loq}
            tag = f"LOD={'inf' if not np.isfinite(lod) else f'{lod:.2e}'}  LOQ={'inf' if not np.isfinite(loq) else f'{loq:.2e}'}"
            print(f"  orig  {pep:<40s}  {tag}")
        except Exception as exc:
            print(f"  orig  {pep:<40s}  ERROR: {exc}", file=sys.stderr)
            results[pep] = {'lod': np.inf, 'loq': np.inf}

    # original saves its own plots named "{peptide}.png" — rename to *_orig.png
    for pep in peptides:
        src = out_orig / f"{pep}.png"
        dst = out_orig.parent / f"{pep}_orig.png"
        if src.exists():
            src.rename(dst)

    return results


def _loqculate_plots(peptides, data, bootreps, std_mult, cv_thresh, out_dir, sliding_window):
    """Fit loqculate models and produce plot_calibration + plot_model_comparison."""
    results = {}
    for pep in peptides:
        mask = data.peptide == pep
        x = data.concentration[mask]
        y = data.area[mask]

        pw = PiecewiseWLS(
            n_boot_reps=bootreps,
            sliding_window=sliding_window,
        ).fit(x, y)

        # EmpiricalCV requires ≥2 replicates per level — skip silently if it fails
        try:
            cv = EmpiricalCV().fit(x, y)
            has_cv = True
        except ValueError:
            has_cv = False

        lod = pw.lod(std_mult)
        loq = pw.loq(cv_thresh)
        results[pep] = {'lod': lod, 'loq': loq}

        tag = f"LOD={'inf' if not np.isfinite(lod) else f'{lod:.2e}'}  LOQ={'inf' if not np.isfinite(loq) else f'{loq:.2e}'}"
        print(f"  lq  {pep:<40s}  {tag}")

        # ── single-model plot ──────────────────────────────────────────────────
        plot_calibration(
            pw, x, y,
            peptide_name=pep,
            output_path=out_dir / f"{pep}_new.png",
            std_mult=std_mult,
            cv_thresh=cv_thresh,
        )

        # ── comparison plot (PiecewiseWLS + EmpiricalCV) ──────────────────────
        models = {'Piecewise WLS': pw}
        if has_cv:
            models['Empirical CV'] = cv
        plot_model_comparison(
            models, x, y,
            peptide_name=pep,
            output_path=out_dir / f"{pep}_comparison.png",
            std_mult=std_mult,
            cv_thresh=cv_thresh,
        )

    return results


def _print_summary(peptides, orig_res, new_res):
    """Print a LOD/LOQ comparison table."""
    hdr = f"{'Peptide':<40s}  {'orig LOD':>12s}  {'orig LOQ':>12s}  {'lq LOD':>12s}  {'lq LOQ':>12s}  {'LOD agree':>10s}  {'LOQ agree':>10s}"
    print()
    print(hdr)
    print('-' * len(hdr))
    for pep in peptides:
        r1, r2 = orig_res[pep], new_res[pep]

        def _fmt(v):
            return 'inf' if not np.isfinite(v) else f'{v:.3e}'

        def _agree(a, b):
            if not np.isfinite(a) and not np.isfinite(b):
                return 'both=∞'
            if not np.isfinite(a):
                return 'orig=∞'
            if not np.isfinite(b):
                return 'new=∞'
            rdiff = abs(a - b) / max(abs(a), 1e-30)
            return 'YES' if rdiff < 0.02 else f'~{rdiff:.0%}'

        print(
            f"{pep:<40s}  {_fmt(r1['lod']):>12s}  {_fmt(r1['loq']):>12s}"
            f"  {_fmt(r2['lod']):>12s}  {_fmt(r2['loq']):>12s}"
            f"  {_agree(r1['lod'], r2['lod']):>10s}  {_agree(r1['loq'], r2['loq']):>10s}"
        )


# ── main ─────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Generate comparison plots for all demo peptides.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--bootreps',        type=int,   default=DEFAULT_BOOTREPS)
    p.add_argument('--std_mult',        type=float, default=DEFAULT_STD_MULT)
    p.add_argument('--cv_thresh',       type=float, default=DEFAULT_CV_THRESH)
    p.add_argument('--sliding_window',  type=int,   default=3,
                   help='sliding-window size for LOQ search')
    p.add_argument('--out',             type=Path,  default=DEFAULT_OUT,
                   help='output directory for all PNGs')
    p.add_argument('--peptide',         type=str,   default=None,
                   help='run for a single peptide only (exact name)')
    return p.parse_args()


def main():
    args = _parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_orig  = out_dir / '_orig_tmp'          # original saves {pep}.png here; we rename
    out_orig.mkdir(exist_ok=True)

    print(f"Output directory : {out_dir}")
    print(f"Bootstrap reps   : {args.bootreps}")
    print(f"Sliding window   : {args.sliding_window}")
    print()

    # ── load data ──────────────────────────────────────────────────────────
    print("Loading loqculate data …")
    data = read_calibration_data(DEMO_DATA, DEMO_MAP)

    print("Loading original module + data …")
    orig = load_original_calc()
    df = orig.read_input(str(DEMO_DATA), str(DEMO_MAP))

    all_peptides = list(sorted(set(data.peptide)))
    if args.peptide:
        if args.peptide not in all_peptides:
            sys.exit(f"Peptide '{args.peptide}' not found. Available:\n  " +
                     '\n  '.join(all_peptides))
        peptides = [args.peptide]
    else:
        peptides = all_peptides

    print(f"Peptides to process: {len(peptides)}\n")

    # ── original plots ──────────────────────────────────────────────────────────
    print("=== Original (lmfit L-M, window=1) ===")
    warnings.filterwarnings('ignore')   # suppress lmfit convergence warnings
    orig_res = _original_plots(orig, peptides, df, args.bootreps,
                       args.std_mult, args.cv_thresh, out_orig)
    warnings.resetwarnings()

    # ── loqculate plots ─────────────────────────────────────────────────────────
    print("\n=== loqculate (scipy TRF, window={}) ===".format(args.sliding_window))
    new_res = _loqculate_plots(
        peptides, data, args.bootreps,
        args.std_mult, args.cv_thresh, out_dir, args.sliding_window,
    )

    # ── summary table ──────────────────────────────────────────────────────
    _print_summary(peptides, orig_res, new_res)

    # clean up the tmp original subdirectory (now empty after renames)
    try:
        out_orig.rmdir()
    except OSError:
        pass   # not empty (extra files) — leave it

    total = len(peptides)
    print(f"\nDone. {total * 3} PNGs written to: {out_dir}")
    print("  *_orig.png         → original build_plots (reference)")
    print("  *_new.png          → loqculate plot_calibration")
    print("  *_comparison.png   → loqculate Piecewise WLS + Empirical CV overlay")


if __name__ == '__main__':
    main()
